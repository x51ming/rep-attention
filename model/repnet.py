from functools import partial
from lib.common import TrainArg
from model.senet import RealSEModule, SEBottleneck, ResNet
from torch import nn
import torch
from lib.common import gState
from rich import print


def reparameterize(
        self,
        prior_mean=0.,
        prior_variance=1.,
        posterior_mu_init=0.,
        posterior_rho_init=-3.):

    self.prior_mean = prior_mean
    self.prior_variance = prior_variance
    self.posterior_mu_init = posterior_mu_init
    self.posterior_rho_init = posterior_rho_init

    self.rho_weight = \
        nn.Parameter(self.posterior_rho_init +
                     0.1 * torch.randn_like(self.weight))
    self.register_buffer('prior_weight_mu',
                         torch.tensor(self.prior_mean), persistent=False)
    self.register_buffer('prior_weight_sigma',
                         torch.tensor(self.prior_variance), persistent=False)

    if self.bias is not None:
        self.rho_bias = \
            nn.Parameter(self.posterior_rho_init +
                         0.1 * torch.randn_like(self.bias))
        self.register_buffer('prior_bias_mu',
                             torch.tensor(self.prior_mean), persistent=False)
        self.register_buffer('prior_bias_sigma',
                             torch.tensor(self.prior_variance), persistent=False)


class Conv2dReparameterization(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 *,
                 prior_mean=0.,
                 prior_variance=1.,
                 posterior_mu_init=0.,
                 posterior_rho_init=-3.):
        nn.Conv2d.__init__(self,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias,
                           padding_mode=padding_mode)

        reparameterize(self,
                       prior_mean,
                       prior_variance,
                       posterior_mu_init,
                       posterior_rho_init)

    def forward(self, input):
        if self.training:
            output = self.bayesian_forward(input, n=1)
        else:
            output = self.bayesian_forward(input, n=0)
        return output

    def bayesian_forward(self, input, n=1):
        if n == 0:
            return self.deterministic_forward(input)
        sigma_weight = torch.exp(self.rho_weight)
        if self.bias is not None:
            # sigma_bias = F.softplus(self.rho_bias)
            sigma_bias = torch.exp(self.rho_bias)
        channels = self.weight.shape[0]
        eps = torch.randn(n, *sigma_weight.shape,
                          device=sigma_weight.device,
                          dtype=sigma_weight.dtype
                          )
        weight = self.weight.unsqueeze(0) + \
            sigma_weight.unsqueeze(0) * eps * gState.args.sscale
        weight = weight.flatten(0, 1)

        if self.bias is not None:
            eps = torch.randn(n, *sigma_bias.shape,
                              device=sigma_bias.device,
                              dtype=sigma_bias.dtype)
            bias = self.bias.unsqueeze(0) + \
                sigma_bias.unsqueeze(0) * eps * gState.args.sscale
            bias = bias.flatten(0, 1)
        else:
            bias = None

        output = self._conv_forward(input, weight, bias)
        output = output.unflatten(1, (n, channels))
        output = output.mean(1, keepdim=False)
        return output

    def deterministic_forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)

    @classmethod
    def from_conv2d(cls, obj: nn.Conv2d, *,
                    prior_mean=0.,
                    prior_variance=1.,
                    posterior_mu_init=0.,
                    posterior_rho_init=-3.):
        assert isinstance(obj, nn.Conv2d)
        return cls(
            in_channels=obj.in_channels,
            out_channels=obj.out_channels,
            kernel_size=obj.kernel_size,
            stride=obj.stride,
            padding=obj.padding,
            dilation=obj.dilation,
            groups=obj.groups,
            bias=obj.bias is not None,
            padding_mode=obj.padding_mode,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )


class RepSEModule(RealSEModule):
    def __init__(self, channels, reduction=16, *, size=None):
        super().__init__(channels, reduction, size=size)
        self.sp_loss_value = 0
        args: TrainArg = gState.args
        if self.block_id in args.full_skip:
            return
        if args.sp_pos in [0, 1]:
            from model.AdaptiveBS import AdaptiveShapingModule
            self.shaping = AdaptiveShapingModule(
                0, sp_arg=args.sp_arg, mean_loss=args.sp_mean)
        if self.block_id in args.skip_layers:
            return
        self.fc1 = Conv2dReparameterization.from_conv2d(
            self.fc1, prior_variance=args.pri_sigma, posterior_rho_init=args.rho_init)
        self.fc2 = Conv2dReparameterization.from_conv2d(
            self.fc2, prior_variance=args.pri_sigma, posterior_rho_init=args.rho_init)
        print("[red]apply reparameterize {}".format(self.block_id))

    def forward(self, x: torch.Tensor):
        args: TrainArg = gState.args

        module_input = x
        x = self.reduce(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if args.sp_pos == 0 and self.block_id not in args.full_skip:
            self.sp_loss_value = self.shaping(x)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        x = self.sigmoid(x) * gState.args.sigscale
        if args.sp_pos == 1 and self.block_id not in args.full_skip:
            self.sp_loss_value = self.shaping(x - 0.5)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        return x * module_input


def repnet50(num_classes=1000, pretrained=False):
    rep_block = partial(SEBottleneck, semodule=RepSEModule)
    rep_block.expansion = SEBottleneck.expansion
    return ResNet(rep_block,
                  gState.args.blocks, num_classes)
