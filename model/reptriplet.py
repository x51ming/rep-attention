import torch
import torch.nn as nn

from model.senet import SEBottleneck, ResNet
from model.repnet import Conv2dReparameterization

from lib.common import gState


class TripletRepAttentionBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = TripletAttention(args[1] * self.expansion, use_rep=True)


class TripletAttentionBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = TripletAttention(args[1] * self.expansion, use_rep=False)


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
        use_rep=False
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_rep = use_rep
        if use_rep:
            self.conv = Conv2dReparameterization.from_conv2d(
                self.conv,
                prior_variance=gState.args.pri_sigma,
                posterior_rho_init=gState.args.rho_init
            )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self, use_rep=False):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, use_rep=use_rep
        )
        self.use_rep = use_rep
        self.shaping = None
        from model.AdaptiveBS import AdaptiveShapingModule
        if self.use_rep:
            self.shaping = AdaptiveShapingModule(
                0, sp_arg=gState.args.sp_arg, mean_loss=gState.args.sp_mean
            )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        if self.use_rep and gState.args.sp_pos == 0:
            self.sp_loss_value = self.shaping(x_out)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        scale = torch.sigmoid_(x_out)
        if self.use_rep and gState.args.sp_pos == 1:
            self.sp_loss_value = self.shaping(scale)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
        use_rep=False
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate(use_rep=use_rep)
        self.ChannelGateW = SpatialGate(use_rep=use_rep)
        self.no_spatial = no_spatial

        if not no_spatial:
            self.SpatialGate = SpatialGate(use_rep=use_rep)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


def triplet_resnet50(num_classes=1000, pretrained=False):
    return ResNet(TripletAttentionBottleneck, gState.args.blocks, num_classes)


def triplet_repnet50(num_classes=1000, pretrained=False):
    return ResNet(TripletRepAttentionBottleneck, gState.args.blocks, num_classes)
