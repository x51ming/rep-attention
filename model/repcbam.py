import torch
import torch.nn as nn
import torch.nn.functional as F

from model.senet import SEBottleneck, ResNet
from model.repnet import Conv2dReparameterization
from model.AdaptiveBS import AdaptiveShapingModule

from lib.common import gState

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, use_rep=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.use_rep = use_rep
        if self.use_rep:
            self.conv = Conv2dReparameterization.from_conv2d(
                self.conv,
                prior_variance=gState.args.pri_sigma,
                posterior_rho_init=gState.args.rho_init
            )
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], use_rep=False):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.use_rep = use_rep
        if self.use_rep:
            self.mlp = nn.Sequential(
                # Flatten(),
                Conv2dReparameterization.from_conv2d(
                    nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1),
                    prior_variance=gState.args.pri_sigma,
                    posterior_rho_init=gState.args.rho_init
                ),
                nn.ReLU(),
                Conv2dReparameterization.from_conv2d(
                    nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1),
                    prior_variance=gState.args.pri_sigma,
                    posterior_rho_init=gState.args.rho_init
                ),
            )
            self.shaping = AdaptiveShapingModule(
                0, sp_arg=gState.args.sp_arg, mean_loss=gState.args.sp_mean
            )
        else:
            self.mlp = nn.Sequential(
                #　Flatten(),
                nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1),
                nn.ReLU(),
                nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1)
            )
            self.shaping = None
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        if self.use_rep and gState.args.sp_pos == 0:
            self.sp_loss_value = self.shaping(channel_att_sum)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        scale = F.sigmoid( channel_att_sum ).expand_as(x)
        
        if self.use_rep and gState.args.sp_pos == 1:
            self.sp_loss_value = self.shaping(scale)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, use_rep=False):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False, use_rep=use_rep)
        self.use_rep = use_rep

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

        scale = F.sigmoid(x_out) # broadcasting

        if self.use_rep and gState.args.sp_pos == 1:
            self.sp_loss_value = self.shaping(scale)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, spatial_rep=False, channel_rep=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, use_rep=channel_rep)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(use_rep=spatial_rep)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAMBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = CBAM(args[1] * self.expansion, 16, spatial_rep=False, channel_rep=False)

class CBAMRepBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = CBAM(args[1] * self.expansion, 16, spatial_rep=True, channel_rep=True)

class CBAMChannelBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = CBAM(args[1] * self.expansion, 16, spatial_rep=False, channel_rep=True)

class CBAMSpatialBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = CBAM(args[1] * self.expansion, 16, spatial_rep=True, channel_rep=False)

def cbam_resnet50(num_classes=1000, pretrained=False):
    return ResNet(CBAMBottleneck, gState.args.blocks, num_classes)

def cbam_repnet50(num_classes=1000, pretrained=False):
    return ResNet(CBAMRepBottleneck, gState.args.blocks, num_classes)

def cbam_repnet50_channel(num_classes=1000, pretrained=False):
    return ResNet(CBAMChannelBottleneck, gState.args.blocks, num_classes)

def cbam_repnet50_spatial(num_classes=1000, pretrained=False):
    return ResNet(CBAMSpatialBottleneck, gState.args.blocks, num_classes)