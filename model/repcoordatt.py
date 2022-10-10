import torch
import torch.nn as nn
import torch.nn.functional as F

from model.senet import SEBottleneck, ResNet
from model.repnet import Conv2dReparameterization
from model.AdaptiveBS import AdaptiveShapingModule

from lib.common import gState

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32, use_rep=False):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.use_rep = use_rep

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.shaping_h = None
        self.shaping_w = None

        if self.use_rep:
            self.conv1 = Conv2dReparameterization.from_conv2d(
                self.conv1,
                prior_variance=gState.args.pri_sigma,
                posterior_rho_init=gState.args.rho_init
            )
            self.conv_h = Conv2dReparameterization.from_conv2d(
                self.conv_h,
                prior_variance=gState.args.pri_sigma,
                posterior_rho_init=gState.args.rho_init
            )
            self.conv_w = Conv2dReparameterization.from_conv2d(
                self.conv_w,
                prior_variance=gState.args.pri_sigma,
                posterior_rho_init=gState.args.rho_init
            )
            self.shaping_h = AdaptiveShapingModule(
                0, sp_arg=gState.args.sp_arg, mean_loss=gState.args.sp_mean
            )
            self.shaping_w = AdaptiveShapingModule(
                0, sp_arg=gState.args.sp_arg, mean_loss=gState.args.sp_mean
            )

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)

        # Simply add loss for a_h and a_w
        if self.use_rep and gState.args.sp_pos == 0:
            self.sp_loss_value = self.shaping_h(a_h) + self.shaping_w(a_w)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        a_h = a_h.sigmoid()
        a_w = a_w.sigmoid()

        if self.use_rep and gState.args.sp_pos == 1:
            self.sp_loss_value = self.shaping_h(a_h) + self.shaping_w(a_w)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        out = identity * a_w * a_h

        return out

class CoordAttBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = CoordAtt(args[1] * self.expansion, args[1] * self.expansion, use_rep=False)

class CoordAttRepBottleneck(SEBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, semodule=None)
        self.se_module = CoordAtt(args[1] * self.expansion, args[1] * self.expansion, use_rep=True)


def coordatt_resnet50(num_classes=1000, pretrained=False):
    return ResNet(CoordAttBottleneck, gState.args.blocks, num_classes)

def coordatt_repnet50(num_classes=1000, pretrained=False):
    return ResNet(CoordAttRepBottleneck, gState.args.blocks, num_classes)