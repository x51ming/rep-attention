# Modified
# original: https://github.com/cuihu1998/GENet-Res50
from model.repnet import RepSEModule
from lib.common import TrainArg
from model.senet import SEBottleneck, ResNet
from functools import partial
from lib.common import gState
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.senet import RealSEModule


class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Downblock, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=2,
                                kernel_size=kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.dwconv(x))


class GEBlock(RealSEModule):
    def __init__(self, channels, reduction=16, *, size=None, extra_params=True, extent=4):
        # extent: 2,4,8
        super().__init__(channels, reduction=reduction, size=size)
        extra_params = gState.args.extra_params
        spatial = size
        if extra_params:
            if extent:
                modules = [Downblock(channels)]
            for i in range((extent-1) // 2):
                modules.append(nn.Sequential(
                    nn.ReLU(inplace=True), Downblock(channels)))
            self.downop = nn.Sequential(
                *modules) if extent else Downblock(channels, kernel_size=spatial)
        else:
            self.downop = nn.AdaptiveAvgPool2d(
                spatial // extent) if extent else nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        map = self.downop(x)
        map = self.fc1(map)
        map = self.relu(map)
        map = self.fc2(map)
        map = F.interpolate(map, x.shape[-1])
        map = self.sigmoid(map)
        return x * map


class RepGEBlock(GEBlock, RepSEModule):
    def forward(self, x: torch.Tensor):
        args: TrainArg = gState.args
        module_input = x
        x = self.downop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.interpolate(x, module_input.shape[-1])

        if args.sp_pos == 0:
            self.sp_loss_value = self.shaping(x)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        x = self.sigmoid(x)
        if args.sp_pos == 1:
            self.sp_loss_value = self.shaping(x - 0.5)
            gState.external_loss = gState.external_loss + \
                self.sp_loss_value * gState.args.stdscale
            self.sp_loss_value = self.sp_loss_value.item()

        return module_input * x


def genet50(num_classes=100):
    rep_block = partial(SEBottleneck, semodule=GEBlock)
    rep_block.expansion = SEBottleneck.expansion
    return ResNet(rep_block,
                  gState.args.blocks, num_classes)


def repge50(num_classes=100):
    rep_block = partial(SEBottleneck, semodule=RepGEBlock)
    rep_block.expansion = SEBottleneck.expansion
    return ResNet(rep_block,
                  gState.args.blocks, num_classes)
