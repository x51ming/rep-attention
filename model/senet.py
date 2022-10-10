from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from collections import defaultdict
from torch import nn
from functools import partial
import torch
from rich import print
from lib.common import gState


class RealSEModule(nn.Module):
    """self.reduce = partial(torch.mean, dim=(2, 3), keepdim=True)"""
    block_id = 0

    def __init__(self, channels, reduction=16, *, size=None):
        super().__init__()

        self.block_id = RealSEModule.block_id
        RealSEModule.block_id += 1

        self.channels = channels
        self.reduction = reduction
        self.size = size
        self.fc1 = nn.Conv2d(
            channels, channels // reduction,
            kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels,
            kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.reduce = partial(torch.mean, dim=(2, 3), keepdim=True)
        print("[blue]RealSEModule {} is ready".format(self.block_id))

    def forward(self, x: torch.Tensor):
        module_input = x
        x = self.reduce(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * gState.args.sigscale
        return module_input * x


class SEBottleneck(Bottleneck):
    GLOBAL_BLOCK_ID = 0

    def __init__(self, *args, semodule=RealSEModule, **kwargs, ):
        reduction = kwargs.pop("reduction", 16)
        super().__init__(*args, **kwargs)
        self.block_id = SEBottleneck.GLOBAL_BLOCK_ID
        SEBottleneck.GLOBAL_BLOCK_ID += 1
        planes = args[1]
        c2wh = defaultdict(int)
        c2wh.update([(64, 56), (128, 28), (256, 14), (512, 7)])
        if semodule is None:
            print("[red]ignore semodule, semodule will be created by a subclass")
        else:
            self.se_module = semodule(
                planes * self.expansion, reduction, size=c2wh[planes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_module(out) + identity
        out = self.relu(out)
        return out


def senet50(num_classes=1000, pretrained=False):
    return ResNet(SEBottleneck, gState.args.blocks, num_classes)
