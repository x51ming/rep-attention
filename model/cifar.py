
from torch import nn
import math


class CResNet(nn.Module):

    @staticmethod
    def LuaConv(a, b, c, d, e, f, g, h):
        """https://github.com/facebookarchive/fb.resnet.torch/blob/master/models/resnet.lua"""
        return nn.Conv2d(in_channels=a,
                         out_channels=b,
                         kernel_size=(c, d),
                         stride=(e, f),
                         padding=(g, h))

    @staticmethod
    def LuaAvgPool(a, b, c, d):
        return nn.AvgPool2d((a, b), (c, d))

    def __init__(self, depth, block, num_classes=1000):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model

        # if issubclass(block, BasicBlock):
        #     assert (depth - 2) % 6 == 0
        #     n = (depth - 2) // 6
        # elif issubclass(block, Bottleneck):
        assert (depth - 2) % 9 == 0
        n = (depth - 2) // 9
        # else:
        #     raise ValueError("depth is not suitable")
        """
        -- The ResNet CIFAR-100 model
            model:add(Convolution(3,16,3,3,1,1,1,1))
            model:add(SBatchNorm(16))
            model:add(ReLU(true))
            model:add(layer(basicblock, 16, n))
            model:add(layer(basicblock, 32, n, 2))
            model:add(layer(basicblock, 64, n, 2))
            model:add(Avg(8, 8, 1, 1))
            model:add(nn.View(64):setNumInputDims(3))
            model:add(nn.Linear(64, 100))
        """
        self.inplanes = 16
        self.conv1 = self.LuaConv(3, 16, 3, 3, 1, 1, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = self.LuaAvgPool(8, 8, 1, 1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or \
                self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def resnet110b(num_classes=100):
    from torchvision.models.resnet import Bottleneck
    return CResNet(110, Bottleneck, num_classes)


def senet110b(num_classes=100):
    from model.senet import SEBottleneck
    return CResNet(110, SEBottleneck, num_classes)


def repnet110b(num_classes=100):
    from model.senet import SEBottleneck
    from model.repnet import RepSEModule
    from functools import partial
    rep_block = partial(SEBottleneck, semodule=RepSEModule)
    rep_block.expansion = SEBottleneck.expansion
    return CResNet(110, rep_block, num_classes)
