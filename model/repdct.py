from model.senet import SEBottleneck
from model.senet import RealSEModule
from lib.common import gState
from model.senet import ResNet
from model.repnet import RepSEModule
from functools import partial
import torch
import math
from torch import nn
from rich import print
from lib.comm import get_device
ordered_filter_map = None


def build_filter(pos, freq, POS):
    result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
    return result if freq == 0 else result * math.sqrt(2)


def freq_filter(u_x, v_y, out):
    dct_h, dct_w = out.shape
    u_x *= dct_h // 7
    v_y *= dct_w // 7
    for t_x in range(dct_h):
        for t_y in range(dct_w):
            out[t_x, t_y] = \
                build_filter(t_x, u_x, dct_h) *\
                build_filter(t_y, v_y, dct_w)


def build_ordered_filter_map():
    global ordered_filter_map
    ordered_filter_map = {}
    all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0,
                         0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
    all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6,
                         3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
    num_freq = 16
    mapper_x = all_top_indices_x[:num_freq]
    mapper_y = all_top_indices_y[:num_freq]
    for hw in [56, 28, 14, 7]:
        filter_ = torch.zeros(num_freq, hw, hw,
                              device=get_device())
        for i, pos in enumerate(zip(mapper_x, mapper_y)):
            freq_filter(pos[0], pos[1], filter_[i])
        ordered_filter_map[hw] = filter_


class DCTReduce(nn.Module):
    def __init__(self, channel, size):
        super().__init__()
        global ordered_filter_map
        if ordered_filter_map is None:
            build_ordered_filter_map()
        self.channel = channel
        self.size = size
        idx = []
        for j in range(16):
            idx += [j] * (self.channel // 16)
        dct_weight = ordered_filter_map[size][idx].unsqueeze(0)
        self.register_buffer("dct_weight", dct_weight, persistent=False)

    def forward(self, x: torch.Tensor):
        x = x * self.dct_weight
        x = x.sum((2, 3), keepdim=True)
        return x


class DCTSEModule(RealSEModule):
    def __init__(self, channels, reduction=16, *, size=None):
        super().__init__(channels, reduction, size=size)
        self.reduce = DCTReduce(channels, size)
        print("[blue]apply dct reduce")


class RepDCTModule(DCTSEModule, RepSEModule):
    pass


def dct50(num_classes=1000, pretrained=False):
    rep_block = partial(SEBottleneck, semodule=DCTSEModule)
    rep_block.expansion = SEBottleneck.expansion
    return ResNet(rep_block,
                  gState.args.blocks, num_classes)


def repdct50(num_classes=1000, pretrained=False):
    rep_block = partial(SEBottleneck, semodule=RepDCTModule)
    rep_block.expansion = SEBottleneck.expansion
    return ResNet(rep_block,
                  gState.args.blocks, num_classes)
