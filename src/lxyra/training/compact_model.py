import math

from torch import nn
from torch.nn import functional as F

from .. import CATEGORIES


def _block(inputs, outputs, stride=1):
    return nn.Sequential(
        nn.Conv2d(inputs, inputs, 3, stride, 1, groups=inputs, bias=False),
        nn.BatchNorm2d(inputs),
        nn.SiLU(inplace=True),
        nn.Conv2d(inputs, outputs, 1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.SiLU(inplace=True),
    )


def _stem(inputs, outputs, stride=1):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.SiLU(inplace=True),
    )


class LxyraCompact(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2 = nn.Sequential(_stem(3, 16, 2), _block(16, 16))
        self.s4 = nn.Sequential(_block(16, 24, 2), _block(24, 24))
        self.s8 = nn.Sequential(_block(24, 40, 2), _block(40, 40))
        self.s16 = nn.Sequential(_block(40, 64, 2), _block(64, 64))
        self.s32 = nn.Sequential(_block(64, 96, 2), _block(96, 96))
        self.p4 = nn.Conv2d(24, 32, 1)
        self.p8 = nn.Conv2d(40, 32, 1)
        self.p16 = nn.Conv2d(64, 32, 1)
        self.p32 = nn.Conv2d(96, 32, 1)
        self.mix = nn.Sequential(_block(32, 48), _block(48, 48))
        self.cls = nn.Conv2d(48, len(CATEGORIES), 1)
        self.size = nn.Conv2d(48, 2, 1)
        self.center = nn.Conv2d(48, 2, 1)
        nn.init.constant_(self.cls.bias, -3.0)
        nn.init.constant_(self.size.bias, math.log(0.06))

    def forward(self, image):
        s4 = self.s4(self.s2(image))
        s8 = self.s8(s4)
        s16 = self.s16(s8)
        s32 = self.s32(s16)
        p16 = self.p16(s16) + F.interpolate(
            self.p32(s32), size=s16.shape[-2:], mode="nearest"
        )
        p8 = self.p8(s8) + F.interpolate(p16, size=s8.shape[-2:], mode="nearest")
        p4 = self.p4(s4) + F.interpolate(p8, size=s4.shape[-2:], mode="nearest")
        features = self.mix(p4)
        return self.cls(features), self.size(features), self.center(features).sigmoid()
