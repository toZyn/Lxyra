from torch import nn
from torch.nn import functional as F

from .detector_model import conv


CROP_SIZE = 96


class TextRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv(3, 16, 2),
            conv(16, 24, 2),
            conv(24, 40, 2),
            conv(40, 48, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48, 64), nn.SiLU(), nn.Linear(64, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images):
        return self.head(self.features(images))


def roi_loss(output, delta, quality, pos):
    weight = pos.float()
    reg = (
        F.smooth_l1_loss(output[:, :4], delta, reduction="none", beta=0.2).sum(dim=1) * weight
    ).sum() / weight.sum().clamp(min=1)
    obj = F.binary_cross_entropy_with_logits(output[:, 4], quality.float())
    return reg * 2 + obj, reg.detach(), obj.detach()
