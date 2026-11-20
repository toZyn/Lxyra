import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .. import CATEGORIES
from .datasets import ScreenDataset
from ..preprocess import letterbox


SIZE = 448
STRIDE = 4


def conv(a, b, stride=1):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(b),
        nn.SiLU(inplace=True),
    )


class LxyraV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2 = nn.Sequential(conv(3, 24, 2), conv(24, 24))
        self.s4 = nn.Sequential(conv(24, 48, 2), conv(48, 48), conv(48, 48))
        self.s8 = nn.Sequential(conv(48, 80, 2), conv(80, 80), conv(80, 80))
        self.s16 = nn.Sequential(conv(80, 112, 2), conv(112, 112))
        self.s32 = nn.Sequential(conv(112, 144, 2), conv(144, 144))
        self.p4 = nn.Conv2d(48, 48, 1)
        self.p8 = nn.Conv2d(80, 48, 1)
        self.p16 = nn.Conv2d(112, 48, 1)
        self.p32 = nn.Conv2d(144, 48, 1)
        self.mix = nn.Sequential(conv(48, 64), conv(64, 64))
        self.cls = nn.Conv2d(64, len(CATEGORIES), 1)
        self.size = nn.Conv2d(64, 2, 1)
        self.center = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.cls.bias, -3.0)
        nn.init.constant_(self.size.bias, math.log(0.06))

    def forward(self, x):
        s4 = self.s4(self.s2(x))
        s8 = self.s8(s4)
        s16 = self.s16(s8)
        s32 = self.s32(s16)
        p16 = self.p16(s16) + F.interpolate(self.p32(s32), size=s16.shape[-2:], mode="nearest")
        p8 = self.p8(s8) + F.interpolate(p16, size=s8.shape[-2:], mode="nearest")
        p4 = self.p4(s4) + F.interpolate(p8, size=s4.shape[-2:], mode="nearest")
        f = self.mix(p4)
        return self.cls(f), self.size(f), self.center(f).sigmoid()


def target(anns, transform, size=SIZE):
    n = size // STRIDE
    heat = torch.zeros((len(CATEGORIES), n, n), dtype=torch.float32)
    wh = torch.zeros((2, n, n), dtype=torch.float32)
    offsets = torch.zeros((2, n, n), dtype=torch.float32)
    mask = torch.zeros((1, n, n), dtype=torch.float32)
    selected = {}
    sx, sy, px, py = transform
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x, y, w, h = x * sx + px, y * sy + py, w * sx, h * sy
        if min(w, h) < 1.8:
            continue
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        ix, iy = int(cx), int(cy)
        if ix < 0 or iy < 0 or ix >= n or iy >= n:
            continue
        category = CATEGORIES.index(ann["category"])
        radius = min(5, max(1, int(math.sqrt(w * h) / 26)))
        x0, y0, x1, y1 = (
            max(0, ix - radius),
            max(0, iy - radius),
            min(n, ix + radius + 1),
            min(n, iy + radius + 1),
        )
        xx, yy = torch.meshgrid(torch.arange(x0, x1), torch.arange(y0, y1), indexing="xy")
        gaussian = torch.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * max(0.6, radius / 2) ** 2))
        heat[category, y0:y1, x0:x1] = torch.maximum(heat[category, y0:y1, x0:x1], gaussian)
        area = w * h
        if (iy, ix) in selected and selected[(iy, ix)] <= area:
            continue
        selected[(iy, ix)] = area
        wh[:, iy, ix] = torch.tensor([math.log(max(w / size, 1e-5)), math.log(max(h / size, 1e-5))])
        offsets[:, iy, ix] = torch.tensor([cx - ix, cy - iy])
        mask[:, iy, ix] = 1
    return heat, wh, offsets, mask


def loss(outputs, targets):
    logits, wh, offsets = outputs
    heat, true_wh, true_offsets, mask = targets
    score = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = heat.eq(1)
    negative = heat.lt(1)
    weights = (1 - heat).pow(4)
    focal = (
        -torch.log(score) * (1 - score).pow(2) * positive
        - torch.log(1 - score) * score.pow(2) * weights * negative
    )
    divisor = mask.sum().clamp(min=1)
    classification = focal.sum() / divisor
    box_loss = (F.smooth_l1_loss(wh, true_wh, reduction="none", beta=0.2) * mask).sum() / divisor
    offset_loss = (
        F.smooth_l1_loss(offsets, true_offsets, reduction="none", beta=0.1) * mask
    ).sum() / divisor
    return classification + 2 * box_loss + offset_loss, (
        float(classification.detach()),
        float(box_loss.detach()),
        float(offset_loss.detach()),
    )


def crop_image(image, anns):
    w, h = image.size
    if random.random() > 0.65:
        return image, anns
    cw = max(96, int(w * random.uniform(0.65, 1)))
    ch = max(96, int(h * random.uniform(0.48, 0.90)))
    cw = min(cw, w)
    ch = min(ch, h)
    x0 = random.randint(0, w - cw)
    y0 = random.randint(0, h - ch)
    out = []
    for ann in anns:
        x, y, aw, ah = ann["bbox"]
        ix0, iy0, ix1, iy1 = max(x, x0), max(y, y0), min(x + aw, x0 + cw), min(y + ah, y0 + ch)
        visible = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if visible >= 0.65 * aw * ah and visible >= 6:
            out.append({**ann, "bbox": [ix0 - x0, iy0 - y0, ix1 - ix0, iy1 - iy0], "parent": None})
    return image.crop((x0, y0, x0 + cw, y0 + ch)), out


class V2Dataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.base = ScreenDataset(root, split)
        self.root = Path(root)
        self.records = self.base.records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = ImageOps.exif_transpose(Image.open(self.root / r["image"])).convert("RGB")
        anns = r["annotations"]
        if self.augment:
            image, anns = crop_image(image, anns)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.08:
                image = ImageOps.grayscale(image).convert("RGB")
        image, transform = letterbox(image, SIZE)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255
        return torch.from_numpy(array), target(anns, transform)
