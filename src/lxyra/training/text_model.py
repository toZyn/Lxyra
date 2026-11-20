import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .datasets import ScreenDataset
from ..preprocess import letterbox


SIZE = 448
STRIDE = 4


def block(a, b, stride=1):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(b),
        nn.SiLU(inplace=True),
    )


class TextDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2 = nn.Sequential(block(3, 16, 2), block(16, 16))
        self.s4 = nn.Sequential(block(16, 32, 2), block(32, 32))
        self.s8 = nn.Sequential(block(32, 48, 2), block(48, 48))
        self.s16 = nn.Sequential(block(48, 64, 2), block(64, 64))
        self.p4 = nn.Conv2d(32, 32, 1)
        self.p8 = nn.Conv2d(48, 32, 1)
        self.p16 = nn.Conv2d(64, 32, 1)
        self.out = nn.Sequential(block(32, 32), nn.Conv2d(32, 1, 1))

    def forward(self, image):
        a = self.s4(self.s2(image))
        b = self.s8(a)
        c = self.s16(b)
        p = self.p4(a) + F.interpolate(
            self.p8(b) + F.interpolate(self.p16(c), size=b.shape[-2:], mode="nearest"),
            size=a.shape[-2:],
            mode="nearest",
        )
        return self.out(p)


def text_mask(annotations, transform, input_size=SIZE):
    mask = Image.new("L", (input_size // STRIDE, input_size // STRIDE), 0)
    draw = ImageDraw.Draw(mask)
    sx, sy, px, py = transform
    for ann in annotations:
        if ann["category"] != "text":
            continue
        x, y, w, h = ann["bbox"]
        x1 = max(0, min(input_size // STRIDE, int((x * sx + px) / STRIDE)))
        y1 = max(0, min(input_size // STRIDE, int((y * sy + py) / STRIDE)))
        x2 = max(0, min(input_size // STRIDE, round(((x + w) * sx + px) / STRIDE)))
        y2 = max(0, min(input_size // STRIDE, round(((y + h) * sy + py) / STRIDE)))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    return torch.from_numpy(np.asarray(mask, dtype=np.float32)[None].copy() / 255)


def text_loss(logits, mask):
    p = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(
        logits, mask, pos_weight=torch.tensor(3.0, device=logits.device)
    )
    intersection = (p * mask).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (union + 1)).mean()
    return bce + dice


class TextData(Dataset):
    def __init__(self, root, split, augment):
        self.root = Path(root)
        self.dataset = ScreenDataset(root, split)
        self.records = [
            record
            for record in self.dataset.records
            if sum(a["category"] == "text" for a in record["annotations"]) >= 2
        ]
        if not self.records:
            raise ValueError("No annotated text examples")
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = ImageOps.exif_transpose(Image.open(self.root / record["image"])).convert("RGB")
        annotations = record["annotations"]
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))
            if random.random() < 0.65:
                w, h = image.size
                if min(w, h) > 200:
                    crop_w = min(w, max(180, int(w * random.uniform(0.65, 1.0))))
                    crop_h = min(h, max(180, int(h * random.uniform(0.48, 0.95))))
                    x = random.randrange(w - crop_w + 1)
                    y = random.randrange(h - crop_h + 1)
                    region = (x, y, x + crop_w, y + crop_h)
                    kept = []
                    for ann in annotations:
                        ax, ay, aw, ah = ann["bbox"]
                        bx1 = max(ax, x)
                        by1 = max(ay, y)
                        bx2 = min(ax + aw, x + crop_w)
                        by2 = min(ay + ah, y + crop_h)
                        if (bx2 - bx1) * (by2 - by1) >= 0.70 * aw * ah:
                            kept.append({**ann, "bbox": [bx1 - x, by1 - y, bx2 - bx1, by2 - by1]})
                    image = image.crop(region)
                    annotations = kept
        canvas, transform = letterbox(image, SIZE)
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255
        return torch.from_numpy(array), text_mask(annotations, transform)
