import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .. import CATEGORIES, IMAGE_SIZE, STRIDE
from ..preprocess import letterbox


def validate_record(record):
    if not isinstance(record.get("group"), str) or not record["group"]:
        raise ValueError("Every screen needs a nonempty group for leakage-safe splits")
    if not isinstance(record.get("image"), str) or not record["image"]:
        raise ValueError("Image path is missing")
    anns = record.get("annotations")
    if not isinstance(anns, list):
        raise ValueError("annotations must be a list")
    parents = []
    for idx, ann in enumerate(anns):
        if ann.get("category") not in CATEGORIES:
            raise ValueError(f"Unknown category at {idx}: {ann.get('category')}")
        box = ann.get("bbox")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box)
        ):
            raise ValueError(f"Invalid bbox at {idx}")
        if min(box[2:]) <= 0 or min(box[:2]) < 0:
            raise ValueError(f"Negative coordinates or non-positive size at {idx}")
        parent = ann.get("parent")
        if parent is not None and (
            type(parent) is not int or not 0 <= parent < len(anns) or parent == idx
        ):
            raise ValueError(f"Invalid parent at {idx}")
        parents.append(parent)
    for i in range(len(anns)):
        seen = set()
        j = i
        while j is not None:
            if j in seen:
                raise ValueError(f"Cyclic hierarchy at {i}")
            seen.add(j)
            j = parents[j]


def make_targets(annotations, transform, size=IMAGE_SIZE, stride=STRIDE):
    n = size // stride
    heat = torch.zeros((len(CATEGORIES), n, n))
    boxes = torch.zeros((2, n, n))
    offsets = torch.zeros((2, n, n))
    positive = torch.zeros((1, n, n))
    sx, sy, pad_x, pad_y = transform
    selected = {}
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x, y, w, h = x * sx + pad_x, y * sy + pad_y, w * sx, h * sy
        cx, cy = (x + w / 2) / stride, (y + h / 2) / stride
        ix, iy = int(cx), int(cy)
        if not 0 <= ix < n or not 0 <= iy < n:
            continue
        area = w * h
        if (iy, ix) in selected and selected[(iy, ix)] <= area:
            continue
        selected[(iy, ix)] = area
        heat[:, iy, ix] = 0
        heat[CATEGORIES.index(ann["category"]), iy, ix] = 1
        boxes[:, iy, ix] = torch.tensor([w / size, h / size]).clamp(0, 1)
        offsets[:, iy, ix] = torch.tensor([cx - ix, cy - iy])
        positive[:, iy, ix] = 1
    return heat, boxes, offsets, positive


class ScreenDataset(Dataset):
    def __init__(self, root, split, size=IMAGE_SIZE, augment=False):
        self.root = Path(root)
        self.size = size
        self.augment = augment
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing dataset manifest: {manifest}")
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        for record in records:
            validate_record(record)
        seen = {}
        for record in records:
            group = record["group"]
            if group in seen and seen[group] != record["split"]:
                raise ValueError(f"Group leakage: {group}")
            seen[group] = record["split"]
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"No records for {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = ImageOps.exif_transpose(Image.open(self.root / record["image"])).convert("RGB")
        width, height = image.size
        for ann in record["annotations"]:
            x, y, w, h = ann["bbox"]
            if x + w > width + 1 or y + h > height + 1:
                raise ValueError(f"Bounding box outside {record['image']}")
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image, transform = letterbox(image, self.size)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        targets = make_targets(record["annotations"], transform, self.size)
        return torch.from_numpy(pixels), targets
