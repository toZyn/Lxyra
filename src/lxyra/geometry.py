import numpy as np

from . import CATEGORIES
from .schema import iou

STRIDE = 4


def decode(outputs, image_size, transform, threshold=0.3, top_k=200):
    logits, wh, offsets = outputs
    if hasattr(logits, "detach"):
        logits, wh, offsets = (x.detach().cpu().numpy() for x in (logits, wh, offsets))
    logits, wh, offsets = [a[0] if a.ndim == 4 else a for a in (logits, wh, offsets)]
    probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    local = np.ones_like(probs, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                local &= probs >= np.roll(np.roll(probs, dy, 1), dx, 2)
    probs = probs * local
    flat = probs.ravel()
    count = min(top_k, flat.size)
    inds = np.argpartition(flat, -count)[-count:]
    inds = sorted(inds, key=lambda i: (-float(flat[i]), int(i)))
    n_h, n_w = probs.shape[-2:]
    sx, sy, px, py = transform
    original_w, original_h = image_size
    candidates = []
    for index in inds:
        score = float(flat[index])
        if score < threshold:
            break
        cls, iy, ix = np.unravel_index(index, probs.shape)
        cx = (ix + float(offsets[0, iy, ix])) * STRIDE
        cy = (iy + float(offsets[1, iy, ix])) * STRIDE
        bw = float(np.exp(np.clip(wh[0, iy, ix], -8, 1))) * n_w * STRIDE
        bh = float(np.exp(np.clip(wh[1, iy, ix], -8, 1))) * n_h * STRIDE
        x1 = max(0, min(original_w, (cx - bw / 2 - px) / sx))
        y1 = max(0, min(original_h, (cy - bh / 2 - py) / sy))
        x2 = max(0, min(original_w, (cx + bw / 2 - px) / sx))
        y2 = max(0, min(original_h, (cy + bh / 2 - py) / sy))
        if min(x2 - x1, y2 - y1) < 1:
            continue
        candidates.append(
            {"category": CATEGORIES[cls], "bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": score}
        )
    return candidates


def nms(candidates, limit=160, overlap=0.4):
    results = []
    for item in sorted(candidates, key=lambda e: -e["confidence"]):
        if all(
            item["category"] != kept["category"] or iou(item["bbox"], kept["bbox"]) < overlap
            for kept in results
        ):
            results.append(item)
            if len(results) >= limit:
                break
    return results


def windows(image):
    w, h = image.size
    if max(w, h) / max(1, min(w, h)) < 1.6:
        return [(0, 0, w, h)]
    side = max(256, int(min(w, h) * 1.15))
    windows = []
    if h > w:
        span = min(h, side)
        start = 0
        while start + span < h:
            windows.append((0, start, w, start + span))
            start += max(1, int(span * 0.75))
        windows.append((0, h - span, w, h))
    else:
        span = min(w, side)
        start = 0
        while start + span < w:
            windows.append((start, 0, start + span, h))
            start += max(1, int(span * 0.75))
        windows.append((w - span, 0, w, h))
    return list(dict.fromkeys(windows))
