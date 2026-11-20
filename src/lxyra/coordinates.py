import math

import numpy as np
from PIL import Image

CROP_SIZE = 96


def crop_rect(image_size, box, scale=1.8):
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    aw = max(36.0, w * scale)
    ah = max(30.0, h * scale)
    return (cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2)


def make_crop(image, box):
    rect = crop_rect(image.size, box)
    crop = image.crop(rect).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.uint8).copy()


def get_delta(before, truth):
    x, y, w, h = before
    gx, gy, gw, gh = truth
    return np.array(
        [
            ((gx + gw / 2) - (x + w / 2)) / max(w, 1),
            ((gy + gh / 2) - (y + h / 2)) / max(h, 1),
            math.log(max(gw, 1) / max(w, 1)),
            math.log(max(gh, 1) / max(h, 1)),
        ],
        dtype=np.float32,
    )


def change_box(box, delta, alpha, image_size):
    x, y, w, h = map(float, box)
    cx = x + w / 2 + float(delta[0]) * w * alpha
    cy = y + h / 2 + float(delta[1]) * h * alpha
    nw = w * math.exp(max(-1, min(1, float(delta[2]) * alpha)))
    nh = h * math.exp(max(-1, min(1, float(delta[3]) * alpha)))
    iw, ih = image_size
    x1 = max(0, min(iw, cx - nw / 2))
    y1 = max(0, min(ih, cy - nh / 2))
    x2 = max(0, min(iw, cx + nw / 2))
    y2 = max(0, min(ih, cy + nh / 2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return box
    return [x1, y1, x2 - x1, y2 - y1]
