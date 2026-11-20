import math

import numpy as np
import pytest
from PIL import Image

from lxyra.detector import infer_onnx
from lxyra.geometry import decode, windows
from lxyra.preprocess import letterbox
from lxyra.schema import validate_document


def prediction(image_size, bbox, category=0, logit=12.0):
    image = Image.new("RGB", image_size, "white")
    _, (sx, sy, px, py) = letterbox(image)
    x, y, width, height = bbox
    cx = (x + width / 2) * sx + px
    cy = (y + height / 2) * sy + py
    grid_x = min(111, max(0, math.floor(cx / 4)))
    grid_y = min(111, max(0, math.floor(cy / 4)))
    logits = np.full((1, 10, 112, 112), -30.0, dtype=np.float32)
    sizes = np.zeros((1, 2, 112, 112), dtype=np.float32)
    offsets = np.zeros((1, 2, 112, 112), dtype=np.float32)
    logits[0, category, grid_y, grid_x] = logit
    sizes[0, 0, grid_y, grid_x] = math.log(width * sx / 448)
    sizes[0, 1, grid_y, grid_x] = math.log(height * sy / 448)
    offsets[0, 0, grid_y, grid_x] = cx / 4 - grid_x
    offsets[0, 1, grid_y, grid_x] = cy / 4 - grid_y
    return (logits, sizes, offsets)


@pytest.mark.parametrize(
    ("size", "bbox"),
    [
        ((448, 448), (70, 90, 120, 70)),
        ((686, 1536), (120, 630, 160, 42)),
        ((1920, 1080), (900, 400, 200, 52)),
        ((360, 2400), (70, 1800, 110, 38)),
    ],
)
def test_letterbox_inverse_produces_original_pixels(size, bbox):
    image = Image.new("RGB", size, "white")
    _, transform = letterbox(image)
    candidates = decode(prediction(size, bbox), size, transform, threshold=0.8)
    assert len(candidates) == 1
    assert candidates[0]["bbox"] == pytest.approx(bbox, abs=0.01)


@pytest.mark.parametrize("size", [(686, 1536), (1920, 1080), (360, 4000), (100, 2100)])
def test_crop_windows_cover_screen_without_gaps(size):
    image = Image.new("RGB", size, "white")
    boxes = windows(image)
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= size[0]
        assert 0 <= y1 < y2 <= size[1]
    if size[1] > size[0]:
        intervals = sorted((a[1], a[3]) for a in boxes)
    else:
        intervals = sorted((a[0], a[2]) for a in boxes)
    end = 0
    for start, stop in intervals:
        assert start <= end
        end = max(end, stop)
    assert end == max(size)
    assert len(boxes) == len(set(boxes))


class FixedLocalPrediction:
    def run(self, output_names, inputs):
        return prediction((448, 448), (160, 180, 68, 36), category=0)


def test_tiling_restores_image_coordinates_not_tile_coordinates():
    image = Image.new("RGB", (448, 1350), "white")
    tiles = windows(image)
    document = infer_onnx(image, FixedLocalPrediction(), threshold=0.5, tiled=True)
    validate_document(document)
    detected_centers = [
        (element["bbox"][0] + element["bbox"][2] / 2, element["bbox"][1] + element["bbox"][3] / 2)
        for element in document["elements"]
    ]
    for left, top, right, bottom in tiles:
        _, (sx, sy, px, py) = letterbox(Image.new("RGB", (right - left, bottom - top)))
        x = left + (160 + 68 / 2 - px) / sx
        y = top + (180 + 36 / 2 - py) / sy
        assert any(abs(cx - x) < 2 and abs(cy - y) < 2 for cx, cy in detected_centers)
