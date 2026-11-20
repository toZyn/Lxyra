import numpy as np
from PIL import Image

from lxyra.detector import _visible_within_tile, infer_onnx
from lxyra.geometry import decode


def test_internal_seam_rejects_partial_objects_but_not_screen_edges():
    image_size = (400, 1200)
    rect = (400, 460)
    at_top = [140, 1, 80, 35]
    assert _visible_within_tile(at_top, (0, 0), rect, image_size)
    assert not _visible_within_tile(at_top, (0, 240), rect, image_size)
    at_bottom = [140, 430, 80, 30]
    assert not _visible_within_tile(at_bottom, (0, 240), rect, image_size)
    at_middle = [140, 100, 80, 35]
    assert _visible_within_tile(at_middle, (0, 240), rect, image_size)


def test_nonwrapping_peak_detection_does_not_connect_opposite_screen_edges():
    logits = np.full((1, 10, 112, 112), -30.0, dtype=np.float32)
    size = np.full((1, 2, 112, 112), np.log(0.06), dtype=np.float32)
    offsets = np.full((1, 2, 112, 112), 0.5, dtype=np.float32)
    logits[0, 0, 56, 0] = 11.0
    logits[0, 0, 56, 111] = 12.0
    output = (logits, size, offsets)
    wrapped = decode(output, (448, 448), (1, 1, 0, 0), 0.8)
    proper = decode(output, (448, 448), (1, 1, 0, 0), 0.8, nonwrapping_peaks=True)
    assert len(wrapped) == 1
    assert len(proper) == 2


class NearTop:
    def run(self, outputs, inputs):
        logits = np.full((1, 10, 112, 112), -30.0, dtype=np.float32)
        size = np.full((1, 2, 112, 112), np.log(0.08), dtype=np.float32)
        offsets = np.full((1, 2, 112, 112), 0.5, dtype=np.float32)
        logits[0, 0, 2, 50] = 13
        return logits, size, offsets


def test_guard_is_opt_in_and_preserves_full_screen_detection():
    image = Image.new("RGB", (448, 1400), "white")
    default = infer_onnx(image, NearTop(), 0.5, tiled=True)
    guarded = infer_onnx(image, NearTop(), 0.5, tiled=True, tile_edge_guard=True)
    assert len(default["elements"]) >= len(guarded["elements"])
    assert len(guarded["elements"]) < len(default["elements"])
    assert len(guarded["elements"]) >= 1
