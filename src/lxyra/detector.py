import numpy as np
import onnxruntime as ort

from .schema import make_document
from .preprocess import letterbox
from .geometry import decode, nms, windows


def _visible_within_tile(box, origin, tile_size, image_size, margin=4):
    x, y, width, height = box
    left, top = origin
    tile_width, tile_height = tile_size
    full_width, full_height = image_size
    return (
        (left == 0 or x > margin)
        and (top == 0 or y > margin)
        and (left + tile_width == full_width or x + width < tile_width - margin)
        and (top + tile_height == full_height or y + height < tile_height - margin)
    )


def infer_onnx(
    image,
    model,
    threshold=0.20,
    tiled=False,
    *,
    tile_edge_guard=False,
    nonwrapping_peaks=False,
):
    session = (
        model
        if hasattr(model, "run")
        else ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    )
    proposals = []
    for x1, y1, x2, y2 in windows(image) if tiled else [(0, 0, *image.size)]:
        crop = image.crop((x1, y1, x2, y2))
        canvas, transform = letterbox(crop, 448)
        inputs = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy()[None] / 255.0
        raw = session.run(None, {"images": inputs})
        boxes = decode(
            raw,
            crop.size,
            transform,
            threshold=threshold,
            top_k=250,
            nonwrapping_peaks=nonwrapping_peaks,
        )
        if tiled and tile_edge_guard:
            boxes = [
                box
                for box in boxes
                if _visible_within_tile(box["bbox"], (x1, y1), crop.size, image.size)
            ]
        for box in boxes:
            box["bbox"][0] += x1
            box["bbox"][1] += y1
        proposals.extend(boxes)
    return make_document(nms(proposals), image.size, threshold)
