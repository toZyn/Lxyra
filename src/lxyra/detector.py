import numpy as np
import onnxruntime as ort

from .schema import make_document
from .preprocess import letterbox
from .geometry import decode, nms, windows


def infer_onnx(image, model, threshold=0.20, tiled=False):
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
        boxes = decode(raw, crop.size, transform, threshold=threshold, top_k=250)
        for box in boxes:
            box["bbox"][0] += x1
            box["bbox"][1] += y1
        proposals.extend(boxes)
    return make_document(nms(proposals), image.size, threshold)
