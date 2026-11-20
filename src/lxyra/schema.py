import numpy as np
from PIL import ImageDraw

from . import CATEGORIES, STRIDE


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def decode_arrays(
    logits, sizes, offsets, image_size, transform, score_threshold=0.12, max_detections=120
):
    if hasattr(logits, "detach"):
        logits, sizes, offsets = (t.detach().cpu().numpy() for t in (logits, sizes, offsets))
    logits, sizes, offsets = (t[0] if t.ndim == 4 else t for t in (logits, sizes, offsets))
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    height, width = probabilities.shape[1:]
    candidates = []
    count = min(max_detections * 4, probabilities.size)
    flat = probabilities.ravel()
    indices = np.argpartition(flat, -count)[-count:]
    indices = sorted(indices, key=lambda i: (-float(flat[i]), int(i)))
    sx, sy, pad_x, pad_y = transform
    ow, oh = image_size
    for index in indices:
        confidence = float(flat[index])
        if confidence < score_threshold:
            break
        category, iy, ix = np.unravel_index(index, probabilities.shape)
        cx = (ix + float(offsets[0, iy, ix])) * STRIDE
        cy = (iy + float(offsets[1, iy, ix])) * STRIDE
        bw = float(sizes[0, iy, ix]) * (width * STRIDE)
        bh = float(sizes[1, iy, ix]) * (height * STRIDE)
        x1 = max(0.0, min(float(ow), (cx - bw / 2 - pad_x) / sx))
        y1 = max(0.0, min(float(oh), (cy - bh / 2 - pad_y) / sy))
        x2 = max(0.0, min(float(ow), (cx + bw / 2 - pad_x) / sx))
        y2 = max(0.0, min(float(oh), (cy + bh / 2 - pad_y) / sy))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        box = [x1, y1, x2 - x1, y2 - y1]
        candidates.append({"category": CATEGORIES[category], "bbox": box, "confidence": confidence})
    accepted = []
    for candidate in candidates:
        if any(
            other["category"] == candidate["category"]
            and iou(other["bbox"], candidate["bbox"]) > 0.5
            for other in accepted
        ):
            continue
        accepted.append(candidate)
        if len(accepted) >= max_detections:
            break
    return accepted


def add_hierarchy(detections, image_size):
    width, height = image_size
    result = sorted(
        detections, key=lambda item: (item["bbox"][1], item["bbox"][0], -item["confidence"])
    )
    for i, node in enumerate(result):
        node["id"] = f"e{i:04d}"
        node["reading_order"] = i
        node["parent_id"] = None
        node["children"] = []
        x, y, w, h = node["bbox"]
        node["bbox"] = [round(float(v), 3) for v in (x, y, w, h)]
        node["bbox_normalized"] = [
            round(x / width, 6),
            round(y / height, 6),
            round(w / width, 6),
            round(h / height, 6),
        ]
        node["confidence"] = round(float(node["confidence"]), 6)
    for node in result:
        box = node["bbox"]
        parents = []
        for candidate in result:
            if candidate is node or candidate["category"] not in ("container", "navigation"):
                continue
            cb = candidate["bbox"]
            if cb[2] * cb[3] <= box[2] * box[3] * 1.1:
                continue
            intersection = iou(box, cb)
            cover = (
                intersection
                * (box[2] * box[3] + cb[2] * cb[3])
                / (box[2] * box[3] * (1 + intersection))
                if box[2] * box[3]
                else 0
            )
            if cover >= 0.90:
                parents.append(candidate)
        if parents:
            parent = min(parents, key=lambda a: a["bbox"][2] * a["bbox"][3])
            node["parent_id"] = parent["id"]
            parent["children"].append(node["id"])
    return result


def make_document(detections, image_size, threshold):
    result = {
        "schema_version": "1.0",
        "image": {"width": image_size[0], "height": image_size[1]},
        "config": {"threshold": threshold},
        "elements": add_hierarchy(detections, image_size),
        "ocr_available": False,
    }
    validate_document(result)
    return result


def validate_document(document):
    if document.get("schema_version") != "1.0" or document.get("ocr_available") is not False:
        raise ValueError("Invalid schema version or OCR capability")
    width, height = document["image"]["width"], document["image"]["height"]
    if not (type(width) is int and type(height) is int and width > 0 and height > 0):
        raise ValueError("Invalid image dimensions")
    elements = document["elements"]
    ids = {element["id"] for element in elements}
    if len(ids) != len(elements):
        raise ValueError("Duplicate element ID")
    by_id = {element["id"]: element for element in elements}
    for i, e in enumerate(elements):
        if (
            e["category"] not in CATEGORIES
            or not isinstance(e["confidence"], (int, float))
            or not 0 <= e["confidence"] <= 1
        ):
            raise ValueError("Invalid category or confidence")
        if type(e["reading_order"]) is not int or e["reading_order"] != i:
            raise ValueError("Invalid reading order")
        if len(e["bbox_normalized"]) != 4 or not all(0 <= v <= 1 for v in e["bbox_normalized"]):
            raise ValueError("Invalid normalized coordinates")
        x, y, w, h = e["bbox"]
        if not (
            0 <= x <= width
            and 0 <= y <= height
            and w > 0
            and h > 0
            and x + w <= width + 0.01
            and y + h <= height + 0.01
        ):
            raise ValueError("Invalid output bounding box")
        expected = [x / width, y / height, w / width, h / height]
        if any(abs(a - b) > 0.00002 for a, b in zip(expected, e["bbox_normalized"])):
            raise ValueError("Normalized coordinates disagree with pixel coordinates")
        parent = e["parent_id"]
        if parent is not None and parent not in ids:
            raise ValueError("Missing parent")
        for child in e["children"]:
            if child not in ids or by_id[child]["parent_id"] != e["id"]:
                raise ValueError("Inconsistent children")
        visited = set()
        while parent is not None:
            if parent in visited or parent == e["id"]:
                raise ValueError("Cycle detected")
            visited.add(parent)
            parent = by_id[parent]["parent_id"]


def draw_overlay(image, document):
    result = image.convert("RGB").copy()
    d = ImageDraw.Draw(result)
    for e in document["elements"]:
        x, y, w, h = e["bbox"]
        color = "#1474e5" if e["category"] in ("container", "navigation") else "#e44c64"
        d.rectangle((x, y, x + w, y + h), outline=color, width=2)
        d.text((x, max(0, y - 12)), f"{e['category']} {e['confidence']:.2f}", fill=color)
    return result
