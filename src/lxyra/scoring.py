import numpy as np

from .preprocess import letterbox


def text_evidence(image, session, boxes):
    canvas, (sx, sy, px, py) = letterbox(image, 448)
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy()[None] / 255
    logits = session.run(None, {"images": array})[0]
    probability = 1 / (1 + np.exp(-np.clip(logits[0, 0], -30, 30)))
    height, width = probability.shape
    evidence = []
    for box in boxes:
        if box["category"] != "text":
            evidence.append((0, 0, 0))
            continue
        x, y, w, h = box["bbox"]
        x1 = max(0, min(width - 1, int((x * sx + px) / 4)))
        y1 = max(0, min(height - 1, int((y * sy + py) / 4)))
        x2 = max(x1 + 1, min(width, round(((x + w) * sx + px) / 4)))
        y2 = max(y1 + 1, min(height, round(((y + h) * sy + py) / 4)))
        region = probability[y1:y2, x1:x2]
        evidence.append(
            (float(np.mean(region)), float(np.percentile(region, 75)), float(np.max(region)))
        )
    return evidence


def apply_scores(boxes, evidences, alpha, measure):
    result = []
    for item, signal in zip(boxes, evidences):
        changed = dict(item)
        if changed["category"] == "text":
            changed["confidence"] = float(item["confidence"]) * (
                (1 - alpha) + alpha * signal[measure]
            )
        result.append(changed)
    return result
