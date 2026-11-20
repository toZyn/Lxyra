from collections import defaultdict

import numpy as np

from .. import CATEGORIES
from ..schema import iou


def average_precision(predictions, targets, category, threshold):
    truth = defaultdict(list)
    for image_id, ann in targets:
        if ann["category"] == category:
            truth[image_id].append(ann["bbox"])
    total = sum(map(len, truth.values()))
    if total == 0:
        return None
    matched = {key: set() for key in truth}
    preds = sorted(
        ((image_id, p) for image_id, p in predictions if p["category"] == category),
        key=lambda item: -item[1]["confidence"],
    )
    tp, fp = [], []
    for image_id, pred in preds:
        boxes = truth.get(image_id, [])
        overlaps = [
            iou(pred["bbox"], box) if j not in matched.get(image_id, set()) else 0.0
            for j, box in enumerate(boxes)
        ]
        best = int(np.argmax(overlaps)) if overlaps else -1
        hit = best >= 0 and overlaps[best] >= threshold
        if hit:
            matched[image_id].add(best)
        tp.append(1 if hit else 0)
        fp.append(0 if hit else 1)
    if not tp:
        return 0.0
    precision = np.cumsum(tp) / np.maximum(1, np.cumsum(tp) + np.cumsum(fp))
    recall = np.cumsum(tp) / total
    p = np.concatenate(([0.0], precision, [0.0]))
    r = np.concatenate(([0.0], recall, [1.0]))
    for j in range(len(p) - 2, -1, -1):
        p[j] = max(p[j], p[j + 1])
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def precision_recall(predictions, targets):
    result = {}
    overall_tp, overall_fp, overall_gt = 0, 0, 0
    for category in CATEGORIES:
        truth = defaultdict(list)
        for image_id, ann in targets:
            if ann["category"] == category:
                truth[image_id].append(ann["bbox"])
        count = sum(len(values) for values in truth.values())
        matched = {image_id: set() for image_id in truth}
        candidates = sorted(
            ((image_id, p) for image_id, p in predictions if p["category"] == category),
            key=lambda item: -item[1]["confidence"],
        )
        tp = 0
        overlaps = []
        for image_id, pred in candidates:
            options = [
                (index, iou(pred["bbox"], box))
                for index, box in enumerate(truth.get(image_id, []))
                if index not in matched.get(image_id, set())
            ]
            if options:
                best_index, overlap = max(options, key=lambda pair: pair[1])
                if overlap >= 0.5:
                    matched[image_id].add(best_index)
                    overlaps.append(overlap)
                    tp += 1
        fp = len(candidates) - tp
        overall_tp += tp
        overall_fp += fp
        overall_gt += count
        result[category] = {
            "gt": count,
            "predicted": len(candidates),
            "tp": tp,
            "fp": fp,
            "fn": count - tp,
            "precision": tp / len(candidates) if candidates else None,
            "recall": tp / count if count else None,
            "mean_matched_iou": float(np.mean(overlaps)) if overlaps else None,
        }
    return {
        "by_category": result,
        "micro_precision": overall_tp / (overall_tp + overall_fp)
        if (overall_tp + overall_fp)
        else None,
        "micro_recall": overall_tp / overall_gt if overall_gt else None,
    }


def detection_metrics(predictions, targets):
    ap50 = {
        category: average_precision(predictions, targets, category, 0.5) for category in CATEGORIES
    }
    valid = [x for x in ap50.values() if x is not None]
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = [
        score
        for threshold in thresholds
        for category in CATEGORIES
        if (score := average_precision(predictions, targets, category, float(threshold)))
        is not None
    ]
    return {
        "map50": float(np.mean(valid)) if valid else None,
        "map50_95": float(np.mean(aps)) if aps else None,
        "ap50_by_category": ap50,
        "precision_recall": precision_recall(predictions, targets),
        "ground_truth_elements": len(targets),
        "predicted_elements": len(predictions),
    }


def parent_metrics(records):
    gt_edges, pred_edges = 0, 0
    tp_edges = 0
    for predicted, truth in records:
        matched = {}
        used = set()
        for pi, p in enumerate(sorted(enumerate(predicted), key=lambda t: -t[1]["confidence"])):
            original, pred = p
            choices = [
                (j, iou(pred["bbox"], gt["bbox"]))
                for j, gt in enumerate(truth)
                if j not in used and pred["category"] == gt["category"]
            ]
            if choices:
                j, score = max(choices, key=lambda t: t[1])
                if score >= 0.5:
                    used.add(j)
                    matched[original] = j
        gt_edges += sum(1 for gt in truth if gt.get("parent") is not None)
        pred_edges += sum(1 for p in predicted if p.get("parent_id") is not None)
        for pi, pred in enumerate(predicted):
            gi = matched.get(pi)
            if gi is None or pred.get("parent_id") is None:
                continue
            parent_pi = next(
                (k for k, v in enumerate(predicted) if v.get("id") == pred["parent_id"]), None
            )
            if (
                parent_pi is not None
                and parent_pi in matched
                and truth[gi].get("parent") == matched[parent_pi]
            ):
                tp_edges += 1
    return {
        "parent_precision": tp_edges / pred_edges if pred_edges and gt_edges else None,
        "parent_recall": tp_edges / gt_edges if gt_edges else None,
        "evaluable": gt_edges > 0,
        "ground_truth_edges": gt_edges,
        "predicted_edges": pred_edges,
        "correct_edges": tp_edges,
    }
