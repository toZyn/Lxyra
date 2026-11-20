import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from lxyra.detector import infer_onnx
from lxyra.pipeline import ScreenParser
from lxyra.scoring import apply_scores, text_evidence
from lxyra.training.metrics import detection_metrics


CONFIGURATIONS = {
    "default": {"tiled": False, "tile_edge_guard": False, "nonwrapping_peaks": False},
    "tiled": {"tiled": True, "tile_edge_guard": False, "nonwrapping_peaks": False},
    "tile_guard": {"tiled": True, "tile_edge_guard": True, "nonwrapping_peaks": False},
    "tile_guard_peaks": {"tiled": True, "tile_edge_guard": True, "nonwrapping_peaks": True},
}


def detection(image, parser, options, threshold):
    output = infer_onnx(image, parser.detector, threshold=threshold, **options)
    boxes = output["elements"]
    confidence = text_evidence(image, parser.text, boxes)
    adjusted = apply_scores(
        boxes, confidence, parser.reranking["alpha"], parser.reranking["measure"]
    )
    for before, after in zip(boxes, adjusted):
        if before["category"] == "text":
            before["confidence"] = round(after["confidence"], 6)
    return boxes


def metrics(predictions, truths):
    result = detection_metrics(predictions, truths)
    pr = result["precision_recall"]
    return {
        "map50": result["map50"],
        "text_ap50": result["ap50_by_category"]["text"],
        "per_category_ap50": result["ap50_by_category"],
        "precision": pr["micro_precision"],
        "recall": pr["micro_recall"],
        "text_precision": pr["by_category"]["text"]["precision"],
        "text_recall": pr["by_category"]["text"]["recall"],
        "predicted": result["predicted_elements"],
        "truths": result["ground_truth_elements"],
    }


def run(root, split, threshold=0.15, limit=None):
    root = Path(root)
    manifest = root / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [r for r in records if r["split"] == split and r["family"] != "launcher_synthetic"]
    if limit:
        records = records[:limit]
    parser = ScreenParser()
    results = {
        name: defaultdict(lambda: {"prediction": [], "truth": [], "time_ms": []})
        for name in CONFIGURATIONS
    }
    for index, record in enumerate(records):
        with Image.open(root / record["image"]) as item:
            image = item.convert("RGB")
        for name, config in CONFIGURATIONS.items():
            started = time.perf_counter()
            detections = detection(image, parser, config, threshold)
            elapsed = 1000 * (time.perf_counter() - started)
            for family in ("all", record["family"]):
                group = results[name][family]
                group["prediction"].extend((index, element) for element in detections)
                group["truth"].extend((index, element) for element in record["annotations"])
                group["time_ms"].append(elapsed)
        if (index + 1) % 20 == 0:
            print("SCREEN", split, index + 1, "/", len(records), flush=True)
    report = {"split": split, "screen_count": len(records), "variants": {}}
    for name, families in results.items():
        report["variants"][name] = {}
        for family, group in families.items():
            summary = metrics(group["prediction"], group["truth"])
            summary["latency_median_ms"] = float(np.median(group["time_ms"]))
            report["variants"][name][family] = summary
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--limit", type=int)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = run(args.root, args.split, args.threshold, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "RESULT",
        json.dumps(
            {
                name: {
                    key: metric["all"][key] for key in ("map50", "text_ap50", "precision", "recall")
                }
                for name, metric in report["variants"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
