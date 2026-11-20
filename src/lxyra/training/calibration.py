import argparse
import json
from pathlib import Path

import onnxruntime as ort
from PIL import Image

from .metrics import detection_metrics
from .detector_model import V2Dataset
from ..detector import infer_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", default="data/mixed_launcher_v2_fixed")
    parser.add_argument("--out", default="artifacts/v2_thresholds.json")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()
    ds = V2Dataset(args.data, "val")
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    thresholds = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40]
    result = {}
    for threshold in thresholds:
        preds = []
        truth = []
        for idx, record in enumerate(ds.records):
            if args.real_only and record["family"] == "launcher_synthetic":
                continue
            image = Image.open(ds.root / record["image"]).convert("RGB")
            doc = infer_onnx(image, session, threshold, tiled=args.tile)
            preds.extend((idx, e) for e in doc["elements"])
            truth.extend((idx, e) for e in record["annotations"])
        m = detection_metrics(preds, truth)
        precision = m["precision_recall"]["micro_precision"] or 0
        recall = m["precision_recall"]["micro_recall"] or 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        result[str(threshold)] = {
            "mAP50": m["map50"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted": len(preds),
        }
    best = max(result, key=lambda k: (result[k]["f1"], result[k]["mAP50"]))
    report = {
        "selected_on": "real-only validation" if args.real_only else "all validation",
        "best_f1_threshold": float(best),
        "results": result,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
