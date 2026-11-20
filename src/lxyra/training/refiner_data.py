import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ..schema import iou
from ..coordinates import get_delta, make_crop
from .detector_model import V2Dataset
from ..detector import infer_onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/mixed_v3")
    p.add_argument("--onnx", default="artifacts/lxyra_v3.onnx")
    p.add_argument("--output", default="data/text_roi_v5")
    p.add_argument("--max-train", type=int, default=720)
    p.add_argument("--max-val", type=int, default=160)
    p.add_argument("--seed", type=int, default=21042)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    rng = random.Random(args.seed)
    for split, cap in [("train", args.max_train), ("val", args.max_val)]:
        ds = V2Dataset(args.root, split)
        indices = [i for i, r in enumerate(ds.records) if r["family"] != "launcher_synthetic"]
        rng.shuffle(indices)
        indices = indices[:cap]
        records = []
        counts = Counter()
        for j, index in enumerate(indices):
            record = ds.records[index]
            image = Image.open(ds.root / record["image"]).convert("RGB")
            pred = [
                x
                for x in infer_onnx(image, session, threshold=0.15)["elements"]
                if x["category"] == "text"
            ]
            gt = [x["bbox"] for x in record["annotations"] if x["category"] == "text"]
            matches = []
            for item in pred:
                overlaps = [iou(item["bbox"], x) for x in gt]
                if overlaps:
                    top = max(range(len(overlaps)), key=overlaps.__getitem__)
                    score = overlaps[top]
                else:
                    top = 0
                    score = 0
                matches.append((score, item, gt[top] if gt else item["bbox"]))
            positive = sorted((x for x in matches if x[0] >= 0.20), key=lambda x: -x[0])[:24]
            negative = sorted(
                (x for x in matches if x[0] < 0.15), key=lambda x: -x[1]["confidence"]
            )[: max(6, len(positive) // 2)]
            for sample_id, (score, item, truth) in enumerate(positive + negative):
                pos = score >= 0.20
                delta = get_delta(item["bbox"], truth) if pos else np.zeros(4, dtype=np.float32)
                if pos and max(abs(float(x)) for x in delta) > 1.25:
                    continue
                crop = make_crop(image, item["bbox"])
                filename = f"{split}_{j:04}_{sample_id:02}.jpg"
                (root / filename).parent.mkdir(exist_ok=True, parents=True)
                Image.fromarray(crop).save(root / filename, quality=89)
                records.append(
                    {
                        "image": filename,
                        "delta": delta.tolist(),
                        "pos": int(pos),
                        "quality": int(score >= 0.5),
                        "iou": score,
                        "family": record["family"],
                        "group": record["group"],
                        "source": record["image"],
                    }
                )
                counts[("positive" if pos else "negative") + "/" + record["family"]] += 1
            if (j + 1) % 50 == 0:
                print("BUILD", split, j + 1, "/", len(indices), "samples", len(records), flush=True)
        with (root / (split + ".jsonl")).open("w") as out:
            for item in records:
                out.write(json.dumps(item, separators=(",", ":")) + "\n")
        print(
            "DONE",
            split,
            "images",
            len(indices),
            "crops",
            len(records),
            "counts",
            dict(counts),
            flush=True,
        )


if __name__ == "__main__":
    main()
