import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from lxyra.pipeline import ScreenParser
from lxyra.screen_coordinates import CANDIDATE_CATEGORIES, agent_targets


def contains(point, rectangle):
    px, py = point
    x, y, width, height = rectangle
    return x <= px < x + width and y <= py < y + height


def evaluate_one(candidates, truth):
    ground_truth = [label for label in truth if label["category"] in CANDIDATE_CATEGORIES]
    claimed = set()
    hits = 0
    for candidate in sorted(candidates, key=lambda item: -item["confidence"]):
        center = candidate["point_on_screenshot"]
        matches = [
            (index, label)
            for index, label in enumerate(ground_truth)
            if index not in claimed
            and label["category"] == candidate["category"]
            and contains(center, label["bbox"])
        ]
        if matches:
            index, _ = min(
                matches,
                key=lambda pair: pair[1]["bbox"][2] * pair[1]["bbox"][3],
            )
            claimed.add(index)
            hits += 1
    return {
        "candidate_count": len(candidates),
        "truth_count": len(ground_truth),
        "center_hits": hits,
    }


def evaluate_data(root, split, threshold=0.25, limit=None, tiled=False):
    root = Path(root)
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [r for r in records if r["split"] == split and r["family"] != "launcher_synthetic"]
    if limit is not None:
        records = records[:limit]

    parser = ScreenParser()
    stats = defaultdict(lambda: {"screens": 0, "predicted": 0, "annotated": 0, "center_hits": 0})
    for index, item in enumerate(records):
        with Image.open(root / item["image"]) as source:
            screenshot = source.convert("RGB")
        parsed = parser.predict(screenshot, threshold=min(threshold, 0.15), tiled=tiled)
        targets = agent_targets(parsed, minimum_confidence=threshold)["targets"]
        counts = evaluate_one(targets, item["annotations"])
        for family in ("all", item["family"]):
            group = stats[family]
            group["screens"] += 1
            group["predicted"] += counts["candidate_count"]
            group["annotated"] += counts["truth_count"]
            group["center_hits"] += counts["center_hits"]
        if (index + 1) % 25 == 0:
            print("EVALUATED", index + 1, "/", len(records), flush=True)
    result = {}
    for family, count in stats.items():
        result[family] = {
            **count,
            "center_match_precision": count["center_hits"] / count["predicted"]
            if count["predicted"]
            else 0.0,
            "center_match_recall": count["center_hits"] / count["annotated"]
            if count["annotated"]
            else 0.0,
        }
    return {
        "split": split,
        "threshold": threshold,
        "tiled": tiled,
        "groups": result,
        "method": "one-to-one same-category screenshot center inside annotated box",
        "warning": "Annotation overlap is not proof of an actionable control or a successful device tap",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_data(args.data_root, args.split, args.threshold, args.limit, args.tiled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("RESULT", json.dumps(result["groups"]["all"]), flush=True)


if __name__ == "__main__":
    main()
