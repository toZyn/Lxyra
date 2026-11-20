import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from lxyra import CATEGORIES


ACTIONABLE = frozenset({"button", "icon", "checkbox", "switch", "select", "input"})


def inspect_annotations(record, image_size):
    width, height = image_size
    issues = Counter()
    seen = set()
    for ann in record["annotations"]:
        box = ann.get("bbox")
        category = ann.get("category")
        if category not in CATEGORIES or not isinstance(box, list) or len(box) != 4:
            issues["invalid_annotation"] += 1
            continue
        x, y, w, h = box
        if not all(isinstance(n, (float, int)) for n in box):
            issues["invalid_annotation"] += 1
            continue
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
            issues["invalid_box"] += 1
            continue
        if (category, tuple(box)) in seen:
            issues["exact_duplicate"] += 1
        seen.add((category, tuple(box)))
        if min(w, h) < 2:
            issues["tiny_annotation"] += 1
        if category in ACTIONABLE and w * h / (width * height) >= 0.30:
            issues["large_actionable_region"] += 1
        if category == "text" and w / width >= 0.75:
            issues["page_width_text_region"] += 1
    return issues


def audit(root, split=None):
    root = Path(root)
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split:
        records = [record for record in records if record["split"] == split]
    group_splits = defaultdict(set)
    families = defaultdict(lambda: {"images": 0, "annotations": 0, "issues": Counter()})
    global_issues = Counter()
    for record in records:
        group_splits[record["group"]].add(record["split"])
        path = root / record["image"]
        with Image.open(path) as image:
            width, height = image.size
        issues = inspect_annotations(record, (width, height))
        family = families[record["family"]]
        family["images"] += 1
        family["annotations"] += len(record["annotations"])
        family["issues"].update(issues)
        global_issues.update(issues)
    leaking = sum(len(splits) > 1 for splits in group_splits.values())
    return {
        "screen_count": len(records),
        "groups": len(group_splits),
        "groups_crossing_splits": leaking,
        "warning": "Oversized controls are heuristics for manual review, not proof of incorrect labels",
        "issues": dict(global_issues),
        "families": {
            key: {**data, "issues": dict(data["issues"])} for key, data in sorted(families.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_root, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "screens": report["screen_count"],
                "issues": report["issues"],
                "leakage": report["groups_crossing_splits"],
            }
        )
    )
    if report["groups_crossing_splits"]:
        raise RuntimeError("Dataset group leakage was found")


if __name__ == "__main__":
    main()
