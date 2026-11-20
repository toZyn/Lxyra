import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .datasets import validate_record


def merge(output, datasets):
    output = Path(output)
    if (output / "manifest.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    (output / "images").mkdir(parents=True, exist_ok=True)
    seen_files = set()
    seen_groups = {}
    records = []
    stats = Counter()
    for i, source in enumerate(datasets):
        source = Path(source)
        source_name = f"s{i}"
        (output / "images" / source_name).symlink_to(
            (source / "images").resolve(), target_is_directory=True
        )
        with (source / "manifest.jsonl").open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                original = source / record["image"]
                if not original.exists():
                    stats["missing_image"] += 1
                    continue
                group = record["group"]
                split = record["split"]
                if group in seen_groups and seen_groups[group] != split:
                    raise ValueError(
                        f"Split leakage: {group} was {seen_groups[group]}, now {split}"
                    )
                seen_groups[group] = split
                with Image.open(original) as image:
                    width, height = image.size
                repaired = []
                for a in record["annotations"]:
                    x, y, w, h = a["bbox"]
                    x1 = max(0, min(width, x))
                    y1 = max(0, min(height, y))
                    x2 = max(0, min(width, x + w))
                    y2 = max(0, min(height, y + h))
                    if x2 - x1 < 3 or y2 - y1 < 3:
                        stats["invalid_box_skipped"] += 1
                        continue
                    repaired.append({**a, "bbox": [x1, y1, x2 - x1, y2 - y1], "parent": None})
                if len(repaired) < 2:
                    stats["too_few_boxes"] += 1
                    continue
                record["annotations"] = repaired
                record["image"] = "images/" + source_name + "/" + original.name
                key = hashlib.sha256(original.read_bytes()).hexdigest()
                if key in seen_files:
                    stats["duplicate_image"] += 1
                    continue
                seen_files.add(key)
                validate_record(record)
                records.append(record)
                stats[split + "/" + record["family"]] += 1
    if not records:
        raise ValueError("No input records")
    with (output / "manifest.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = {
        "sources": [str(x) for x in datasets],
        "total": len(records),
        "counts": dict(stats),
        "groups": len(seen_groups),
        "images_sha256_deduplicated": True,
        "group_based_splits_verified": True,
        "bbox_clipped_and_validated": True,
        "parent_edges_discarded_without_consistent_cross_dataset_semantics": True,
    }
    (output / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/mixed_v3")
    p.add_argument("sources", nargs="+")
    args = p.parse_args()
    merge(args.output, args.sources)
