import argparse
import json
import random
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .hf_builder import download, fetch_json, parse_rico, parse_mrtoy, parse_web, split_for


WEB_TAGS = {
    "button": "button",
    "input": "input",
    "textarea": "input",
    "select": "select",
    "option": "select",
    "input-search": "input",
    "nav": "navigation",
    "header": "navigation",
    "footer": "navigation",
    "img": "image",
    "picture": "image",
    "canvas": "image",
    "video": "image",
    "svg": "icon",
    "i": "icon",
    "p": "text",
    "h1": "text",
    "h2": "text",
    "h3": "text",
    "h4": "text",
    "h5": "text",
    "h6": "text",
    "label": "text",
    "span": "text",
    "strong": "text",
    "small": "text",
    "em": "text",
    "code": "text",
    "a": "text",
    "section": "container",
    "article": "container",
    "main": "container",
    "aside": "container",
    "form": "container",
    "dialog": "container",
}


def parse_webui(row):
    d = row["row"]
    boxes = d["bboxes"]
    host = urllib.parse.urlparse(d["source_url"]).netloc.lower().removeprefix("www.")
    if not host:
        return None
    group = f"web:{host}"
    annotations = []
    canonical = {}
    all_indices = {}
    sizes = zip(
        boxes["tag"],
        boxes["x"],
        boxes["y"],
        boxes["width"],
        boxes["height"],
        boxes["parent_index"],
        boxes["text"],
    )
    for original_id, (tag, x, y, w, h, parent, text) in enumerate(sizes):
        tag = str(tag).lower()
        kind = WEB_TAGS.get(tag)
        if not kind:
            continue
        try:
            x, y, w, h = float(x), float(y), float(w), float(h)
        except (ValueError, TypeError):
            continue
        if min(w, h) < 3 or x < 0 or y < 0:
            continue
        if kind == "text" and not str(text).strip():
            continue
        if kind == "container" and w * h < 300:
            continue
        if kind == "icon" and w * h > 90000:
            continue
        key = (kind, round(x), round(y), round(w), round(h))
        if key in canonical:
            all_indices[original_id] = canonical[key]
            continue
        ann = {"category": kind, "bbox": [x, y, w, h], "parent": None}
        canonical[key] = len(annotations)
        all_indices[original_id] = len(annotations)
        annotations.append(ann)
    if len(annotations) < 4:
        return None
    return {
        "image_url": d["image"]["src"],
        "source": "MorganTitcher/webui, MIT",
        "family": "webui",
        "group": group,
        "split": split_for(group),
        "record_id": "webui-" + d["sample_id"],
        "annotations": annotations,
    }


def fetch_rows(dataset, split, offset, length):
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return fetch_json("https://datasets-server.huggingface.co/rows?" + query, attempts=3)["rows"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/hf_expand")
    p.add_argument("--webui-pages", type=int, default=22)
    p.add_argument("--screenparse-pages", type=int, default=18)
    p.add_argument("--rico-pages", type=int, default=15)
    p.add_argument("--mobile-pages", type=int, default=10)
    args = p.parse_args()
    root = Path(args.root)
    if (root / "manifest.jsonl").exists():
        raise FileExistsError("Refusing to overwrite")
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(14143)
    choices = [
        ("MorganTitcher/webui", "train", 29409, args.webui_pages, 28, parse_webui),
        ("docling-project/screenparse", "train", 1447100, args.screenparse_pages, 70, parse_web),
        ("seeeeiii/RICO-WidgetCaptioning", "train", 41221, args.rico_pages, 45, parse_rico),
        ("mrtoy/mobile-ui-design", "train", 7846, args.mobile_pages, 45, parse_mrtoy),
    ]
    records = []
    failures = []
    duplicates = set()
    for dataset, split, total, page_count, page_size, parser in choices:
        offsets = sorted(rng.sample(range(0, total - page_size, page_size), page_count))
        print("DATASET", dataset, "pages", len(offsets), flush=True)
        for page_idx, offset in enumerate(offsets):
            try:
                chunk = fetch_rows(dataset, split, offset, page_size)
            except Exception as exc:
                failures.append({"dataset": dataset, "offset": offset, "error": str(exc)[:100]})
                print("FETCH_ERROR", dataset, offset, str(exc)[:130], flush=True)
                continue
            batch = []
            for elem in chunk:
                elem["row_idx"] = offset + elem["row_idx"] % page_size
                try:
                    item = parser(elem)
                    if item and item["record_id"] not in duplicates:
                        duplicates.add(item["record_id"])
                        batch.append(item)
                except Exception as exc:
                    if len(failures) < 100:
                        failures.append(
                            {"dataset": dataset, "offset": offset, "error": str(exc)[:130]}
                        )
            with ThreadPoolExecutor(max_workers=4) as pool:
                for item, result in zip(batch, pool.map(lambda x: download(x, root), batch)):
                    record, error = result
                    if record:
                        records.append(record)
                    elif error and len(failures) < 100:
                        failures.append(
                            {"dataset": dataset, "record_id": item["record_id"], "error": error}
                        )
            print(
                "PAGE",
                dataset,
                f"{page_idx + 1}/{page_count}",
                "valid_total",
                len(records),
                flush=True,
            )
            time.sleep(0.20)
    if not records:
        raise RuntimeError("Zero usable records")
    with (root / "manifest.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    counts = Counter((x["family"], x["split"]) for x in records)
    report = {
        "sources": {
            "MorganTitcher/webui": "MIT",
            "docling-project/screenparse": "CC-BY-4.0",
            "seeeeiii/RICO-WidgetCaptioning": "CC-BY-4.0",
            "mrtoy/mobile-ui-design": "Apache-2.0",
        },
        "records": len(records),
        "counts": {"/".join(k): v for k, v in counts.items()},
        "failures": failures,
        "seed": 14143,
        "annotations_from_html_hierarchy_only_during_data_creation": True,
        "at_inference_uses_only_screenshot": True,
    }
    (root / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "DONE",
        json.dumps(
            {"records": len(records), "counts": report["counts"], "failures": len(failures)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
