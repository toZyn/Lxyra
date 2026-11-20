import argparse
import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .datasets import validate_record
from .source_mapping import MAPPING


def fetch_json(url, attempts=5):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Lxyra-Dataset-Builder/2.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 >= attempts:
                raise
            time.sleep(1.3 * (attempt + 1))


def rows(dataset, config, split, offset, length):
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return fetch_json("https://datasets-server.huggingface.co/rows?" + query)["rows"]


def split_for(key):
    n = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "train" if n < 78 else "val" if n < 89 else "test"


def image_from_url(url):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def box_from_rect(rect, width, height):
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        x, y, w, h = map(float, rect)
    except (TypeError, ValueError):
        return None
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, width - x), min(h, height - y)
    if w < 3 or h < 3:
        return None
    return [x, y, w, h]


def rico_category(node):
    cls = str(node.get("class", "")).lower()
    if "textview" in cls or "checkedtext" in cls:
        return "text"
    if "edittext" in cls or "autocomplete" in cls or "searchview" in cls:
        return "input"
    if "imagebutton" in cls or "floatingactionbutton" in cls:
        return "button"
    if "button" in cls:
        return "button"
    if "checkbox" in cls or "radiobutton" in cls:
        return "checkbox"
    if "switch" in cls or "toggle" in cls:
        return "switch"
    if "spinner" in cls or "seekbar" in cls:
        return "select"
    if "imageview" in cls:
        return "icon" if node.get("clickable") else "image"
    if "toolbar" in cls or "bottomnavigationview" in cls:
        return "navigation"
    if "cardview" in cls:
        return "container"
    return None


def rico_annotations(root, width, height):
    root_box = root.get("bounds") or [0, 0, width, height]
    old_w, old_h = max(1, float(root_box[2])), max(1, float(root_box[3]))
    sx, sy = width / old_w, height / old_h
    found = []
    seen = set()

    def walk(node):
        if node.get("visibility") not in (None, "visible") or node.get("visible-to-user") is False:
            return
        bounds = node.get("bounds")
        if bounds and len(bounds) == 4:
            x1, y1, x2, y2 = bounds
            box = box_from_rect(
                [
                    float(x1) * sx,
                    float(y1) * sy,
                    (float(x2) - float(x1)) * sx,
                    (float(y2) - float(y1)) * sy,
                ],
                width,
                height,
            )
            category = rico_category(node)
            if box and category:
                key = (category, *map(lambda v: round(v), box))
                if key not in seen:
                    seen.add(key)
                    found.append({"category": category, "bbox": box, "parent": None})
        for child in node.get("children", []) or []:
            walk(child)

    walk(root)
    return found


def parse_rico(row):
    d = row["row"]
    app = d["app_package_name"]
    if not app:
        return None
    hierarchy = json.loads(d["view_hierarchy"])["activity"]["root"]
    return {
        "image_url": d["image"]["src"],
        "source": "RICO-WidgetCaptioning, CC-BY-4.0",
        "family": "android",
        "group": f"rico:{app}",
        "split": split_for(f"rico:{app}"),
        "record_id": f"rico-{d['screenId']}",
        "annotation_kind": "hierarchy_class_heuristic",
        "annotation_source": hierarchy,
    }


def parse_mrtoy(row):
    d = row["row"]
    width, height = int(d["width"]), int(d["height"])
    group = f"mobile-design-series:{row['row_idx'] // 500}"
    anns = []
    mapping = {"text": "text", "image": "image", "group": "container", "icon": "icon"}
    seen = set()
    for category, bounds in zip(d["objects"]["category"], d["objects"]["bbox"]):
        kind = mapping.get(category.lower())
        box = box_from_rect(bounds, width, height)
        if not kind or not box:
            continue
        if kind == "container" and box[2] * box[3] > width * height * 0.75:
            continue
        key = (kind, *map(round, box))
        if key not in seen:
            seen.add(key)
            anns.append({"category": kind, "bbox": box, "parent": None})
    if len(anns) < 3:
        return None
    return {
        "image_url": d["image"]["src"],
        "source": "mrtoy/mobile-ui-design, Apache-2.0",
        "family": "mobile-design",
        "group": group,
        "split": "train",
        "record_id": f"mobile-design-{row['row_idx']}",
        "annotations": anns,
    }


def parse_web(row):
    d = row["row"]
    width, height = int(d["width"]), int(d["height"])
    domain = urllib.parse.urlparse(d["url"]).netloc.lower().removeprefix("www.")
    anns = []
    for label, bbox in zip(d["labels"], d["bboxes"]):
        kind = MAPPING.get(label)
        box = box_from_rect(bbox, width, height)
        if kind and box:
            anns.append({"category": kind, "bbox": box, "parent": None})
    if not anns or not domain:
        return None
    return {
        "image_url": d["image"]["src"],
        "source": "ScreenParse v2, CC-BY-4.0",
        "family": "web",
        "group": f"web:{domain}",
        "split": split_for(f"web:{domain}"),
        "record_id": f"web-{d['id']}",
        "annotations": anns,
    }


def download(item, root):
    try:
        image = image_from_url(item["image_url"])
        if "annotation_source" in item:
            annotations = rico_annotations(item["annotation_source"], *image.size)
        else:
            annotations = item["annotations"]
        if len(annotations) < 3:
            return None, "few-annotations"
        width, height = image.size
        if max(width, height) > 1280:
            factor = 1280 / max(width, height)
            shape = max(1, round(width * factor)), max(1, round(height * factor))
            sx, sy = shape[0] / width, shape[1] / height
            image = image.resize(shape, Image.Resampling.BILINEAR)
            annotations = [
                {
                    **a,
                    "bbox": [
                        a["bbox"][0] * sx,
                        a["bbox"][1] * sy,
                        a["bbox"][2] * sx,
                        a["bbox"][3] * sy,
                    ],
                }
                for a in annotations
            ]
        name = hashlib.sha256(item["record_id"].encode()).hexdigest()[:24] + ".jpg"
        path = root / "images" / name
        image.save(path, quality=84, optimize=True)
        record = {k: item[k] for k in ("source", "family", "group", "split")}
        record.update({"image": "images/" + name, "annotations": annotations})
        validate_record(record)
        return record, None
    except Exception as exc:
        return None, str(exc)[:140]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/mixed_v2")
    p.add_argument("--rico-pages", type=int, default=14)
    p.add_argument("--web-pages", type=int, default=20)
    p.add_argument("--design-pages", type=int, default=14)
    p.add_argument("--page-size", type=int, default=30)
    args = p.parse_args()
    root = Path(args.root)
    if (root / "manifest.jsonl").exists():
        raise FileExistsError("Refusing to overwrite a dataset")
    (root / "images").mkdir(parents=True, exist_ok=True)
    plan = [
        ("seeeeiii/RICO-WidgetCaptioning", "default", "train", args.rico_pages, 41221, parse_rico),
        ("docling-project/screenparse", "default", "train", args.web_pages, 1447100, parse_web),
        ("mrtoy/mobile-ui-design", "default", "train", args.design_pages, 7846, parse_mrtoy),
    ]
    rng = random.Random(1842)
    items = []
    failures = []
    for dataset, config, split, n, length, parser in plan:
        offsets = sorted(rng.sample(range(0, length - args.page_size, args.page_size), n))
        for offset in offsets:
            try:
                page = rows(dataset, config, split, offset, args.page_size)
                for elem in page:
                    elem["row_idx"] = offset + elem["row_idx"] % args.page_size
                    item = parser(elem)
                    if item:
                        items.append(item)
                print(
                    json.dumps({"dataset": dataset, "offset": offset, "parsed": len(items)}),
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"{dataset}:offset={offset}:{type(exc).__name__}:{exc}")
                print(f"FETCH_FAILURE {dataset} offset={offset} {str(exc)[:100]}", flush=True)
    dedup = {}
    for item in items:
        dedup[item["record_id"]] = item
    records = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(download, item, root) for item in dedup.values()]
        for i, future in enumerate(futures):
            record, error = future.result()
            if record:
                records.append(record)
            elif error and len(failures) < 30:
                failures.append(error)
            if (i + 1) % 100 == 0:
                print("downloaded", i + 1, "valid", len(records), flush=True)
    with (root / "manifest.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    counts = Counter((r["split"], r["family"]) for r in records)
    report = {
        "counts": {f"{a}/{b}": v for (a, b), v in counts.items()},
        "records": len(records),
        "failures": failures[:30],
        "selected": len(dedup),
    }
    (root / "builder_report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
