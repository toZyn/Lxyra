import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from .datasets import validate_record


MAPPING = {
    "Button": "button",
    "Utility Button": "button",
    "Text": "text",
    "Heading": "text",
    "Link": "text",
    "Code snippet": "text",
    "Text Input": "input",
    "Search Field": "input",
    "Search Bar": "input",
    "Checkbox": "checkbox",
    "Radiobox": "checkbox",
    "Switch": "switch",
    "Toggles": "switch",
    "App Icon": "icon",
    "File Icon": "icon",
    "Logo": "icon",
    "Avatar": "icon",
    "Image": "image",
    "Video": "image",
    "Chart": "image",
    "Card": "container",
    "Window": "container",
    "Modal": "container",
    "List": "container",
    "List Item": "container",
    "Table": "container",
    "Navigation Bar": "navigation",
    "Toolbar": "navigation",
    "Bottom navigation": "navigation",
    "Tab Bar": "navigation",
    "Side Bar": "navigation",
    "Menu": "navigation",
    "Select": "select",
    "Picker": "select",
    "Date-Time picker": "select",
    "Slider": "select",
}


def read_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Lxyra-Dataset-Importer/1.0"})
    with urllib.request.urlopen(request, timeout=35) as response:
        return json.load(response)


def fetch_rows(offset, count):
    query = urllib.parse.urlencode(
        {
            "dataset": "docling-project/screenparse",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": count,
        }
    )
    return read_json("https://datasets-server.huggingface.co/rows?" + query)["rows"]


def split_for(domain):
    number = int(hashlib.sha256(domain.encode("utf8")).hexdigest()[:8], 16) % 100
    return "train" if number < 75 else ("val" if number < 88 else "test")


def import_screenparse(root, offsets, page_size=24):
    root = Path(root)
    if (root / "manifest.jsonl").exists():
        raise FileExistsError("Refusing to overwrite a dataset manifest")
    (root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    ids = set()
    failures = []
    for offset in offsets:
        try:
            rows = fetch_rows(offset, page_size)
        except Exception as exc:
            failures.append({"offset": offset, "error": str(exc)[:150]})
            continue
        for entry in rows:
            data = entry["row"]
            if data["id"] in ids:
                continue
            ids.add(data["id"])
            width, height = int(data["width"]), int(data["height"])
            domain = urllib.parse.urlparse(data["url"]).netloc.lower().removeprefix("www.")
            if not domain:
                continue
            annotations = []
            for label, bbox in zip(data["labels"], data["bboxes"]):
                if label not in MAPPING or len(bbox) != 4:
                    continue
                x, y, w, h = (float(v) for v in bbox)
                x, y = max(x, 0), max(y, 0)
                w, h = min(w, width - x), min(h, height - y)
                if w >= 2 and h >= 2:
                    annotations.append(
                        {"category": MAPPING[label], "bbox": [x, y, w, h], "parent": None}
                    )
            if not annotations:
                continue
            filename = f"images/{hashlib.sha256(data['id'].encode()).hexdigest()[:20]}.jpg"
            try:
                request = urllib.request.Request(
                    data["image"]["src"], headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(request, timeout=35) as response:
                    image = Image.open(response).convert("RGB")
                if image.size != (width, height):
                    failures.append(
                        {"id": data["id"], "error": "image/annotation dimension mismatch"}
                    )
                    continue
                image.save(root / filename, quality=86)
            except Exception as exc:
                failures.append({"id": data["id"], "error": str(exc)[:100]})
                continue
            record = {
                "image": filename,
                "source": "ScreenParse v2, CC-BY-4.0",
                "family": "web",
                "group": domain,
                "split": split_for(domain),
                "annotations": annotations,
            }
            validate_record(record)
            records.append(record)
        print(f"offset={offset} imported={len(records)}", flush=True)
    counts = {part: sum(r["split"] == part for r in records) for part in ("train", "val", "test")}
    if min(counts.values()) < 3:
        raise RuntimeError(
            f"Insufficient domain-separated samples: {counts}; failures={failures[:3]}"
        )
    with (root / "manifest.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = {
        "dataset": "docling-project/screenparse",
        "license": "CC-BY-4.0",
        "counts": counts,
        "failed": failures,
        "offsets": offsets,
        "date_unix": time.time(),
    }
    (root / "import_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"counts": counts, "failed": len(failures)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/real")
    parser.add_argument(
        "--offsets", nargs="+", type=int, default=[0, 1500, 12000, 43000, 85000, 180000]
    )
    parser.add_argument("--page-size", type=int, default=24)
    args = parser.parse_args()
    import_screenparse(args.root, args.offsets, args.page_size)
