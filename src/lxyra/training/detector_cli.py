import argparse
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .. import CATEGORIES
from ..schema import make_document, draw_overlay
from ..preprocess import letterbox
from .metrics import detection_metrics
from .detector_model import LxyraV2, SIZE, V2Dataset, loss
from .compact_model import LxyraCompact
from ..geometry import decode, nms, windows


def infer(net, image, threshold=0.2, tiled=False, is_onnx=False):
    candidates = []
    crops = windows(image) if tiled else [(0, 0, *image.size)]
    for x1, y1, x2, y2 in crops:
        crop = image.crop((x1, y1, x2, y2))
        canvas, transform = letterbox(crop, SIZE)
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy()[None] / 255
        if is_onnx:
            output = net.run(None, {"images": array})
        else:
            with torch.inference_mode():
                output = net(torch.from_numpy(array))
        found = decode(output, crop.size, transform, threshold=threshold, top_k=250)
        for item in found:
            item["bbox"][0] += x1
            item["bbox"][1] += y1
        candidates.extend(found)
    return make_document(nms(candidates), image.size, threshold)


def create_model(architecture):
    if architecture == "v2":
        return LxyraV2()
    if architecture == "compact":
        return LxyraCompact()
    raise ValueError(f"Unknown detector architecture: {architecture}")


def load_model(checkpoint):
    d = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        d.get("version") != 2
        or d.get("categories") != list(CATEGORIES)
        or d.get("input_size") != SIZE
    ):
        raise ValueError("Incompatible Lxyra v2 checkpoint")
    net = create_model(d.get("architecture", "v2"))
    net.load_state_dict(d["model"])
    net.eval()
    return net, d


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    training = V2Dataset(args.data, "train", augment=True)
    validation = V2Dataset(args.data, "val")
    train_loader = DataLoader(training, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch, shuffle=False, num_workers=0)
    net = create_model(args.architecture)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    best_score = -1
    best_loss = float("inf")
    start = 0
    if args.resume:
        d = torch.load(args.resume, map_location="cpu", weights_only=True)
        if d.get("architecture", "v2") != args.architecture:
            raise ValueError("Cannot resume checkpoint with a different architecture")
        net.load_state_dict(d["model"])
        start = int(d["epoch"])
        if args.resume_optimizer:
            opt.load_state_dict(d["optimizer"])
            schedule.load_state_dict(d["schedule"])
    report = {
        "device": "cpu",
        "architecture": args.architecture,
        "input_size": SIZE,
        "batch": args.batch,
        "train": len(training),
        "validation": len(validation),
        "parameters": sum(p.numel() for p in net.parameters()),
        "seed": args.seed,
        "start_epoch": start,
    }
    print("START", json.dumps(report), flush=True)
    for epoch in range(start, args.epochs):
        began = time.perf_counter()
        net.train()
        losses = []
        for image, target in train_loader:
            opt.zero_grad(set_to_none=True)
            value, parts = loss(net(image), target)
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            opt.step()
            losses.append(value.item())
        schedule.step()
        net.eval()
        val_losses = []
        with torch.inference_mode():
            for image, target in val_loader:
                value, _ = loss(net(image), target)
                val_losses.append(value.item())
        val = float(np.mean(val_losses))
        map50 = None
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            preds = []
            truth = []
            for index, r in enumerate(validation.records):
                image = Image.open(validation.root / r["image"]).convert("RGB")
                document = infer(net, image, threshold=args.threshold)
                preds.extend((index, a) for a in document["elements"])
                truth.extend((index, a) for a in r["annotations"])
            map50 = detection_metrics(preds, truth)["map50"]
        checkpoint = {
            "version": 2,
            "architecture": args.architecture,
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "schedule": schedule.state_dict(),
            "epoch": epoch + 1,
            "categories": list(CATEGORIES),
            "input_size": SIZE,
            "seed": args.seed,
            "dataset": str(args.data),
        }
        torch.save(checkpoint, dst / "last.pt")
        if map50 is not None and (map50 > best_score or (map50 == best_score and val < best_loss)):
            best_score = map50
            best_loss = val
            torch.save(checkpoint, dst / "best.pt")
        row = {
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(losses)), 5),
            "val_loss": round(val, 5),
            "val_map50": map50,
            "seconds": round(time.perf_counter() - began, 2),
            "lr": opt.param_groups[0]["lr"],
        }
        with (dst / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("EPOCH", json.dumps(row), flush=True)


def evaluate(args):
    torch.set_num_threads(args.threads)
    data = V2Dataset(args.data, args.split)
    if args.onnx:
        net = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        onnx_runtime = True
    else:
        net, _ = load_model(args.checkpoint)
        onnx_runtime = False
    preds = []
    gt = []
    per_family = {}
    elapsed = []
    overlay_path = Path(args.out).with_suffix("")
    for index, r in enumerate(data.records):
        image = Image.open(data.root / r["image"]).convert("RGB")
        started = time.perf_counter()
        doc = infer(net, image, threshold=args.threshold, tiled=args.tile, is_onnx=onnx_runtime)
        elapsed.append((time.perf_counter() - started) * 1000)
        preds.extend((index, a) for a in doc["elements"])
        gt.extend((index, a) for a in r["annotations"])
        domain = per_family.setdefault(
            r["family"], {"predictions": [], "truth": [], "samples": 0}
        )
        domain["predictions"].extend((index, a) for a in doc["elements"])
        domain["truth"].extend((index, a) for a in r["annotations"])
        domain["samples"] += 1
        if index < args.overlays:
            overlay_path.mkdir(parents=True, exist_ok=True)
            draw_overlay(image, doc).save(overlay_path / f"{index:03d}.png")
            write_json(overlay_path / f"{index:03d}.json", doc)
    metrics = detection_metrics(preds, gt)
    output = {
        "dataset": args.data,
        "split": args.split,
        "samples": len(data),
        "tiled": args.tile,
        "threshold": args.threshold,
        "metrics": metrics,
        "by_family": {
            name: {
                "samples": records["samples"],
                "metrics": detection_metrics(records["predictions"], records["truth"]),
            }
            for name, records in sorted(per_family.items())
        },
        "latency_cpu_ms_median": round(float(np.median(elapsed)), 3),
        "latency_cpu_ms_p95": round(float(np.percentile(elapsed, 95)), 3),
        "peak_memory_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }
    write_json(args.out, output)
    print("EVALUATION", json.dumps(output, allow_nan=False), flush=True)


def export(args):
    net, checkpoint = load_model(args.checkpoint)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        net,
        x,
        str(path),
        input_names=["images"],
        output_names=["class_logits", "log_box_wh", "center_offsets"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "class_logits": {0: "batch", 2: "grid_h", 3: "grid_w"},
            "log_box_wh": {0: "batch", 2: "grid_h", 3: "grid_w"},
            "center_offsets": {0: "batch", 2: "grid_h", 3: "grid_w"},
        },
    )
    obj = onnx.load(str(path))
    for name, value in {
        "name": "Lxyra " + checkpoint.get("architecture", "v2"),
        "weights": "random initialization, supervised from scratch",
        "classes": json.dumps(CATEGORIES),
        "input": "RGB 448x448; [0,1]; letterbox(127)",
        "output": "logits, log(width,height) relative to input, sigmoid cell offsets",
    }.items():
        prop = obj.metadata_props.add()
        prop.key, prop.value = name, value
    onnx.save(obj, str(path))
    onnx.checker.check_model(str(path))
    ort_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for size in ((448, 448), (448, 512)):
        x = torch.rand(1, 3, *size)
        with torch.inference_mode():
            expected = [y.numpy() for y in net(x)]
        actual = ort_session.run(None, {"images": x.numpy()})
        errors.extend(float(np.max(np.abs(a - b))) for a, b in zip(expected, actual))
    result = {
        "bytes": path.stat().st_size,
        "max_absolute_error": max(errors),
        "passed": max(errors) < 1e-4,
    }
    write_json(path.with_suffix(".verification.json"), result)
    print("EXPORTED", json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("ONNX and torch differ")


def predict(args):
    image = Image.open(args.image).convert("RGB")
    if args.onnx:
        net = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    else:
        net, _ = load_model(args.checkpoint)
    started = time.perf_counter()
    doc = infer(net, image, args.threshold, tiled=args.tile, is_onnx=bool(args.onnx))
    write_json(args.out, doc)
    if args.overlay:
        draw_overlay(image, doc).save(args.overlay)
    print(
        json.dumps(
            {
                "elements": len(doc["elements"]),
                "milliseconds": round((time.perf_counter() - started) * 1000, 2),
                "out": args.out,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--data", default="data/mixed_v2")
    train_p.add_argument("--out", default="artifacts/v2")
    train_p.add_argument("--epochs", type=int, default=12)
    train_p.add_argument("--batch", type=int, default=5)
    train_p.add_argument("--architecture", choices=("v2", "compact"), default="v2")
    train_p.add_argument("--lr", type=float, default=0.0008)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--threads", type=int, default=4)
    train_p.add_argument("--threshold", type=float, default=0.20)
    train_p.add_argument("--resume")
    train_p.add_argument("--resume-optimizer", action="store_true")
    test_p = sub.add_parser("evaluate")
    test_p.add_argument("--data", default="data/mixed_v2")
    test_p.add_argument("--split", default="test", choices=("train", "val", "test"))
    test_p.add_argument("--checkpoint", default="artifacts/v2/best.pt")
    test_p.add_argument("--onnx")
    test_p.add_argument("--threshold", type=float, default=0.20)
    test_p.add_argument("--tile", action="store_true")
    test_p.add_argument("--overlays", type=int, default=3)
    test_p.add_argument("--threads", type=int, default=4)
    test_p.add_argument("--out", default="artifacts/v2_metrics.json")
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint", default="artifacts/v2/best.pt")
    exp.add_argument("--out", default="artifacts/lxyra_v2.onnx")
    pred = sub.add_parser("infer")
    pred.add_argument("image")
    pred.add_argument("--checkpoint", default="artifacts/v2/best.pt")
    pred.add_argument("--onnx")
    pred.add_argument("--threshold", type=float, default=0.20)
    pred.add_argument("--tile", action="store_true")
    pred.add_argument("--out", default="artifacts/v2_predict.json")
    pred.add_argument("--overlay")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    elif args.cmd == "export":
        export(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
