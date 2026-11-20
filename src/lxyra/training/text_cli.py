import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .text_model import SIZE, TextData, TextDetector, text_loss


FAMILY_WEIGHTS = {
    "web": 0.35,
    "webui": 0.28,
    "android": 0.22,
    "launcher_synthetic": 0.07,
    "mobile-design": 0.08,
}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train_data = TextData(args.data, "train", True)
    val_data = TextData(args.data, "val", False)
    counts = Counter(r["family"] for r in train_data.records)
    weights = [FAMILY_WEIGHTS[r["family"]] / counts[r["family"]] for r in train_data.records]
    sampling = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=args.steps * args.batch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch, sampler=sampling, num_workers=0)
    real_val = [i for i, r in enumerate(val_data.records) if r["family"] != "launcher_synthetic"]
    random.Random(args.seed).shuffle(real_val)
    val_indices = real_val[: args.val_count]
    model = TextDetector()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best = -1
    print(
        "START",
        json.dumps(
            {
                "train": len(train_data),
                "val": len(real_val),
                "val_samples": len(val_indices),
                "families": dict(counts),
                "parameters": sum(x.numel() for x in model.parameters()),
                "epochs": args.epochs,
                "batch": args.batch,
            }
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            value = text_loss(model(x), y)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()
            losses.append(float(value.detach()))
        sched.step()
        model.eval()
        intersection, union = 0, 0
        val_losses = []
        with torch.inference_mode():
            for idx in val_indices:
                x, y = val_data[idx]
                logits = model(x.unsqueeze(0))
                val_losses.append(float(text_loss(logits, y.unsqueeze(0))))
                pred = logits.sigmoid()[0, 0] >= 0.5
                truth = y[0] >= 0.5
                intersection += float((pred & truth).sum())
                union += float((pred | truth).sum())
        iou = intersection / union if union else 0.0
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_mask_iou": iou,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "classes": ["text"],
            "size": SIZE,
            "seed": args.seed,
            "data": args.data,
        }
        torch.save(ckpt, out / "last.pt")
        if iou > best:
            best = iou
            torch.save(ckpt, out / "best.pt")
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print("EPOCH", json.dumps(record), flush=True)


def export(args):
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TextDetector()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model,
        x,
        str(path),
        opset_version=17,
        dynamo=False,
        input_names=["images"],
        output_names=["text_logits"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "text_logits": {0: "batch", 2: "grid_h", 3: "grid_w"},
        },
    )
    loaded = onnx.load(str(path))
    meta = loaded.metadata_props.add()
    meta.key = "description"
    meta.value = "Lxyra text mask head, random initialization and supervised training"
    onnx.save(loaded, str(path))
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for shape in ((448, 448), (448, 512)):
        x = torch.rand((1, 3, *shape))
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(None, {"images": x.numpy()})[0]
        errors.append(float(np.max(np.abs(expected - actual))))
    report = {
        "onnx_bytes": path.stat().st_size,
        "max_absolute_error": max(errors),
        "passed": max(errors) < 0.0001,
    }
    path.with_suffix(".verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise RuntimeError("ONNX divergence")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/mixed_v3")
    t.add_argument("--output", default="artifacts/v4_text")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--steps", type=int, default=120)
    t.add_argument("--batch", type=int, default=6)
    t.add_argument("--val-count", type=int, default=80)
    t.add_argument("--seed", type=int, default=5321)
    t.add_argument("--lr", type=float, default=0.0008)
    t.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", default="artifacts/v4_text/best.pt")
    e.add_argument("--output", default="artifacts/v4_text/text.onnx")
    e.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
