import argparse
import json
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .refiner_model import TextRefiner, roi_loss


class ROIData(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        self.rows = [
            json.loads(line) for line in (self.root / (split + ".jsonl")).read_text().splitlines()
        ]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        pixels = (
            np.asarray(Image.open(self.root / row["image"]).convert("RGB"), dtype=np.float32)
            .transpose(2, 0, 1)
            .copy()
            / 255
        )
        if self.augment:
            pixels = np.clip(pixels * random.uniform(0.83, 1.17), 0, 1)
        return (
            torch.from_numpy(pixels.copy()),
            torch.tensor(row["delta"], dtype=torch.float32),
            torch.tensor(row["quality"], dtype=torch.float32),
            torch.tensor(row["pos"], dtype=torch.float32),
        )


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    ds = ROIData(args.data, "train", augment=True)
    val = ROIData(args.data, "val")
    pos = sum(q["pos"] for q in ds.rows)
    neg = len(ds) - pos
    if pos < 100:
        raise ValueError("Not enough real text matches to train")
    if args.sampling == "balanced":
        family_mix = {"web": 0.35, "webui": 0.30, "android": 0.30, "mobile-design": 0.05}
        counts = Counter((r["family"], r["pos"]) for r in ds.rows)
        weights = torch.tensor(
            [
                family_mix[r["family"]]
                * (0.5 if r["pos"] else 0.5)
                / max(1, counts[(r["family"], r["pos"])])
                for r in ds.rows
            ],
            dtype=torch.double,
        )
    else:
        weights = torch.tensor(
            [0.5 / pos if r["pos"] else 0.5 / max(1, neg) for r in ds.rows], dtype=torch.double
        )
    sampler = WeightedRandomSampler(
        weights, len(ds), replacement=True, generator=torch.Generator().manual_seed(args.seed)
    )
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    vl = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0)
    model = TextRefiner()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.1)
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    best = 1e8
    print(
        "TRAIN",
        json.dumps(
            {
                "screens_crops": len(ds),
                "positive": pos,
                "negative": neg,
                "val_crops": len(val),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        start = time.perf_counter()
        model.train()
        loss_hist = []
        for images, delta, quality, positive in loader:
            opt.zero_grad(set_to_none=True)
            loss, _, _ = roi_loss(model(images), delta, quality, positive)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            loss_hist.append(float(loss.detach()))
        model.eval()
        valid = []
        with torch.inference_mode():
            for images, delta, quality, posmask in vl:
                loss, _, _ = roi_loss(model(images), delta, quality, posmask)
                valid.append(float(loss))
        sched.step()
        v = float(np.mean(valid))
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "data": args.data,
            "seed": args.seed,
            "input": 96,
        }
        torch.save(checkpoint, output / "last.pt")
        if v < best:
            best = v
            torch.save(checkpoint, output / "best.pt")
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(loss_hist)),
            "val_loss": v,
            "best_val_loss": best,
            "seconds": round(time.perf_counter() - start, 2),
        }
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("EPOCH", json.dumps(row), flush=True)


def export(args):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TextRefiner()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.rand(1, 3, 96, 96)
    torch.onnx.export(
        model,
        tensor,
        str(dest),
        opset_version=17,
        dynamo=False,
        input_names=["crops"],
        output_names=["refinement"],
        dynamic_axes={"crops": {0: "batch"}, "refinement": {0: "batch"}},
    )
    onnx.checker.check_model(str(dest))
    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    errors = []
    for n in (1, 7):
        t = torch.rand(n, 3, 96, 96)
        with torch.inference_mode():
            expected = model(t).numpy()
        found = session.run(None, {"crops": t.numpy()})[0]
        errors.append(float(np.abs(expected - found).max()))
    report = {
        "max_abs_error": max(errors),
        "passed": max(errors) < 1e-4,
        "bytes": dest.stat().st_size,
    }
    dest.with_suffix(".verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ONNX", json.dumps(report), flush=True)
    if not report["passed"]:
        raise ValueError("ONNX parity failure")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/text_roi_v5")
    t.add_argument("--output", default="artifacts/v5_roi")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=0.0007)
    t.add_argument("--seed", type=int, default=2241)
    t.add_argument("--sampling", choices=["original", "balanced"], default="original")
    t.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", default="artifacts/v5_roi/best.pt")
    e.add_argument("--output", default="artifacts/v5_roi/text_roi.onnx")
    e.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    train(args) if args.task == "train" else export(args)


if __name__ == "__main__":
    main()
