import argparse
import json
import random
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from .metrics import detection_metrics
from .detector_model import V2Dataset, loss
from .detector_cli import infer, load_model


FAMILY_MIX = {
    "web": 0.30,
    "webui": 0.22,
    "android": 0.27,
    "mobile-design": 0.16,
    "launcher_synthetic": 0.05,
}


def validation(net, data, samples, threshold):
    pred = []
    gt = []
    family_samples = Counter()
    net.eval()
    for i in samples:
        rec = data.records[i]
        image = Image.open(data.root / rec["image"]).convert("RGB")
        doc = infer(net, image, threshold=threshold, tiled=False)
        pred.extend((i, e) for e in doc["elements"])
        gt.extend((i, e) for e in rec["annotations"])
        family_samples[rec["family"]] += 1
    d = detection_metrics(pred, gt)
    return {
        "map50": d["map50"],
        "map50_95": d["map50_95"],
        "precision": d["precision_recall"]["micro_precision"],
        "recall": d["precision_recall"]["micro_recall"],
        "n": len(samples),
        "families": dict(family_samples),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    ds = V2Dataset(args.data, "train", augment=True)
    val = V2Dataset(args.data, "val")
    counts = Counter(x["family"] for x in ds.records)
    if len(ds) < 100:
        raise ValueError("Need more training samples")
    active = {k: v for k, v in FAMILY_MIX.items() if counts.get(k)}
    total_weight = sum(active.values())
    weights = [(active[r["family"]] / total_weight) / counts[r["family"]] for r in ds.records]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=args.steps * args.batch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    all_real_indices = [i for i, r in enumerate(val.records) if r["family"] != "launcher_synthetic"]
    random.Random(9177).shuffle(all_real_indices)
    selected = all_real_indices[: args.max_validation]
    if len(selected) < 15:
        raise ValueError("Insufficient validation screens")
    model, previous = load_model(args.checkpoint)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.15
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = -1
    print(
        "SETUP",
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "dataset": args.data,
                "train_records": len(ds),
                "val_records": len(val),
                "real_val_evaluated": len(selected),
                "sampled_batches_per_epoch": len(loader),
                "families": dict(counts),
                "active_family_ratios": active,
                "parameters": sum(x.numel() for x in model.parameters()),
                "epochs": args.epochs,
                "seed": args.seed,
            }
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        start = time.perf_counter()
        model.train()
        train_loss = []
        for images, targets in loader:
            opt.zero_grad(set_to_none=True)
            total, _ = loss(model(images), targets)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()
            train_loss.append(float(total.detach()))
        model.eval()
        measures = validation(model, val, selected, args.threshold)
        scheduler.step()
        checkpoint = {
            "version": 2,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "schedule": scheduler.state_dict(),
            "epoch": epoch + 1,
            "categories": previous["categories"],
            "input_size": previous["input_size"],
            "seed": args.seed,
            "dataset": args.data,
            "starting_checkpoint": args.checkpoint,
            "checkpoint_selection": "real-only validation mAP50",
        }
        torch.save(checkpoint, output / "last.pt")
        if measures["map50"] is not None and measures["map50"] > best:
            best = measures["map50"]
            torch.save(checkpoint, output / "best.pt")
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(train_loss)),
            "real_val": measures,
            "seconds": time.perf_counter() - start,
            "lr": opt.param_groups[0]["lr"],
            "best_val_map50": best,
            "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("EPOCH", json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mixed_v3")
    p.add_argument("--checkpoint", default="artifacts/v2_final/best.pt")
    p.add_argument("--output", default="artifacts/v3")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--steps", type=int, default=280)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--max-validation", type=int, default=150)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=0.00035)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=9184)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
