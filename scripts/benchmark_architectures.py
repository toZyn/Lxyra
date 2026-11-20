import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lxyra.training.compact_model import LxyraCompact
from lxyra.training.detector_model import LxyraV2


def benchmark(model, pixels, repetitions):
    image = torch.rand((1, 3, pixels, pixels), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(image)
        durations = []
        for _ in range(repetitions):
            start = time.perf_counter()
            outputs = model(image)
            durations.append((time.perf_counter() - start) * 1000)
    assert outputs[0].shape[-2:] == (pixels // 4, pixels // 4)
    return {
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(sorted(durations)[int((repetitions - 1) * 0.95)], 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--resolution", type=int, default=448)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 5 or args.resolution % 32:
        parser.error("Threads must be positive; repeats >=5; resolution a multiple of 32")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    records = {}
    for name, cls in (("v2", LxyraV2), ("compact", LxyraCompact)):
        network = cls()
        stats = benchmark(network, args.resolution, args.repetitions)
        records[name] = {
            "parameters": sum(p.numel() for p in network.parameters()),
            **stats,
        }
    result = {
        "input_shape": [1, 3, args.resolution, args.resolution],
        "cpu_threads": args.threads,
        "repetitions": args.repetitions,
        "framework": str(torch.__version__),
        "hardware": "current runner, no cross-hardware comparison implied",
        "weights": "randomly initialized, not detector accuracy benchmark",
        "models": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
