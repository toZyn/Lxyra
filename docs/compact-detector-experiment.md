# Compact detector experiment

This experiment adds a depthwise-separable multi-scale detector that retains the existing 10-category, stride-four tensor contract. It is deliberately separate from the shipped ONNX models.

## Motivation

Lxyra's current component detector has limited text and icon localization on dense and high-resolution screens. Replacing it with a smaller network without measuring detection accuracy would be an invalid optimization.

Related designs:

- [ScreenParser / ScreenParse v2](https://huggingface.co/docling-project/ScreenParser): trains a large detector on dense, filtered GUI annotations at 1280-pixel input resolution. Its publicly documented training set is predominantly web-oriented. The leaf-element filtering and high-resolution supervision are more transferable than its parameter count.
- [OmniParser V2](https://github.com/microsoft/OmniParser): uses a detector and separate OCR/icon semantics. Lxyra retains the same modularity without claiming OCR where it has not been trained.
- [ScreenParse project](https://saidgurbuz.github.io/screenparse/): demonstrates the importance of dense labels and measures cross-domain transfer. Web results alone are not sufficient evidence for native Android or desktop performance.

These are engineering references, not transferred pretrained weights. Lxyra's compact model has independent randomly initialized parameters.

## Architecture

The experimental detector uses depthwise-separable convolution blocks, a conventional convolutional stem, four-scale feature fusion and a stride-four detection head. It produces the same three outputs as the legacy detector: class logits, logarithmic box sizes and center offsets.

The model contains **44,302 parameters**, compared with **818,934** in the original architecture. The parameter reduction is approximately 94.6%. This does not imply a similar reduction in runtime or memory on every device.

A local CPU-only PyTorch microbenchmark at 448 × 448 pixels and two threads measured median single-image forward passes of **52.97 ms** for the original randomly initialized architecture and **22.38 ms** for the compact one. The run used one transient CPU environment, without preprocessing, postprocessing or ONNX Runtime. These figures are diagnostic, not reproducible mobile performance guarantees, and neither model was trained for this speed comparison.

## Real supervised comparison

The `Detector architecture experiment` GitHub Actions workflow downloads a bounded sample from three public sources:

- ScreenParse v2 (CC BY 4.0), web interface screenshots
- RICO WidgetCaptioning (CC BY 4.0), native Android screenshots
- Mobile UI Design (Apache 2.0), annotated mobile design screenshots

The workflow reuses the existing source-specific parsers and per-app/domain split assignments. It trains **both architectures from scratch** for the same number of epochs, compares their validation metrics, and exports both models to ONNX. Runtime measures, checkpoints, evaluation JSON and ONNX numerical verification files are stored in a short-lived Actions artifact.

The workflow does not modify the repository's default ONNX weights. This deliberately limited experiment is a smoke-level quality comparison, not sufficient to declare either model superior across unseen apps and websites.

To run a larger representative experiment, increase the number and diversity of source screenshots, train sufficiently for convergence, use an independent test set, and compare mAP@0.50 and per-category precision and recall separately for Android and web. Do not use validation to select a model and then report the same validation as a blind test.

## Commands

```bash
pip install -e ".[train,dev]"
python -m pytest -q tests/test_compact_model.py
python scripts/benchmark_architectures.py --resolution 448 --threads 2
python -m lxyra.training.detector_cli train --architecture compact --data /path/to/real-dataset --out experiment/compact --epochs 3 --batch 4
python -m lxyra.training.detector_cli evaluate --data /path/to/real-dataset --split val --checkpoint experiment/compact/best.pt --out experiment/metrics.json
python -m lxyra.training.detector_cli export --checkpoint experiment/compact/best.pt --out experiment/model.onnx
```

The existing `v2` architecture remains the training default. Checkpoint metadata records the architecture so exporting and resuming cannot silently mix incompatible tensor shapes.

## Acceptance gate

Promote a new model only after it demonstrably improves a genuinely held-out benchmark in the target platform families without unacceptable per-category regressions. Measure CPU latency and ONNX parity separately. The mere presence of smaller weights or faster random-weight forward passes must not be used as a quality claim.
