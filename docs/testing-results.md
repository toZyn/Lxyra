# Coordinate integrity and crop inference test report

Date: 2026-11-20. The tests were run on CPU with the existing frozen ONNX model files. No weights were retrained or replaced as part of this branch.

## Automated checks

- 65 local unit and integration tests passed, including crop offsets, letterbox inversion, aspect-ratio changes, pixel-space remapping, EXIF orientation, explicit image padding and rotated capture mapping, overlap reporting, frame fingerprints and model output validation.
- Ruff lint and formatting checks passed.
- Packaged ONNX models, external tensor data and SHA-256 checksums passed verification with CPU inference.
- Source and wheel distributions built successfully, including the new coordinate-mapping module.

These checks verify implementation correctness and model loadability. They do not establish neural detection accuracy.

## Frozen-model detection comparison

A local group-separated real validation set of 167 screenshots was evaluated at confidence 0.15:

| Inference mode | mAP@0.50 overall | AP@0.50 text |
|---|---:|---:|
| Whole screenshot, default | 0.033411 | 0.099533 |
| Overlapping tiles, existing | 0.023742 | 0.107597 |
| Tiles with internal-edge filtering | 0.023858 | 0.108124 |
| Edge filtering with nonwrapping local maxima | 0.023858 | 0.108124 |

The experimental tile-edge filter slightly improves the existing tiled mode but neither tiled mode matches the default's general detection accuracy. Therefore the new filter is not enabled in the default public pipeline.

## Agent target center diagnostic

On 103 real test screenshots, with candidate confidence at least 0.25, the comparison measured whether a predicted candidate's center fell inside a same-category annotated bounding box. Matches were one-to-one.

| Measurement | Full image | Tiled |
|---|---:|---:|
| Candidates | 700 | 765 |
| Matched candidate centers | 188 | 175 |
| Candidate-center precision | 26.9% | 22.9% |
| Annotation coverage | 37.2% | 34.7% |

This is not a real click-success benchmark. It does not establish control clickability, visibility beneath overlays, post-click state or OCR correctness. Annotations from the WebUI subset often use wide text regions, which affects the suitability of those labels for fine-grained localization.

## Annotation quality diagnostic

An audit of the locally held 1,781-screenshot mixed dataset reported:

- 1,297 text regions occupying at least 75% of image width;
- 12 exact duplicate annotations;
- 4 unusually large actionable boxes;
- 0 groups found across different train/validation/test splits.

These flags are intended for manual review and do not prove label errors. Screenshots and datasets are not stored in the repository.

## Agent coordinate contract

The separate agent-output format retains the original screenshot pixel space. Physical or logical display coordinates are returned **only** when the caller explicitly provides display size and asserts either a full-display capture or a capture rectangle. For correct input geometry, the integer coordinate transform has less than one target-pixel quantization error per axis.

Every response includes a frame fingerprint and frame-scoped element IDs. All target positions are estimates from detected boxes and are marked `interactive_verified: false`. Correct coordinate transforms cannot compensate for incorrect neural localization.

## Reproduction

```bash
pip install -e ".[dev]"
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python scripts/verify_models.py
python -m build
python scripts/compare_tiling.py --root /path/to/mixed_v3 --split val --out validation_tiling.json
python scripts/evaluate_agent_targets.py --data-root /path/to/mixed_v3 --split test --threshold 0.25 --output agent_centers.json
python scripts/audit_annotations.py --data-root /path/to/mixed_v3 --output dataset_audit.json
```

For consistent numerical comparisons, use the same dataset manifests, image bytes, model hashes, confidence thresholds and source-family groupings. The image datasets are not distributed by this repository.
