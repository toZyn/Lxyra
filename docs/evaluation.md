# Evaluation

## Protocol

Evaluate frozen weights on screenshots separated by application or web domain. Report AP@0.50 by category, overall mAP@0.50, precision and recall, resource consumption and PyTorch-to-ONNX numerical differences. Compare models on the same exact files, categories, thresholds and postprocessing configuration.

## Recorded results

The original real test set contains 69 screenshots: 11 Android and 58 web. Results at a 0.15 detection threshold:

| System | Overall mAP@0.50 | Text AP@0.50 |
|---|---:|---:|
| Component detector | 0.036449 | 0.104217 |
| With text score adjustment | 0.038026 | 0.119985 |
| With optional text box refinement | 0.038452 | 0.124247 |

The optional refinement model was also evaluated on an expanded set of 103 real screenshots (11 Android, 74 web and 18 WebUI). Although overall text AP increased from 0.074573 to 0.078374, WebUI text AP decreased from 0.001108 to 0.000935. This fails the project's no-regression promotion criterion; the refiner remains disabled by default.

The development process has previously inspected the original 69 test screenshots. They should not be treated as a completely blind independent leaderboard benchmark. Synthetic launcher results are separately reported and must not be combined with real-interface scores to imply generalization.

## Model integrity

Every distributed ONNX file is checked for model validity and input/output compatibility. `scripts/verify_models.py` verifies hashes, graphs and a real CPU inference pass. The model checksum manifest is in `src/lxyra/models/checksums.json`.

## Release criteria

A new default model must pass unit tests, ONNX validation, reproducible same-set comparisons and per-domain precision/recall checks. Improvements in one subset do not justify silently replacing the default if another domain regresses.

## Agent target coordinate evaluation

For use by computer-control agents, `scripts/evaluate_agent_targets.py` computes a separate and stricter diagnostic: the proportion of predicted component centers that land inside a same-category annotated bounding box, without matching the same annotation twice. Both precision and annotation coverage are reported by source family. This is **not** a success rate for clicking, since real clickability, occlusion and active UI state are not annotated.

```bash
python scripts/evaluate_agent_targets.py --data-root /path/to/annotated-data --split test --threshold 0.25 --output center-metrics.json
```

This tool requires an existing local dataset manifest and image files. No private screenshots or datasets are committed to the repository.

## Crop inference comparison

On 167 real validation screenshots from the existing mixed benchmark, these frozen-model modes were compared at the same 0.15 threshold:

| Mode | Overall mAP@0.50 | Text AP@0.50 |
|---|---:|---:|
| Whole screenshot | 0.033411 | 0.099533 |
| Current overlapping crop mode | 0.023742 | 0.107597 |
| With internal crop-edge filtering | 0.023858 | 0.108124 |

Crop-edge filtering improves the tiled mode slightly but does not restore general detection quality; the candidate change remains experimental. The extra local-peak boundary correction did not change AP on this validation set. The default full-screenshot mode and model weights remain unchanged.

The geometric crop coordinate transform is tested independently from detection quality. Correctly returning a bounding box to full-image pixel coordinates is not evidence that the neural model located the right element.

## Agent target center diagnostics

On 103 previously inspected real test screenshots, with a minimum candidate confidence of 0.25, 188 of 700 candidate centers from full-frame inference fell within a same-category annotated actionable box (26.9% one-to-one center-match precision and 37.2% annotation coverage). Tiled inference yielded 175 of 765 candidates (22.9% center-match precision and 34.7% coverage).

This diagnostic is *not* a measured click-success rate. Category labels may be coarse and some WebUI text annotations cover full rows rather than individual rendered words or controls. Automated interaction should not treat these centers as guaranteed hit targets; uncertain targets require a fresh observation and independent confirmation.
