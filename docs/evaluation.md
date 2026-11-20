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
