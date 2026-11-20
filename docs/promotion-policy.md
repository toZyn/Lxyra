# Candidate detector promotion

Lxyra does not promote detector weights based on parameter count, CPU speed, or a single aggregate detection metric. Its default inference models remain unchanged until a candidate passes reproducible comparisons.

## Current protection

V6 increased overall mAP on the existing 103-image real test set from 0.02883 to 0.03385 when evaluated with the text model. It also reduced text AP in the small Android and WebUI test subsets. The latter regressions disqualify V6 as a whole-model replacement.

The optional classwise promotion mechanism keeps **all baseline text detections**, including their coordinates and confidence, and allows an experimental detector to replace other categories only when all measured source families demonstrate nondecreasing AP@0.50. It is not enabled in the default `ScreenParser`.

## Quality gate

A candidate needs two same-dataset comparison pairs: baseline/candidate validation reports and baseline/candidate test reports. Both must use the same threshold, tiling configuration, class vocabulary, sample count and domain splits. Android, web and WebUI are mandatory.

On validation, a category may be selected if it improves global AP@0.50 and does not reduce AP@0.50 in any evaluated domain. Text is never selected. On test, **every selected category must again avoid AP regressions** in every domain. If any selected category fails, the entire candidate is rejected. This gate tests an exact model artifact via a recorded SHA-256 checksum.

Run the gate with reports produced by the training evaluation CLI:

```bash
python scripts/guard_candidate_release.py \
  --baseline-val reports/baseline_val.json \
  --candidate-val reports/candidate_val.json \
  --baseline-test reports/baseline_test.json \
  --candidate-test reports/candidate_test.json \
  --candidate-onnx models/candidate.onnx \
  --output reports/candidate_policy.json
```

The command exits unsuccessfully on rejection. It does not modify deployed weights.

## Opt-in inference

Only a policy that passed validation and the subsequent test, with the candidate model's matching SHA-256, can be loaded:

```python
from PIL import Image
from lxyra.candidate_parser import ValidatedCandidateParser

parser = ValidatedCandidateParser(
    candidate_model="models/candidate.onnx",
    promotion_policy="reports/candidate_policy.json",
)
with Image.open("screen.png") as image:
    result = parser.predict(image)
```

This mode requires running both detectors and therefore costs extra time and memory. Unselected categories are copied from V4. The default pipeline remains a single detector plus the separately trained text scoring model.

## Interpretation

No finite benchmark can guarantee nonregression for every unknown screenshot. Classwise promotion is an auditable, conservative release check, not proof of universal accuracy or successful GUI interaction. Source-family test subsets are small and the historical test data has already been inspected during development. Model releases still need fresh, application-disjoint data and reliable text and word annotations.

The original V4 weights and model checksums must not be changed as part of a candidate experiment. Do not use test metrics to choose training epochs or select candidate categories.
