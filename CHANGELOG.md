# Changelog

## Unreleased

- Added frame-scoped agent target data, explicit capture-to-display coordinate mapping, and stale-frame fingerprints.
- Added tests for crop offsets, image orientation, projected bounding boxes, and invalid mapping.
- Added annotation-quality and action-center diagnostics.
- Tested alternate tile-edge handling against real validation data; retained the previous default due to measured regression.


## 0.1.0

- Added a packaged ONNX screen parser with structured JSON output.
- Included the component detector and text confidence model in the default runtime.
- Kept neural text-box refinement optional after domain-specific regression checks.
- Added training utilities, dataset documentation, model integrity verification and automated tests.
- Added GitHub Actions for linting, testing, model verification, security analysis and releases.
