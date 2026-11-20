# Architecture

## Inputs and outputs

The component detector receives RGB screenshots normalized to float32 values in [0, 1]. The default preprocessing letterboxes a screenshot to 448 × 448 with RGB (127, 127, 127) padding. Long screenshots may be processed in overlapping crops. Predictions are decoded back into original pixel coordinates and filtered using class-specific non-maximum suppression.

The public output is JSON schema version 1.0. Each element has a stable identifier within the prediction, category, bounding box [x, y, width, height], normalized bounding box, confidence, reading order and optional parent. The parent hierarchy is based on geometric containment, not a trained relationship network. The same input, models and configuration produce deterministic output on the same runtime.

## Processing stages

1. **Component detection.** A convolutional multiscale backbone predicts class logits, bounding-box dimensions and cell-center offsets over a stride-four grid. Categories are button, text, input, checkbox, switch, icon, image, container, navigation and select.
2. **Text confidence.** An independent convolutional text-region model predicts a stride-four probability mask. The maximum text probability inside each detected text box is used to adjust its score. This stage never adds or removes boxes.
3. **Optional text refinement.** A small neural model takes a 96 × 96 crop around each candidate text box. It predicts center and scale corrections. This stage is opt-in because broader evaluation found regressions.
4. **Document generation.** Pixel boxes are validated, reading order and containment-based relationships are added, and the result is serialized.

All models are shipped as ONNX with standard operators. The component detector uses the ONNX external-data specification to store tensor weights in adjacent binary shards; no custom runtime or reconstruction is needed. ONNX Runtime CPU is the default execution backend. The training code is separated from the importable inference package and uses PyTorch only in `lxyra.training`.

## Screen-control agent integration

The separate agent-targets output uses frame-local IDs, optional full-display coordinate transforms and a screenshot content fingerprint. The mapping requires declared crop geometry or an explicit full-display capture assertion. Target centers are estimates, not verified clickable positions. See [agent integration](agent-integration.md) for the coordinate contract and validation tests.

## Limitations

The detector does not recover inaccessible or hidden elements, understand behaviors, read text strings or reliably infer control semantics. The text stage localizes regions rather than producing word-level transcription or polygons. The current evaluation does not justify use as an autonomous GUI interaction engine.
