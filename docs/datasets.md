# Data and training

Lxyra models were trained from random initialization with supervised annotation data. The repository includes model definitions and data preparation utilities; screenshot corpora are not redistributed.

## Sources

| Dataset | Contribution | License reported by its dataset card |
|---|---|---|
| ScreenParse V2 | Web component boxes and semantic categories | CC BY 4.0 |
| RICO WidgetCaptioning | Android screenshots and visible widget annotations | CC BY 4.0 |
| Mobile UI Design | Mobile design component annotations | Apache 2.0 |
| WebUI | Website boxes and layout context | MIT |
| Original launcher generator | Additional generated launcher images | Project-generated |

Review the upstream dataset cards and their terms before downloading, redistributing or publishing derived annotations. Source labels do not share a perfect taxonomy; text-region annotations can represent different granularities between datasets.

## Data preparation

The training utilities support fetching and normalizing annotations, verifying bounding boxes, generating splits by application or website domain, deduplicating exact screenshots and building a training manifest. Dataset preparation code is under `src/lxyra/training`:

```bash
python -m lxyra.training.hf_import --help
python -m lxyra.training.merge --help
python -m lxyra.training.refiner_data --help
```

The importers use the Hugging Face Dataset Viewer API and may encounter rate limits. The original development dataset and imported training images are not part of this repository. A complete training reproduction requires the relevant source data, identical manifests, dependency versions and random seeds. For this reason, a local retraining run with newly downloaded samples may not reproduce historical metrics bit-for-bit.

## Training commands

The component detector, text-mask model and bounding-box refiner each have their own training entry points:

```bash
python -m lxyra.training.detector_cli train --help
python -m lxyra.training.finetune --help
python -m lxyra.training.text_cli train --help
python -m lxyra.training.refiner_cli train --help
```

Training relies on PyTorch and is intentionally not part of the inference installation. Exported ONNX models are checked for numerical correspondence with their PyTorch counterparts during each experiment.

## Data integrity

Keep datasets, downloaded images, checkpoints and local reports outside Git. Avoid mixing repeated screenshots or images from one app/domain between training and test splits. Reserve an independent benchmark for quality claims. Do not upload user screenshots, private application screens or access credentials.
