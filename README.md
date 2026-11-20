# Lxyra

Lxyra is a portable screen parsing library for detecting visible interface components and exporting their structure as JSON. It runs directly on screenshots without application-specific APIs, accessibility trees or DOM access.

The project includes a component detector, a text confidence model and an optional text bounding-box refiner. All inference models are distributed in ONNX format, including standard external tensor data for the component detector. Lxyra currently provides experimental detection quality; it is not an OCR engine or a production-grade universal UI parser.

## Installation

Python 3.11 or later is required.

```bash
git clone https://github.com/toZyn/Lxyra.git
cd Lxyra
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

The package includes its ONNX models, standards-compliant external tensor data and validation policies. An NVIDIA GPU is not required.

## Usage

```bash
lxyra screenshot.png --output results.json --overlay results.png
```

For long or narrow screenshots, overlapping crops can preserve more detail, but current benchmarks show domain-specific accuracy regressions with this mode:

```bash
lxyra screenshot.png --tile --output results.json
```

The experimental bounding-box refiner is disabled by default because it regresses on part of the independent WebUI evaluation set:

```bash
lxyra screenshot.png --refine-text --output refined.json
```

For integration with an agent that needs coordinates in its screen-control API:

```bash
lxyra screenshot.png --agent-output targets.json
lxyra resized-full-screen.png --agent-output targets.json --display-size 1080x2400 --full-display-capture
lxyra cropped.png --agent-output targets.json --display-size 1080x2400 --capture-rect 100,200,800,1200
```

The agent output always identifies estimated targets, never verified clickable controls. If screen size and crop placement are unknown, device coordinates remain unavailable. See [agent integration](docs/agent-integration.md).

The Python API reuses loaded inference sessions across screenshots:

```python
from PIL import Image
from lxyra.pipeline import ScreenParser

parser = ScreenParser()
with Image.open("screenshot.png") as image:
    document = parser.predict(image)
```

The output contains detected categories, pixel-space bounding boxes, normalized boxes, confidence, reading order and conservative containment-based parent links:

```json
{
  "schema_version": "1.0",
  "image": {"width": 1080, "height": 1920},
  "config": {"threshold": 0.15, "text_reranking": true, "box_refinement": false},
  "elements": [],
  "ocr_available": false
}
```

An empty `elements` array is valid for screenshots where no candidates meet the threshold. The example above illustrates the schema, not a model prediction.

## Project layout

```text
src/lxyra/
  models/         ONNX models, policies and checksums
  training/       dataset preparation, architectures, training and evaluation
  pipeline.py     public inference API and CLI
  detector.py     visual component inference
  reranking.py    text confidence adjustment
  coordinates.py  text-region geometry
  geometry.py     box decoding and suppression
  schema.py       output validation and visualization
tests/            offline and ONNX regression checks
docs/             architecture, datasets and evaluation
.github/workflows/ automated validation and releases
```

## Models and quality

The default pipeline uses a convolutional component detector (448 × 448 input) and a separately trained text confidence model. Both run with ONNX Runtime. An optional additional model refines existing text boxes but is not enabled by default.

The 69-screen real test set used during development produced the following results at the original evaluation threshold:

| Model | mAP@0.50 overall | AP@0.50 text |
|---|---:|---:|
| Detector | 0.03645 | 0.10422 |
| Detector + text reranking | 0.03803 | 0.11998 |
| Experimental box refinement | 0.03845 | 0.12425 |

These are small, previously inspected development test sets. Results do not establish state-of-the-art performance, and text localization is not equivalent to reading text. On a broader WebUI sample, the optional refiner showed a regression. See [evaluation methodology](docs/evaluation.md).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
python -m pytest -q
python scripts/verify_models.py
python -m build
```

Training needs optional dependencies and source datasets:

```bash
pip install -e ".[train,dev]"
```

See [agent coordinate mapping](docs/agent-integration.md), [architecture](docs/architecture.md), [dataset and training notes](docs/datasets.md), [evaluation](docs/evaluation.md), [testing results](docs/testing-results.md), and [contributing](CONTRIBUTING.md) for details.

## License

Code is released under the [MIT License](LICENSE). Third-party datasets remain governed by their respective licenses; they are not redistributed by this repository.
