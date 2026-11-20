import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from lxyra.pipeline import MODEL_DIRECTORY, ScreenParser
from lxyra.schema import validate_document


@pytest.fixture(scope="module")
def parser():
    return ScreenParser()


def test_default_inference_is_portable(parser):
    image = Image.new("RGB", (480, 850), "white")
    ImageDraw.Draw(image).text((40, 60), "Settings", fill="black")
    document = parser.predict(image, threshold=0.2)
    validate_document(document)
    assert document["image"] == {"width": 480, "height": 850}
    assert document["config"]["box_refinement"] is False
    assert document["config"]["text_reranking"] is True


def test_deterministic_inference(parser):
    image = Image.new("RGB", (448, 448), "#d9e1ed")
    first = parser.predict(image, threshold=0.2)
    second = parser.predict(image, threshold=0.2)
    assert first == second


def test_optional_refiner_preserves_other_components(parser):
    image = Image.new("RGB", (448, 448), "white")
    base = parser.predict(image, threshold=0.15)
    refined = ScreenParser(enable_refinement=True).predict(image, threshold=0.15)
    assert len(base["elements"]) == len(refined["elements"])
    for before, after in zip(base["elements"], refined["elements"]):
        assert before["id"] == after["id"]
        assert before["category"] == after["category"]
        if before["category"] != "text":
            assert before["bbox"] == after["bbox"]


def test_missing_model_directory_fails():
    with pytest.raises(FileNotFoundError):
        ScreenParser(model_directory=Path("/nonexistent/lxyra/models"))


def test_cli_generates_json(tmp_path):
    image_path = tmp_path / "screen.png"
    output = tmp_path / "results.json"
    Image.new("RGB", (320, 640), "white").save(image_path)
    result = subprocess.run(
        [sys.executable, "-m", "lxyra", str(image_path), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    doc = json.loads(output.read_text(encoding="utf-8"))
    validate_document(doc)


def test_inference_import_does_not_load_training_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lxyra.pipeline; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_packaged_models_exist():
    for item in ("detector.onnx", "text.onnx", "refiner.onnx", "rerank.json", "refine.json"):
        assert (MODEL_DIRECTORY / item).is_file()


def test_tiled_model_output_stays_in_full_frame_coordinates(parser):
    screenshot = Image.new("RGB", (360, 1280), "#dedede")
    document = parser.predict(screenshot, threshold=0.25, tiled=True)
    validate_document(document)
    for node in document["elements"]:
        x, y, width, height = node["bbox"]
        assert 0 <= x and 0 <= y
        assert x + width <= screenshot.width + 0.01
        assert y + height <= screenshot.height + 0.01
