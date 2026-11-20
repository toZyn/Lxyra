import json

from lxyra.pipeline import MODEL_DIRECTORY


def test_model_manifest_matches_files():
    import hashlib

    manifest = json.loads((MODEL_DIRECTORY / "checksums.json").read_text(encoding="utf-8"))
    for filename, entry in manifest.items():
        content = (MODEL_DIRECTORY / filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert len(content) == entry["bytes"]


def test_model_names_and_shapes():
    import onnx

    models = {
        "detector.onnx": 3,
        "text.onnx": 1,
        "refiner.onnx": 1,
    }
    for filename, output_count in models.items():
        model = onnx.load(str(MODEL_DIRECTORY / filename))
        onnx.checker.check_model(model)
        assert len(model.graph.output) == output_count
