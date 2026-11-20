import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from lxyra.pipeline import MODEL_DIRECTORY, ScreenParser
from lxyra.schema import validate_document


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    expected = {"detector.onnx", "text.onnx", "refiner.onnx", "rerank.json", "refine.json"}
    if not expected.issubset(manifest):
        raise ValueError("Missing required model files")
    if not any(
        filename.startswith("detector.weights.") and filename.endswith(".bin")
        for filename in manifest
    ):
        raise ValueError("External ONNX weights are missing")
    if set(manifest) != {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.json"
    }:
        raise ValueError("Model manifest does not cover all files")

    for filename, details in manifest.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != details["sha256"] or len(data) != details["bytes"]:
            raise ValueError(f"Model checksum mismatch: {filename}")
        if filename.endswith(".onnx"):
            model = onnx.load(str(path))
            onnx.checker.check_model(model)

    runtime = ScreenParser(model_directory=directory)
    samples = (
        Image.new("RGB", (448, 448), "#ffffff"),
        Image.new("RGB", (360, 780), "#dedede"),
    )
    for image in samples:
        document = runtime.predict(image, threshold=0.20)
        validate_document(document)

    optional = ScreenParser(model_directory=directory, enable_refinement=True)
    document = optional.predict(samples[0], threshold=0.20)
    validate_document(document)

    model = ort.InferenceSession(
        str(directory / "detector.onnx"),
        providers=["CPUExecutionProvider"],
    )
    output = model.run(None, {"images": np.zeros((1, 3, 448, 448), dtype=np.float32)})
    if len(output) != 3 or not all(np.isfinite(tensor).all() for tensor in output):
        raise ValueError("Model output is invalid")

    print(json.dumps({"models": len(manifest), "cpu_inference": "passed", "checksums": "passed"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=MODEL_DIRECTORY)
    arguments = parser.parse_args()
    verify(arguments.directory)


if __name__ == "__main__":
    main()
