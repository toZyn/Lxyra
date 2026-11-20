import argparse
import json
import sys
from pathlib import Path

import onnxruntime as ort
from PIL import Image, ImageOps

from .coordinates import change_box, make_crop
from .reranking import infer_v4, load_policy
from .schema import draw_overlay, validate_document
from .screen_coordinates import ScreenTransform, agent_targets, fingerprint

import math

import numpy as np


MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"


def _session(path, providers):
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Model file not found: {file}")
    return ort.InferenceSession(str(file), providers=providers)


def _policy(path):
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    selected = document.get("selected")
    if not selected:
        raise ValueError("No validated refinement policy is available")
    config = selected["config"]
    if not (
        0 <= config["alpha"] <= 1 and 0 <= config["quality"] <= 1 and 0 < config["restrict"] <= 1
    ):
        raise ValueError("Refinement policy contains invalid parameters")
    return config


def _refine(image, document, session, policy):
    width, height = image.size
    elements = document["elements"]
    indices = [index for index, node in enumerate(elements) if node["category"] == "text"]
    changed = 0

    for start in range(0, len(indices), 64):
        batch_indices = indices[start : start + 64]
        batch = (
            np.asarray(
                [make_crop(image, elements[index]["bbox"]) for index in batch_indices],
                dtype=np.float32,
            )
            .transpose(0, 3, 1, 2)
            .copy()
            / 255
        )
        outputs = session.run(None, {"crops": batch})[0]
        for index, vector in zip(batch_indices, outputs):
            probability = 1 / (1 + math.exp(-float(np.clip(vector[4], -30, 30))))
            if probability < policy["quality"]:
                continue
            node = elements[index]
            delta = np.clip(vector[:4], -policy["restrict"], policy["restrict"])
            updated = change_box(node["bbox"], delta, policy["alpha"], image.size)
            if updated == node["bbox"]:
                continue
            node["bbox_before_refinement"] = node["bbox"][:]
            node["bbox"] = [round(float(value), 3) for value in updated]
            x, y, w, h = node["bbox"]
            node["bbox_normalized"] = [
                round(x / width, 6),
                round(y / height, 6),
                round(w / width, 6),
                round(h / height, 6),
            ]
            changed += 1

    document["config"]["box_refinement"] = True
    document["config"]["box_refinement_changed"] = changed
    validate_document(document)
    return document


class ScreenParser:
    def __init__(self, model_directory=None, providers=None, enable_refinement=False):
        directory = Path(model_directory) if model_directory else MODEL_DIRECTORY
        providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.detector = _session(directory / "detector.onnx", providers)
        self.text = _session(directory / "text.onnx", providers)
        self.reranking = load_policy(directory / "rerank.json")
        self.enable_refinement = enable_refinement
        self.refiner = None
        self.refinement = None
        if enable_refinement:
            self.refiner = _session(directory / "refiner.onnx", providers)
            self.refinement = _policy(directory / "refine.json")

    def predict(self, image, threshold=0.15, tiled=False):
        if not isinstance(image, Image.Image):
            raise TypeError("predict expects a PIL Image")
        image = ImageOps.exif_transpose(image).convert("RGB")
        document = infer_v4(
            image,
            self.detector,
            self.text,
            self.reranking,
            threshold=threshold,
            tiled=tiled,
        )
        if self.refiner is None:
            document["config"]["box_refinement"] = False
            return document
        return _refine(image, document, self.refiner, self.refinement)


def _parse_dimensions(value):
    try:
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        width, height = (int(part) for part in parts)
        if width <= 0 or height <= 0:
            raise ValueError
        return (width, height)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT with positive integers") from error


def _parse_rectangle(value):
    try:
        parts = value.split(",")
        if len(parts) != 4:
            raise ValueError
        return tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LEFT,TOP,WIDTH,HEIGHT in display pixels") from error


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lxyra",
        description="Parse screenshot components into structured JSON.",
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--refine-text", action="store_true")
    parser.add_argument("--agent-output", type=Path)
    parser.add_argument("--display-size", type=_parse_dimensions)
    region = parser.add_mutually_exclusive_group()
    region.add_argument("--capture-rect", type=_parse_rectangle)
    region.add_argument("--full-display-capture", action="store_true")
    parser.add_argument("--agent-confidence", type=float, default=0.25)
    arguments = parser.parse_args(argv)

    if not 0 <= arguments.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    if arguments.capture_rect or arguments.full_display_capture:
        if not arguments.display_size or not arguments.agent_output:
            parser.error("Display mapping requires --display-size and --agent-output")
    if arguments.display_size and not (arguments.capture_rect or arguments.full_display_capture):
        parser.error("Declare --capture-rect or --full-display-capture explicitly")
    if arguments.agent_output is None and arguments.agent_confidence != 0.25:
        parser.error("--agent-confidence requires --agent-output")

    with Image.open(arguments.image) as screenshot:
        image = ImageOps.exif_transpose(screenshot).convert("RGB")

    engine = ScreenParser(
        model_directory=arguments.model_directory,
        enable_refinement=arguments.refine_text,
    )
    document = engine.predict(image, threshold=arguments.threshold, tiled=arguments.tile)
    content = json.dumps(document, indent=2, allow_nan=False) + "\n"

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)

    if arguments.overlay:
        arguments.overlay.parent.mkdir(parents=True, exist_ok=True)
        draw_overlay(image, document).save(arguments.overlay)

    if arguments.agent_output:
        geometry = None
        if arguments.display_size:
            if arguments.full_display_capture:
                geometry = ScreenTransform.full_screen(image.size, arguments.display_size)
            else:
                geometry = ScreenTransform(
                    image.size, arguments.display_size, arguments.capture_rect
                )
        agent_document = agent_targets(
            document,
            geometry,
            arguments.agent_confidence,
            frame_sha256=fingerprint(image),
        )
        arguments.agent_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.agent_output.write_text(
            json.dumps(agent_document, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
