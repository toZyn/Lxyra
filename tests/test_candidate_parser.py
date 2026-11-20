import hashlib
import json

from PIL import Image

from lxyra.candidate_parser import ValidatedCandidateParser
from lxyra.pipeline import MODEL_DIRECTORY, ScreenParser
from lxyra.schema import validate_document


def test_candidate_parser_preserves_baseline_categories(tmp_path):
    model = MODEL_DIRECTORY / "detector.onnx"
    policy_path = tmp_path / "candidate_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_split": "val",
                "independent_test_passed": True,
                "enabled": True,
                "candidate_categories": ["button"],
                "candidate_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    screenshot = Image.new("RGB", (448, 448), "#f5f5f5")
    original = ScreenParser().predict(screenshot, threshold=0.2)
    combined = ValidatedCandidateParser(model, policy_path).predict(screenshot, threshold=0.2)
    validate_document(combined)
    for category in ("text", "icon", "input", "container"):
        before = [
            (item["bbox"], item["confidence"])
            for item in original["elements"]
            if item["category"] == category
        ]
        after = [
            (item["bbox"], item["confidence"])
            for item in combined["elements"]
            if item["category"] == category
        ]
        assert before == after
