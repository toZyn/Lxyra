import json
from pathlib import Path

import onnxruntime as ort

from .schema import validate_document
from .scoring import apply_scores, text_evidence
from .detector import infer_onnx


def load_policy(path):
    with Path(path).open() as stream:
        policy = json.load(stream)
    selected = policy.get("selected")
    if not selected or not policy.get("passing_configs"):
        raise ValueError("Text reranking was not validated")
    if not (0 <= selected["alpha"] <= 1 and selected["measure"] in (0, 1, 2)):
        raise ValueError("Invalid reranking policy")
    return selected


def infer_v4(image, detector, text_model=None, policy=None, threshold=0.15, tiled=False):
    general = (
        detector
        if hasattr(detector, "run")
        else ort.InferenceSession(str(detector), providers=["CPUExecutionProvider"])
    )
    result = infer_onnx(image, general, threshold, tiled)
    if text_model is None or policy is None:
        result["config"]["text_reranking"] = False
        validate_document(result)
        return result
    selected = load_policy(policy) if isinstance(policy, (str, Path)) else policy
    text = (
        text_model
        if hasattr(text_model, "run")
        else ort.InferenceSession(str(text_model), providers=["CPUExecutionProvider"])
    )
    before = result["elements"]
    signals = text_evidence(image, text, before)
    after = apply_scores(before, signals, selected["alpha"], selected["measure"])
    for old, new in zip(before, after):
        if old["category"] == "text":
            old["confidence_original"] = old["confidence"]
            old["confidence"] = round(float(new["confidence"]), 6)
    result["config"]["text_reranking"] = True
    result["config"]["text_reranking_alpha"] = selected["alpha"]
    result["config"]["text_evidence_mode"] = ["mean", "percentile75", "max"][selected["measure"]]
    validate_document(result)
    return result
