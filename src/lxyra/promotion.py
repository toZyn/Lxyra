"""Validation-gated, class-specific detector promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import CATEGORIES
from .schema import make_document, validate_document


class PromotionRejected(ValueError):
    """A model or policy fails the release integrity contract."""


def sha256_file(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def select_categories(
    baseline: dict, candidate: dict, required_families=("android", "web", "webui")
) -> dict:
    """Choose only classes improved across every measured domain.

    Selection must use validation data. An independent test remains mandatory.
    """
    for key in ("dataset", "split", "samples", "threshold", "tiled"):
        if baseline.get(key) != candidate.get(key):
            raise PromotionRejected(f"Evaluation metadata mismatch: {key}")
    if baseline.get("split") != "val":
        raise PromotionRejected("Never select a model using test results")

    before = baseline.get("by_family", {})
    after = candidate.get("by_family", {})
    if set(before) != set(after) or not set(required_families).issubset(before):
        raise PromotionRejected("The required domain evaluations are missing")
    if sum(group["samples"] for group in before.values()) != baseline["samples"]:
        raise PromotionRejected("Inconsistent baseline sample counts")
    if sum(group["samples"] for group in after.values()) != candidate["samples"]:
        raise PromotionRejected("Inconsistent candidate sample counts")

    chosen = []
    decisions = {}
    for category in CATEGORIES:
        if category == "text":
            decisions[category] = "baseline: text preservation"
            continue

        original_ap = baseline["metrics"]["ap50_by_category"].get(category)
        candidate_ap = candidate["metrics"]["ap50_by_category"].get(category)
        if original_ap is None or candidate_ap is None or candidate_ap <= original_ap:
            decisions[category] = "baseline: no global AP improvement"
            continue

        regressions = []
        for family in sorted(before):
            original_group = before[family]
            candidate_group = after[family]
            if original_group["samples"] != candidate_group["samples"]:
                raise PromotionRejected(f"Different sample counts for {family}")
            a = original_group["metrics"]["ap50_by_category"].get(category)
            b = candidate_group["metrics"]["ap50_by_category"].get(category)
            if (a is None) != (b is None):
                raise PromotionRejected(f"Inconsistent class support in {family}/{category}")
            if a is not None and b + 1e-12 < a:
                regressions.append(family)
        if regressions:
            decisions[category] = "baseline: regression in " + ", ".join(regressions)
            continue
        chosen.append(category)
        decisions[category] = "candidate: improved in validation"

    return {
        "schema_version": 1,
        "selection_split": "val",
        "required_families": list(required_families),
        "candidate_categories": chosen,
        "decisions": decisions,
        "enabled": bool(chosen),
        "status": "experimental; independent held-out tests still required",
    }


def load_policy(path: str | Path, candidate_path: str | Path) -> dict:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = policy.get("candidate_categories")
    if (
        policy.get("schema_version") != 1
        or policy.get("selection_split") != "val"
        or not isinstance(categories, list)
        or policy.get("independent_test_passed") is not True
        or policy.get("enabled") is not True
        or len(set(categories)) != len(categories)
        or any(category not in CATEGORIES or category == "text" for category in categories)
    ):
        raise PromotionRejected("Invalid or unvalidated candidate category policy")
    expected = policy.get("candidate_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or sha256_file(candidate_path) != expected
    ):
        raise PromotionRejected("Candidate ONNX weights do not match the validated policy")
    return policy


def combine_documents(baseline: dict, candidate: dict, selected_categories: list[str]) -> dict:
    """Preserve entire baseline categories not explicitly promoted."""
    validate_document(baseline)
    validate_document(candidate)
    if baseline["image"] != candidate["image"]:
        raise PromotionRejected("Candidate and baseline use different screenshot dimensions")
    selected = set(selected_categories)
    if "text" in selected or not selected.issubset(CATEGORIES):
        raise PromotionRejected("Text cannot be replaced; unknown classes are forbidden")
    if not selected:
        return baseline

    elements = []
    for document, replace_selected in ((baseline, False), (candidate, True)):
        for element in document["elements"]:
            if (element["category"] in selected) != replace_selected:
                continue
            elements.append(
                {
                    "category": element["category"],
                    "bbox": list(element["bbox"]),
                    "confidence": element["confidence"],
                }
            )
    width, height = baseline["image"]["width"], baseline["image"]["height"]
    merged = make_document(elements, (width, height), baseline["config"]["threshold"])
    merged["config"].update(baseline["config"])
    merged["config"]["experimental_candidate_categories"] = sorted(selected)
    return merged
