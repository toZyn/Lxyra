import hashlib
import json

import pytest

from lxyra import CATEGORIES
from lxyra.promotion import (
    PromotionRejected,
    combine_documents,
    load_policy,
    select_categories,
)
from lxyra.schema import make_document


def report(split="val", button=0.1, text=0.2):
    def record(b, t):
        return {
            "metrics": {"ap50_by_category": {**{c: None for c in CATEGORIES}, "button": b, "text": t}},
            "samples": 10,
        }

    return {
        "dataset": "held-separate",
        "split": split,
        "samples": 30,
        "threshold": 0.15,
        "tiled": False,
        "metrics": record(button, text)["metrics"],
        "by_family": {
            key: record(button, text) for key in ("android", "web", "webui")
        },
    }


def test_candidate_button_selected_only_when_all_domains_improve():
    baseline = report(button=0.1)
    candidate = report(button=0.2)
    policy = select_categories(baseline, candidate)
    assert policy["candidate_categories"] == ["button"]
    assert policy["enabled"] is True
    assert policy["decisions"]["text"].startswith("baseline")


def test_domain_regression_blocks_candidate_even_when_average_improves():
    baseline = report(button=0.1)
    candidate = report(button=0.2)
    candidate["by_family"]["android"]["metrics"]["ap50_by_category"]["button"] = 0.099
    policy = select_categories(baseline, candidate)
    assert policy["candidate_categories"] == []
    assert "android" in policy["decisions"]["button"]


def test_faster_but_worse_model_is_rejected():
    result = select_categories(report(button=0.2), report(button=0.1))
    assert result["enabled"] is False


@pytest.mark.parametrize("bad_field", ["dataset", "samples", "threshold", "tiled"])
def test_incompatible_evaluations_are_rejected(bad_field):
    baseline = report()
    candidate = report(button=0.3)
    candidate[bad_field] = {"dataset": "other", "samples": 50, "threshold": 0.9, "tiled": True}[bad_field]
    with pytest.raises(PromotionRejected):
        select_categories(baseline, candidate)


def test_test_set_must_not_be_used_to_select():
    with pytest.raises(PromotionRejected):
        select_categories(report("test"), report("test", button=0.4))


def test_required_domains_must_be_present():
    candidate = report(button=0.3)
    del candidate["by_family"]["webui"]
    with pytest.raises(PromotionRejected):
        select_categories(report(), candidate)


def test_text_is_always_preserved_even_if_candidate_is_better():
    policy = select_categories(report(text=0.1), report(text=0.9))
    assert "text" not in policy["candidate_categories"]


def test_policy_tied_to_exact_candidate_onnx_hash(tmp_path):
    candidate = tmp_path / "detector.onnx"
    candidate.write_bytes(b"onnx-placeholder-for-hash-check")
    policy = {
        "schema_version": 1,
        "selection_split": "val",
        "enabled": True,
        "independent_test_passed": True,
        "candidate_categories": ["button"],
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    assert load_policy(path, candidate)["candidate_categories"] == ["button"]
    candidate.write_bytes(b"changed")
    with pytest.raises(PromotionRejected):
        load_policy(path, candidate)


def document(elements):
    return make_document(
        [
            {"category": category, "bbox": bbox, "confidence": confidence}
            for category, bbox, confidence in elements
        ],
        (200, 250),
        0.15,
    )


def test_unchanged_categories_preserve_exact_boxes_and_scores():
    baseline = document([
        ("text", [12, 20, 90, 18], 0.8),
        ("button", [50, 80, 100, 30], 0.75),
        ("icon", [35, 170, 24, 24], 0.9),
    ])
    candidate = document([
        ("text", [20, 15, 50, 18], 0.95),
        ("button", [55, 85, 92, 29], 0.92),
        ("icon", [30, 160, 50, 50], 0.88),
    ])
    output = combine_documents(baseline, candidate, ["button"])
    for category in ("text", "icon"):
        got = [e for e in output["elements"] if e["category"] == category]
        old = [e for e in baseline["elements"] if e["category"] == category]
        assert [(e["bbox"], e["confidence"]) for e in got] == [
            (e["bbox"], e["confidence"]) for e in old
        ]
    assert [e["bbox"] for e in output["elements"] if e["category"] == "button"] == [
        e["bbox"] for e in candidate["elements"] if e["category"] == "button"
    ]


def test_no_selected_categories_return_original_document():
    original = document([("text", [40, 50, 70, 20], 0.7)])
    candidate = document([("button", [40, 50, 70, 20], 0.9)])
    assert combine_documents(original, candidate, []) is original


def test_invalid_category_selection_rejected():
    original = document([])
    with pytest.raises(PromotionRejected):
        combine_documents(original, original, ["text"])
    with pytest.raises(PromotionRejected):
        combine_documents(original, original, ["not-real"])
