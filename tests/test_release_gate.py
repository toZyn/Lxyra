import hashlib
import pytest

from lxyra import CATEGORIES
from scripts.guard_candidate_release import make_release_policy, verify_holdout


def report(split, button=0.10, text=0.20):
    def score(b, t):
        return {
            "metrics": {
                "ap50_by_category": {
                    **{category: None for category in CATEGORIES},
                    "button": b,
                    "text": t,
                }
            },
            "samples": 12,
        }

    return {
        "dataset": "same-benchmark",
        "split": split,
        "samples": 36,
        "threshold": 0.15,
        "tiled": False,
        "metrics": score(button, text)["metrics"],
        "by_family": {
            family: score(button, text) for family in ("android", "web", "webui")
        },
    }


def test_downgrade_in_test_cancels_promotion(tmp_path):
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-only candidate")
    bv, cv = report("val"), report("val", button=0.30)
    bt, ct = report("test"), report("test", button=0.25)
    ct["by_family"]["webui"]["metrics"]["ap50_by_category"]["button"] = 0.09
    policy = make_release_policy(bv, cv, bt, ct, model)
    assert policy["enabled"] is False
    assert policy["candidate_categories"] == []
    assert policy["independent_test_passed"] is False
    assert policy["holdout_regressions"][0]["family"] == "webui"


def test_improvement_in_all_domains_allows_opt_in_only(tmp_path):
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-only candidate")
    policy = make_release_policy(
        report("val"),
        report("val", button=0.22),
        report("test"),
        report("test", button=0.18),
        model,
    )
    assert policy["enabled"] is True
    assert policy["independent_test_passed"] is True
    assert policy["candidate_categories"] == ["button"]
    assert policy["candidate_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()


def test_holdout_comparisons_are_not_used_as_validation():
    with pytest.raises(Exception):
        verify_holdout(report("val"), report("val"), ["button"])


def test_missing_candidate_quality_metrics_fail_closed(tmp_path):
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-only candidate")
    candidate = report("test", button=0.30)
    candidate["by_family"]["web"]["metrics"]["ap50_by_category"]["button"] = None
    policy = make_release_policy(
        report("val"),
        report("val", button=0.35),
        report("test"),
        candidate,
        model,
    )
    assert not policy["independent_test_passed"]


def test_guard_does_not_promote_without_any_better_category(tmp_path):
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-only candidate")
    policy = make_release_policy(
        report("val"),
        report("val", button=0.05),
        report("test"),
        report("test", button=0.05),
        model,
    )
    assert not policy["enabled"]
    assert not policy["independent_test_passed"]
