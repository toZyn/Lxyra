"""Block model promotion unless validation and independent tests agree."""

import argparse
import json
from pathlib import Path

from lxyra.promotion import PromotionRejected, select_categories, sha256_file


def verify_holdout(baseline: dict, candidate: dict, categories: list[str]) -> list[dict]:
    for key in ("dataset", "split", "samples", "threshold", "tiled"):
        if baseline.get(key) != candidate.get(key):
            raise PromotionRejected(f"Holdout metadata mismatch: {key}")
    if baseline["split"] != "test":
        raise PromotionRejected("Holdout reports must use the test split")
    before, after = baseline["by_family"], candidate["by_family"]
    if set(before) != set(after) or not {"android", "web", "webui"}.issubset(before):
        raise PromotionRejected("Missing required holdout domains")
    problems = []
    for family in ("all", *sorted(before)):
        previous = baseline["metrics"] if family == "all" else before[family]["metrics"]
        current = candidate["metrics"] if family == "all" else after[family]["metrics"]
        if family != "all" and before[family]["samples"] != after[family]["samples"]:
            raise PromotionRejected(f"Holdout domain sample mismatch: {family}")
        for category in categories:
            a = previous["ap50_by_category"].get(category)
            b = current["ap50_by_category"].get(category)
            if a is None or b is None or b + 1e-12 < a:
                problems.append(
                    {
                        "family": family,
                        "category": category,
                        "baseline_ap50": a,
                        "candidate_ap50": b,
                    }
                )
    return problems


def make_release_policy(
    baseline_val: dict,
    candidate_val: dict,
    baseline_test: dict,
    candidate_test: dict,
    candidate_model: Path,
) -> dict:
    selected = select_categories(baseline_val, candidate_val)
    problems = verify_holdout(
        baseline_test,
        candidate_test,
        selected["candidate_categories"],
    )
    if problems:
        selected["candidate_categories"] = []
        selected["enabled"] = False
    selected["independent_test_passed"] = bool(selected["candidate_categories"]) and not problems
    selected["holdout_regressions"] = problems
    selected["candidate_sha256"] = sha256_file(candidate_model)
    selected["status"] = (
        "experimental candidate accepted on checked benchmarks"
        if selected["independent_test_passed"]
        else "rejected; default weights remain unchanged"
    )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-val", required=True, type=Path)
    parser.add_argument("--candidate-val", required=True, type=Path)
    parser.add_argument("--baseline-test", required=True, type=Path)
    parser.add_argument("--candidate-test", required=True, type=Path)
    parser.add_argument("--candidate-onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            args.baseline_val,
            args.candidate_val,
            args.baseline_test,
            args.candidate_test,
        )
    ]
    policy = make_release_policy(*reports, args.candidate_onnx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "enabled": policy["enabled"],
                "candidate_categories": policy["candidate_categories"],
                "holdout_regressions": policy["holdout_regressions"],
            }
        )
    )
    if not policy["independent_test_passed"]:
        raise SystemExit("REJECTED: model quality gate failed")


if __name__ == "__main__":
    main()
