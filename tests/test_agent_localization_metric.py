from scripts.evaluate_agent_targets import evaluate_one


def test_center_metric_requires_correct_category_and_location():
    truth = [
        {"category": "button", "bbox": [20, 20, 50, 20]},
        {"category": "icon", "bbox": [130, 80, 30, 30]},
    ]
    targets = [
        {"category": "button", "point_on_screenshot": [45, 30], "confidence": 0.91},
        {"category": "button", "point_on_screenshot": [45, 30], "confidence": 0.80},
        {"category": "icon", "point_on_screenshot": [145, 90], "confidence": 0.75},
        {"category": "icon", "point_on_screenshot": [60, 30], "confidence": 0.73},
    ]
    value = evaluate_one(targets, truth)
    assert value == {"candidate_count": 4, "truth_count": 2, "center_hits": 2}


def test_center_metric_does_not_claim_control_clickability():
    result = evaluate_one(
        [{"category": "button", "point_on_screenshot": [10, 10], "confidence": 0.9}],
        [{"category": "input", "bbox": [0, 0, 20, 20]}],
    )
    assert result["center_hits"] == 0
