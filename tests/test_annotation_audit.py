from scripts.audit_annotations import inspect_annotations


def test_flags_unrealistically_large_actionable_box():
    record = {
        "annotations": [
            {"category": "button", "bbox": [0, 0, 950, 600]},
            {"category": "text", "bbox": [20, 20, 150, 24]},
        ]
    }
    issues = inspect_annotations(record, (1000, 800))
    assert issues["large_actionable_region"] == 1
    assert issues["invalid_box"] == 0


def test_flags_duplicate_and_outside_annotations():
    record = {
        "annotations": [
            {"category": "icon", "bbox": [20, 20, 30, 30]},
            {"category": "icon", "bbox": [20, 20, 30, 30]},
            {"category": "input", "bbox": [180, 180, 40, 40]},
        ]
    }
    issues = inspect_annotations(record, (200, 200))
    assert issues["exact_duplicate"] == 1
    assert issues["invalid_box"] == 1


def test_correct_small_buttons_are_not_flagged():
    record = {"annotations": [{"category": "button", "bbox": [20, 40, 80, 20]}]}
    assert not inspect_annotations(record, (500, 800))


def test_page_wide_text_region_flagged_for_review():
    record = {
        "annotations": [
            {"category": "text", "bbox": [8, 21, 1264, 74]},
            {"category": "text", "bbox": [100, 160, 300, 27]},
        ]
    }
    issues = inspect_annotations(record, (1280, 720))
    assert issues["page_width_text_region"] == 1
    assert issues["invalid_box"] == 0
