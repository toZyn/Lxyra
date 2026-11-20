import copy

import pytest

from lxyra.coordinates import change_box, get_delta
from lxyra.schema import make_document, validate_document


def test_document_has_valid_pixel_and_normalized_bounds():
    elements = [
        {"category": "container", "bbox": [0, 0, 300, 500], "confidence": 0.91},
        {"category": "text", "bbox": [40, 30, 120, 24], "confidence": 0.88},
    ]
    document = make_document(elements, (300, 500), 0.15)
    validate_document(document)
    assert len(document["elements"]) == 2
    assert all(len(item["bbox_normalized"]) == 4 for item in document["elements"])
    assert document["ocr_available"] is False


def test_schema_rejects_invalid_boxes():
    document = make_document(
        [{"category": "text", "bbox": [10, 10, 80, 24], "confidence": 0.8}],
        (200, 200),
        0.2,
    )
    broken = copy.deepcopy(document)
    broken["elements"][0]["bbox"][2] = 300
    with pytest.raises(ValueError):
        validate_document(broken)


def test_text_coordinate_round_trip():
    original = [18, 20, 70, 20]
    target = [22, 24, 60, 16]
    result = change_box(original, get_delta(original, target), 1.0, (120, 120))
    assert result == pytest.approx(target)


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        make_document(
            [{"category": "unknown", "bbox": [10, 10, 15, 15], "confidence": 0.8}],
            (200, 200),
            0.2,
        )
