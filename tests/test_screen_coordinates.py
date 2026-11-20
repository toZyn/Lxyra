import pytest

from lxyra.schema import make_document
from lxyra.screen_coordinates import ScreenTransform, agent_targets, fingerprint


@pytest.mark.parametrize(
    ("image", "display", "rect", "point", "expected"),
    [
        ((1080, 2400), (1080, 2400), (0, 0, 1080, 2400), (540, 1200), (540, 1200)),
        ((270, 600), (1080, 2400), (0, 0, 1080, 2400), (135, 300), (540, 1200)),
        ((200, 100), (1080, 2400), (200, 400, 600, 300), (100, 50), (500, 550)),
        ((480, 270), (1920, 1080), (0, 0, 1920, 1080), (240, 135), (960, 540)),
        ((100, 100), (1000, 2000), (250, 700, 200, 400), (99.5, 99.5), (449, 1098)),
    ],
)
def test_explicit_mapping_handles_crops_and_resizes(image, display, rect, point, expected):
    mapping = ScreenTransform(image, display, rect)
    assert mapping.map_point(*point) == expected


def test_rejects_invalid_crop_metadata():
    with pytest.raises(ValueError):
        ScreenTransform((200, 100), (1080, 2400), (900, 2000, 400, 500))
    with pytest.raises(ValueError):
        ScreenTransform((200, 100), (1080, 2400), (-1, 0, 200, 100))
    with pytest.raises(ValueError):
        ScreenTransform((200, 100), (1080, 2400), (0, 0, 0, 100))


def test_rejects_outside_point_instead_of_guessing():
    transform = ScreenTransform.full_screen((100, 50), (1080, 2400))
    for x, y in [(-1, 1), (100, 20), (50, 50), (float("nan"), 0)]:
        with pytest.raises(ValueError):
            transform.map_point(x, y)


def test_agent_target_point_requires_calibrated_transform():
    document = make_document(
        [{"category": "button", "bbox": [40, 30, 100, 50], "confidence": 0.95}],
        (200, 100),
        0.2,
    )
    without = agent_targets(document)
    assert without["targets"][0]["point_on_display"] is None
    assert without["coordinates_authoritative_for"] == "screenshot"

    geometry = ScreenTransform((200, 100), (1080, 2400), (200, 400, 600, 300))
    with_geometry = agent_targets(document, geometry)
    assert with_geometry["targets"][0]["point_on_display"] == [470, 565]
    assert with_geometry["targets"][0]["interactive_verified"] is False


def test_agent_candidates_report_overlap_instead_of_promising_safe_click():
    document = make_document(
        [
            {"category": "button", "bbox": [40, 40, 80, 70], "confidence": 0.9},
            {"category": "icon", "bbox": [55, 55, 50, 50], "confidence": 0.8},
            {"category": "text", "bbox": [30, 140, 90, 20], "confidence": 0.9},
        ],
        (200, 200),
        0.2,
    )
    result = agent_targets(document)
    assert len(result["targets"]) == 2
    assert any(target["ambiguous"] for target in result["targets"])
    assert all(target["interactive_verified"] is False for target in result["targets"])


def test_mismatched_frame_rejected():
    document = make_document([], (100, 200), 0.2)
    geometry = ScreenTransform.full_screen((200, 100), (1080, 2400))
    with pytest.raises(ValueError):
        agent_targets(document, geometry)


def test_transform_error_bounded_by_one_display_pixel():
    transform = ScreenTransform((333, 887), (1080, 2400), (40, 100, 720, 2000))
    for x, y in [(1, 1), (123.4, 203.1), (332.99, 886.99), (22.52, 333.39)]:
        xx, yy = transform.map_point(x, y)
        target_x = 40 + x * 720 / 333
        target_y = 100 + y * 2000 / 887
        assert abs(xx - target_x) < 1.0
        assert abs(yy - target_y) < 1.0


def test_fingerprint_changes_when_screen_content_changes():
    from PIL import Image

    a = Image.new("RGB", (100, 200), "#ffffff")
    b = Image.new("RGB", (100, 200), "#ffffff")
    assert fingerprint(a) == fingerprint(b)
    b.putpixel((50, 100), (0, 0, 0))
    assert fingerprint(a) != fingerprint(b)


def test_rejects_invalid_frame_fingerprint():
    doc = make_document([], (100, 200), 0.2)
    with pytest.raises(ValueError):
        agent_targets(doc, frame_sha256="not-a-sha256")


def test_mapped_display_point_matches_published_screenshot_point():
    doc = make_document(
        [{"category": "button", "bbox": [41, 31, 92, 43], "confidence": 0.96}],
        (201, 101),
        0.2,
    )
    transform = ScreenTransform((201, 101), (1080, 2400), (135, 400, 750, 1400))
    target = agent_targets(doc, transform)["targets"][0]
    assert target["point_on_display"] == list(transform.map_point(*target["point_on_screenshot"]))
