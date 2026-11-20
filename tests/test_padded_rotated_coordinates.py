import pytest

from lxyra.schema import make_document
from lxyra.screen_coordinates import ScreenTransform, agent_targets


@pytest.mark.parametrize(
    ("rotation", "point", "expected"),
    [
        (0, (50, 25), (500, 250)),
        (90, (50, 25), (250, 500)),
        (180, (50, 25), (500, 750)),
        (270, (50, 25), (750, 500)),
    ],
)
def test_explicit_rotation_mapping(rotation, point, expected):
    transform = ScreenTransform(
        (100, 100), (1000, 1000), (0, 0, 1000, 1000), rotation_degrees=rotation
    )
    assert transform.map_point(*point) == expected


def test_content_padding_is_removed_before_mapping():
    transform = ScreenTransform(
        (448, 448), (1080, 1920), (0, 0, 1080, 1920), screenshot_content_rect=(98, 0, 252, 448)
    )
    assert transform.map_point(224, 224) == (540, 960)
    assert transform.map_point(98, 0) == (0, 0)
    assert transform.map_bbox([98, 112, 252, 224]) == [0, 480, 1080, 960]
    with pytest.raises(ValueError):
        transform.map_point(50, 224)


def test_outside_content_never_produces_display_click_coordinate():
    transform = ScreenTransform(
        (448, 448), (1080, 1920), (0, 0, 1080, 1920), screenshot_content_rect=(98, 0, 252, 448)
    )
    document = make_document(
        [
            {"category": "button", "bbox": [15, 20, 50, 25], "confidence": 0.95},
            {"category": "button", "bbox": [155, 200, 100, 45], "confidence": 0.88},
        ],
        (448, 448),
        0.15,
    )
    by_status = {
        target["mapping_status"]: target for target in agent_targets(document, transform)["targets"]
    }
    assert by_status["outside_content_rect"]["point_on_display"] is None
    assert by_status["outside_content_rect"]["bbox_on_display"] is None
    assert by_status["mapped"]["point_on_display"] is not None


def test_rotated_bbox_uses_display_rectangle_without_overflow():
    transform = ScreenTransform((200, 100), (1080, 1920), (100, 250, 700, 600), rotation_degrees=90)
    assert transform.map_bbox([50, 25, 100, 50]) == [275, 400, 350, 300]


def test_bad_padding_and_rotation_rejected():
    with pytest.raises(ValueError):
        ScreenTransform(
            (200, 100), (1080, 1920), (0, 0, 1080, 1920), screenshot_content_rect=(180, 10, 30, 30)
        )
    with pytest.raises(ValueError):
        ScreenTransform((200, 100), (1080, 1920), (0, 0, 1080, 1920), rotation_degrees=45)


def test_randomized_transform_never_exits_capture_rectangle():
    import random

    rng = random.Random(7182)
    for _ in range(200):
        screen_width = rng.randint(70, 1200)
        screen_height = rng.randint(70, 1200)
        display_width = rng.randint(200, 2400)
        display_height = rng.randint(200, 2400)
        content_left = rng.randint(0, screen_width // 4)
        content_top = rng.randint(0, screen_height // 4)
        content_width = rng.randint(1, screen_width - content_left)
        content_height = rng.randint(1, screen_height - content_top)
        capture_left = rng.randint(0, display_width - 1)
        capture_top = rng.randint(0, display_height - 1)
        capture_width = rng.randint(1, display_width - capture_left)
        capture_height = rng.randint(1, display_height - capture_top)
        rotation = rng.choice((0, 90, 180, 270))
        transform = ScreenTransform(
            (screen_width, screen_height),
            (display_width, display_height),
            (capture_left, capture_top, capture_width, capture_height),
            screenshot_content_rect=(content_left, content_top, content_width, content_height),
            rotation_degrees=rotation,
        )
        image_x = content_left + rng.random() * content_width
        image_y = content_top + rng.random() * content_height
        mapped_x, mapped_y = transform.map_point(image_x, image_y)
        assert capture_left <= mapped_x < capture_left + capture_width
        assert capture_top <= mapped_y < capture_top + capture_height
        bx, by, bw, bh = transform.map_bbox(
            [content_left, content_top, content_width, content_height]
        )
        assert bx == capture_left
        assert by == capture_top
        assert bw == capture_width
        assert bh == capture_height
