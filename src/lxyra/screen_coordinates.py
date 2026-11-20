import hashlib
import math
from dataclasses import dataclass

from .schema import validate_document


CANDIDATE_CATEGORIES = frozenset({"button", "input", "checkbox", "switch", "select", "icon"})


@dataclass(frozen=True)
class ScreenTransform:
    screenshot_size: tuple[int, int]
    display_size: tuple[int, int]
    capture_rect: tuple[int, int, int, int]

    def __post_init__(self):
        for pair in (self.screenshot_size, self.display_size):
            if len(pair) != 2 or any(type(value) is not int or value <= 0 for value in pair):
                raise ValueError("Image and display dimensions must be positive integers")
        rect = self.capture_rect
        if len(rect) != 4 or any(type(value) is not int for value in rect):
            raise ValueError("Capture rectangle must contain four integers")
        left, top, width, height = rect
        display_width, display_height = self.display_size
        if (
            left < 0
            or top < 0
            or width <= 0
            or height <= 0
            or left + width > display_width
            or top + height > display_height
        ):
            raise ValueError("Capture rectangle lies outside the display")

    @classmethod
    def full_screen(cls, screenshot_size, display_size):
        return cls(
            screenshot_size=tuple(screenshot_size),
            display_size=tuple(display_size),
            capture_rect=(0, 0, *display_size),
        )

    def map_point(self, screenshot_x, screenshot_y):
        image_width, image_height = self.screenshot_size
        if (
            not math.isfinite(screenshot_x)
            or not math.isfinite(screenshot_y)
            or not 0 <= screenshot_x < image_width
            or not 0 <= screenshot_y < image_height
        ):
            raise ValueError("Point lies outside the screenshot")
        left, top, width, height = self.capture_rect
        x = left + math.floor(screenshot_x * width / image_width)
        y = top + math.floor(screenshot_y * height / image_height)
        return (min(left + width - 1, x), min(top + height - 1, y))

    def as_dict(self):
        return {
            "screenshot_size": list(self.screenshot_size),
            "display_size": list(self.display_size),
            "capture_rect_on_display": list(self.capture_rect),
            "mapping": "explicit_crop_or_full_frame_scale",
        }


def _inside(point, rectangle):
    x, y = point
    left, top, width, height = rectangle
    return left <= x < left + width and top <= y < top + height


def fingerprint(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    frame_hash = hashlib.sha256()
    frame_hash.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    frame_hash.update(image.tobytes())
    return frame_hash.hexdigest()


def agent_targets(document, geometry=None, minimum_confidence=0.25, frame_sha256=None):
    validate_document(document)
    if not 0 <= minimum_confidence <= 1 or not math.isfinite(minimum_confidence):
        raise ValueError("Confidence threshold must be finite and between zero and one")
    if frame_sha256 is not None and (
        not isinstance(frame_sha256, str)
        or len(frame_sha256) != 64
        or any(character not in "0123456789abcdef" for character in frame_sha256)
    ):
        raise ValueError("Frame fingerprint must be a SHA-256 hex digest")
    image = document["image"]
    image_size = (image["width"], image["height"])
    if geometry is not None and geometry.screenshot_size != image_size:
        raise ValueError("Provided transform does not match the screenshot dimensions")

    nodes = [
        node
        for node in document["elements"]
        if node["category"] in CANDIDATE_CATEGORIES and node["confidence"] >= minimum_confidence
    ]
    result = []
    for node in nodes:
        x, y, width, height = node["bbox"]
        center = (x + width / 2, y + height / 2)
        image_point = [
            min(image_size[0] - 1, int(center[0])),
            min(image_size[1] - 1, int(center[1])),
        ]
        overlaps = [
            other["id"]
            for other in nodes
            if other["id"] != node["id"] and _inside(center, other["bbox"])
        ]
        result.append(
            {
                "id": node["id"],
                "category": node["category"],
                "confidence": node["confidence"],
                "bbox_on_screenshot": list(node["bbox"]),
                "point_on_screenshot": image_point,
                "point_on_display": list(geometry.map_point(*image_point)) if geometry else None,
                "overlapping_candidate_ids": overlaps,
                "ambiguous": bool(overlaps),
                "interactive_verified": False,
                "point_kind": "estimated_bbox_center",
            }
        )

    return {
        "schema_version": "1.0",
        "screenshot": {"width": image_size[0], "height": image_size[1]},
        "frame_sha256": frame_sha256,
        "id_scope": "current_frame_only",
        "display_mapping": geometry.as_dict() if geometry else None,
        "coordinates_authoritative_for": "display" if geometry else "screenshot",
        "candidate_count": len(result),
        "targets": result,
        "limitations": [
            "Geometric mapping is deterministic given correct capture metadata.",
            "Component localization and clickability are visual estimates, not guaranteed.",
            "Element IDs and target points are valid only for the referenced frame.",
            "Revalidate against a fresh screenshot after scrolling or window movement.",
        ],
    }
