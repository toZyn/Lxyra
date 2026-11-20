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
    screenshot_content_rect: tuple[int, int, int, int] | None = None
    rotation_degrees: int = 0

    def __post_init__(self):
        for pair in (self.screenshot_size, self.display_size):
            if len(pair) != 2 or any(type(value) is not int or value <= 0 for value in pair):
                raise ValueError("Image and display dimensions must be positive integers")
        self._validate_rectangle(
            self.capture_rect,
            self.display_size,
            "Capture rectangle lies outside the display",
        )
        if self.screenshot_content_rect is not None:
            self._validate_rectangle(
                self.screenshot_content_rect,
                self.screenshot_size,
                "Content rectangle lies outside the screenshot",
            )
        if type(self.rotation_degrees) is not int or self.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270 degrees clockwise")

    @staticmethod
    def _validate_rectangle(rect, bounds, message):
        if len(rect) != 4 or any(type(value) is not int for value in rect):
            raise ValueError(message)
        left, top, width, height = rect
        if (
            left < 0
            or top < 0
            or width <= 0
            or height <= 0
            or left + width > bounds[0]
            or top + height > bounds[1]
        ):
            raise ValueError(message)

    @classmethod
    def full_screen(
        cls, screenshot_size, display_size, screenshot_content_rect=None, rotation_degrees=0
    ):
        return cls(
            screenshot_size=tuple(screenshot_size),
            display_size=tuple(display_size),
            capture_rect=(0, 0, *display_size),
            screenshot_content_rect=screenshot_content_rect,
            rotation_degrees=rotation_degrees,
        )

    @property
    def content_rect(self):
        return self.screenshot_content_rect or (0, 0, *self.screenshot_size)

    def covers_point(self, x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            return False
        left, top, width, height = self.content_rect
        return left <= x < left + width and top <= y < top + height

    def _to_display(self, x, y):
        left, top, width, height = self.content_rect
        u = (x - left) / width
        v = (y - top) / height
        if self.rotation_degrees == 90:
            u, v = v, 1 - u
        elif self.rotation_degrees == 180:
            u, v = 1 - u, 1 - v
        elif self.rotation_degrees == 270:
            u, v = 1 - v, u
        display_left, display_top, display_width, display_height = self.capture_rect
        return (
            display_left + u * display_width,
            display_top + v * display_height,
        )

    def map_point(self, screenshot_x, screenshot_y):
        if not self.covers_point(screenshot_x, screenshot_y):
            raise ValueError("Point lies outside the screenshot content")
        x, y = self._to_display(screenshot_x, screenshot_y)
        left, top, width, height = self.capture_rect
        return (
            min(left + width - 1, max(left, math.floor(x))),
            min(top + height - 1, max(top, math.floor(y))),
        )

    def map_bbox(self, bbox):
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ValueError("Invalid screenshot bounding box")
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            raise ValueError("Bounding box dimensions must be positive")
        left, top, cw, ch = self.content_rect
        x0 = max(x, left)
        y0 = max(y, top)
        x1 = min(x + width, left + cw)
        y1 = min(y + height, top + ch)
        if x1 <= x0 or y1 <= y0:
            return None
        points = [self._to_display(px, py) for px in (x0, x1) for py in (y0, y1)]
        display_left, display_top, display_width, display_height = self.capture_rect
        bx0 = min(
            display_left + display_width, max(display_left, math.floor(min(p[0] for p in points)))
        )
        by0 = min(
            display_top + display_height, max(display_top, math.floor(min(p[1] for p in points)))
        )
        bx1 = min(
            display_left + display_width, max(display_left, math.ceil(max(p[0] for p in points)))
        )
        by1 = min(
            display_top + display_height, max(display_top, math.ceil(max(p[1] for p in points)))
        )
        return [bx0, by0, bx1 - bx0, by1 - by0] if bx1 > bx0 and by1 > by0 else None

    def as_dict(self):
        return {
            "screenshot_size": list(self.screenshot_size),
            "display_size": list(self.display_size),
            "capture_rect_on_display": list(self.capture_rect),
            "content_rect_on_screenshot": list(self.content_rect),
            "rotation_degrees_clockwise": self.rotation_degrees,
            "mapping": "explicit_cropped_scaled_and_rotated_capture",
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
        mapped_point = None
        mapped_box = None
        mapping_status = "uncalibrated"
        if geometry is not None:
            mapped_box = geometry.map_bbox(node["bbox"])
            if geometry.covers_point(*image_point):
                mapped_point = list(geometry.map_point(*image_point))
                mapping_status = "mapped"
            else:
                mapping_status = "outside_content_rect"
        result.append(
            {
                "id": node["id"],
                "category": node["category"],
                "confidence": node["confidence"],
                "bbox_on_screenshot": list(node["bbox"]),
                "point_on_screenshot": image_point,
                "point_on_display": mapped_point,
                "bbox_on_display": mapped_box,
                "mapping_status": mapping_status,
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
