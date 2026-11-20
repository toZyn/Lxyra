import json
import subprocess
import sys

from PIL import Image


def run(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "lxyra", *[str(arg) for arg in arguments]],
        capture_output=True,
        text=True,
    )


def test_cli_exposes_screenshot_coordinates_without_screen_guess(tmp_path):
    screenshot = tmp_path / "screen.png"
    output = tmp_path / "targets.json"
    Image.new("RGB", (250, 450), "white").save(screenshot)
    process = run(screenshot, "--agent-output", output)
    assert process.returncode == 0, process.stderr
    data = json.loads(output.read_text())
    assert data["display_mapping"] is None
    assert data["coordinates_authoritative_for"] == "screenshot"
    assert len(data["frame_sha256"]) == 64
    assert data["id_scope"] == "current_frame_only"


def test_cli_requires_explicit_capture_mapping(tmp_path):
    screenshot = tmp_path / "cropped.png"
    Image.new("RGB", (250, 450), "white").save(screenshot)
    process = run(
        screenshot, "--agent-output", tmp_path / "targets.json", "--display-size", "1080x2400"
    )
    assert process.returncode != 0
    assert "--capture-rect" in process.stderr


def test_cli_accepts_cropped_display_mapping(tmp_path):
    screenshot = tmp_path / "cropped.png"
    target = tmp_path / "targets.json"
    Image.new("RGB", (250, 450), "white").save(screenshot)
    process = run(
        screenshot,
        "--agent-output",
        target,
        "--display-size",
        "1080x2400",
        "--capture-rect",
        "60,120,960,2050",
    )
    assert process.returncode == 0, process.stderr
    data = json.loads(target.read_text())
    assert data["display_mapping"]["capture_rect_on_display"] == [60, 120, 960, 2050]
    assert data["coordinates_authoritative_for"] == "display"


def test_cli_rejects_invalid_mapping_and_size(tmp_path):
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (250, 450), "white").save(screenshot)
    process = run(
        screenshot,
        "--agent-output",
        tmp_path / "result.json",
        "--display-size",
        "1080x2400",
        "--capture-rect",
        "1000,2200,400,300",
    )
    assert process.returncode != 0
    assert "outside the display" in process.stderr


def test_cli_normalizes_exif_orientation_before_mapping(tmp_path):
    screenshot = tmp_path / "rotated.jpg"
    output = tmp_path / "targets.json"
    image = Image.new("RGB", (180, 320), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(screenshot, exif=exif)
    process = run(screenshot, "--agent-output", output)
    assert process.returncode == 0, process.stderr
    data = json.loads(output.read_text())
    assert data["screenshot"] == {"width": 320, "height": 180}


def test_cli_handles_rotated_and_padded_capture(tmp_path):
    screenshot = tmp_path / "rotated-with-padding.png"
    target = tmp_path / "coordinates.json"
    Image.new("RGB", (448, 448), "white").save(screenshot)
    process = run(
        screenshot,
        "--agent-output",
        target,
        "--display-size",
        "1080x2400",
        "--full-display-capture",
        "--screenshot-content-rect",
        "98,0,252,448",
        "--capture-rotation",
        "90",
    )
    assert process.returncode == 0, process.stderr
    document = json.loads(target.read_text())
    assert document["display_mapping"]["rotation_degrees_clockwise"] == 90
    assert document["display_mapping"]["content_rect_on_screenshot"] == [98, 0, 252, 448]


def test_cli_rejects_content_rectangle_without_display_mapping(tmp_path):
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (320, 600), "white").save(screenshot)
    process = run(
        screenshot,
        "--agent-output",
        tmp_path / "targets.json",
        "--screenshot-content-rect",
        "0,0,300,500",
    )
    assert process.returncode != 0
    assert "require explicit display mapping" in process.stderr
