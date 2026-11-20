# Coordinate mapping for screen-control agents

Lxyra interprets screenshots and provides estimated locations of visible components. It does not execute actions, guarantee clickability, read text strings or confirm that the state of a GUI has not changed.

A bounding box is a **model prediction**. Transforming a predicted point between two explicitly calibrated pixel coordinate systems is deterministic, but cannot make the underlying detection semantically or spatially infallible.

## Output coordinate spaces

Every parsing response uses `bbox: [x, y, width, height]` measured in **pixels of the exact screenshot passed into the model**. When `--tile` is used, tile offsets and letterbox scaling are already inverted; coordinates remain in the full input screenshot, not in a tile.

For agents that act using a separate display coordinate system, use `--agent-output` and declare the capture geometry:

```bash
lxyra screenshot.png --output elements.json --agent-output targets.json
```

Without display calibration, `point_on_display` is `null`. This is deliberate: a screenshot alone does not establish the resolution, device-pixel ratio, offset or cropping of an agent's input system.

When an image is a resized capture of the entire display:

```bash
lxyra resized.png --agent-output targets.json --display-size 1080x2400 --full-display-capture
```

When an image is a crop or resized crop of the display:

```bash
lxyra cropped.png --agent-output targets.json --display-size 1080x2400 --capture-rect 100,200,800,1200
```

`--capture-rect` is `LEFT,TOP,WIDTH,HEIGHT` in the display coordinate units used by the action driver, not the screenshot. These arguments are explicit assertions by the caller. An image with unknown crop offset cannot be mapped to accurate display coordinates.

## Mapping

For a screenshot of size `W × H` representing a display rectangle `(L, T, CW, CH)`, an image point `(x, y)` maps to:

```text
display_x = L + floor(x * CW / W)
display_y = T + floor(y * CH / H)
```

Both coordinates must lie within the explicit capture rectangle. Inputs outside the screenshot, invalid rectangles and geometry that disagrees with the parsed document are rejected, never guessed.

Examples:

| Screenshot | Display rectangle | Screenshot point | Display point |
|---|---|---|---|
| 270×600 | (0,0,1080,2400) | (135,300) | (540,1200) |
| 200×100 | (200,400,600,300) | (100,50) | (500,550) |
| 480×270 | (0,0,1920,1080) | (240,135) | (960,540) |

If an agent acts in logical coordinates, use **the logical display coordinate system that the driver actually accepts**. Do not substitute the hardware resolution without verifying device-pixel scaling. The model cannot infer coordinate conventions from a screenshot.

## Target candidates

The separate targets document reports candidate IDs, categories, detector confidence, screenshot bounding boxes, estimated centers, optional display points and overlapping candidate IDs. An icon may be decorative; a button-like shape may be an image. Every target has `interactive_verified: false`, even if its location is mapped precisely.

Element IDs are **frame-scoped**, not persistent IDs. The `frame_sha256` fingerprint binds the coordinates to a particular RGB screenshot. After any action, scroll, rotation or layout change, capture and analyze the updated screen again; compare the frame digest and avoid reusing stale coordinates.

The agent should verify the intended target and user permissions before acting, and use observation after each action to confirm the result. If the target is ambiguous, overlapping, missing or low-confidence, it should seek another observation rather than guessing.

## Testing and limitations

Geometry round-trip, crop offsets, resized captures, screen rotations and invalid calibration are checked in `tests/test_crop_coordinates.py` and `tests/test_screen_coordinates.py`. These tests establish **coordinate-transform correctness** for explicitly defined geometry, not pixel-perfect neural localization.

The training datasets contain annotation noise, and some text regions merge words or lines. Current model performance on real-world interfaces remains experimental. Avoid promoting new crop inference modes without category- and domain-specific validation.
