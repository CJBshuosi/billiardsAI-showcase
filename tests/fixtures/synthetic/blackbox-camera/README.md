# Black-box camera synthetic fixture

The integration test generates its source video in pytest's temporary directory. No
binary fixture is committed and no private field asset is used.

The generated source is a 160×120, 60 FPS MJPG/AVI containing a gray rectangle on a
black background. It is stationary through frame 90, then moves horizontally. The
480-frame form contains enough stable tail for production visual-motion settlement
and the standard 120-frame post-roll. The 105-frame form ends while motion is active
and is used to prove fail-closed incomplete-shot behavior.

This rectangle is only a deterministic pixel-motion stimulus. It is not a synthetic
billiards result and does not assert that the product's trained detector recognizes
balls. The test's observation-only backend returns no detections, so the real rules
path must abstain (`eight_status_unknown`) instead of inventing gameplay facts.

Renderer parameters and expected outcomes are separate frozen values in the test.
Neither expected frame boundaries nor rule truth is passed into capture, perception,
the event channel, rule facts, `LiveMatchCoordinator`, or `USBLiveProjectionSource`.
Events may be published only by `ProductRealtimeRuntime._emit_event`.

The fixture proves software wiring and safety through the existing shared runtime. It
does not close the 1920×1080@60 approved-model full-match gate, the physical RER U3
field gate, canonical postgame clip completion, or cloud publication.

## Independent ball appearance snapshots

`tests.support.blackbox_scene_renderer` supplies offline, deterministic OpenCV/NumPy
artwork for future video builders. It imports no product code and returns only raw
BGR pixels; independent physics and evaluation must own motion, timing and truth.
For example, from the repository root with the existing OpenCV/NumPy environment:

```python
import cv2
from tests.support.blackbox_scene_renderer import BallVisual, render_frame

frame = render_frame([BallVisual(0, 635, 254), BallVisual(9, 1270, 762)])
cv2.imwrite("<TEMP_ROOT>/blackbox-snapshot.png", frame)
```

`BallVisual(number, x_mm, y_mm, visible=True)` is immutable. Coordinates are on the
2540×1270 mm renderer plane: x points right and **y points down**, without a product
coordinate conversion. Output is writable C-contiguous 1920×1080 uint8 BGR; the cloth
runs from (240,147) through (1640,847), inclusive. Scale is 1400/2540 px/mm and ball
radius is 28.575 mm (15.75 px). Finite off-table positions are preserved and artwork
is cropped at image boundaries; invisible balls are omitted. Duplicate numbers,
malformed values and nonfinite coordinates are rejected, including invisible inputs.

Numbers 0/1–7/8/9–15 mean unnumbered white cue/solids/black eight/striped balls. Each
stripe shares the corresponding solid's fixed hue, with a horizontal colored middle
and white caps. Balls 1–15 have ordinary white numeral disks with black digits.
`draw_ball(frame, number=9, center_px=(100.5, 100.5), radius_px=16)` also draws in place,
validating all arguments before changing pixels. Frames must be nonempty writable
C-contiguous H×W×3 uint8; radii must be finite and at least one pixel. Boolean ball
numbers are rejected. The helper does not clamp balls back onto the table.

This is simple stylized artwork, with no photorealism, rolling/orientation simulation
or proven trained-model recognition. It is a renderer prerequisite, not a complete
scene, native input approval, product-routing test or full-game acceptance result.
No snapshot/truth metadata is a product input; future integration must encode pixels
as video and retain the approved shared capture/perception/rules path.
