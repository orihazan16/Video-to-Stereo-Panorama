# Video Stabilization & Dynamic Panorama

A classical computer-vision pipeline (no deep learning) that turns a handheld video of a panning camera into a **stabilized, stitched panorama**, and generates a **sequence of panoramas from slightly different viewpoints**, which, played back as a video, gives a parallax-like "move around the scene" effect.

Built with OpenCV and NumPy.

## How It Works

The pipeline has two stages, implemented in `main.py`.

### 1. Video stabilization (`Stabilizer`)
1. **Motion estimation**: detect corners with `goodFeaturesToTrack` and track them between consecutive frames using sparse **Lucas-Kanade optical flow** (pyramidal, 21x21 window, 3 levels).
2. **Transform fitting**: fit a *partial affine* transform (translation + rotation + uniform scale, via `estimateAffinePartial2D`) with RANSAC to each frame pair. A full affine is deliberately avoided to prevent skew artifacts.
3. **Trajectory integration**: accumulate the per-frame motion (dx, dy, da) into a global camera trajectory.
4. **Trajectory smoothing**: apply a **moving-average filter** (radius of 25 frames, edge-padded) to remove high-frequency jitter.
5. **Frame warping**: warp each frame by the difference between the smoothed and the original trajectory.

### 2. Panorama stitching (`PanoramaBuilder`)
1. **Step estimation**: measure the horizontal displacement between consecutive stabilized frames with the same optical-flow + partial-affine approach.
2. **Direction and filtering**: take the median step to find the main pan direction, then drop frames that move backwards or barely move.
3. **Strip mosaicking**: from every kept frame, cut a **vertical strip whose width equals that frame's displacement** and paste the strips side by side on a canvas.
4. **Viewpoint bias**: the horizontal position of the strip inside each frame is controlled by a `bias` in [0, 1] (left, center, right). The value is remapped to the 0.1-0.9 range of the frame to avoid distorted edges.
5. **Post-processing**: crop black borders, flip if needed so the result reads left to right, and pad all outputs to a common size.

### Dynamic viewpoint effect
`generate_panorama` builds `n_out_frames` panoramas with `bias` swept from 0 to 1. Because strips are taken from different parts of each frame, nearby objects shift relative to distant ones, and the sequence of panoramas, played as a video, appears as a camera moving through the scene.

## Project Structure

```
.
├── main.py        # Stabilizer, PanoramaBuilder, generate_panorama()
├── images/        # Sample outputs used in this README
└── videos/        # Example input and result videos
```

## Getting Started

```bash
git clone <repo-url>
cd <repo-name>
pip install opencv-python numpy pillow
```

The input is a directory of frames named `frame_*.jpg` (for example, extracted from a video with `ffmpeg -i input.mp4 frames/frame_%04d.jpg`).

```python
from main import generate_panorama

# Returns a list of PIL images, one panorama per viewpoint
panos = generate_panorama("frames/", n_out_frames=100)

# Save as an animated GIF
panos[0].save("result.gif", save_all=True, append_images=panos[1:], duration=50, loop=0)
```

## Configuration

Tunable constants at the top of `main.py`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `MAX_CORNERS` | 200 | Max number of tracked features per frame |
| `QUALITY_LEVEL` | 0.02 | Corner quality threshold |
| `MIN_DISTANCE` | 30 | Min pixel distance between features |
| `OPTICAL_FLOW_WIN_SIZE` | (21, 21) | Lucas-Kanade search window |
| `OPTICAL_FLOW_MAX_LEVEL` | 3 | Pyramid levels for optical flow |
| `SMOOTH_RADIUS` | 25 | Frames on each side used for trajectory smoothing |

## Limitations

- The strip-based approach assumes a **mostly horizontal camera pan** and a scene that is relatively far from the camera.
- With **close-range foreground objects and strong parallax**, such as the fence in `bad_input.mp4`, plus handheld vertical motion, the stitched result becomes wavy and has black gaps (see `images/panorama_failure_case.jpg`). Estimating a single global motion per frame cannot account for depth.
- Strips are pasted without blending, so exposure changes between frames can leave visible seams.

Possible improvements: blend strips with multi-band blending, estimate vertical drift separately, or use per-region (depth-aware) motion estimation.

## Tech Stack

Python, OpenCV, NumPy, Pillow