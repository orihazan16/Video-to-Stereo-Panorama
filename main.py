import cv2
import numpy as np
import math
from PIL import Image
import os
import glob
from typing import List

# ---------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------

# Feature tracking parameters
MAX_CORNERS = 200
QUALITY_LEVEL = 0.02
MIN_DISTANCE = 30

# Optical flow parameters (Lucas-Kanade)
OPTICAL_FLOW_WIN_SIZE = (21, 21)
OPTICAL_FLOW_MAX_LEVEL = 3

# Stabilization settings
SMOOTH_RADIUS = 25  # Number of frames to look ahead/behind for smoothing
SCALE_FACTOR = 1.0


class Motion:
    """
    Data structure representing the relative transformation between two consecutive frames.
    """

    def __init__(self, dx=0, dy=0, da=0):
        self.dx = dx  # Horizontal translation
        self.dy = dy  # Vertical translation
        self.da = da  # Rotation angle (radians)


class Pose:
    """
    Data structure representing the cumulative global position of the camera.
    """

    def __init__(self, x=0, y=0, a=0):
        self.x = x  # Global Horizontal position
        self.y = y  # Global Vertical position
        self.a = a  # Global Rotation angle (radians)


class VideoReader:
    """
    Handles loading and extracting frames from a specific directory.
    """

    def __init__(self, path: str):
        self.path = path

    def read_frames(self) -> List[np.ndarray]:
        """
        Reads all images matching 'frame_*.jpg' from the directory.

        Returns:
            List[np.ndarray]: A list of loaded images (BGR format).
        """
        search_pattern = os.path.join(self.path, "frame_*.jpg")
        frame_files = sorted(glob.glob(search_pattern))

        if not frame_files:
            # Returns empty list if no frames are found
            return []

        frames = []
        for f in frame_files:
            img = cv2.imread(f)
            if img is not None:
                frames.append(img)

        return frames


class Stabilizer:
    """
    Implements a video stabilization pipeline using:
    1. Motion Estimation (Optical Flow)
    2. Trajectory Integration
    3. Trajectory Smoothing (Moving Average)
    4. Frame Warping
    """

    def __init__(self, frames: List[np.ndarray]):
        self.frames = frames
        self.motions = []
        self.trajectory = []
        self.smooth_trajectory = []

    def _estimate_motion(self):
        """Calculates frame-to-frame motion using sparse Lucas-Kanade optical flow."""
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames]

        # Initialize with identity motion for the first frame
        self.motions.append(Motion())

        for i in range(len(gray_frames) - 1):
            # Detect good features to track (corners)
            p0 = cv2.goodFeaturesToTrack(
                gray_frames[i],
                maxCorners=MAX_CORNERS,
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE
            )

            if p0 is None:
                self.motions.append(Motion())
                continue

            # Calculate optical flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                gray_frames[i], gray_frames[i + 1], p0, None,
                winSize=OPTICAL_FLOW_WIN_SIZE,
                maxLevel=OPTICAL_FLOW_MAX_LEVEL
            )

            # Filter only valid tracked points
            if p1 is None:
                self.motions.append(Motion())
                continue

            p0_valid = p0[st == 1]
            p1_valid = p1[st == 1]

            if len(p0_valid) < 3:
                self.motions.append(Motion())
                continue

            # Estimate partial affine transform (Translation + Rotation)
            # I avoid full affine to prevent skewing artifacts
            matrix, _ = cv2.estimateAffinePartial2D(p0_valid, p1_valid)

            if matrix is None:
                self.motions.append(Motion())
                continue

            # Extract transformation components
            dx = matrix[0, 2]
            dy = matrix[1, 2]
            da = math.atan2(matrix[1, 0], matrix[0, 0])
            self.motions.append(Motion(dx, dy, da))

    def _integrate(self):
        """Accumulates relative motions to create a global camera trajectory."""
        x, y, a = 0, 0, 0
        for m in self.motions:
            x += m.dx
            y += m.dy
            a += m.da
            self.trajectory.append(Pose(x, y, a))

    def _smooth(self):
        """Applies a moving average filter to smooth out camera jitter."""

        def moving_avg(data, r):
            size = 2 * r + 1
            kernel = np.ones(size) / size
            # Pad edges to handle boundary conditions
            data_padded = np.pad(data, (r, r), mode="edge")
            return np.convolve(data_padded, kernel, mode="same")[r:-r]

        xs = moving_avg([p.x for p in self.trajectory], SMOOTH_RADIUS)
        ys = moving_avg([p.y for p in self.trajectory], SMOOTH_RADIUS)
        as_ = moving_avg([p.a for p in self.trajectory], SMOOTH_RADIUS)

        self.smooth_trajectory = [Pose(xs[i], ys[i], as_[i]) for i in range(len(xs))]

    def stabilize(self) -> List[np.ndarray]:
        """
        Executes the full stabilization process.

        Returns:
            List[np.ndarray]: A list of stabilized frames.
        """
        self._estimate_motion()
        self._integrate()
        self._smooth()

        h, w = self.frames[0].shape[:2]
        center = (w // 2, h // 2)
        output = []

        for i, frame in enumerate(self.frames):
            # Calculate the correction needed (Smooth - Original)
            dx = self.smooth_trajectory[i].x - self.trajectory[i].x
            dy = self.smooth_trajectory[i].y - self.trajectory[i].y
            da = self.smooth_trajectory[i].a - self.trajectory[i].a

            # Construct the transformation matrix
            M = cv2.getRotationMatrix2D(center, math.degrees(da), 1.0)
            M[0, 2] += dx
            M[1, 2] += dy

            # Warp the frame
            output.append(cv2.warpAffine(frame, M, (w, h)))

        return output


class PanoramaBuilder:
    """
    Stitches a long panoramic image from a sequence of video frames
    by taking vertical strips from each frame.
    """

    def __init__(self, frames: List[np.ndarray]):
        self.frames = frames
        self.steps = []  # Stores horizontal displacement per frame

    def estimate_steps(self):
        """
        Calculates the horizontal displacement (step size) between consecutive frames.
        This determines the width of the strip to take from each frame.
        """
        g0 = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(len(self.frames) - 1):
            g1 = cv2.cvtColor(self.frames[i + 1], cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(g0, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE)
            if p0 is None:
                self.steps.append(0)
                continue

            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)

            if p1 is not None:
                p0, p1 = p0[st == 1], p1[st == 1]

            if p1 is None or len(p0) < 3:
                self.steps.append(0)
                continue

            # Estimate translation
            M, _ = cv2.estimateAffinePartial2D(p0, p1)
            self.steps.append(M[0, 2] if M is not None else 0)

            g0 = g1

    def build(self, bias: float = 0.5) -> np.ndarray:
        """
        Stitches the panorama.

        Args:
            bias (float): Controls the viewing angle (0.0 to 1.0).
                          Maps to a safe range to pick strips from left/center/right.

        Returns:
            np.ndarray: The stitched panorama image (BGR).
        """

        # 1. Validation and Filter Logic
        if not self.frames or not self.steps:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Filter out noise (very small movements)
        significant_steps = [s for s in self.steps if abs(s) > 0.5]
        if not significant_steps:
            return self.frames[0]

        # Determine main direction of camera movement
        avg_step = np.median(significant_steps)
        direction = 1 if avg_step > 0 else -1

        valid_steps = []
        filtered_frames = []

        # Always include the first frame
        filtered_frames.append(self.frames[0])

        for i, step in enumerate(self.steps):
            # Skip backward motion (jitter against the main direction)
            if step * direction <= 0:
                continue
            # Skip negligible movement
            if abs(step) < 1.0:
                continue

            valid_steps.append(step)
            filtered_frames.append(self.frames[i + 1])

        if not valid_steps:
            return self.frames[0]

        # 2. Prepare Canvas
        h, w = filtered_frames[0].shape[:2]
        total_w = int(sum(abs(s) for s in valid_steps)) + w
        canvas = np.zeros((h, total_w, 3), dtype=np.uint8)

        # 3. Calculate Effective Bias (Safety Margin)
        # Remap bias 0.0-1.0 to 0.1-0.9 to avoid distorted frame edges
        safe_margin = 0.1
        effective_bias = safe_margin + (bias * (1.0 - 2 * safe_margin))

        curr_x = 0

        # 4. Stitching Loop
        for i, step in enumerate(valid_steps):
            strip_w = int(abs(step))
            if strip_w <= 0:
                continue

            # Determine crop window based on bias
            max_start = w - strip_w
            start_px = int(effective_bias * max_start)

            # Clamp to image boundaries
            start_px = max(0, min(start_px, max_start))

            end_x = min(curr_x + strip_w, total_w)
            actual_w = end_x - curr_x

            if actual_w > 0:
                canvas[:, curr_x:end_x] = filtered_frames[i][:, start_px:start_px + actual_w]
                curr_x = end_x

        # 5. Post-process: Crop black borders
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w_c, h_c = cv2.boundingRect(coords)
            canvas = canvas[y:y + h_c, x:x + w_c]

        return canvas


def generate_panorama(input_frames_path: str, n_out_frames: int) -> List[Image.Image]:
    """
    Main entry point.

    Args:
        input_frames_path (str): Path to directory containing frame images.
        n_out_frames (int): Number of panorama variations to generate.

    Returns:
        List[Image.Image]: List of generated panoramas as PIL images.
    """

    # 1. Load frames
    frames = VideoReader(input_frames_path).read_frames()
    if len(frames) == 0:
        return []

    # 2. Stabilize frames
    print("Stabilizing video frames...")
    stable_frames = Stabilizer(frames).stabilize()

    # 3. Build panorama variations
    print("Building panorama...")
    builder = PanoramaBuilder(stable_frames)
    builder.estimate_steps()

    # Determine direction for flipping result if needed
    significant_steps = [s for s in builder.steps if abs(s) > 0.5]
    avg_step = np.median(significant_steps) if significant_steps else 0
    should_flip = avg_step > 0

    list_panos = []
    max_w, max_h = 0, 0

    # Generate N variations with different view biases
    for b in np.linspace(0, 1, n_out_frames):
        img_bgr = builder.build(b)

        if should_flip:
            # Flip horizontally to match natural reading direction
            img_bgr = cv2.flip(img_bgr, 1)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        list_panos.append(pil_img)

        # Track max dimensions for padding later
        max_w = max(max_w, pil_img.width)
        max_h = max(max_h, pil_img.height)

    # 4. Standardize dimensions (center and pad with black)
    final_images = []
    for img in list_panos:
        new_canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
        offset = ((max_w - img.width) // 2, (max_h - img.height) // 2)
        new_canvas.paste(img, offset)
        final_images.append(new_canvas)

    return final_images