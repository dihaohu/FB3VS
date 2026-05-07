"""
Step 1: Optical Flow Computation & Tile Weight Generation
- Reads 360 video, computes dense optical flow on downsampled frames
- Applies spherical projection correction (equirectangular cos(lat) weighting)
- Divides into tile grid, outputs per-frame tile-level flow magnitudes
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = "e:/research/myself/360video/data/360 VR ｜ POV Ducati Diavel cruising in Miami.webm"
OUTPUT_DIR = Path("e:/research/myself/360video/output")
OUTPUT_DIR.mkdir(exist_ok=True)

TILE_ROWS = 5        # vertical tiles
TILE_COLS = 9        # horizontal tiles (9x5 = 45 tiles)
FLOW_SCALE = 0.25    # downscale factor for optical flow (0.25 = 640x360)
FRAME_STEP = 3       # process every Nth frame (3 = ~10 fps effective)

# ============================================================
# SPHERICAL CORRECTION MASK (equirectangular -> sphere)
# ============================================================
def build_spherical_mask(h, w):
    """Build cos(latitude) weight mask for equirectangular projection."""
    ys = np.arange(h)
    # latitude: -pi/2 (top) to +pi/2 (bottom)
    lat = np.pi * (ys / (h - 1) - 0.5)
    weight = np.cos(lat)           # cos(lat), shape (h,)
    weight = np.abs(weight)         # always positive
    weight = np.clip(weight, 0.05, 1.0)  # avoid zero at poles
    # broadcast to (h, w)
    mask = np.tile(weight[:, np.newaxis], (1, w))
    return mask.astype(np.float32)

# ============================================================
# MAIN
# ============================================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flow_w = int(orig_w * FLOW_SCALE)
    flow_h = int(orig_h * FLOW_SCALE)
    tile_h = flow_h // TILE_ROWS
    tile_w = flow_w // TILE_COLS

    print(f"Video: {orig_w}x{orig_h}, {fps:.1f} fps, {total_frames} frames")
    print(f"Flow res: {flow_w}x{flow_h}, Tile grid: {TILE_COLS}x{TILE_ROWS}, "
          f"tile size: {tile_w}x{tile_h}")
    print(f"Frame step: {FRAME_STEP}, estimated output frames: {total_frames // FRAME_STEP}")

    # Build spherical correction mask at flow resolution
    sphere_mask = build_spherical_mask(flow_h, flow_w)
    print(f"Spherical mask range: [{sphere_mask.min():.3f}, {sphere_mask.max():.3f}]")

    # State
    prev_gray = None
    frame_idx = 0
    output_idx = 0
    results = []       # list of {frame, tiles: [flow_mean, ...]}

    pbar = tqdm(total=total_frames, desc="Computing optical flow")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        # Downsample + grayscale
        small = cv2.resize(frame, (flow_w, flow_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            # Magnitude
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            # Apply spherical correction
            mag_corrected = mag * sphere_mask

            # Aggregate per tile
            tile_flows = []
            for r in range(TILE_ROWS):
                for c in range(TILE_COLS):
                    y0, y1 = r * tile_h, (r + 1) * tile_h
                    x0, x1 = c * tile_w, (c + 1) * tile_w
                    tile_mean = float(np.mean(mag_corrected[y0:y1, x0:x1]))
                    tile_flows.append(tile_mean)

            results.append({
                "frame": frame_idx,
                "output_idx": output_idx,
                "tiles": tile_flows,
            })
            output_idx += 1

        prev_gray = gray
        frame_idx += 1

    pbar.close()
    cap.release()

    # ============================================================
    # Normalize & Save
    # ============================================================
    # Collect all flow values for global normalization
    all_flows = []
    for r in results:
        all_flows.extend(r["tiles"])
    flow_max = max(all_flows) if all_flows else 1.0
    flow_min = min(all_flows)

    # Normalize each frame's tiles
    for r in results:
        r["flow_norm"] = [(v - flow_min) / (flow_max - flow_min + 1e-8) for v in r["tiles"]]

    # Save as JSON
    output_path = OUTPUT_DIR / "flow_data.json"
    metadata = {
        "video_path": VIDEO_PATH,
        "orig_resolution": [orig_w, orig_h],
        "fps": fps,
        "total_frames": total_frames,
        "flow_scale": FLOW_SCALE,
        "flow_resolution": [flow_w, flow_h],
        "tile_grid": [TILE_ROWS, TILE_COLS],
        "tile_size": [tile_h, tile_w],
        "frame_step": FRAME_STEP,
        "output_frames": output_idx,
        "flow_global_max": flow_max,
        "flow_global_min": flow_min,
    }
    output_data = {"metadata": metadata, "frames": results}

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {output_path}")
    print(f"Frames processed: {output_idx}, Tile count: {TILE_ROWS * TILE_COLS}")
    print(f"Flow range: [{flow_min:.2f}, {flow_max:.2f}]")


if __name__ == "__main__":
    main()
