"""
Step 3: Tile-Adaptive Encoding & Comparison
- Reads original video and QP table from Step 2
- Generates two versions:
  a) Uniform: all tiles same quality
  b) Flow-guided: tile-adaptive blur based on QP offsets (simulates quality diff)
- Encodes both with FFmpeg and compares bitrate + PSNR
"""
import cv2
import numpy as np
import json
import subprocess
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = "e:/research/myself/360video/data/360 VR ｜ POV Ducati Diavel cruising in Miami.webm"
QP_TABLE_PATH = Path("e:/research/myself/360video/output/qp_table.json")
OUTPUT_DIR = Path("e:/research/myself/360video/output")
TEMP_DIR = OUTPUT_DIR / "temp_frames"
UNIFORM_DIR = TEMP_DIR / "uniform"
FLOW_GUIDED_DIR = TEMP_DIR / "flow_guided"

# Processing range: use a segment for faster turnaround
START_SEC = 30      # start at 30 seconds (skip intro)
DURATION_SEC = 60   # process 60 seconds

# Tile grid (must match previous steps)
TILE_ROWS = 5
TILE_COLS = 9

# CRF for final encoding (lower = better quality)
CRF_VALUE = 23


def qp_delta_to_blur_sigma(delta_qp):
    """Map QP delta to Gaussian blur sigma.
    delta_qp < 0 (high priority) -> sigma = 0 (keep sharp)
    delta_qp > 0 (low priority)  -> sigma proportional to delta
    """
    # delta_qp range: [-6, +6]
    # Map to sigma: delta_qp = -6 -> 0, delta_qp = 0 -> 0.3, delta_qp = +6 -> 1.5
    return max(0, (delta_qp + 6) / 12 * 2.0)


def apply_tile_blur(frame, tile_qp_deltas, tile_h, tile_w):
    """Apply tile-adaptive Gaussian blur to frame based on QP offsets."""
    result = frame.copy().astype(np.float32)
    h, w = frame.shape[:2]

    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            idx = r * TILE_COLS + c
            sigma = qp_delta_to_blur_sigma(tile_qp_deltas[idx])
            if sigma < 0.05:
                continue  # skip sharp tiles

            y0, y1 = r * tile_h, min((r + 1) * tile_h, h)
            x0, x1 = c * tile_w, min((c + 1) * tile_w, w)

            tile = frame[y0:y1, x0:x1]
            # Blur tile
            ksize = int(2 * np.ceil(3 * sigma) + 1)
            if ksize >= 3:
                blurred = cv2.GaussianBlur(tile, (ksize, ksize), sigma)
                result[y0:y1, x0:x1] = blurred

    return np.clip(result, 0, 255).astype(np.uint8)


def encode_frames_to_video(frames_dir, output_path, fps, crf, preset="medium"):
    """Encode PNG frames to H.264 video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def compute_psnr(frame1, frame2):
    """Compute PSNR between two frames."""
    mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    # Load QP table
    with open(QP_TABLE_PATH) as f:
        qp_data = json.load(f)

    qp_frames = qp_data["frames"]
    qp_lookup = {f["frame"]: f for f in qp_frames}

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tile_h = orig_h // TILE_ROWS
    tile_w = orig_w // TILE_COLS

    # Calculate frame range
    frame_step = qp_data["metadata"].get("frame_step_from_flow", 3)
    start_frame = int(START_SEC * fps)
    end_frame = int((START_SEC + DURATION_SEC) * fps)

    print(f"Video: {orig_w}x{orig_h}, {fps:.1f} fps")
    print(f"Processing frames {start_frame} to {end_frame} "
          f"({DURATION_SEC}s, ~{end_frame - start_frame} frames)")
    print(f"Tile grid: {TILE_COLS}x{TILE_ROWS}")

    # Create output directories
    UNIFORM_DIR.mkdir(parents=True, exist_ok=True)
    FLOW_GUIDED_DIR.mkdir(parents=True, exist_ok=True)

    # Process frames
    frame_idx = 0
    output_idx = 0
    psnr_uniform_list = []
    psnr_flow_guided_list = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pbar = tqdm(total=end_frame - start_frame, desc="Processing frames")
    metrics_log = []

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < start_frame:
            frame_idx += 1
            continue

        # Find nearest QP data frame (flow computed every FRAME_STEP frames)
        flow_frame = (frame_idx // frame_step) * frame_step
        qp_info = qp_lookup.get(flow_frame)
        if qp_info is None:
            # Find nearest
            nearest = min(qp_lookup.keys(),
                          key=lambda k: abs(k - frame_idx))
            qp_info = qp_lookup[nearest]

        tile_qp_deltas = qp_info["delta_qp"]
        frame_qp_base = qp_info["qp_base"]

        # --- Uniform version: apply uniform blur (all tiles same treatment) ---
        # Uniform = all tiles get delta_qp = 0 (no differentiation)
        uniform_frame = apply_tile_blur(frame, [0] * (TILE_ROWS * TILE_COLS),
                                        tile_h, tile_w)

        # --- Flow-guided version: tile-adaptive blur ---
        flow_guided_frame = apply_tile_blur(frame, tile_qp_deltas,
                                            tile_h, tile_w)

        # Save frames
        cv2.imwrite(str(UNIFORM_DIR / f"frame_{output_idx:06d}.png"), uniform_frame)
        cv2.imwrite(str(FLOW_GUIDED_DIR / f"frame_{output_idx:06d}.png"),
                    flow_guided_frame)

        # Compute PSNR vs original (perceptual metric: difference in quality)
        psnr_u = compute_psnr(frame, uniform_frame)
        psnr_fg = compute_psnr(frame, flow_guided_frame)
        psnr_uniform_list.append(psnr_u)
        psnr_flow_guided_list.append(psnr_fg)

        # Log every 30 frames
        if output_idx % 30 == 0:
            metrics_log.append({
                "frame_idx": frame_idx,
                "output_idx": output_idx,
                "psnr_uniform": round(psnr_u, 2),
                "psnr_flow_guided": round(psnr_fg, 2),
                "qp_base": frame_qp_base,
            })

        frame_idx += 1
        output_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # --- Encode both to video ---
    print("\nEncoding uniform video...")
    uniform_video = OUTPUT_DIR / "uniform_baseline.mp4"
    encode_frames_to_video(UNIFORM_DIR, uniform_video, fps, CRF_VALUE)

    print("Encoding flow-guided video...")
    flow_guided_video = OUTPUT_DIR / "flow_guided.mp4"
    encode_frames_to_video(FLOW_GUIDED_DIR, flow_guided_video, fps, CRF_VALUE)

    # --- Compare results ---
    uniform_size = uniform_video.stat().st_size if uniform_video.exists() else 0
    fg_size = flow_guided_video.stat().st_size if flow_guided_video.exists() else 0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Uniform baseline:     {uniform_size / 1e6:.2f} MB")
    print(f"Flow-guided:          {fg_size / 1e6:.2f} MB")
    if uniform_size > 0:
        saving = (1 - fg_size / uniform_size) * 100
        print(f"Size saving:          {saving:.1f}%")
    print(f"Uniform avg PSNR:     {np.mean(psnr_uniform_list):.2f} dB")
    print(f"Flow-guided avg PSNR: {np.mean(psnr_flow_guided_list):.2f} dB")

    # Save metrics
    metrics_path = OUTPUT_DIR / "encode_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": {
                "start_sec": START_SEC,
                "duration_sec": DURATION_SEC,
                "crf": CRF_VALUE,
                "tile_grid": [TILE_ROWS, TILE_COLS],
            },
            "results": {
                "uniform_size_bytes": uniform_size,
                "flow_guided_size_bytes": fg_size,
                "uniform_psnr_mean": round(float(np.mean(psnr_uniform_list)), 2),
                "flow_guided_psnr_mean": round(float(np.mean(psnr_flow_guided_list)), 2),
                "size_saving_pct": round((1 - fg_size / uniform_size) * 100, 1) if uniform_size > 0 else 0,
            },
            "per_frame_log": metrics_log,
        }, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
