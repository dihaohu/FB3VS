"""
Step 5: Real-Time Tile-Differentiated 360° Live Streaming (RK3576)
- Captures from Insta360 X5 via USB (/dev/video25)
- Computes optical flow on downsampled frames
- Fuses flow + viewport proxy + bandwidth into per-tile QP
- Applies spatial quality degradation per tile (lightweight, no JPEG overhead)
- Encodes with h264_rkmpp hardware encoder via FFmpeg pipe
- Pushes single RTMP stream to SRS server
"""
import cv2
import numpy as np
import subprocess
import time
import signal
import sys
from pathlib import Path

from pipeline_utils import (
    build_spherical_mask, build_tile_distance_matrix,
    compute_tile_qp, apply_tile_quality_degrade,
    simulate_bandwidth_trace, qp_base_from_bandwidth,
    TILE_ROWS, TILE_COLS,
)

# ============================================================
# CONFIGURATION
# ============================================================
CAMERA_DEV = "/dev/video25"
CAMERA_W = 2880
CAMERA_H = 1440
CAMERA_FPS = 30

RTMP_URL = "rtmp://192.168.1.100/live/360stream"
STREAM_BITRATE = "30M"

FLOW_SCALE = 0.25
FLOW_FRAME_STEP = 3
N_WARMUP = 90              # frames to calibrate speed thresholds (~3s)

# Bandwidth: "simulate" or "fixed"
BW_MODE = "simulate"
BW_FIXED_Mbps = 50.0

running = True


def signal_handler(sig, frame):
    global running
    running = False
    print("\nShutting down...")


def compute_flow_per_tile(raw_flow, sphere_mask, tile_h, tile_w):
    """Compute per-tile spherically-corrected raw flow magnitude."""
    mag = np.sqrt(raw_flow[..., 0]**2 + raw_flow[..., 1]**2)
    mag_corrected = mag * sphere_mask
    tile_flows = np.empty(TILE_ROWS * TILE_COLS, dtype=np.float32)
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            tile_flows[r * TILE_COLS + c] = float(
                np.mean(mag_corrected[y0:y1, x0:x1]))
    return tile_flows


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Camera ---
    print(f"Opening camera: {CAMERA_DEV}")
    cap = cv2.VideoCapture(CAMERA_DEV)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {CAMERA_DEV}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

    # --- Precompute shared data ---
    flow_w = int(actual_w * FLOW_SCALE)
    flow_h = int(actual_h * FLOW_SCALE)
    tile_h_full = actual_h // TILE_ROWS
    tile_w_full = actual_w // TILE_COLS
    tile_h_flow = flow_h // TILE_ROWS
    tile_w_flow = flow_w // TILE_COLS

    sphere_mask = build_spherical_mask(flow_h, flow_w)
    dist_matrix = build_tile_distance_matrix(TILE_ROWS, TILE_COLS)

    # Bandwidth trace
    if BW_MODE == "simulate":
        bw_duration_sec = 7200
        bw_trace = simulate_bandwidth_trace(bw_duration_sec,
                                            CAMERA_FPS / FLOW_FRAME_STEP)
    else:
        bw_trace = None

    # --- State ---
    prev_gray = None
    current_qp_tile = None
    qp_base = 25
    bw_mbps = BW_FIXED_Mbps
    warmup_flow_means = []
    speed_thresholds = (0.02, 0.05)  # fallback defaults
    frame_idx = 0
    frame_times = []
    total_start = time.time()
    stream_start = None  # set when FFmpeg starts

    # FFmpeg starts after warmup
    proc = None

    print(f"Warmup ({N_WARMUP} frames) + streaming to {RTMP_URL}")
    print("Ctrl+C to stop\n")

    while running:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying...")
            time.sleep(0.01)
            continue

        # --- Optical flow (every FLOW_FRAME_STEP) ---
        if frame_idx % FLOW_FRAME_STEP == 0:
            small = cv2.resize(frame, (flow_w, flow_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                raw_flow = compute_flow_per_tile(
                    flow, sphere_mask, tile_h_flow, tile_w_flow)

                # Per-frame normalization (robust for live stream)
                frame_max = float(np.max(raw_flow))
                flow_norm = np.clip(raw_flow / max(frame_max, 1e-6), 0, 1)

                if proc is None:
                    # Still warming up — collect flow stats
                    warmup_flow_means.append(float(np.mean(flow_norm)))
                else:
                    # Streaming — compute QP
                    elapsed = time.time() - stream_start
                    if BW_MODE == "simulate":
                        bw_idx = min(int(elapsed * (CAMERA_FPS / FLOW_FRAME_STEP)),
                                     len(bw_trace) - 1)
                        bw_mbps = float(bw_trace[bw_idx])
                    else:
                        bw_mbps = BW_FIXED_Mbps

                    qp_base = int(qp_base_from_bandwidth(bw_mbps).item())
                    current_qp_tile, _, _, _ = compute_tile_qp(
                        flow_norm, qp_base, dist_matrix, speed_thresholds)

            prev_gray = gray

        # --- Start FFmpeg after warmup ---
        if proc is None and frame_idx >= N_WARMUP:
            if len(warmup_flow_means) >= 2:
                lo = np.percentile(warmup_flow_means, 33)
                hi = np.percentile(warmup_flow_means, 66)
                speed_thresholds = (lo, hi)
                print(f"Speed thresholds: slow<{lo:.4f} medium<{hi:.4f} fast>={hi:.4f}")
            else:
                print("Warning: insufficient warmup data, using defaults")

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{actual_w}x{actual_h}",
                "-r", str(CAMERA_FPS),
                "-i", "-",
                "-c:v", "h264_rkmpp",
                "-b:v", STREAM_BITRATE,
                "-pix_fmt", "yuv420p",
                "-f", "flv",
                RTMP_URL,
            ]
            print(f"FFmpeg: h264_rkmpp {STREAM_BITRATE} → {RTMP_URL}")
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            stream_start = time.time()

        # --- Quality degradation ---
        if proc is not None and current_qp_tile is not None:
            processed = apply_tile_quality_degrade(
                frame, current_qp_tile, tile_h_full, tile_w_full)
        else:
            processed = frame

        # --- Push to FFmpeg ---
        if proc is not None:
            try:
                proc.stdin.write(processed.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe broken, exiting")
                break

        frame_idx += 1
        frame_times.append(time.time() - t0)

        # --- Periodic status ---
        if frame_idx % 90 == 0:
            recent = frame_times[-90:]
            avg_ms = np.mean(recent) * 1000
            fps_actual = 1000 / avg_ms if avg_ms > 0 else 0
            phase = "streaming" if proc is not None else "warmup"
            qp_str = f"qp_base={qp_base}" if current_qp_tile is not None else "qp=-"
            print(f"[frame {frame_idx:5d}] {fps_actual:.1f}fps | {phase} | "
                  f"{qp_str} | bw={bw_mbps:.0f}Mbps | {avg_ms:.0f}ms/frame")

    # --- Cleanup ---
    elapsed = time.time() - total_start
    stream_frames = frame_idx - N_WARMUP if frame_idx > N_WARMUP else 0
    print(f"\nStreamed {stream_frames} frames in {elapsed:.1f}s "
          f"({stream_frames / max(elapsed - N_WARMUP/CAMERA_FPS, 0.1):.1f} avg fps)")

    cap.release()
    if proc is not None:
        proc.stdin.close()
        proc.wait(timeout=5)
    print("Done.")


if __name__ == "__main__":
    main()
