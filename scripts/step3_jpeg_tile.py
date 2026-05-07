"""
Step 3 (Revised): Per-Tile JPEG Compression Simulation
- Simulates tile-level differentiated encoding using JPEG quality per tile
- For each frame: split into tiles, JPEG each at per-tile quality, decode, reassemble
- Uniform baseline: all tiles use same QP_base-derived quality
- Flow-guided: each tile uses QP_tile[i]-derived quality
- Tracks total bytes and PSNR for both versions
"""
import cv2
import numpy as np
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = "e:/research/myself/360video/data/360 VR ｜ POV Ducati Diavel cruising in Miami.webm"
QP_TABLE_PATH = Path("e:/research/myself/360video/output/qp_table.json")
OUTPUT_DIR = Path("e:/research/myself/360video/output")
TEMP_DIR = OUTPUT_DIR / "temp_frames_v2"
UNIFORM_DIR = TEMP_DIR / "uniform"
FLOW_GUIDED_DIR = TEMP_DIR / "flow_guided"

START_SEC = 30
DURATION_SEC = 30         # 30 seconds for faster turnaround
TILE_ROWS = 5
TILE_COLS = 9
CRF_VALUE = 23            # FFmpeg CRF for final encoding
VIDEO_FPS = 30


def qp_to_jpeg_quality(qp):
    """Map QP value to JPEG quality (0-100)."""
    q = int(round(100 - (qp - 18) * 3.0))
    return max(5, min(100, q))


def encode_tile_jpeg(tile, quality):
    """JPEG encode a tile at given quality, return encoded bytes and decoded tile."""
    _, enc = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return enc, dec


def process_frame(frame, qp_tile_list, qp_base, tile_h, tile_w):
    """
    Process a single frame for both uniform and flow-guided.
    Returns:
      uniform_frame, fg_frame, uniform_bytes, fg_bytes
    """
    h, w = frame.shape[:2]
    n_tiles = TILE_ROWS * TILE_COLS

    uniform_tiles = np.zeros_like(frame)
    fg_tiles = np.zeros_like(frame)
    uniform_bytes = 0
    fg_bytes = 0

    # QP_base quality for uniform
    uniform_quality = qp_to_jpeg_quality(qp_base)

    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            idx = r * TILE_COLS + c
            y0, y1 = r * tile_h, min((r + 1) * tile_h, h)
            x0, x1 = c * tile_w, min((c + 1) * tile_w, w)
            tile = frame[y0:y1, x0:x1]

            # Uniform: all tiles same quality
            u_enc, u_dec = encode_tile_jpeg(tile, uniform_quality)
            uniform_tiles[y0:y1, x0:x1] = u_dec
            uniform_bytes += len(u_enc)

            # Flow-guided: per-tile quality
            fg_qp = qp_tile_list[idx]
            fg_quality = qp_to_jpeg_quality(fg_qp)
            f_enc, f_dec = encode_tile_jpeg(tile, fg_quality)
            fg_tiles[y0:y1, x0:x1] = f_dec
            fg_bytes += len(f_enc)

    return uniform_tiles, fg_tiles, uniform_bytes, fg_bytes


def encode_frames_to_video(frames_dir, output_path, fps, crf):
    """Encode PNG frames to H.264 video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
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
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tile_h = orig_h // TILE_ROWS
    tile_w = orig_w // TILE_COLS

    start_frame = int(START_SEC * VIDEO_FPS)
    end_frame = int((START_SEC + DURATION_SEC) * VIDEO_FPS)
    frame_step = qp_data["metadata"].get("frame_step", 6)  # from flow computation

    print(f"Processing frames {start_frame}-{end_frame} ({DURATION_SEC}s)")
    print(f"Tile grid: {TILE_COLS}x{TILE_ROWS}, tile: {tile_w}x{tile_h}")
    print(f"QP→JPEG: QP18→{qp_to_jpeg_quality(18)}, "
          f"QP25→{qp_to_jpeg_quality(25)}, QP38→{qp_to_jpeg_quality(38)}")

    UNIFORM_DIR.mkdir(parents=True, exist_ok=True)
    FLOW_GUIDED_DIR.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    output_idx = 0

    psnr_uniform_vals = []
    psnr_fg_vals = []
    byte_uniform_vals = []
    byte_fg_vals = []
    per_frame_log = []

    pbar = tqdm(total=end_frame - start_frame, desc="JPEG tile encoding")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Find nearest QP data
        flow_frame = (frame_idx // frame_step) * frame_step
        qp_info = qp_lookup.get(flow_frame)
        if qp_info is None:
            keys = sorted(qp_lookup.keys())
            nearest = min(keys, key=lambda k: abs(k - frame_idx))
            qp_info = qp_lookup[nearest]

        uniform_frame, fg_frame, u_bytes, fg_bytes = process_frame(
            frame, qp_info["qp_tile"], qp_info["qp_base"], tile_h, tile_w
        )

        # Save PNG frames
        cv2.imwrite(str(UNIFORM_DIR / f"frame_{output_idx:06d}.png"), uniform_frame)
        cv2.imwrite(str(FLOW_GUIDED_DIR / f"frame_{output_idx:06d}.png"), fg_frame)

        # Metrics
        psnr_u = compute_psnr(frame, uniform_frame)
        psnr_f = compute_psnr(frame, fg_frame)
        psnr_uniform_vals.append(psnr_u)
        psnr_fg_vals.append(psnr_f)
        byte_uniform_vals.append(u_bytes)
        byte_fg_vals.append(fg_bytes)

        if output_idx % 30 == 0:
            per_frame_log.append({
                "frame_idx": frame_idx,
                "output_idx": output_idx,
                "psnr_uniform": round(psnr_u, 2),
                "psnr_flow_guided": round(psnr_f, 2),
                "jpeg_bytes_uniform": u_bytes,
                "jpeg_bytes_flow_guided": fg_bytes,
                "byte_ratio": round(fg_bytes / max(u_bytes, 1), 3),
            })

        frame_idx += 1
        output_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # --- Encode to video ---
    print("\nEncoding uniform video (FFmpeg x264 CRF={})...".format(CRF_VALUE))
    uniform_video = OUTPUT_DIR / "uniform_jpeg_baseline.mp4"
    encode_frames_to_video(UNIFORM_DIR, uniform_video, VIDEO_FPS, CRF_VALUE)

    print("Encoding flow-guided video...")
    fg_video = OUTPUT_DIR / "flow_guided_jpeg.mp4"
    encode_frames_to_video(FLOW_GUIDED_DIR, fg_video, VIDEO_FPS, CRF_VALUE)

    # --- Results ---
    u_video_size = uniform_video.stat().st_size
    fg_video_size = fg_video.stat().st_size
    u_jpeg_total = sum(byte_uniform_vals)
    fg_jpeg_total = sum(byte_fg_vals)
    u_psnr_mean = np.mean(psnr_uniform_vals)
    fg_psnr_mean = np.mean(psnr_fg_vals)

    print(f"\n{'='*60}")
    print("RESULTS: Per-Tile JPEG Simulation")
    print(f"{'='*60}")
    print(f"{'':25} {'Uniform':>15} {'Flow-Guided':>15}")
    print(f"{'─'*55}")
    print(f"{'JPEG total bytes':25} {u_jpeg_total:>15,} {fg_jpeg_total:>15,}")
    print(f"{'JPEG avg bytes/frame':25} {u_jpeg_total/output_idx:>15.0f} "
          f"{fg_jpeg_total/output_idx:>15.0f}")
    print(f"{'Video file size':25} {u_video_size/1e6:>14.2f} MB "
          f"{fg_video_size/1e6:>14.2f} MB")
    print(f"{'Avg PSNR vs original':25} {u_psnr_mean:>14.2f} dB "
          f"{fg_psnr_mean:>14.2f} dB")

    jpeg_saving = (1 - fg_jpeg_total / u_jpeg_total) * 100
    video_saving = (1 - fg_video_size / u_video_size) * 100
    print(f"\nJPEG bytes saving: {jpeg_saving:.1f}%")
    print(f"Video file saving: {video_saving:.1f}%")

    # Save metrics
    metrics = {
        "method": "per_tile_jpeg",
        "config": {"start_sec": START_SEC, "duration_sec": DURATION_SEC,
                   "crf": CRF_VALUE, "tile_grid": [TILE_ROWS, TILE_COLS]},
        "results": {
            "jpeg_total_uniform": u_jpeg_total,
            "jpeg_total_flow_guided": fg_jpeg_total,
            "jpeg_bytes_saving_pct": round(jpeg_saving, 1),
            "video_size_uniform_bytes": u_video_size,
            "video_size_flow_guided_bytes": fg_video_size,
            "video_size_saving_pct": round(video_saving, 1),
            "psnr_uniform_mean": round(float(u_psnr_mean), 2),
            "psnr_flow_guided_mean": round(float(fg_psnr_mean), 2),
        },
        "per_frame_log": per_frame_log,
    }
    metrics_path = OUTPUT_DIR / "encode_metrics_v2.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
