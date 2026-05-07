"""
Step 4: Visualization
- Optical flow heatmap overlay on equirectangular frame
- Tile weight maps (flow, viewport, combined)
- Bandwidth trace + QP_base over time
- PSNR comparison
"""
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = "e:/research/myself/360video/data/360 VR ｜ POV Ducati Diavel cruising in Miami.webm"
FLOW_DATA_PATH = Path("e:/research/myself/360video/output/flow_data.json")
QP_TABLE_PATH = Path("e:/research/myself/360video/output/qp_table.json")
METRICS_PATH = Path("e:/research/myself/360video/output/encode_metrics.json")
OUTPUT_DIR = Path("e:/research/myself/360video/output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_ROWS = 5
TILE_COLS = 9
FLOW_SCALE = 0.25

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
})


def draw_tile_grid(ax, values, rows, cols, title, cmap="RdYlGn_r"):
    """Draw tile weight grid as heatmap overlay on equirectangular layout."""
    # Reshape flat list to (rows, cols)
    grid = np.array(values).reshape(rows, cols)
    # Repeat each tile cell to create image at tile resolution
    h, w = grid.shape
    img = np.kron(grid, np.ones((20, 40)))  # scale up for display
    im = ax.imshow(img, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw grid lines
    for i in range(1, w):
        ax.axvline(x=i * 40 - 0.5, color="white", linewidth=0.5, alpha=0.5)
    for i in range(1, h):
        ax.axhline(y=i * 20 - 0.5, color="white", linewidth=0.5, alpha=0.5)
    return im


def visualize_weights():
    """Plot figure 1: Tile weight maps for a sample frame."""
    with open(QP_TABLE_PATH) as f:
        qp_data = json.load(f)

    frames = qp_data["frames"]
    # Pick frames from slow, medium, fast segments
    speed_samples = {}
    for fdata in frames:
        sl = fdata["speed_level"]
        if sl not in speed_samples:
            speed_samples[sl] = fdata
    sample_frames = list(speed_samples.values())

    fig, axes = plt.subplots(len(sample_frames), 4,
                              figsize=(14, 3 * len(sample_frames)))
    if len(sample_frames) == 1:
        axes = axes.reshape(1, -1)

    for row, fdata in enumerate(sample_frames):
        draw_tile_grid(axes[row, 0], fdata["w_flow"], TILE_ROWS, TILE_COLS,
                       f"Flow Weight (1-flow_norm)\nFrame {fdata['frame']} [{fdata['speed_level']}]")
        draw_tile_grid(axes[row, 1], fdata["w_view"], TILE_ROWS, TILE_COLS,
                       "Viewport Proxy Weight")
        draw_tile_grid(axes[row, 2], fdata["w_tile"], TILE_ROWS, TILE_COLS,
                       "Combined Weight (W_tile)")
        # QP offsets: use RdYlGn (green=negative delta=better, red=positive=worse)
        qp_grid = np.array(fdata["delta_qp"]).reshape(TILE_ROWS, TILE_COLS)
        img = np.kron(qp_grid, np.ones((20, 40)))
        vmax = max(abs(qp_grid.min()), abs(qp_grid.max()))
        im = axes[row, 3].imshow(img, cmap="RdYlGn", aspect="auto",
                                 vmin=-vmax, vmax=vmax)
        axes[row, 3].set_title("Delta QP (green=better, red=worse)")
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    plt.tight_layout()
    path = OUTPUT_DIR / "tile_weights.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def visualize_flow_overlay():
    """Plot figure 2: Optical flow on a real equirectangular frame."""
    with open(FLOW_DATA_PATH) as f:
        flow_data = json.load(f)

    # Get a sample frame from the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    sample_frame_idx = flow_data["frames"][len(flow_data["frames"]) // 2]["frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read sample frame for flow overlay")
        return

    # Downsample for flow display
    flow_w = int(frame.shape[1] * FLOW_SCALE)
    flow_h = int(frame.shape[0] * FLOW_SCALE)
    frame_small = cv2.resize(frame, (flow_w, flow_h))

    # Get flow data for this frame
    sample = flow_data["frames"][len(flow_data["frames"]) // 2]
    flow_norm = np.array(sample["flow_norm"])

    # Reshape to tile grid for overlay
    flow_grid = flow_norm.reshape(TILE_ROWS, TILE_COLS)
    tile_h = flow_h // TILE_ROWS
    tile_w = flow_w // TILE_COLS
    flow_img = np.kron(flow_grid, np.ones((tile_h, tile_w)))
    # Resize to match exact frame dimensions (kron may be off by 1 pixel)
    flow_img = cv2.resize(flow_img, (flow_w, flow_h))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Frame (downsampled)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(flow_img, cmap="hot", aspect="auto")
    axes[1].set_title("Optical Flow Magnitude (normalized)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay
    flow_colored = cv2.applyColorMap(
        (flow_img * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(frame_small, 0.5, flow_colored, 0.5, 0)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Flow Overlay")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()
    path = OUTPUT_DIR / "flow_overlay.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def visualize_bandwidth_and_qp():
    """Plot figure 3: Bandwidth trace and QP_base over time."""
    with open(QP_TABLE_PATH) as f:
        qp_data = json.load(f)

    bw = np.array(qp_data["bandwidth_trace_mbps"])
    qp_base = np.array(qp_data["qp_base_trace"])
    n = len(bw)
    fps_eff = 30 / 3  # approximate effective fps
    time_sec = np.arange(n) / fps_eff

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax1 = axes[0]
    ax1.plot(time_sec, bw, linewidth=0.8, color="#2196F3")
    ax1.axhline(y=50, color="green", linestyle="--", alpha=0.5, label="Target 50 Mbps")
    ax1.axhline(y=15, color="orange", linestyle="--", alpha=0.5, label="Low 15 Mbps")
    ax1.fill_between(time_sec, 0, bw, alpha=0.15, color="#2196F3")
    ax1.set_ylabel("Bandwidth (Mbps)")
    ax1.set_title("Simulated 5G Uplink Bandwidth")
    ax1.legend(fontsize=7)
    ax1.set_ylim(0, 65)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(time_sec, qp_base, linewidth=0.8, color="#E91E63")
    ax2.fill_between(time_sec, 18, qp_base, alpha=0.1, color="#E91E63")
    ax2.set_ylabel("QP_base")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Adaptive QP_base Response")
    ax2.set_ylim(16, 40)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "bandwidth_qp.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def visualize_psnr():
    """Plot figure 4: PSNR comparison over time."""
    if not METRICS_PATH.exists():
        print("Metrics file not found, skipping PSNR plot")
        return

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    log = metrics["per_frame_log"]
    frames = [l["output_idx"] for l in log]
    psnr_u = [l["psnr_uniform"] for l in log]
    psnr_fg = [l["psnr_flow_guided"] for l in log]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frames, psnr_u, linewidth=0.6, alpha=0.7, label="Uniform", color="#666")
    ax.plot(frames, psnr_fg, linewidth=0.8, label="Flow-Guided", color="#E91E63")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("PSNR (dB) vs Original")
    ax.set_title("Per-Frame PSNR: Uniform vs Flow-Guided")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add mean lines
    mu = np.mean(psnr_u)
    mfg = np.mean(psnr_fg)
    ax.axhline(y=mu, color="#666", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(y=mfg, color="#E91E63", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(len(frames) * 0.98, mu, f" μ={mu:.1f}", fontsize=7, va="bottom", ha="right")
    ax.text(len(frames) * 0.98, mfg, f" μ={mfg:.1f}", fontsize=7, va="bottom", ha="right")

    plt.tight_layout()
    path = OUTPUT_DIR / "psnr_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("Generating visualizations...")

    print("\n[1/4] Tile weight maps...")
    visualize_weights()

    print("[2/4] Flow overlay...")
    visualize_flow_overlay()

    print("[3/4] Bandwidth & QP trace...")
    visualize_bandwidth_and_qp()

    print("[4/4] PSNR comparison...")
    visualize_psnr()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
