"""
Step 2: QP Offset Table Generation (Optical Flow + Viewport Proxy + Bandwidth)
- Reads flow_data.json from Step 1
- Computes viewport proxy weights (optical-flow-minimum direction)
- Fuses flow, viewport, and simulated bandwidth signals into per-tile QP offsets
- Outputs QP offset table for encoding
"""
import json
import numpy as np
from pathlib import Path
# (no external deps beyond numpy, json, pathlib)

# ============================================================
# CONFIGURATION
# ============================================================
FLOW_DATA_PATH = Path("e:/research/myself/360video/output/flow_data.json")
OUTPUT_DIR = Path("e:/research/myself/360video/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Tile grid (must match Step 1)
TILE_ROWS = 5
TILE_COLS = 9

# Fusion weights
ALPHA_VIEW = 0.35     # viewport proxy weight
BETA_FLOW = 0.45      # optical flow weight
GAMMA_TASK = 0.20     # task semantic weight (0 for generic video, will be all-equal)

# Viewport proxy: Gaussian sigma (in tile units) for viewport spread around min-flow direction
SIGMA_VIEW_FAST = 1.5   # fast movement: narrow viewport
SIGMA_VIEW_SLOW = 4.0   # slow movement: wide viewport

# QP parameters
QP_BASE_DEFAULT = 25     # default QP when bandwidth is sufficient
QP_MIN = 18
QP_MAX = 38
DELTA_QP_MAX = 6         # max QP deviation between tiles

# Bandwidth simulation
# Simulate a 5G bandwidth trace: start good, drop in middle, recover
BW_TARGET_Mbps = 50


def build_tile_distance_matrix(rows, cols):
    """Precompute tile-to-tile distance matrix (in tile units)."""
    n = rows * cols
    dist = np.zeros((n, n))
    for i in range(n):
        r1, c1 = i // cols, i % cols
        for j in range(n):
            r2, c2 = j // cols, j % cols
            # Euclidean distance in tile space
            # Account for horizontal wrap-around in equirectangular
            dc = min(abs(c1 - c2), cols - abs(c1 - c2))
            dr = abs(r1 - r2)
            dist[i, j] = np.sqrt(dr**2 + dc**2)
    return dist


def compute_viewport_weights(tile_flows, tile_idx_min_flow, dist_matrix, sigma):
    """Gaussian viewport weight centered on min-flow tile."""
    n = len(tile_flows)
    d = dist_matrix[tile_idx_min_flow]  # distance from center to all tiles
    w = np.exp(-0.5 * (d / sigma)**2)
    w = w / (w.max() + 1e-8)  # normalize to [0, 1]
    return w.tolist()


# Speed thresholds computed dynamically from data distribution
_speed_thresholds = None  # set in main()


def estimate_speed(tile_flows):
    """Estimate 'speed level' from tile flows: mean flow across all tiles."""
    global _speed_thresholds
    mean_flow = np.mean(tile_flows)
    lo, hi = _speed_thresholds
    if mean_flow < lo:
        return "slow", SIGMA_VIEW_SLOW
    elif mean_flow < hi:
        return "medium", (SIGMA_VIEW_SLOW + SIGMA_VIEW_FAST) / 2
    else:
        return "fast", SIGMA_VIEW_FAST


def simulate_bandwidth_trace(n_frames, fps_effective):
    """
    Generate a simulated 5G bandwidth trace.
    Good (50 Mbps) -> drop (15 Mbps) -> recover (40 Mbps) -> drop (8 Mbps) -> recover
    """
    duration_sec = n_frames / fps_effective
    t = np.linspace(0, duration_sec, n_frames)

    # Base: 50 Mbps
    bw = np.full(n_frames, 50.0)

    # Drop 1: 60-120s, down to 15 Mbps
    mask1 = (t > 60) & (t <= 120)
    bw[mask1] = 15.0

    # Recovery 1: 120-180s, back to 40 Mbps
    mask2 = (t > 120) & (t <= 180)
    bw[mask2] = 40.0

    # Drop 2: 300-420s, down to 8 Mbps (severe)
    mask3 = (t > 300) & (t <= 420)
    bw[mask3] = 8.0

    # Add small random fluctuation
    np.random.seed(42)
    bw += np.random.normal(0, 2, n_frames)
    bw = np.clip(bw, 3, 60)

    return bw


def qp_base_from_bandwidth(bw, bw_target=50, qp_default=25,
                           qp_min=18, qp_max=38, k_high=0.6, k_low=0.2):
    """
    Map bandwidth to global QP base.
    - Bandwidth drops -> QP rises quickly (k_high)
    - Bandwidth recovers -> QP falls slowly (k_low)
    """
    bw_ratio = bw / bw_target
    delta = np.where(bw_ratio < 1,
                     (1 - bw_ratio) * qp_default * k_high,   # BW below target
                     (1 - bw_ratio) * qp_default * k_low)     # BW above target
    qp = qp_default + delta
    return np.clip(np.round(qp).astype(int), qp_min, qp_max)


def main():
    # Load flow data
    with open(FLOW_DATA_PATH) as f:
        data = json.load(f)

    meta = data["metadata"]
    frames = data["frames"]
    n_frames = len(frames)
    n_tiles = TILE_ROWS * TILE_COLS
    fps_eff = meta["fps"] / meta["frame_step"]

    print(f"Loaded {n_frames} frames, {n_tiles} tiles")
    print(f"Effective fps: {fps_eff:.1f}")

    # Precompute tile distance matrix
    dist_matrix = build_tile_distance_matrix(TILE_ROWS, TILE_COLS)

    # Compute speed thresholds from flow data distribution (33rd and 66th percentiles)
    global _speed_thresholds
    all_norm_means = [np.mean(f["flow_norm"]) for f in frames]
    lo_thresh = np.percentile(all_norm_means, 33)
    hi_thresh = np.percentile(all_norm_means, 66)
    _speed_thresholds = (lo_thresh, hi_thresh)
    print(f"Speed thresholds: slow < {lo_thresh:.4f}, "
          f"medium < {hi_thresh:.4f}, fast >= {hi_thresh:.4f}")

    # Simulate bandwidth trace
    bw_trace = simulate_bandwidth_trace(n_frames, fps_eff)
    qp_base_trace = qp_base_from_bandwidth(bw_trace)

    # Process each frame
    output_frames = []
    speed_stats = {"slow": 0, "medium": 0, "fast": 0}

    for fdata in frames:
        flow_norm = np.array(fdata["flow_norm"])

        # --- Viewport proxy: Gaussian around min-flow tile ---
        idx_min = int(np.argmin(flow_norm))
        speed_level, sigma = estimate_speed(flow_norm)
        speed_stats[speed_level] += 1
        w_view = np.array(compute_viewport_weights(flow_norm, idx_min,
                                                    dist_matrix, sigma))

        # --- Flow weight: invert normalized flow ---
        w_flow = 1.0 - flow_norm

        # --- Task weight: uniform for generic video ---
        w_task = np.ones(n_tiles) / n_tiles

        # --- Fusion ---
        w_tile = (ALPHA_VIEW * w_view +
                  BETA_FLOW * w_flow +
                  GAMMA_TASK * w_task)
        w_tile = w_tile / (w_tile.max() + 1e-8)  # normalize

        # --- QP offset: high priority -> negative delta QP (better quality) ---
        # Map W ∈ [0,1] to ΔQP ∈ [+DELTA_QP_MAX, -DELTA_QP_MAX]
        # W=1 (important) -> ΔQP = -DELTA_QP_MAX; W=0 (unimportant) -> ΔQP = +DELTA_QP_MAX
        delta_qp = DELTA_QP_MAX * (1 - 2 * w_tile)

        # Bandwidth-aware scaling of delta_qp range
        # When bandwidth is tight, increase delta_qp range to concentrate bits
        frame_qp_base = int(qp_base_trace[fdata["output_idx"]])
        if frame_qp_base > 30:
            bw_factor = 1.5  # severe compression: larger differentiation
        elif frame_qp_base > 25:
            bw_factor = 1.2
        else:
            bw_factor = 1.0
        delta_qp = delta_qp * bw_factor

        # Final QP per tile
        qp_tile = np.clip(np.round(frame_qp_base + delta_qp).astype(int),
                          QP_MIN, QP_MAX)

        output_frames.append({
            "frame": fdata["frame"],
            "output_idx": fdata["output_idx"],
            "speed_level": speed_level,
            "qp_base": frame_qp_base,
            "bandwidth_mbps": round(float(bw_trace[fdata["output_idx"]]), 1),
            "w_view": w_view.tolist(),
            "w_flow": w_flow.tolist(),
            "w_tile": w_tile.tolist(),
            "delta_qp": delta_qp.tolist(),
            "qp_tile": qp_tile.tolist(),
        })

    # Save
    output_path = OUTPUT_DIR / "qp_table.json"
    output_meta = {
        "tile_grid": [TILE_ROWS, TILE_COLS],
        "n_tiles": n_tiles,
        "alpha_view": ALPHA_VIEW,
        "beta_flow": BETA_FLOW,
        "gamma_task": GAMMA_TASK,
        "delta_qp_max": DELTA_QP_MAX,
        "qp_base_default": QP_BASE_DEFAULT,
        "qp_range": [QP_MIN, QP_MAX],
        "flow_data_source": str(FLOW_DATA_PATH),
    }
    out = {"metadata": output_meta, "frames": output_frames,
           "bandwidth_trace_mbps": bw_trace.tolist(),
           "qp_base_trace": qp_base_trace.tolist()}

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {output_path}")
    print(f"Speed distribution: {speed_stats}")
    print(f"QP_base range: [{min(qp_base_trace)}, {max(qp_base_trace)}]")
    print(f"Bandwidth trace range: [{bw_trace.min():.1f}, {bw_trace.max():.1f}] Mbps")


if __name__ == "__main__":
    main()
