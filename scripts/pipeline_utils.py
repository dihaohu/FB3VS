"""
Shared utilities for 360° tile-adaptive video pipeline.
Functions extracted from step1_optical_flow.py and step2_qp_table.py.
"""
import cv2
import numpy as np

# ============================================================
# Constants (match step1 & step2)
# ============================================================
TILE_ROWS = 5
TILE_COLS = 9
DELTA_QP_MAX = 6
QP_MIN = 18
QP_MAX = 38
QP_BASE_DEFAULT = 25

SIGMA_VIEW_FAST = 1.5
SIGMA_VIEW_SLOW = 4.0

ALPHA_VIEW = 0.35
BETA_FLOW = 0.45
GAMMA_TASK = 0.20


# ============================================================
# Spherical correction (from step1)
# ============================================================
def build_spherical_mask(h, w):
    """Build cos(latitude) weight mask for equirectangular projection."""
    ys = np.arange(h)
    lat = np.pi * (ys / (h - 1) - 0.5)
    weight = np.cos(lat)
    weight = np.abs(weight)
    weight = np.clip(weight, 0.05, 1.0)
    mask = np.tile(weight[:, np.newaxis], (1, w))
    return mask.astype(np.float32)


# ============================================================
# Tile geometry (from step2)
# ============================================================
def build_tile_distance_matrix(rows, cols):
    """Precompute tile-to-tile distance matrix (in tile units)."""
    n = rows * cols
    dist = np.zeros((n, n))
    for i in range(n):
        r1, c1 = i // cols, i % cols
        for j in range(n):
            r2, c2 = j // cols, j % cols
            dc = min(abs(c1 - c2), cols - abs(c1 - c2))
            dr = abs(r1 - r2)
            dist[i, j] = np.sqrt(dr**2 + dc**2)
    return dist


def compute_viewport_weights(tile_flows, min_flow_idx, dist_matrix, sigma):
    """Gaussian viewport weight centered on min-flow tile."""
    d = dist_matrix[min_flow_idx]
    w = np.exp(-0.5 * (d / sigma)**2)
    w = w / (w.max() + 1e-8)
    return w


def estimate_speed(tile_flows, speed_thresholds):
    """Estimate speed level from tile flows. Returns (level_str, sigma)."""
    mean_flow = np.mean(tile_flows)
    lo, hi = speed_thresholds
    if mean_flow < lo:
        return "slow", SIGMA_VIEW_SLOW
    elif mean_flow < hi:
        return "medium", (SIGMA_VIEW_SLOW + SIGMA_VIEW_FAST) / 2
    else:
        return "fast", SIGMA_VIEW_FAST


# ============================================================
# Bandwidth simulation (from step2)
# ============================================================
def simulate_bandwidth_trace(n_frames, fps_effective):
    """Generate a simulated 5G bandwidth trace."""
    duration_sec = n_frames / fps_effective
    t = np.linspace(0, duration_sec, n_frames)
    bw = np.full(n_frames, 50.0)
    mask1 = (t > 60) & (t <= 120)
    bw[mask1] = 15.0
    mask2 = (t > 120) & (t <= 180)
    bw[mask2] = 40.0
    mask3 = (t > 300) & (t <= 420)
    bw[mask3] = 8.0
    np.random.seed(42)
    bw += np.random.normal(0, 2, n_frames)
    bw = np.clip(bw, 3, 60)
    return bw


def qp_base_from_bandwidth(bw, bw_target=50, qp_default=25,
                           qp_min=18, qp_max=38, k_high=0.6, k_low=0.2):
    """Map bandwidth to global QP base. Asymmetric response."""
    bw_ratio = bw / bw_target
    delta = np.where(bw_ratio < 1,
                     (1 - bw_ratio) * qp_default * k_high,
                     (1 - bw_ratio) * qp_default * k_low)
    qp = qp_default + delta
    return np.clip(np.round(qp).astype(int), qp_min, qp_max)


def qp_to_jpeg_quality(qp):
    """Map QP value to JPEG quality (0-100). For logging reference only."""
    q = int(round(100 - (qp - 18) * 3.0))
    return max(5, min(100, q))


# ============================================================
# QP computation (per-frame, core logic from step2 main loop)
# ============================================================
def compute_tile_qp(flow_norm, frame_qp_base, dist_matrix, speed_thresholds,
                    alpha_view=ALPHA_VIEW, beta_flow=BETA_FLOW,
                    gamma_task=GAMMA_TASK, delta_qp_max=DELTA_QP_MAX,
                    qp_min=QP_MIN, qp_max=QP_MAX):
    """
    Compute per-tile QP for a single frame given normalized flow.

    Args:
        flow_norm: np.array of shape (n_tiles,) normalized flow magnitudes [0, 1]
        frame_qp_base: int, global QP base for this frame
        dist_matrix: precomputed tile distance matrix
        speed_thresholds: (lo, hi) tuple for speed classification
        alpha_view, beta_flow, gamma_task: fusion weights
        delta_qp_max: max QP deviation between tiles
        qp_min, qp_max: QP bounds

    Returns:
        qp_tile: np.array of per-tile QP values
        w_tile: np.array of combined tile weights
        delta_qp: np.array of per-tile QP offsets
        speed_level: str, "slow"/"medium"/"fast"
    """
    n_tiles = len(flow_norm)

    # Viewport proxy: Gaussian around min-flow tile
    idx_min = int(np.argmin(flow_norm))
    speed_level, sigma = estimate_speed(flow_norm, speed_thresholds)
    w_view = np.array(compute_viewport_weights(flow_norm, idx_min,
                                                dist_matrix, sigma))

    # Flow weight: invert normalized flow
    w_flow = 1.0 - flow_norm

    # Task weight: uniform for generic video
    w_task = np.ones(n_tiles) / n_tiles

    # Fusion
    w_tile = (alpha_view * w_view +
              beta_flow * w_flow +
              gamma_task * w_task)
    w_tile = w_tile / (w_tile.max() + 1e-8)

    # QP offset
    delta_qp = delta_qp_max * (1 - 2 * w_tile)

    # Bandwidth-aware scaling
    if frame_qp_base > 30:
        bw_factor = 1.5
    elif frame_qp_base > 25:
        bw_factor = 1.2
    else:
        bw_factor = 1.0
    delta_qp = delta_qp * bw_factor

    qp_tile = np.clip(np.round(frame_qp_base + delta_qp).astype(int),
                      qp_min, qp_max)

    return qp_tile, w_tile, delta_qp, speed_level


# ============================================================
# Quality degradation (spatial downscale/upscale — real-time only)
# ============================================================
def apply_tile_quality_degrade(frame, qp_tile, tile_h, tile_w):
    """
    Apply tile-adaptive spatial quality degradation.
    Important tiles (low QP) kept sharp; unimportant tiles (high QP)
    are downscaled then upscaled back to lose high-frequency detail.

    No boundary artifacts — each tile is processed independently
    with smooth interpolation from cv2.resize.
    """
    result = frame.copy()
    h, w = frame.shape[:2]

    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            idx = r * TILE_COLS + c
            qp = qp_tile[idx]
            scale = 1.0 - (qp - QP_MIN) / (QP_MAX - QP_MIN) * 0.6
            scale = max(0.3, min(1.0, scale))

            if scale >= 0.98:
                continue

            y0, y1 = r * tile_h, min((r + 1) * tile_h, h)
            x0, x1 = c * tile_w, min((c + 1) * tile_w, w)
            tile = frame[y0:y1, x0:x1]
            th, tw = tile.shape[:2]

            small = cv2.resize(tile, (max(4, int(tw * scale)), max(4, int(th * scale))),
                               interpolation=cv2.INTER_LINEAR)
            degraded = cv2.resize(small, (tw, th), interpolation=cv2.INTER_LINEAR)
            result[y0:y1, x0:x1] = degraded

    return result
