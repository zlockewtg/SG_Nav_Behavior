"""Pure geometry / math utilities for SG-Nav.

Quaternion conversions, angle wrapping, coordinate transforms between
GPS-frame, map-grid, and world-meters.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw convention, OmniGibson-style)
# ---------------------------------------------------------------------------

def quat_xyzw_to_rpy(qx: float, qy: float, qz: float, qw: float):
    """Roll / pitch / yaw (radians) from quaternion (x, y, z, w)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(n) or n < 1e-9:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = (qx / n, qy / n, qz / n, qw / n)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_xyzw_rotate_vec(qx: float, qy: float, qz: float, qw: float,
                          vx: float, vy: float, vz: float):
    """Rotate 3-D vector by quaternion (xyzw)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(n) or n < 1e-9:
        return float(vx), float(vy), float(vz)
    x, y, z, w = (qx / n, qy / n, qz / n, qw / n)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return (
        float(r00 * vx + r01 * vy + r02 * vz),
        float(r10 * vx + r11 * vy + r12 * vz),
        float(r20 * vx + r21 * vy + r22 * vz),
    )


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap to (-pi, pi]."""
    x = float(angle_rad)
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap to (-180, 180]."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Pinhole camera
# ---------------------------------------------------------------------------

def horiz_angle_from_pixel_u(u_pixel: float, sensor_width: int, hfov_deg: float) -> float:
    """Horizontal viewing angle (deg) from pixel column, linear pinhole model."""
    cx = (sensor_width - 1) / 2.0
    return (float(u_pixel) - cx) * hfov_deg / float(sensor_width)


def depth_m_at_xy(depth_hw1: np.ndarray, x_col: int, y_row: int) -> float:
    h, w = int(depth_hw1.shape[0]), int(depth_hw1.shape[1])
    xc = int(np.clip(x_col, 0, w - 1))
    yr = int(np.clip(y_row, 0, h - 1))
    return float(depth_hw1[yr, xc, 0])


# ---------------------------------------------------------------------------
# Map ↔ GPS coordinate conversions
# ---------------------------------------------------------------------------

def goal_gps_to_map_rc(goal_gps, map_size_cm: int, resolution: int, map_size: int):
    """GPS-frame goal → raw goal_map (row, col)."""
    gx, gy = float(goal_gps[0]), float(goal_gps[1])
    half_cells = float(map_size_cm) / 10.0
    r = int(round(half_cells - gy * 100.0 / float(resolution)))
    c = int(round(half_cells + gx * 100.0 / float(resolution)))
    return max(0, min(map_size - 1, r)), max(0, min(map_size - 1, c))


def map_rc_to_goal_gps(r: int, c: int, map_size_cm: int, resolution: int):
    """Inverse: raw goal_map (row, col) → GPS-frame (x, y)."""
    half_cells = float(map_size_cm) / 10.0
    gx = (float(c) - half_cells) / 100.0 * float(resolution)
    gy = (half_cells - float(r)) / 100.0 * float(resolution)
    return np.array([gx, gy], dtype=np.float32)


def world_xy_to_grid_rc(tx: float, ty: float, resolution: int, map_size: int):
    """Absolute world meters → grid (row, col)."""
    cell_m = float(resolution) / 100.0
    r = int(round(float(ty) / cell_m))
    c = int(round(float(tx) / cell_m))
    return max(0, min(map_size - 1, r)), max(0, min(map_size - 1, c))


def get_goal_gps(agent_gps, agent_compass, angle_deg, distance_m):
    """Bearing + range → goal point in Habitat-style GPS frame."""
    angle = np.asarray(angle_deg)
    if hasattr(angle, 'cpu'):
        angle = angle.cpu().numpy()
    goal_direction = agent_compass - float(angle) / 180.0 * np.pi
    return np.array([
        (agent_gps[0] + np.cos(goal_direction) * distance_m).item(),
        (agent_gps[1] - np.sin(goal_direction) * distance_m).item(),
    ])


def get_relative_goal_gps(agent_gps, agent_compass, goal_gps):
    """Polar (rho, phi) point-goal relative to agent."""
    if goal_gps is None or len(goal_gps) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    d = np.asarray(goal_gps[:2]) - np.array([float(agent_gps[0]), float(agent_gps[1])])
    rho = float(np.sqrt(d[0] ** 2 + d[1] ** 2))
    phi_world = np.arctan2(d[1], d[0])
    phi = phi_world - agent_compass
    return np.array([rho, float(phi)], dtype=np.float32)


def get_goal_stop_status(agent_pose, goal_gps, map_size_cm: int):
    """Distance + bearing vs heading for final approach."""
    if goal_gps is None or len(goal_gps) < 2:
        return None
    try:
        gx, gy = float(goal_gps[0]), float(goal_gps[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(gx) and math.isfinite(gy)):
        return None
    half = float(map_size_cm) / 200.0
    ax = float(agent_pose[0]) - half
    ay = half - float(agent_pose[1])
    dx, dy = gx - ax, gy - ay
    rho = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx))
    heading = float(agent_pose[2])
    return {
        "distance_m": rho,
        "bearing_deg": bearing,
        "heading_deg": heading,
        "heading_error_deg": wrap_angle_deg(bearing - heading),
    }


# ---------------------------------------------------------------------------
# Map frame transforms (hardcoded flipud for the simplified pipeline)
# ---------------------------------------------------------------------------

def apply_flipud(arr):
    """Flipud transform — the only frame mode after simplification."""
    return np.flipud(np.asarray(arr))


def transform_rc_flipud(r: int, c: int, h: int, _w: int):
    return max(0, min(h - 1, h - 1 - r)), c


def inverse_transform_rc_flipud(r: int, c: int, h: int, _w: int):
    return max(0, min(h - 1, h - 1 - r)), c


def disk_mask(h: int, w: int, cy: int, cx: int, radius_cells: int):
    if radius_cells <= 0:
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius_cells ** 2)


def robot_map_rc(full_pose, map_size_cm: int, resolution: int, h: int, w: int):
    """full_pose → grid (row, col)."""
    px = float(full_pose[0].detach().cpu().item()) if hasattr(full_pose[0], 'detach') else float(full_pose[0])
    py = float(full_pose[1].detach().cpu().item()) if hasattr(full_pose[1], 'detach') else float(full_pose[1])
    sy = int(round((float(map_size_cm) / 100.0 - py) * 100.0 / float(resolution)))
    sx = int(round(px * 100.0 / float(resolution)))
    return max(0, min(h - 1, sy)), max(0, min(w - 1, sx))
