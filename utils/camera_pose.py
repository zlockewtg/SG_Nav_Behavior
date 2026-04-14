"""Camera extrinsics synchronisation for SG-Nav HTTP / OmniGibson mode.

Extracts pitch and height from ``camera_pose_world`` quaternion + position
and applies them to ``Semantic_Mapping`` modules before each depth→map update.
Also handles fall detection and mapping suspension.
"""
from __future__ import annotations

import math

import numpy as np

from utils.geometry import quat_xyzw_to_rpy, quat_xyzw_rotate_vec


# ── Hardcoded defaults (previously YAML runtime params) ──────────────
_TILT_FALLBACK_TO_CAM_Z_DOWN = True
_PITCH_MAPPING_SCALE = 1.0
_PITCH_MAPPING_OFFSET_DEG = 0.0
_REJECT_OUTLIER = True
_HEIGHT_REJECT_MIN_CM = 120.0
_HEIGHT_REJECT_MAX_CM = 210.0
_PITCH_REJECT_ABS_DEG = 55.0
_REJECT_MOTION_DELTA = True
_PITCH_REJECT_DELTA_DEG = 6.0
_HEIGHT_REJECT_DELTA_CM = 8.0
_HEIGHT_FROM_WORLD_Z = True
_HEIGHT_Z_OFFSET_M = 0.0
_OBSTACLE_ABOVE_CAMERA_CM = 50.0
_MAP_FREE_MIN_Z_CM = -150.0
_MAP_FREE_MAX_Z_CM = 10.0
_SUSPEND_STEPS_ON_FALL = 4
_SUSPEND_ROLL_DEG = 85.0
_SUSPEND_PITCH_DEG = 50.0
_SUSPEND_HEIGHT_CM = 95.0
_SUSPEND_HEIGHT_DELTA_CM = 20.0


# ── Public helpers ────────────────────────────────────────────────────

def client_camera_pose_active(observations: dict) -> bool:
    cpp = observations.get("camera_pose_world")
    if not isinstance(cpp, dict):
        return False
    pos = cpp.get("position")
    quat = cpp.get("quaternion")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return False
    if not isinstance(quat, (list, tuple)) or len(quat) < 4:
        return False
    return True


def sync_camera_extrinsics(
    observations: dict,
    modules,
    *,
    camera_tilt_source: str,
    camera_height_fallback_cm: float,
    camera_pitch_min_deg: float,
    camera_pitch_max_deg: float,
    camera_height_min_cm: float,
    camera_height_max_cm: float,
    map_obstacle_min_z_cm: float,
    last_good_height_cm: float | None,
    last_good_pitch_deg: float | None,
    logger=None,
) -> dict:
    """Sync mapping modules with client camera pose.

    Returns a ``camera_diag`` dict and updated last-good values packed as::

        {"diag": {...}, "last_good_height_cm": ..., "last_good_pitch_deg": ...}
    """
    sem_mod, free_mod, room_mod = modules

    if not client_camera_pose_active(observations):
        h_fb = None
        if camera_height_fallback_cm > 0.0:
            h_fb = float(np.clip(camera_height_fallback_cm, camera_height_min_cm, camera_height_max_cm))
        for m in (sem_mod, free_mod, room_mod):
            m._extrinsic_height_cm = h_fb
        _sync_height_band(sem_mod, free_mod, map_obstacle_min_z_cm)
        diag = {
            "active": False,
            "source": "fallback",
            "tilt_source": "",
            "pitch_deg_used": 0.0,
            "height_cm_used": h_fb or 0.0,
            "roll_deg": 0.0,
        }
        return {"diag": diag, "last_good_height_cm": last_good_height_cm, "last_good_pitch_deg": last_good_pitch_deg}

    cpp = observations["camera_pose_world"]
    px, py, pz = (float(cpp["position"][i]) for i in range(3))
    qx, qy, qz, qw = (float(cpp["quaternion"][i]) for i in range(4))
    roll, pitch, yaw = quat_xyzw_to_rpy(qx, qy, qz, qw)

    z_w = quat_xyzw_rotate_vec(qx, qy, qz, qw, 0.0, 0.0, 1.0)
    y_w = quat_xyzw_rotate_vec(qx, qy, qz, qw, 0.0, 1.0, 0.0)
    z_h = math.hypot(z_w[0], z_w[1])
    y_h = math.hypot(y_w[0], y_w[1])
    cam_z_down_pitch_deg = -math.degrees(math.atan2(z_w[2], max(z_h, 1e-9)))
    cam_y_down_pitch_deg = -math.degrees(math.atan2(y_w[2], max(y_h, 1e-9)))

    tilt_fallback = None
    tilt_src = camera_tilt_source
    if tilt_src == "roll":
        tilt_src_deg = math.degrees(roll)
    elif tilt_src == "yaw":
        tilt_src_deg = math.degrees(yaw)
    elif tilt_src == "cam_z_down":
        tilt_src_deg = cam_z_down_pitch_deg
    elif tilt_src == "cam_y_down":
        tilt_src_deg = cam_y_down_pitch_deg
    else:
        tilt_src = "pitch"
        tilt_src_deg = math.degrees(pitch)

    pitch_deg_raw = tilt_src_deg * _PITCH_MAPPING_SCALE + _PITCH_MAPPING_OFFSET_DEG

    if (
        tilt_src == "roll"
        and _TILT_FALLBACK_TO_CAM_Z_DOWN
        and math.isfinite(cam_z_down_pitch_deg)
        and abs(pitch_deg_raw) > _PITCH_REJECT_ABS_DEG
    ):
        pitch_deg_raw = cam_z_down_pitch_deg
        tilt_fallback = "roll_outlier_to_cam_z_down"

    pitch_deg_clamped = float(np.clip(pitch_deg_raw, camera_pitch_min_deg, camera_pitch_max_deg))

    # ── Height from world Z ──
    h_cm_raw = None
    if _HEIGHT_FROM_WORLD_Z:
        h_cm_raw = (pz + _HEIGHT_Z_OFFSET_M) * 100.0
    h_cm_clamped = None
    if h_cm_raw is not None:
        h_cm_clamped = float(np.clip(h_cm_raw, camera_height_min_cm, camera_height_max_cm))

    # ── Outlier rejection ──
    rejected = False
    if _REJECT_OUTLIER and h_cm_raw is not None:
        if h_cm_raw < _HEIGHT_REJECT_MIN_CM or h_cm_raw > _HEIGHT_REJECT_MAX_CM:
            rejected = True
        if abs(pitch_deg_raw) > _PITCH_REJECT_ABS_DEG and tilt_fallback is None:
            rejected = True

    if _REJECT_MOTION_DELTA and not rejected:
        if last_good_pitch_deg is not None and math.isfinite(pitch_deg_clamped):
            if abs(pitch_deg_clamped - last_good_pitch_deg) > _PITCH_REJECT_DELTA_DEG:
                rejected = True
        if last_good_height_cm is not None and h_cm_clamped is not None and math.isfinite(h_cm_clamped):
            if abs(h_cm_clamped - last_good_height_cm) > _HEIGHT_REJECT_DELTA_CM:
                rejected = True

    if rejected:
        pitch_used = last_good_pitch_deg if last_good_pitch_deg is not None else pitch_deg_clamped
        h_used = last_good_height_cm if last_good_height_cm is not None else h_cm_clamped
    else:
        pitch_used = pitch_deg_clamped
        h_used = h_cm_clamped
        if math.isfinite(pitch_used):
            last_good_pitch_deg = pitch_used
        if h_used is not None and math.isfinite(h_used):
            last_good_height_cm = h_used

    view_angle_cmd = -pitch_used
    for m in (sem_mod, free_mod, room_mod):
        m.set_view_angles(view_angle_cmd)
        m._extrinsic_height_cm = h_used if h_used is not None else None

    _sync_height_band(sem_mod, free_mod, map_obstacle_min_z_cm)

    diag = {
        "active": True,
        "source": str(cpp.get("source", "camera_pose_world")),
        "tilt_source": tilt_src,
        "pitch_deg_used": pitch_used,
        "height_cm_used": h_used if h_used is not None else 0.0,
        "roll_deg": math.degrees(roll),
        "rejected": rejected,
    }
    return {"diag": diag, "last_good_height_cm": last_good_height_cm, "last_good_pitch_deg": last_good_pitch_deg}


def _sync_height_band(sem_mod, free_mod, obstacle_min_z_cm: float):
    h_cm = float(sem_mod._camera_height_cm_for_projection())
    obs_max = h_cm + _OBSTACLE_ABOVE_CAMERA_CM
    obs_max = max(obs_max, obstacle_min_z_cm + 5.0)
    sem_mod.min_z_consider = obstacle_min_z_cm
    sem_mod.max_z_consider = obs_max
    free_mod.min_z_consider = _MAP_FREE_MIN_Z_CM
    free_mod.max_z_consider = _MAP_FREE_MAX_Z_CM


# ── Fall detection & mapping suspension ──────────────────────────────

def check_fall_suspected(camera_diag: dict | None, last_good_height_cm: float | None) -> tuple[bool, str]:
    if not isinstance(camera_diag, dict) or not camera_diag.get("active", False):
        return False, "camera_inactive"
    roll_abs = abs(float(camera_diag.get("roll_deg", 0.0)))
    pitch_abs = abs(float(camera_diag.get("pitch_deg_used", 0.0)))
    h_used = camera_diag.get("height_cm_used")
    reasons = []
    if roll_abs >= _SUSPEND_ROLL_DEG:
        reasons.append(f"roll={roll_abs:.1f}")
    if pitch_abs >= _SUSPEND_PITCH_DEG:
        reasons.append(f"pitch={pitch_abs:.1f}")
    if h_used is not None and float(h_used) <= _SUSPEND_HEIGHT_CM:
        reasons.append(f"height={float(h_used):.1f}cm")
    if (
        h_used is not None
        and last_good_height_cm is not None
        and math.isfinite(float(h_used))
        and math.isfinite(float(last_good_height_cm))
        and abs(float(h_used) - float(last_good_height_cm)) >= _SUSPEND_HEIGHT_DELTA_CM
    ):
        reasons.append(f"height_delta={abs(float(h_used) - float(last_good_height_cm)):.1f}cm")
    return (len(reasons) > 0), ",".join(reasons) if reasons else "ok"


def maybe_suspend_mapping(camera_diag: dict | None, last_good_height_cm: float | None,
                           current_countdown: int, client_collision: dict | None) -> int:
    """Return updated suspend countdown."""
    suspected, _reason = check_fall_suspected(camera_diag, last_good_height_cm)
    if suspected:
        return max(current_countdown, _SUSPEND_STEPS_ON_FALL)
    return current_countdown


def consume_mapping_suspend(countdown: int) -> tuple[bool, int]:
    """Returns (should_skip, new_countdown)."""
    if countdown <= 0:
        return False, 0
    return True, max(0, countdown - 1)


# ── Health check flags ───────────────────────────────────────────────

def compute_health_flags(
    *,
    traversible_free: float,
    depth_zero_frac: float,
    depth_median: float,
    camera_diag: dict | None,
) -> list[str]:
    flags = []
    if math.isfinite(traversible_free) and traversible_free >= 0.95:
        flags.append("TRAVERSIBLE_TOO_FREE")
    if math.isfinite(depth_zero_frac) and depth_zero_frac >= 0.25:
        flags.append("DEPTH_TOO_MANY_ZEROS")
    if math.isfinite(depth_median) and (depth_median < 0.08 or depth_median > 8.0):
        flags.append("DEPTH_MEDIAN_OUT_OF_RANGE")
    if isinstance(camera_diag, dict) and camera_diag.get("active", False):
        pitch_abs = abs(float(camera_diag.get("pitch_deg_used", 0.0)))
        h_used = camera_diag.get("height_cm_used")
        if pitch_abs >= 35.0:
            flags.append("CAMERA_PITCH_LARGE")
        if h_used is not None and (float(h_used) < 70.0 or float(h_used) > 210.0):
            flags.append("CAMERA_HEIGHT_OUTLIER")
        roll_abs = abs(float(camera_diag.get("roll_deg", 0.0)))
        if h_used is not None and float(h_used) < 90.0 and (roll_abs > 60.0 or pitch_abs > 45.0):
            flags.append("FALL_SUSPECTED")
    return flags
