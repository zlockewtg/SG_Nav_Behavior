"""
Build Habitat-shaped observation dicts for SG_Nav_Agent.act() and map discrete
action indices (ObjectNav v1) to OmniGibson robot base motion.

Indices match HabitatSimActions / configs/challenge_objectnav2021.local.rgbd.yaml:
  0 STOP, 1 MOVE_FORWARD, 2 TURN_LEFT, 3 TURN_RIGHT,
  4 LOOK_UP, 5 LOOK_DOWN, 6 TURN_RIGHT_2 (same turn as 3 in v1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# Action indices (Habitat ObjectNav v1 + challenge YAML)
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
LOOK_UP = 4
LOOK_DOWN = 5
TURN_RIGHT_2 = 6


@dataclass
class HabitatShapedObsConfig:
    """Depth handling aligned with Habitat challenge / SG_Nav.act."""

    min_depth_m: float = 0.5
    max_depth_m: float = 5.0


def compass_from_base_yaw(yaw_rad: float) -> np.ndarray:
    """Single-element compass in radians (same layout as Habitat GPS/compass sensors)."""
    return np.array([yaw_rad], dtype=np.float64)


def build_habitat_shaped_observations(
    rgb: np.ndarray,
    depth: np.ndarray,
    gps_xy: np.ndarray,
    compass_rad: float,
    *,
    cfg: Optional[HabitatShapedObsConfig] = None,
) -> dict:
    """
    Args:
        rgb: HxWx3 uint8 or float in [0,255], RGB order (SG_Nav swaps to BGR for GLIP internally).
        depth: HxW or HxWx1 float32 meters.
        gps_xy: shape (2,) episode-frame horizontal coordinates (meters), consistent across steps.
        compass_rad: robot yaw in radians (must match how gps moves when the robot turns).
    """
    cfg = cfg or HabitatShapedObsConfig()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if depth.ndim == 2:
        depth = depth[..., np.newaxis]
    depth = depth.astype(np.float32, copy=False)
    depth = np.clip(depth, 0.0, cfg.max_depth_m)
    obs = {
        "rgb": rgb,
        "depth": depth,
        "gps": np.asarray(gps_xy, dtype=np.float64).reshape(2),
        "compass": compass_from_base_yaw(compass_rad),
    }
    return obs


def apply_sgnav_discrete_action(
    robot,
    action_idx: int,
    *,
    forward_m: float = 0.25,
    turn_deg: float = 30.0,
    look_up_fn: Optional[Callable[[float], None]] = None,
    look_down_fn: Optional[Callable[[float], None]] = None,
) -> None:
    """
    Apply one SG-Nav discrete action on an OmniGibson locomotion robot.

    Uses kinematic helpers (move_forward / turn_*) from omnigibson.robots.robot.Robot.
    For physics stepping, call env.sim.step() or env.step(...) after this as your app requires.

    LOOK_UP / LOOK_DOWN are optional: provide look_up_fn / look_down_fn(degrees) to tilt
    a camera or head joint; otherwise they are no-ops (map quality may degrade for
    SG-Nav's initial panorama sequence).
    """
    if not getattr(robot, "is_locomotion", False):
        raise RuntimeError("Robot must be a locomotion robot for base navigation actions.")

    turn = math.radians(turn_deg)

    if action_idx == STOP:
        return
    if action_idx == MOVE_FORWARD:
        robot.move_forward(delta=forward_m)
        return
    if action_idx == TURN_LEFT:
        robot.turn_left(delta=turn)
        return
    if action_idx in (TURN_RIGHT, TURN_RIGHT_2):
        robot.turn_right(delta=turn)
        return
    if action_idx == LOOK_UP:
        if look_up_fn is not None:
            look_up_fn(turn_deg)
        return
    if action_idx == LOOK_DOWN:
        if look_down_fn is not None:
            look_down_fn(turn_deg)
        return
    raise ValueError(f"Unknown SG-Nav action index: {action_idx}")


def base_xy_yaw_from_robot(robot) -> Tuple[np.ndarray, float]:
    """
    Convenience: horizontal position (x, y) and yaw from OmniGibson robot root pose.

    Assumes z is up; yaw is rotation about +z from the robot's orientation quaternion.
    """
    import torch as th

    pos, quat = robot.get_position_orientation()
    pos = pos.detach().cpu().numpy().flatten()
    gps_xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)

    q = quat.detach().cpu().numpy().flatten()
    # qw, qx, qy, qz
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return gps_xy, yaw
