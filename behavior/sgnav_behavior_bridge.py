"""
Observation, Habitat-style gps/compass, and discrete base commands for SG-Nav + BEHAVIOR.

Aligned with UniGoal ``src/envs/behavior_omnigibson_env.py`` (RGB/depth keys, flatten obs,
``_habitat_style_gps_compass``). SG-Nav expects depth in **meters** (Habitat-like), not
UniGoal's internal [0,1] normalization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch as th

from omnigibson.utils import transform_utils as T


def tensor_to_numpy(x: np.ndarray | th.Tensor) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _slice_spec_to_indices(spec) -> list[int]:
    if isinstance(spec, slice):
        start = 0 if spec.start is None else int(spec.start)
        stop = start if spec.stop is None else int(spec.stop)
        step = 1 if spec.step is None else int(spec.step)
        return list(range(start, stop, step))
    arr = tensor_to_numpy(spec).reshape(-1)
    if arr.size == 0:
        return []
    return [int(v) for v in arr.tolist()]


def summarize_action_by_slices(
    action: np.ndarray | th.Tensor,
    slice_map: dict,
    *,
    reference_action: np.ndarray | th.Tensor | None = None,
    atol: float = 1e-6,
) -> dict:
    """
    Summarize action values grouped by controller / slice for debug logging.

    This is a pure helper used to compare the actually-sent action with a reference no-op action
    without changing the action-generation logic.
    """
    sent = tensor_to_numpy(action).astype(np.float64, copy=False).reshape(-1)
    if reference_action is None:
        ref = np.zeros_like(sent)
    else:
        ref = tensor_to_numpy(reference_action).astype(np.float64, copy=False).reshape(-1)
        if ref.shape != sent.shape:
            raise ValueError(
                f"reference_action shape {ref.shape} does not match action shape {sent.shape}"
            )

    out = {}
    for name, spec in slice_map.items():
        idx = _slice_spec_to_indices(spec)
        vals = sent[idx] if idx else np.zeros(0, dtype=np.float64)
        ref_vals = ref[idx] if idx else np.zeros(0, dtype=np.float64)
        delta_vals = vals - ref_vals
        out[str(name)] = {
            "indices": [int(i) for i in idx],
            "sent_norm": float(np.linalg.norm(vals)) if vals.size else 0.0,
            "noop_norm": float(np.linalg.norm(ref_vals)) if ref_vals.size else 0.0,
            "sent_minus_noop_norm": float(np.linalg.norm(delta_vals)) if delta_vals.size else 0.0,
            "nonzero_indices": [int(i) for i, v in zip(idx, vals) if abs(float(v)) > float(atol)],
            "values": [float(v) for v in vals.tolist()],
            "noop_values": [float(v) for v in ref_vals.tolist()],
            "delta_values": [float(v) for v in delta_vals.tolist()],
        }
    return out


def _init_action_tensor(action_dim: int, seed_action: np.ndarray | th.Tensor | None = None) -> th.Tensor:
    if seed_action is None:
        return th.zeros(action_dim, dtype=th.float32)
    action = th.as_tensor(seed_action, dtype=th.float32).reshape(-1).clone()
    if int(action.numel()) != int(action_dim):
        raise ValueError(
            f"seed_action dim {int(action.numel())} does not match requested action_dim {int(action_dim)}"
        )
    return action


def _hold_trunk_current_qpos(
    action: th.Tensor,
    *,
    torso_slice: Optional[slice],
    robot,
) -> tuple[th.Tensor, th.Tensor | None]:
    """
    Explicitly pin the trunk controller command to the robot's current trunk qpos.

    Relying on ``seed_action`` alone is brittle when upstream controller ordering / no-op semantics change.
    For non-look SG-Nav actions we always want the torso / head pitch joints to hold their current positions.
    """
    if (
        torso_slice is None
        or robot is None
        or not hasattr(robot, "trunk_control_idx")
        or torso_slice.stop - torso_slice.start != len(robot.trunk_control_idx)
    ):
        return action, None
    qpos = robot.get_joint_positions()
    tidx = robot.trunk_control_idx
    for k in range(tidx.shape[0]):
        qi = int(tidx[k].item())
        action[torso_slice.start + k] = qpos[qi].to(dtype=th.float32)
    return action, qpos


def rgb_to_hwc_uint8(arr: np.ndarray | th.Tensor) -> np.ndarray:
    """Normalize OmniGibson vision tensors to H×W×3 uint8 (UniGoal ``_rgb_to_hwc_uint8``)."""
    x = tensor_to_numpy(arr)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        h, w, c = x.shape[0], x.shape[1], x.shape[2]
        if c not in (1, 3, 4) and h in (1, 3, 4) and h < min(w, c):
            x = np.transpose(x, (1, 2, 0))
            h, w, c = x.shape[0], x.shape[1], x.shape[2]
    else:
        raise ValueError(f"rgb must be 3D or 4D, got shape {x.shape}")

    if x.shape[-1] > 3:
        x = x[..., :3]
    elif x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    if x.dtype == np.uint8:
        return x

    xf = x.astype(np.float32)
    mx = float(np.nanmax(xf)) if xf.size else 0.0
    if mx <= 1.01:
        xf = np.clip(xf * 255.0, 0.0, 255.0)
    else:
        xf = np.clip(xf, 0.0, 255.0)
    return xf.astype(np.uint8)


def depth_to_hw1_meters(
    arr: np.ndarray | th.Tensor,
    *,
    dmin: float = 0.0,
    dmax: float = 10.0,
) -> np.ndarray:
    """``depth_linear`` / depth map → H×W×1 float32 meters (UniGoal depth scaling, no 0–1 norm)."""
    d = tensor_to_numpy(arr).astype(np.float32)
    if d.ndim == 3:
        if d.shape[0] == 1 and d.shape[1] > 1 and d.shape[2] > 1:
            d = d[0]
        elif d.shape[-1] == 1:
            d = d[..., 0]
        else:
            d = np.squeeze(d)
    if d.ndim != 2:
        raise ValueError(f"depth must be 2D after cleanup, got {d.shape}")

    d = np.nan_to_num(d, nan=0.0, posinf=dmax, neginf=0.0)
    finite = d[np.isfinite(d)]
    p95 = float(np.percentile(finite, 95)) if finite.size > 0 else 0.0
    if p95 > dmax * 20.0:
        d = d / 1000.0
    elif p95 > dmax * 2.0:
        d = d / 100.0
    d = np.clip(d, dmin, dmax)
    return d[..., None]


def extract_omnigibson_occupancy_grid(flat: dict) -> Optional[np.ndarray]:
    """
    Read OmniGibson ``ScanSensor`` occupancy from flattened obs (``*::occupancy_grid``).

    Values match ``OccupancyGridState`` after sensor scaling (~0 obstacle, 0.5 unknown, 1 free).
    Returns H×W×1 float32 numpy, or None if no modality present.
    """
    keys = [k for k in flat if isinstance(k, str) and k.endswith("::occupancy_grid")]
    if not keys:
        return None
    k = sorted(keys)[0]
    occ = tensor_to_numpy(flat[k])
    if occ.ndim == 4:
        occ = occ[0]
    if occ.ndim == 2:
        occ = occ[..., np.newaxis]
    return occ.astype(np.float32, copy=False)


def resolve_rgb_depth_keys(flat: dict, head_key: str) -> Tuple[str, str, str]:
    """UniGoal ``_resolve_rgb_depth_keys``."""
    rgb_exact = head_key + "::rgb"
    if rgb_exact in flat:
        rgb_k = rgb_exact
    else:
        cands = [k for k in flat if isinstance(k, str) and k.endswith("::rgb")]
        zed = [k for k in cands if "zed_link" in k and "Camera" in k]
        pick = zed if zed else cands
        if not pick:
            raise KeyError(
                f"No ::rgb in obs (expected {rgb_exact}). First keys: {list(flat.keys())[:32]}"
            )
        rgb_k = sorted(pick, key=len)[-1]

    prefix = rgb_k[: -len("::rgb")]
    lin_k = prefix + "::depth_linear"
    rad_k = prefix + "::depth"
    if lin_k in flat:
        return rgb_k, lin_k, "depth_linear"
    if rad_k in flat:
        return rgb_k, rad_k, "depth"
    raise KeyError(
        f"No depth for sensor prefix {prefix!r}; have linear={lin_k in flat} radial={rad_k in flat}. "
        f"Keys sample: {[k for k in flat if 'depth' in k][:12]}"
    )


def robot_sensor_dict_key_from_eval_head_key(head_key: str) -> str:
    """UniGoal ``_robot_sensor_dict_key_from_eval_head_key`` (for ``robot.sensors[...]``)."""
    parts = head_key.split("::", 1)
    if len(parts) != 2:
        return head_key
    return parts[1]


def habitat_gps_compass(robot, map_size_cm: float) -> Tuple[np.ndarray, np.ndarray]:
    """UniGoal ``_habitat_style_gps_compass`` using ``z_angle_from_quat``."""
    pos, quat = robot.get_position_orientation()
    p = pos.detach().cpu().numpy().reshape(3)
    x, y = float(p[0]), float(p[1])
    q = quat.flatten()[:4]
    o = float(T.z_angle_from_quat(q).item())
    c = float(map_size_cm) / 100.0 / 2.0
    gps = np.asarray([x - c, c - y], dtype=np.float64)
    compass = np.asarray([o + np.pi / 2], dtype=np.float64)
    return gps, compass


def flatten_obs_maybe(env_obs: dict, flatten_fn) -> dict:
    return flatten_fn(env_obs)


def build_sgnav_step_tensors(
    obs: dict,
    robot,
    *,
    head_key: str,
    map_size_cm: float,
    flatten_obs_dict,
    obs_width: int = 640,
    obs_height: int = 480,
    max_depth_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
 Returns rgb uint8 HW3, depth float32 HW1 meters, gps (2,), compass scalar (first element).
 """
    flat = flatten_obs_maybe(obs, flatten_obs_dict)
    rgb_k, depth_k, _mod = resolve_rgb_depth_keys(flat, head_key)
    rgb = rgb_to_hwc_uint8(flat[rgb_k])
    dep = depth_to_hw1_meters(flat[depth_k], dmin=0.0, dmax=max_depth_m)
    if rgb.shape[0] != dep.shape[0] or rgb.shape[1] != dep.shape[1]:
        dep2 = cv2.resize(
            dep[..., 0],
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )[..., None].astype(np.float32)
        dep = np.clip(dep2, 0.0, max_depth_m)
    if rgb.shape[1] != obs_width or rgb.shape[0] != obs_height:
        rgb = cv2.resize(rgb, (obs_width, obs_height), interpolation=cv2.INTER_AREA)
        dep = cv2.resize(dep[..., 0], (obs_width, obs_height), interpolation=cv2.INTER_NEAREST)[
            ..., None
        ].astype(np.float32)
    gps, compass = habitat_gps_compass(robot, map_size_cm)
    return rgb, dep.astype(np.float32), gps, float(compass.reshape(-1)[0])


def discrete_sgnav_to_velocity_action(
    cmd: int,
    action_dim: int,
    base_slice: slice,
    *,
    forward_vel: float = 1.25,
    turn_vel: float = 1.75,
    max_linear_m_s: float = 1.5,
    max_angular_rad_s: float = math.pi,
    torso_slice: Optional[slice] = None,
    robot=None,
    look_pitch_step_rad: float = math.radians(30.0),
    look_pitch_sign: float = -1.0,
    look_pitch_limit_margin_rad: float = 0.0,
    seed_action: np.ndarray | th.Tensor | None = None,
) -> th.Tensor:
    """
    Map SG-Nav / Habitat discrete cmd to R1Pro base slice (UniGoal ``_build_action_tensor``).

    OmniGibson ``HolonomicBaseJointController`` uses **normalized** commands in [-1, 1] (default
    ``command_input_limits``), then scales to joint velocity limits — for R1 holonomic base that is
    about ±1.5 m/s linear and ±π rad/s angular (see ``omnigibson/robots/holonomic_base_robot.py``).
    Passing raw ``turn_vel=5`` or ``25`` used to clip to 1.0, identical to any value ≥1, so changing
    yaml ``behavior_turn_vel`` had no effect on per-step rotation.

    Here ``forward_vel`` / ``turn_vel`` are interpreted as **physical** m/s and rad/s, divided by the
    caps above (then clipped to [-1, 1]) before writing the action tensor.

    0 stop, 1 forward, 2 turn left (+angular), 3 / 6 turn right (-angular),
    4 LOOK_UP / 5 LOOK_DOWN: base stop; optional torso pitch on the head-tilt trunk joint.
    """
    action = _init_action_tensor(action_dim, seed_action=seed_action)
    action, trunk_qpos = _hold_trunk_current_qpos(action, torso_slice=torso_slice, robot=robot)
    base = base_slice
    max_lin = float(max_linear_m_s) if max_linear_m_s > 1e-6 else 1.5
    max_ang = float(max_angular_rad_s) if max_angular_rad_s > 1e-6 else math.pi
    norm_forward = max(-1.0, min(1.0, float(forward_vel) / max_lin))
    norm_turn = max(-1.0, min(1.0, float(turn_vel) / max_ang))
    if cmd == 0:
        pass
    elif cmd == 1:
        action[base.start + 0] = norm_forward
    elif cmd == 2:
        action[base.start + 2] = norm_turn
    elif cmd in (3, 6):
        action[base.start + 2] = -norm_turn
    elif cmd in (4, 5):
        if (
            torso_slice is not None
            and robot is not None
            and hasattr(robot, "trunk_control_idx")
            and torso_slice.stop - torso_slice.start == len(robot.trunk_control_idx)
        ):
            qpos = trunk_qpos if trunk_qpos is not None else robot.get_joint_positions()
            tidx = robot.trunk_control_idx
            pitch_k = int(tidx.shape[0]) - 1
            delta = float(look_pitch_sign) * float(
                look_pitch_step_rad if cmd == 4 else -look_pitch_step_rad
            )
            qi_pitch = int(tidx[pitch_k].item())
            target = qpos[qi_pitch].to(dtype=th.float32) + float(delta)
            lower = getattr(robot, "joint_lower_limits", None)
            upper = getattr(robot, "joint_upper_limits", None)
            if lower is not None and upper is not None:
                lower_t = th.as_tensor(lower, dtype=th.float32).reshape(-1)
                upper_t = th.as_tensor(upper, dtype=th.float32).reshape(-1)
                if qi_pitch < int(lower_t.numel()) and qi_pitch < int(upper_t.numel()):
                    margin = max(0.0, float(look_pitch_limit_margin_rad))
                    lo = float(lower_t[qi_pitch].item()) + margin
                    hi = float(upper_t[qi_pitch].item()) - margin
                    if lo <= hi:
                        target = th.clamp(target, min=lo, max=hi)
            action[torso_slice.start + pitch_k] = target
    return action


def discrete_sgnav_to_position_action(
    cmd: int,
    action_dim: int,
    base_slice: slice,
    *,
    forward_step_m: float = 0.05,
    turn_angle_rad: float = math.radians(10.0),
    torso_slice: Optional[slice] = None,
    robot=None,
    look_pitch_step_rad: float = math.radians(30.0),
    look_pitch_sign: float = -1.0,
    look_pitch_limit_margin_rad: float = 0.0,
    seed_action: np.ndarray | th.Tensor | None = None,
) -> th.Tensor:
    """
    Map SG-Nav / Habitat discrete cmd to absolute local-frame [dx, dy, drz] base commands.

    This matches ``HolonomicBaseJointController`` with ``motor_type="position"`` and
    ``command_input_limits=None``, where base commands are interpreted as local-frame deltas
    in meters / radians for a single env.step.
    """
    action = _init_action_tensor(action_dim, seed_action=seed_action)
    action, trunk_qpos = _hold_trunk_current_qpos(action, torso_slice=torso_slice, robot=robot)
    base = base_slice
    if cmd == 0:
        pass
    elif cmd == 1:
        action[base.start + 0] = float(forward_step_m)
    elif cmd == 2:
        action[base.start + 2] = float(turn_angle_rad)
    elif cmd in (3, 6):
        action[base.start + 2] = -float(turn_angle_rad)
    elif cmd in (4, 5):
        if (
            torso_slice is not None
            and robot is not None
            and hasattr(robot, "trunk_control_idx")
            and torso_slice.stop - torso_slice.start == len(robot.trunk_control_idx)
        ):
            qpos = trunk_qpos if trunk_qpos is not None else robot.get_joint_positions()
            tidx = robot.trunk_control_idx
            pitch_k = int(tidx.shape[0]) - 1
            delta = float(look_pitch_sign) * float(
                look_pitch_step_rad if cmd == 4 else -look_pitch_step_rad
            )
            qi_pitch = int(tidx[pitch_k].item())
            target = qpos[qi_pitch].to(dtype=th.float32) + float(delta)
            lower = getattr(robot, "joint_lower_limits", None)
            upper = getattr(robot, "joint_upper_limits", None)
            if lower is not None and upper is not None:
                lower_t = th.as_tensor(lower, dtype=th.float32).reshape(-1)
                upper_t = th.as_tensor(upper, dtype=th.float32).reshape(-1)
                if qi_pitch < int(lower_t.numel()) and qi_pitch < int(upper_t.numel()):
                    margin = max(0.0, float(look_pitch_limit_margin_rad))
                    lo = float(lower_t[qi_pitch].item()) + margin
                    hi = float(upper_t[qi_pitch].item()) - margin
                    if lo <= hi:
                        target = th.clamp(target, min=lo, max=hi)
            action[torso_slice.start + pitch_k] = target
    return action


def discrete_sgnav_to_base_action(
    cmd: int,
    action_dim: int,
    base_slice: slice,
    *,
    base_action_mode: str = "velocity",
    forward_vel: float = 0.35,
    turn_vel: float = 0.45,
    max_linear_m_s: float = 1.5,
    max_angular_rad_s: float = math.pi,
    forward_step_m: float = 0.05,
    turn_angle_rad: float = math.radians(10.0),
    torso_slice: Optional[slice] = None,
    robot=None,
    look_pitch_step_rad: float = math.radians(30.0),
    look_pitch_sign: float = -1.0,
    look_pitch_limit_margin_rad: float = 0.0,
    seed_action: np.ndarray | th.Tensor | None = None,
) -> th.Tensor:
    mode = str(base_action_mode).strip().lower()
    if mode == "velocity":
        return discrete_sgnav_to_velocity_action(
            cmd,
            action_dim,
            base_slice,
            forward_vel=forward_vel,
            turn_vel=turn_vel,
            max_linear_m_s=max_linear_m_s,
            max_angular_rad_s=max_angular_rad_s,
            torso_slice=torso_slice,
            robot=robot,
            look_pitch_step_rad=look_pitch_step_rad,
            look_pitch_sign=look_pitch_sign,
            look_pitch_limit_margin_rad=look_pitch_limit_margin_rad,
            seed_action=seed_action,
        )
    if mode == "position":
        return discrete_sgnav_to_position_action(
            cmd,
            action_dim,
            base_slice,
            forward_step_m=forward_step_m,
            turn_angle_rad=turn_angle_rad,
            torso_slice=torso_slice,
            robot=robot,
            look_pitch_step_rad=look_pitch_step_rad,
            look_pitch_sign=look_pitch_sign,
            look_pitch_limit_margin_rad=look_pitch_limit_margin_rad,
            seed_action=seed_action,
        )
    raise ValueError(f"Unsupported base_action_mode {base_action_mode!r}; expected 'velocity' or 'position'")
