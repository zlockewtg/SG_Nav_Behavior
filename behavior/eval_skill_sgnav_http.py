#!/usr/bin/env python3
"""
Evaluate move-to style navigation using SG-Nav running in a **separate** process (HTTP API).

Prerequisite: start SG-Nav server in the SG_Nav conda env (from SG-Nav repo root):

  conda activate SG_Nav
  python behavior/sgnav_http_server.py --host 127.0.0.1 --port 8765

Then run this script in the **OmniGibson / openpi-comet** env:

  cd /path/to/openpi-comet
  python scripts/eval_skill_sgnav_http.py --sgnav_url http://127.0.0.1:8765 \\
    --task turning_on_radio --object_id_contains radio --total_segments 1

Environment construction and **observations / gps / compass** follow UniGoal
``UniGoal/src/envs/behavior_omnigibson_env.py`` (``flatten_obs_space``, ``depth_linear``,
``_habitat_style_gps_compass``, ``_resolve_rgb_depth_keys``, sensor 640×480).

**Actions**: discrete SG-Nav indices are mapped to holonomic base commands and
``env.step({ROBOT_NAME: action})`` like UniGoal ``_build_action_tensor``. The default
``--base_action_mode velocity`` preserves the old velocity-control path; ``position`` switches the
base controller to absolute local-frame ``[dx, dy, drz]`` commands where the configured
position-step size is applied per substep and amplified by repeating multiple env.steps. LOOK_UP/DOWN
(4/5) additionally step **torso_joint4** so head camera pitch tracks SG-Nav’s panorama
(see ``--look_pitch_step_deg``).

**Navigation vs BDDL**: By default (``--ignore-bddl-termination``) the segment does **not** end when
the Behavior activity goal is satisfied (e.g. radio already ``toggled_on``). The episode stops on
``--distance_threshold`` (early stop), simulator **timeout** (``truncated``), ``--max_steps``, or
SG-Nav emitting ``STOP`` while target-navigation is active.
Success is **min arm–object distance** under the threshold, not task predicate success.

Use ``--nav_debug_path path.jsonl`` to log RGB/depth stats, GPS, compass, ``gps_delta_m``,
``map_size_cm``, ``compass_deg``, ``world_xy_m`` (OmniGibson world frame), and the SG-Nav action.
Each ``/step`` may send ``target_world_xy`` when the task object is resolved (optional FMM goal if
``gt_pathplan_when_target_on_map`` is enabled in SG-Nav yaml).
``camera_pose_world`` is built from the head-camera parent link world pose plus the fixed local
camera extrinsics used by OmniGibson teleop for R1Pro; camera parameters are still resolved
directly from the exact head VisionSensor in ``robot.sensors`` and the script raises if that
sensor cannot be found (disable with ``--no-send-camera-pose-world``).
Check Y-axis: with ``habitat_gps_compass``, ``gps[1] = c - world_y`` so ``Δgps[1]/Δworld_y ≈ -1``.
If your data shows the same sign for both deltas, set ``SGNAV_RUNTIME.gps_negate_y: true`` on the server.
See ``SGNAV_RUNTIME.log_pose_align_trace`` on the HTTP server for planner heading vs STG.
Post-run: ``python scripts/analyze_sgnav_nav_debug.py /path/to/nav_debug.jsonl``.
each step. Use ``--use_og_occupancy`` to stream OmniGibson ``ScanSensor`` occupancy into SG-Nav (robot must
expose a range-sensor prim). See ``docs/BEHAVIOR_SGNAV_HTTP.md`` for full startup flags.

**YAML defaults**: If sibling ``SG-Nav/configs/sgnav_minimal.rgbd.yaml`` exists (or pass ``--sgnav_config``),
the script loads ``SIMULATOR.RGB_SENSOR`` / ``DEPTH_SENSOR`` / ``TILT_ANGLE`` and optional
``SGNAV_RUNTIME.behavior_*`` keys for camera resolution, depth clip, look pitch, and base velocities.
Explicit CLI flags override yaml.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import torch as th


def _configure_live_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


_configure_live_stdio()

warnings.filterwarnings("ignore", message="Casting input x to numpy array", category=UserWarning, module="gymnasium")

project_root = Path(__file__).resolve().parents[1]
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
og_path = project_root / "BEHAVIOR-1K" / "OmniGibson"
if og_path.exists():
    sys.path.insert(0, str(og_path))
sys.path.insert(0, str(project_root / "BEHAVIOR-1K"))
sys.path.insert(0, str(project_root / "BEHAVIOR-1K" / "joylo"))
sys.path.insert(0, str(project_root / "BEHAVIOR-1K" / "bddl"))

from gello.robots.sim_robot.og_teleop_cfg import (
    DISABLED_TRANSITION_RULES,
    HEAD_CAMERA_LINK_NAME,
    R1PRO_HEAD_CAMERA_LOCAL_ORI,
    R1PRO_HEAD_CAMERA_LOCAL_POS,
    ROBOT_NAME,
)
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
import omnigibson as og
from omnigibson.controllers import ControllerView
from omnigibson.learning.utils.eval_utils import (
    ACTION_QPOS_INDICES,
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.macros import gm
gm.DEFAULT_SIM_STEP_FREQ = 10
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils import transform_utils as T
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.python_utils import recursively_convert_to_torch
from hydra.utils import instantiate

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
gm.NO_OMNI_LOGS = True
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

POST_RESET_PHYSICS_STEPS = 100
MOVE_TO_SUCCESS_DISTANCE_THRESHOLD = 1.2
# Default cap for early-stop once within distance_threshold (must be < typical --max_steps).
MOVE_TO_MIN_STEPS_BEFORE_REACH_EARLY_STOP = 100
DEFAULT_MAX_STEPS = 3000
SGNAV_STOP_ACTION = 0


def _load_eval_skill_flat():
    p = Path(__file__).resolve().parent / "eval_skill_flat.py"
    spec = importlib.util.spec_from_file_location("eval_skill_flat", p)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(m)
    return m


ef = _load_eval_skill_flat()

from sgnav_behavior_bridge import (
    build_sgnav_step_tensors,
    discrete_sgnav_to_base_action,
    extract_omnigibson_occupancy_grid,
    summarize_action_by_slices,
)

VALID_BASE_ACTION_MODES = ("velocity", "position")


def _normalize_base_action_mode(value: str | None) -> str:
    mode = "velocity" if value is None else str(value).strip().lower()
    if mode not in VALID_BASE_ACTION_MODES:
        raise ValueError(
            f"Unsupported base_action_mode {value!r}; expected one of {VALID_BASE_ACTION_MODES}"
        )
    return mode


def _pose_dict_from_position_orientation(pos, quat) -> dict | None:
    try:
        p = pos.detach().cpu().numpy().reshape(-1)
        q = quat.detach().cpu().numpy().reshape(-1)
    except Exception:
        p = np.asarray(pos).reshape(-1)
        q = np.asarray(quat).reshape(-1)
    if p.size < 3 or q.size < 4:
        raise ValueError(
            f"Invalid camera pose shape: position has {p.size} values, quaternion has {q.size} values"
        )
    return {
        "position": [float(p[0]), float(p[1]), float(p[2])],
        "quaternion": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


def _head_sensor_key_from_eval_head_key(head_key: str) -> str:
    return head_key.split("::", 1)[1] if "::" in head_key else head_key


def resolve_head_sensor_or_raise(robot, head_key: str):
    """
    Resolve the exact head VisionSensor used by eval observations.

    Accepts only the canonical eval key or its robot.sensors-local form. Any other fallback
    is treated as a configuration error because SG-Nav mapping must use the real head camera.
    """
    sensors = getattr(robot, "sensors", None)
    if not sensors:
        raise RuntimeError("robot.sensors is empty; cannot resolve the head camera sensor")

    sensor_key = _head_sensor_key_from_eval_head_key(head_key)
    if sensor_key in sensors:
        return sensor_key, sensors[sensor_key]
    if head_key in sensors:
        return head_key, sensors[head_key]

    raise KeyError(
        "Head camera sensor not found in robot.sensors. "
        f"Expected one of {sensor_key!r} or {head_key!r}, available keys={sorted(str(k) for k in sensors.keys())}"
    )


def _quat_xyzw_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_xyzw_angle_deg(q_from: list[float], q_to: list[float]) -> float:
    q1 = np.asarray(q_from, dtype=np.float64)
    q2 = np.asarray(q_to, dtype=np.float64)
    if q1.shape[0] < 4 or q2.shape[0] < 4:
        return float("nan")
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    q1 = q1 / n1
    q2 = q2 / n2
    dq = _quat_xyzw_multiply(q2, _quat_xyzw_conjugate(q1))
    w = float(np.clip(abs(dq[3]), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def collect_head_sensor_pose_world(robot, head_key: str) -> tuple[dict, str]:
    """
    World-frame pose from the actual VisionSensor prim that provides depth/rgb.
    Returns (pose_dict, resolved_sensor_key).
    """
    resolved_key, sensor = resolve_head_sensor_or_raise(robot, head_key)
    pos, quat = sensor.get_position_orientation()
    return _pose_dict_from_position_orientation(pos, quat), resolved_key


def collect_head_camera_pose_world_from_link(
    robot,
    configured_link_name: str | None = None,
) -> tuple[dict, str]:
    """
    Build head camera world pose from the parent link world pose and the fixed local camera
    extrinsics configured in ``gello.robots.sim_robot.og_teleop_cfg`` / ``og_teleop_utils``.

    The composition must be ``link_world @ camera_local``.
    """
    robot_type = robot.__class__.__name__
    if robot_type not in HEAD_CAMERA_LINK_NAME:
        raise KeyError(
            f"No head camera link mapping for robot type {robot_type!r}; "
            f"known types={sorted(HEAD_CAMERA_LINK_NAME.keys())}"
        )
    if robot_type != "R1Pro":
        raise NotImplementedError(
            f"Head camera link+extrinsic pose composition is only implemented for R1Pro, got {robot_type!r}"
        )

    link_name = HEAD_CAMERA_LINK_NAME[robot_type]
    if configured_link_name is not None and str(configured_link_name) != str(link_name):
        raise ValueError(
            f"Configured head link {configured_link_name!r} does not match "
            f"{robot_type} head camera link mapping {link_name!r}"
        )
    link = robot.links[link_name]
    link_pos, link_quat = link.get_position_orientation()
    cam_local_pos = R1PRO_HEAD_CAMERA_LOCAL_POS
    cam_local_quat = R1PRO_HEAD_CAMERA_LOCAL_ORI

    if hasattr(link_pos, "device"):
        cam_local_pos = cam_local_pos.to(device=link_pos.device, dtype=link_pos.dtype)
    if hasattr(link_quat, "device"):
        cam_local_quat = cam_local_quat.to(device=link_quat.device, dtype=link_quat.dtype)
    if not hasattr(link_pos, "device"):
        link_pos = th.as_tensor(link_pos, dtype=th.float32)
    if not hasattr(link_quat, "device"):
        link_quat = th.as_tensor(link_quat, dtype=th.float32)

    cam_world_pos, cam_world_quat = T.pose_transform(link_pos, link_quat, cam_local_pos, cam_local_quat)
    return _pose_dict_from_position_orientation(cam_world_pos, cam_world_quat), link_name


def _safe_contact_body_name(body) -> str:
    try:
        name = str(body)
    except Exception:
        return ""
    return name if name and name != "None" else ""


def _collect_base_collision_info(robot) -> dict:
    """
    Collect current contact information for non-floor-touching base links.

    This approximates "the robot base hit scene/object geometry" while ignoring
    the expected floor contact from locomotion.
    """
    info = {
        "base_contact": False,
        "n_contacts": 0,
        "contact_bodies": [],
        "max_impulse": 0.0,
    }
    links = list(getattr(robot, "non_floor_touching_base_links", []) or [])
    if not links:
        return info

    robot_prim_path = str(getattr(robot, "prim_path", "") or "")
    bodies = []
    n_contacts = 0
    max_impulse = 0.0
    for link in links:
        try:
            contacts = list(link.contact_list())
        except Exception:
            continue
        link_prim_path = str(getattr(link, "prim_path", "") or "")
        for contact in contacts:
            body0 = _safe_contact_body_name(getattr(contact, "body0", ""))
            body1 = _safe_contact_body_name(getattr(contact, "body1", ""))
            other_bodies = []
            for body in (body0, body1):
                if not body:
                    continue
                if link_prim_path and body == link_prim_path:
                    continue
                if robot_prim_path and body.startswith(robot_prim_path):
                    continue
                other_bodies.append(body)
            if not other_bodies:
                continue
            n_contacts += 1
            bodies.extend(other_bodies)
            impulse = getattr(contact, "impulse", None)
            try:
                imp = np.asarray(impulse, dtype=np.float64).reshape(-1)
                if imp.size:
                    max_impulse = max(max_impulse, float(np.linalg.norm(imp)))
            except Exception:
                pass

    uniq_bodies = []
    seen = set()
    for body in bodies:
        if body in seen:
            continue
        seen.add(body)
        uniq_bodies.append(body)

    info["base_contact"] = bool(n_contacts > 0)
    info["n_contacts"] = int(n_contacts)
    info["contact_bodies"] = [str(body) for body in uniq_bodies[:8]]
    info["max_impulse"] = float(max_impulse)
    return info


def _default_sgnav_yaml_path() -> str | None:
    """Sibling SG-Nav repo: …/openpi-comet/../SG-Nav/configs/sgnav_minimal.rgbd.yaml"""
    p = Path(__file__).resolve().parents[2] / "SG-Nav" / "configs" / "sgnav_minimal.rgbd.yaml"
    return str(p) if p.is_file() else None


def eval_defaults_from_sgnav_yaml(path: str | Path) -> dict:
    """
    Read OmniGibson / HTTP-client defaults from SG-Nav OmegaConf yaml.

    SIMULATOR.RGB_SENSOR / DEPTH_SENSOR / TILT_ANGLE: camera & look pitch.
    SIMULATOR.FORWARD_STEP_SIZE / TURN_ANGLE: discrete navigation step targets for motion scaling.
    SGNAV_RUNTIME.behavior_* (optional): base-action mode, velocities, map_size_cm, VisionSensor
    horizontal_aperture, and optional branch-specific motion overrides.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise RuntimeError("omegaconf is required to load --sgnav_config") from e

    cfg = OmegaConf.load(str(path))
    out: dict = {}
    sim = cfg.get("SIMULATOR") or {}
    rgb = sim.get("RGB_SENSOR") or {}
    depth = sim.get("DEPTH_SENSOR") or {}
    if rgb.get("WIDTH") is not None:
        out["obs_width"] = int(rgb.WIDTH)
    if rgb.get("HEIGHT") is not None:
        out["obs_height"] = int(rgb.HEIGHT)
    if depth.get("MAX_DEPTH") is not None:
        out["max_depth_m"] = float(depth.MAX_DEPTH)
    if rgb.get("HFOV") is not None:
        out["vision_hfov_deg"] = float(rgb.HFOV)
    if sim.get("TILT_ANGLE") is not None:
        out["look_pitch_step_deg"] = float(sim.TILT_ANGLE)
    if sim.get("FORWARD_STEP_SIZE") is not None:
        out["forward_step_size_m"] = float(sim.FORWARD_STEP_SIZE)
    if sim.get("TURN_ANGLE") is not None:
        out["turn_angle_deg"] = float(sim.TURN_ANGLE)

    rt = cfg.get("SGNAV_RUNTIME") or {}
    if rt.get("behavior_base_action_mode") is not None:
        out["base_action_mode"] = _normalize_base_action_mode(rt.behavior_base_action_mode)
    if rt.get("behavior_forward_vel") is not None:
        out["forward_vel"] = float(rt.behavior_forward_vel)
    if rt.get("behavior_turn_vel") is not None:
        out["turn_vel"] = float(rt.behavior_turn_vel)
    if rt.get("behavior_velocity_forward_vel") is not None:
        out["velocity_forward_vel"] = float(rt.behavior_velocity_forward_vel)
    if rt.get("behavior_velocity_turn_vel") is not None:
        out["velocity_turn_vel"] = float(rt.behavior_velocity_turn_vel)
    if rt.get("behavior_position_forward_step_size_m") is not None:
        out["position_forward_step_size_m"] = float(rt.behavior_position_forward_step_size_m)
    if rt.get("behavior_position_turn_angle_deg") is not None:
        out["position_turn_angle_deg"] = float(rt.behavior_position_turn_angle_deg)
    if rt.get("behavior_position_physical_look_pitch_step_deg") is not None:
        out["position_physical_look_pitch_step_deg"] = float(rt.behavior_position_physical_look_pitch_step_deg)
    if rt.get("behavior_physics_steps_per_nav_step") is not None:
        out["physics_steps_per_nav_step"] = int(rt.behavior_physics_steps_per_nav_step)
    if rt.get("behavior_position_physics_steps_per_nav_step") is not None:
        out["position_physics_steps_per_nav_step"] = int(
            rt.behavior_position_physics_steps_per_nav_step
        )
    if rt.get("behavior_position_forward_physics_steps_per_nav_step") is not None:
        out["position_forward_physics_steps_per_nav_step"] = int(
            rt.behavior_position_forward_physics_steps_per_nav_step
        )
    if rt.get("behavior_position_turn_physics_steps_per_nav_step") is not None:
        out["position_turn_physics_steps_per_nav_step"] = int(
            rt.behavior_position_turn_physics_steps_per_nav_step
        )
    if rt.get("behavior_position_forward_guard_enable") is not None:
        out["position_forward_guard_enable"] = bool(rt.behavior_position_forward_guard_enable)
    if rt.get("behavior_position_forward_guard_step_delta_m") is not None:
        out["position_forward_guard_step_delta_m"] = float(
            rt.behavior_position_forward_guard_step_delta_m
        )
    if rt.get("behavior_position_forward_guard_roll_pitch_deg") is not None:
        out["position_forward_guard_roll_pitch_deg"] = float(
            rt.behavior_position_forward_guard_roll_pitch_deg
        )
    if rt.get("behavior_position_forward_guard_consecutive_contacts") is not None:
        out["position_forward_guard_consecutive_contacts"] = int(
            rt.behavior_position_forward_guard_consecutive_contacts
        )
    if rt.get("behavior_position_forward_guard_heading_error_deg") is not None:
        out["position_forward_guard_heading_error_deg"] = float(
            rt.behavior_position_forward_guard_heading_error_deg
        )
    if rt.get("map_size_cm") is not None:
        out["map_size_cm"] = float(rt.map_size_cm)
    if rt.get("behavior_vision_horizontal_aperture") is not None:
        out["vision_horizontal_aperture"] = float(rt.behavior_vision_horizontal_aperture)
    if rt.get("behavior_vision_focal_length_mm") is not None:
        out["vision_focal_length_mm"] = float(rt.behavior_vision_focal_length_mm)
    if rt.get("behavior_physical_look_pitch_step_deg") is not None:
        out["physical_look_pitch_step_deg"] = float(rt.behavior_physical_look_pitch_step_deg)

    return out


def _merge_eval_yaml_and_args(yaml_defaults: dict, args: argparse.Namespace) -> dict:
    """CLI wins when the argument was explicitly passed (not None)."""

    def pick(key: str, arg_value, hard_default):
        if arg_value is not None:
            return arg_value
        return yaml_defaults.get(key, hard_default)

    base_action_mode = _normalize_base_action_mode(
        pick("base_action_mode", args.base_action_mode, "velocity")
    )

    if args.forward_vel is not None:
        forward_vel = args.forward_vel
    elif base_action_mode == "velocity":
        forward_vel = yaml_defaults.get("velocity_forward_vel", yaml_defaults.get("forward_vel", 0.75))
    else:
        forward_vel = yaml_defaults.get("forward_vel", 0.75)

    if args.turn_vel is not None:
        turn_vel = args.turn_vel
    elif base_action_mode == "velocity":
        turn_vel = yaml_defaults.get("velocity_turn_vel", yaml_defaults.get("turn_vel", 1.25))
    else:
        turn_vel = yaml_defaults.get("turn_vel", 1.25)

    if args.forward_step_size_m is not None:
        forward_step_size_m = args.forward_step_size_m
    elif base_action_mode == "position":
        forward_step_size_m = yaml_defaults.get(
            "position_forward_step_size_m",
            yaml_defaults.get("forward_step_size_m", 0.1),
        )
    else:
        forward_step_size_m = yaml_defaults.get("forward_step_size_m", 0.1)

    if args.turn_angle_deg is not None:
        turn_angle_deg = args.turn_angle_deg
    elif base_action_mode == "position":
        turn_angle_deg = yaml_defaults.get(
            "position_turn_angle_deg",
            yaml_defaults.get("turn_angle_deg", 30.0),
        )
    else:
        turn_angle_deg = yaml_defaults.get("turn_angle_deg", 30.0)

    if args.physical_look_pitch_step_deg is not None:
        physical_look_pitch_step_deg = args.physical_look_pitch_step_deg
    elif base_action_mode == "position":
        physical_look_pitch_step_deg = yaml_defaults.get(
            "position_physical_look_pitch_step_deg",
            yaml_defaults.get("physical_look_pitch_step_deg"),
        )
    else:
        physical_look_pitch_step_deg = None

    if args.physics_steps_per_nav_step is not None:
        physics_steps_per_nav_step = args.physics_steps_per_nav_step
    elif base_action_mode == "position":
        physics_steps_per_nav_step = yaml_defaults.get(
            "position_physics_steps_per_nav_step",
            yaml_defaults.get("physics_steps_per_nav_step", 15),
        )
    else:
        physics_steps_per_nav_step = yaml_defaults.get("physics_steps_per_nav_step", 15)

    if args.physics_steps_per_nav_step is not None:
        position_forward_physics_steps_per_nav_step = args.physics_steps_per_nav_step
        position_turn_physics_steps_per_nav_step = args.physics_steps_per_nav_step
    elif base_action_mode == "position":
        position_repeat_default = yaml_defaults.get(
            "position_physics_steps_per_nav_step",
            yaml_defaults.get("physics_steps_per_nav_step", 15),
        )
        position_forward_physics_steps_per_nav_step = yaml_defaults.get(
            "position_forward_physics_steps_per_nav_step",
            position_repeat_default,
        )
        position_turn_physics_steps_per_nav_step = yaml_defaults.get(
            "position_turn_physics_steps_per_nav_step",
            1,
        )
    else:
        position_forward_physics_steps_per_nav_step = physics_steps_per_nav_step
        position_turn_physics_steps_per_nav_step = physics_steps_per_nav_step

    if base_action_mode == "position":
        position_forward_guard_enable = bool(
            yaml_defaults.get("position_forward_guard_enable", True)
        )
        position_forward_guard_step_delta_m = float(
            yaml_defaults.get("position_forward_guard_step_delta_m", 0.15)
        )
        position_forward_guard_roll_pitch_deg = float(
            yaml_defaults.get("position_forward_guard_roll_pitch_deg", 10.0)
        )
        position_forward_guard_consecutive_contacts = int(
            yaml_defaults.get("position_forward_guard_consecutive_contacts", 2)
        )
        position_forward_guard_heading_error_deg = float(
            yaml_defaults.get("position_forward_guard_heading_error_deg", 45.0)
        )
    else:
        position_forward_guard_enable = False
        position_forward_guard_step_delta_m = 0.15
        position_forward_guard_roll_pitch_deg = 10.0
        position_forward_guard_consecutive_contacts = 2
        position_forward_guard_heading_error_deg = 45.0

    return {
        "base_action_mode": base_action_mode,
        "obs_width": pick("obs_width", args.obs_width, 640),
        "obs_height": pick("obs_height", args.obs_height, 480),
        "max_depth_m": pick("max_depth_m", args.max_depth_m, 10.0),
        "map_size_cm": pick("map_size_cm", args.map_size_cm, 4000.0),
        "forward_vel": forward_vel,
        "turn_vel": turn_vel,
        "forward_step_size_m": forward_step_size_m,
        "turn_angle_deg": turn_angle_deg,
        "look_pitch_step_deg": pick("look_pitch_step_deg", args.look_pitch_step_deg, 30.0),
        "physical_look_pitch_step_deg": physical_look_pitch_step_deg,
        "physics_steps_per_nav_step": int(physics_steps_per_nav_step),
        "position_forward_physics_steps_per_nav_step": int(
            position_forward_physics_steps_per_nav_step
        ),
        "position_turn_physics_steps_per_nav_step": int(
            position_turn_physics_steps_per_nav_step
        ),
        "position_forward_guard_enable": bool(position_forward_guard_enable),
        "position_forward_guard_step_delta_m": float(position_forward_guard_step_delta_m),
        "position_forward_guard_roll_pitch_deg": float(position_forward_guard_roll_pitch_deg),
        "position_forward_guard_consecutive_contacts": int(
            position_forward_guard_consecutive_contacts
        ),
        "position_forward_guard_heading_error_deg": float(
            position_forward_guard_heading_error_deg
        ),
        "vision_horizontal_aperture": (
            args.vision_horizontal_aperture
            if args.vision_horizontal_aperture is not None
            else yaml_defaults.get("vision_horizontal_aperture")
        ),
        "vision_hfov_deg": (
            args.vision_hfov_deg
            if args.vision_hfov_deg is not None
            else yaml_defaults.get("vision_hfov_deg")
        ),
        "vision_focal_length_mm": pick(
            "vision_focal_length_mm",
            args.vision_focal_length_mm,
            17.0,
        ),
    }


def _arr_blob(a: np.ndarray) -> dict:
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    # Use C-contiguous payload
    a = np.ascontiguousarray(a)
    return {
        "dtype": np.dtype(a.dtype).name,
        "shape": list(a.shape),
        "data": base64.standard_b64encode(a.tobytes()).decode("ascii"),
    }


def _normalize_goal_phrase(text: str) -> str:
    """Convert object ids / labels into compact SG-Nav hint phrases."""
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\.[nvars]\.\d+\b", " ", s)  # e.g. receiver.n.01
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_goal_sg_hint(object_id: str, display_name: str | None) -> str | None:
    phrases: list[str] = []
    for raw in (display_name, object_id):
        p = _normalize_goal_phrase(raw or "")
        if p and p not in phrases:
            phrases.append(p)
    if not phrases:
        return None
    return ". ".join(phrases) + "."


def _resolve_segment_instance_id(seg: dict, override_instance_id: int | None = None) -> int:
    if override_instance_id is not None:
        return int(override_instance_id)
    raw = seg.get("instance_id", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid segment instance_id=%r; fallback to 0 for segment=%s", raw, seg)
        return 0


def _md5_file(path: str | os.PathLike) -> str | None:
    try:
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except OSError:
        return None


def _collect_scene_asset_hash_mismatches(scene) -> list[dict]:
    mismatches: list[dict] = []
    for obj in getattr(scene, "objects", []):
        expected_hash = getattr(obj, "_expected_file_hash", None)
        usd_path = getattr(obj, "usd_path", None)
        if not expected_hash or not usd_path:
            continue
        current_hash = _md5_file(usd_path)
        if current_hash is None or current_hash == expected_hash:
            continue
        mismatches.append(
            {
                "name": str(getattr(obj, "name", "")),
                "category": str(getattr(obj, "category", "")),
                "model": str(getattr(obj, "model", getattr(obj, "_model", ""))),
                "expected_hash": str(expected_hash),
                "current_hash": str(current_hash),
                "usd_path": str(usd_path),
            }
        )
    return mismatches


def _resolve_tro_export_path(base_path: str, task_name: str, instance_id: int) -> str:
    out = os.path.abspath(base_path)
    if os.path.isdir(out):
        safe_task = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(task_name).strip()) or "task"
        return os.path.join(out, f"{safe_task}_instance_{int(instance_id)}_tro_state.json")
    return out


class SGNavHTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e

    def health(self) -> dict:
        url = f"{self.base}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def reset(self, *, behavior_goal: str | None = None, object_category: str | None = None, object_category_sg=None):
        pl: dict = {}
        if behavior_goal:
            pl["behavior_goal"] = behavior_goal
        if object_category:
            pl["object_category"] = object_category
        if object_category_sg is not None:
            pl["object_category_sg"] = object_category_sg
        return self._post("/reset", pl)

    def step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        gps: np.ndarray,
        compass_rad: float,
        *,
        og_occupancy: np.ndarray | None = None,
        og_occupancy_meta: dict | None = None,
        target_world_xy: list[float] | tuple[float, float] | None = None,
        camera_pose_world: dict | None = None,
        collision_info: dict | None = None,
    ) -> int:
        if depth.ndim == 2:
            depth_payload = depth.astype(np.float32)
        else:
            depth_payload = depth[..., 0].astype(np.float32)
        payload = {
            "rgb": _arr_blob(rgb.astype(np.uint8)),
            "depth": _arr_blob(depth_payload.astype(np.float32)),
            "gps": [float(gps[0]), float(gps[1])],
            "compass": [float(compass_rad)],
        }
        if og_occupancy is not None:
            occ = np.ascontiguousarray(og_occupancy.astype(np.float32))
            payload["og_occupancy"] = _arr_blob(occ)
            if og_occupancy_meta:
                payload["og_occupancy_meta"] = {
                    k: (float(v) if k == "range_m" else int(v))
                    for k, v in og_occupancy_meta.items()
                    if k in ("resolution", "range_m")
                }
        if target_world_xy is not None and len(target_world_xy) >= 2:
            payload["target_world_xy"] = [float(target_world_xy[0]), float(target_world_xy[1])]
        if camera_pose_world is not None:
            payload["camera_pose_world"] = camera_pose_world
        if collision_info is not None:
            payload["collision_info"] = {
                "base_contact": bool(collision_info.get("base_contact", False)),
                "n_contacts": int(collision_info.get("n_contacts", 0)),
                "contact_bodies": [str(x) for x in list(collision_info.get("contact_bodies", []))[:8]],
                "max_impulse": float(collision_info.get("max_impulse", 0.0)),
                "prev_action": (
                    None if collision_info.get("prev_action") is None else int(collision_info.get("prev_action"))
                ),
            }
        out = self._post("/step", payload)
        return int(out["action"])


class SGNavHTTPRunner:
    """OmniGibson side: same env wiring as SkillEvaluator in eval_skill_flat.py."""

    def __init__(
        self,
        sgnav_url: str,
        env_wrapper_cfg=None,
        *,
        online_sampling: bool = False,
        partial_scene_load: bool = False,
        rft_style_tro: bool = False,
        use_annotation_object_lookup: bool = False,
        http_timeout_s: float = 600.0,
        obs_width: int = 640,
        obs_height: int = 480,
        map_size_cm: float = 4000.0,
        forward_vel: float = 0.75,
        turn_vel: float = 1.25,
        max_depth_m: float = 10.0,
        nav_debug_path: str | None = None,
        look_pitch_step_rad: float = math.radians(30.0),
        physical_look_pitch_step_rad: float | None = None,
        ignore_bddl_termination: bool = True,
        use_og_occupancy: bool = False,
        og_occupancy_resolution: int = 256,
        og_occupancy_range_m: float = 5,
        vision_horizontal_aperture: float | None = None,
        vision_hfov_deg: float | None = None,
        vision_focal_length_mm: float = 17.0,
        physics_steps_per_nav_step: int = 1,
        position_forward_physics_steps_per_nav_step: int | None = None,
        position_turn_physics_steps_per_nav_step: int | None = None,
        position_forward_guard_enable: bool = True,
        position_forward_guard_step_delta_m: float = 0.15,
        position_forward_guard_roll_pitch_deg: float = 10.0,
        position_forward_guard_consecutive_contacts: int = 2,
        position_forward_guard_heading_error_deg: float = 45.0,
        send_camera_pose_world: bool = True,
        head_link_name: str = "zed_link",
        camera_pose_source: str = "sensor_world",
        base_action_mode: str = "velocity",
        forward_step_size_m: float = 0.1,
        turn_angle_deg: float = 30.0,
        debug_action_semantics: bool = False,
        instance_id_override: int | None = None,
        fail_on_asset_hash_mismatch: bool = False,
        tro_state_path: str | None = None,
        save_tro_state_to: str | None = None,
        settle_after_nav_action: bool = True,
        settle_max_steps: int = 12,
        settle_consecutive_static_steps: int = 2,
        settle_linear_vel_m_s: float = 0.02,
        settle_angular_vel_rad_s: float = 0.08,
        settle_joint_vel_max: float = 0.05,
        settle_contact_extra_steps: int = 2,
    ):
        self.client = SGNavHTTPClient(sgnav_url, timeout_s=http_timeout_s)
        self.env_wrapper_cfg = env_wrapper_cfg
        self.env = None
        self.robot = None
        self.online_sampling = online_sampling
        self.partial_scene_load = partial_scene_load
        self.rft_style_tro = rft_style_tro
        self.use_annotation_object_lookup = use_annotation_object_lookup
        self._cached_tro_state = None
        self.target = None
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.map_size_cm = map_size_cm
        self.forward_vel = forward_vel
        self.turn_vel = turn_vel
        self.max_depth_m = max_depth_m
        self._nav_debug_path = nav_debug_path
        self._look_pitch_step_rad = look_pitch_step_rad
        if physical_look_pitch_step_rad is None:
            physical_look_pitch_step_rad = min(float(look_pitch_step_rad), math.radians(8.0))
        self._physical_look_pitch_step_rad = max(0.0, float(physical_look_pitch_step_rad))
        self._ignore_bddl_termination = ignore_bddl_termination
        self._head_key = ROBOT_CAMERA_NAMES["R1Pro"]["head"]
        self.use_og_occupancy = bool(use_og_occupancy)
        self._og_occupancy_resolution = int(og_occupancy_resolution)
        self._og_occupancy_range_m = float(og_occupancy_range_m)
        self._og_occupancy_meta = {
            "resolution": self._og_occupancy_resolution,
            "range_m": self._og_occupancy_range_m,
        }
        self._nav_prev_gps: np.ndarray | None = None
        self._nav_prev_compass: float | None = None
        self._nav_prev_action: int | None = None
        self._nav_prev_collision_info: dict | None = None
        self._vision_horizontal_aperture = vision_horizontal_aperture
        self._vision_hfov_deg = vision_hfov_deg
        self._vision_focal_length_mm = float(vision_focal_length_mm)
        self._last_vs_kwargs: dict = {}
        self._physics_steps_per_nav_step = max(1, int(physics_steps_per_nav_step))
        if position_forward_physics_steps_per_nav_step is None:
            position_forward_physics_steps_per_nav_step = self._physics_steps_per_nav_step
        if position_turn_physics_steps_per_nav_step is None:
            position_turn_physics_steps_per_nav_step = self._physics_steps_per_nav_step
        self._position_forward_physics_steps_per_nav_step = max(
            1, int(position_forward_physics_steps_per_nav_step)
        )
        self._position_turn_physics_steps_per_nav_step = max(
            1, int(position_turn_physics_steps_per_nav_step)
        )
        self._position_forward_guard_enable = bool(position_forward_guard_enable)
        self._position_forward_guard_step_delta_m = max(
            1e-4, float(position_forward_guard_step_delta_m)
        )
        self._position_forward_guard_roll_pitch_deg = max(
            0.0, float(position_forward_guard_roll_pitch_deg)
        )
        self._position_forward_guard_consecutive_contacts = max(
            1, int(position_forward_guard_consecutive_contacts)
        )
        self._position_forward_guard_heading_error_deg = max(
            0.0, float(position_forward_guard_heading_error_deg)
        )
        self._send_camera_pose_world = bool(send_camera_pose_world)
        self._head_link_name = str(head_link_name)
        self._camera_pose_source_requested = str(camera_pose_source).strip().lower()
        self._base_action_mode = _normalize_base_action_mode(base_action_mode)
        if self._camera_pose_source_requested not in ("sensor_world", "head_link_extrinsic"):
            raise ValueError(
                "--camera_pose_source must be one of "
                f"('sensor_world', 'head_link_extrinsic'), got {self._camera_pose_source_requested!r}"
            )
        self._camera_pose_source = self._camera_pose_source_requested
        self._camera_pose_align_logged = False
        self._last_camera_pose_used: str | None = None
        self._forward_step_size_m = max(1e-4, float(forward_step_size_m))
        self._turn_angle_deg = max(1e-3, float(turn_angle_deg))
        self._debug_action_semantics = bool(debug_action_semantics)
        self._instance_id_override = (
            None if instance_id_override is None else int(instance_id_override)
        )
        self._fail_on_asset_hash_mismatch = bool(fail_on_asset_hash_mismatch)
        self._last_asset_hash_mismatches: list[dict] = []
        self._tro_state_path = None if tro_state_path is None else os.path.abspath(str(tro_state_path))
        self._save_tro_state_to = (
            None if save_tro_state_to is None else os.path.abspath(str(save_tro_state_to))
        )
        self._current_tro_source_path: str | None = None
        self._current_tro_export_path: str | None = None
        self._action_semantics_snapshot_logged = False
        self._settle_after_nav_action = bool(settle_after_nav_action)
        self._settle_max_steps = max(0, int(settle_max_steps))
        self._settle_consecutive_static_steps = max(1, int(settle_consecutive_static_steps))
        self._settle_linear_vel_m_s = max(0.0, float(settle_linear_vel_m_s))
        self._settle_angular_vel_rad_s = max(0.0, float(settle_angular_vel_rad_s))
        self._settle_joint_vel_max = max(0.0, float(settle_joint_vel_max))
        self._settle_contact_extra_steps = max(0, int(settle_contact_extra_steps))
        # Velocity caps used by discrete_sgnav_to_velocity_action() to convert physical m/s,rad/s to controller input.
        # Initialize with historical defaults; load_env() will overwrite from actual base controller limits.
        self._base_cmd_max_linear_m_s = 1.5
        self._base_cmd_max_angular_rad_s = float(math.pi)
        self._motion_scale_log_every = 20
        if self._base_action_mode == "velocity" and self._physics_steps_per_nav_step > 1:
            logger.info(
                "Velocity mode: turn/forward actions can auto-repeat env.step to match "
                "turn_angle_deg / forward_step_size_m given the active base velocity caps. "
                "Minimum configured repeat count=%d. LOOK_UP/DOWN still 1 step.",
                self._physics_steps_per_nav_step,
            )
        elif self._base_action_mode == "position":
            logger.info(
                "SG-Nav base_action_mode=position: configured absolute local-frame [dx, dy, drz] "
                "step sizes are applied per substep and can be amplified via repeated env.step calls. "
                "position repeats [forward=%d, turn=%d], forward_guard=[enable=%d step_delta=%.3f roll_pitch=%.1f contact_n=%d heading_err=%.1f].",
                self._position_forward_physics_steps_per_nav_step,
                self._position_turn_physics_steps_per_nav_step,
                int(self._position_forward_guard_enable),
                self._position_forward_guard_step_delta_m,
                self._position_forward_guard_roll_pitch_deg,
                self._position_forward_guard_consecutive_contacts,
                self._position_forward_guard_heading_error_deg,
            )
        logger.info(
            "SG-Nav look pitch: logical_step_deg=%.3f physical_trunk_step_deg=%.3f",
            math.degrees(float(self._look_pitch_step_rad)),
            math.degrees(float(self._physical_look_pitch_step_rad)),
        )
        logger.info(
            "SG-Nav action settle: enabled=%d max_steps=%d static_steps=%d "
            "lin<=%.3f ang<=%.3f joint<=%.3f extra_after_contact=%d",
            int(self._settle_after_nav_action),
            int(self._settle_max_steps),
            int(self._settle_consecutive_static_steps),
            float(self._settle_linear_vel_m_s),
            float(self._settle_angular_vel_rad_s),
            float(self._settle_joint_vel_max),
            int(self._settle_contact_extra_steps),
        )

    def _refresh_base_velocity_caps_from_controller(self) -> None:
        """
        Read base controller command_output_limits so bridge normalization matches the active robot config.
        """
        lin_cap = 1.5
        ang_cap = float(math.pi)
        base_ctrl, _, _ = self._get_controller_runtime("base")
        try:
            lim = getattr(base_ctrl, "command_output_limits", None)
            if lim is not None and len(lim) == 2:
                lo = np.asarray(lim[0], dtype=np.float64).reshape(-1)
                hi = np.asarray(lim[1], dtype=np.float64).reshape(-1)
                if lo.size >= 3 and hi.size >= 3:
                    lin_cap = max(abs(float(lo[0])), abs(float(hi[0])))
                    ang_cap = max(abs(float(lo[2])), abs(float(hi[2])))
        except Exception:
            pass
        self._base_cmd_max_linear_m_s = max(1e-6, float(lin_cap))
        self._base_cmd_max_angular_rad_s = max(1e-6, float(ang_cap))
        print(
            "[eval_skill_sgnav_http][base_caps] "
            f"mode={getattr(base_ctrl, 'motor_type', 'unknown')} "
            f"max_linear_m_s={self._base_cmd_max_linear_m_s:.3f} "
            f"max_angular_rad_s={self._base_cmd_max_angular_rad_s:.3f}",
            flush=True,
        )

    def _apply_base_controller_mode_to_robot_cfg(self, robot_cfg: dict) -> None:
        controller_cfg = robot_cfg.setdefault("controller_config", {})
        base_cfg = controller_cfg.setdefault("base", {})
        base_cfg["name"] = "HolonomicBaseJointController"
        base_cfg["motor_type"] = self._base_action_mode
        if self._base_action_mode == "position":
            # Pass through physical meters / radians directly, like OmniGibson primitive configs.
            base_cfg["command_input_limits"] = None
            base_cfg["command_output_limits"] = None
            base_cfg.pop("vel_kp", None)
        else:
            base_cfg.setdefault("command_input_limits", "default")

    @staticmethod
    def _indices_list(spec) -> list[int]:
        if isinstance(spec, slice):
            start = 0 if spec.start is None else int(spec.start)
            stop = start if spec.stop is None else int(spec.stop)
            step = 1 if spec.step is None else int(spec.step)
            return list(range(start, stop, step))
        arr = np.asarray(SGNavHTTPRunner._to_python_scalar_or_list(spec), dtype=np.int64).reshape(-1)
        return [int(v) for v in arr.tolist()]

    def _get_controller_runtime(self, name: str):
        binding = getattr(self.robot, "controllers", {}).get(name, None)
        if binding is None:
            return None, None, None
        if isinstance(binding, tuple) and len(binding) == 2:
            group_key, controller_idx = binding
            controller = getattr(ControllerView, "_controller_groups", {}).get(group_key, None)
            return controller, int(controller_idx), group_key
        return binding, None, None

    @staticmethod
    def _preview_preprocess_command(controller, command) -> list[float]:
        cmd = np.asarray(SGNavHTTPRunner._to_python_scalar_or_list(command), dtype=np.float64).reshape(-1)
        input_limits = getattr(controller, "command_input_limits", None)
        output_limits = getattr(controller, "command_output_limits", None)
        if input_limits is None:
            return [float(v) for v in cmd.tolist()]

        lo_in = np.asarray(SGNavHTTPRunner._to_python_scalar_or_list(input_limits[0]), dtype=np.float64).reshape(-1)
        hi_in = np.asarray(SGNavHTTPRunner._to_python_scalar_or_list(input_limits[1]), dtype=np.float64).reshape(-1)
        out = np.clip(cmd, lo_in, hi_in)
        if output_limits is not None:
            lo_out = np.asarray(
                SGNavHTTPRunner._to_python_scalar_or_list(output_limits[0]), dtype=np.float64
            ).reshape(-1)
            hi_out = np.asarray(
                SGNavHTTPRunner._to_python_scalar_or_list(output_limits[1]), dtype=np.float64
            ).reshape(-1)
            scale = np.abs(hi_out - lo_out) / np.abs(hi_in - lo_in)
            out = (out - (hi_in + lo_in) / 2.0) * scale + (hi_out + lo_out) / 2.0
        return [float(v) for v in out.tolist()]

    def _controller_slice_map(self) -> dict:
        return {str(name): idx for name, idx in self.robot.controller_action_idx.items()}

    @staticmethod
    def _wrap_angle_deg(angle_deg: float) -> float:
        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _quat_to_rpy_deg(quat_xyzw) -> list[float]:
        q = th.as_tensor(quat_xyzw, dtype=th.float32).reshape(-1)
        e = T.quat2euler(q).detach().cpu().numpy().reshape(-1)
        return [float(np.degrees(v)) for v in e[:3].tolist()]

    def _get_robot_pose_diag(self) -> dict:
        pos, quat = self.robot.get_position_orientation()
        p = np.asarray(self._to_python_scalar_or_list(pos), dtype=np.float64).reshape(-1)
        q = np.asarray(self._to_python_scalar_or_list(quat), dtype=np.float64).reshape(-1)
        rpy_deg = self._quat_to_rpy_deg(q)
        return {
            "pos_m": [float(v) for v in p[:3].tolist()],
            "quat_xyzw": [float(v) for v in q[:4].tolist()],
            "rpy_deg": [float(v) for v in rpy_deg],
        }

    @staticmethod
    def _merge_collision_infos(*infos: dict | None) -> dict:
        out = {
            "base_contact": False,
            "n_contacts": 0,
            "contact_bodies": [],
            "max_impulse": 0.0,
        }
        bodies = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            out["base_contact"] = bool(out["base_contact"] or info.get("base_contact", False))
            out["n_contacts"] = max(int(out["n_contacts"]), int(info.get("n_contacts", 0)))
            out["max_impulse"] = max(
                float(out["max_impulse"]), float(info.get("max_impulse", 0.0))
            )
            bodies.extend(str(x) for x in list(info.get("contact_bodies", []))[:8])
        uniq_bodies = []
        seen = set()
        for body in bodies:
            if body in seen:
                continue
            seen.add(body)
            uniq_bodies.append(body)
        out["contact_bodies"] = uniq_bodies[:8]
        return out

    def _get_robot_motion_diag(self) -> dict:
        lin = np.asarray(
            self._to_python_scalar_or_list(self.robot.get_linear_velocity()),
            dtype=np.float64,
        ).reshape(-1)
        ang = np.asarray(
            self._to_python_scalar_or_list(self.robot.get_angular_velocity()),
            dtype=np.float64,
        ).reshape(-1)
        joint_vel = np.asarray(
            self._to_python_scalar_or_list(self.robot.get_joint_velocities()),
            dtype=np.float64,
        ).reshape(-1)
        asleep_attr = getattr(self.robot, "is_asleep", None)
        try:
            asleep = bool(asleep_attr() if callable(asleep_attr) else asleep_attr)
        except Exception:
            asleep = False
        lin_norm = float(np.linalg.norm(lin)) if lin.size > 0 else 0.0
        ang_norm = float(np.linalg.norm(ang)) if ang.size > 0 else 0.0
        joint_vel_max = float(np.max(np.abs(joint_vel))) if joint_vel.size > 0 else 0.0
        static = bool(
            asleep
            or (
                lin_norm <= float(self._settle_linear_vel_m_s)
                and ang_norm <= float(self._settle_angular_vel_rad_s)
                and joint_vel_max <= float(self._settle_joint_vel_max)
            )
        )
        return {
            "lin_norm_m_s": lin_norm,
            "ang_norm_rad_s": ang_norm,
            "joint_vel_max": joint_vel_max,
            "is_asleep": bool(asleep),
            "is_static": bool(static),
        }

    def _settle_after_action(self, *, step: int, action_idx: int, had_contact: bool):
        if (
            not bool(self._settle_after_nav_action)
            or int(self._settle_max_steps) <= 0
            or int(action_idx) == SGNAV_STOP_ACTION
        ):
            return None
        noop = self._build_controller_noop_action()
        static_streak = 0
        min_steps = int(self._settle_contact_extra_steps) if had_contact else 0
        settle_samples = []
        last_obs = None
        last_step_info = {}
        terminated = False
        truncated = False
        agg_collision = None
        for settle_idx in range(int(self._settle_max_steps)):
            last_obs, _r, terminated, truncated, last_step_info = self.env.step(
                {ROBOT_NAME: noop}, n_render_iterations=1
            )
            motion = self._get_robot_motion_diag()
            collision = _collect_base_collision_info(self.robot)
            agg_collision = self._merge_collision_infos(agg_collision, collision)
            if motion["is_static"]:
                static_streak += 1
            else:
                static_streak = 0
            sample = {
                "settle_idx": int(settle_idx),
                "lin_norm_m_s": float(motion["lin_norm_m_s"]),
                "ang_norm_rad_s": float(motion["ang_norm_rad_s"]),
                "joint_vel_max": float(motion["joint_vel_max"]),
                "is_asleep": bool(motion["is_asleep"]),
                "is_static": bool(motion["is_static"]),
                "static_streak": int(static_streak),
                "base_contact": bool(collision.get("base_contact", False)),
                "n_contacts": int(collision.get("n_contacts", 0)),
                "max_impulse": float(collision.get("max_impulse", 0.0)),
            }
            settle_samples.append(sample)
            if (
                settle_idx + 1 >= min_steps
                and static_streak >= int(self._settle_consecutive_static_steps)
            ):
                break
            if terminated or truncated:
                break
        if settle_samples:
            print(
                "[eval_skill_sgnav_http][action_settle] "
                + json.dumps(
                    {
                        "step": int(step),
                        "action": int(action_idx),
                        "had_contact_before_settle": bool(had_contact),
                        "n_settle_steps": int(len(settle_samples)),
                        "max_settle_steps": int(self._settle_max_steps),
                        "static_required_steps": int(self._settle_consecutive_static_steps),
                        "final": settle_samples[-1],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return {
            "obs": last_obs,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_info": last_step_info,
            "samples": settle_samples,
            "collision_info": agg_collision,
        }

    def _build_position_controller_truth(self) -> dict:
        return {
            "source": [
                "omnigibson/controllers/holonomic_base_joint_controller.py",
                "omnigibson/robots/holonomic_base_robot.py",
            ],
            "summary": (
                "HolonomicBaseJointController(position) treats [x, y, rz] as absolute commands "
                "expressed in the robot local frame for each env.step update_goal call."
            ),
            "details": [
                "For position mode, local [x, y] is transformed into canonical/world-frame target joints each update.",
                "For position mode, rz is treated as a delta angle added to the current rz joint target.",
                "Repeatedly sending the same forward command does not divide by repeat count; it reissues the same local-frame request each env.step.",
                "HolonomicBaseRobot.q_to_action converts absolute canonical targets back into local-frame action commands.",
            ],
        }

    def _build_controller_noop_action(self) -> th.Tensor:
        parts = []
        control_dict = None
        for name in self.robot.controller_order:
            controller, controller_idx, _ = self._get_controller_runtime(name)
            if controller is None:
                continue
            if controller_idx is None:
                if control_dict is None:
                    control_dict = self.robot.get_control_dict()
                part = controller.compute_no_op_action(control_dict)
            else:
                part = controller.compute_no_op_action(controller_idx)
            parts.append(th.as_tensor(part).reshape(-1).to(dtype=th.float32))
        return th.cat(parts, dim=0) if parts else th.zeros(self.robot.action_dim, dtype=th.float32)

    def _build_noop_reference_action(self) -> tuple[th.Tensor, th.Tensor | None]:
        noop_action = self._build_controller_noop_action()
        all_absolute_joint = True
        for name in self.robot.controller_order:
            controller, _, _ = self._get_controller_runtime(name)
            if getattr(controller, "use_delta_commands", False):
                all_absolute_joint = False
        qpos_action = None
        if all_absolute_joint:
            try:
                qpos_action = self.robot.q_to_action(self.robot.get_joint_positions()).reshape(-1).to(dtype=th.float32)
            except Exception:
                qpos_action = None
        return noop_action, qpos_action

    def _build_controller_snapshot(self, base_slice: slice, torso_slice: slice) -> dict:
        snapshot = {
            "action_dim": int(self.robot.action_dim),
            "controller_order": [str(x) for x in self.robot.controller_order],
            "controller_action_idx": {
                str(k): self._indices_list(v) for k, v in self.robot.controller_action_idx.items()
            },
            "static_bridge_indices": {
                "base": self._indices_list(base_slice),
                "trunk": self._indices_list(torso_slice),
            },
            "controllers": {},
        }
        if self._base_action_mode == "position":
            snapshot["base_position_controller_truth"] = self._build_position_controller_truth()
        for name in self.robot.controller_order:
            controller, _, group_key = self._get_controller_runtime(name)
            snapshot["controllers"][str(name)] = {
                "type": str(type(controller).__name__) if controller is not None else None,
                "name": (
                    str(getattr(controller, "__class__", type(controller)).__name__)
                    if controller is not None
                    else str(ControllerView.get_controller_type_str(group_key)) if group_key is not None else None
                ),
                "motor_type": getattr(controller, "motor_type", None),
                "command_dim": int(getattr(controller, "command_dim", 0)) if controller is not None else 0,
                "use_delta_commands": bool(getattr(controller, "use_delta_commands", False)),
                "command_input_limits": self._to_python_scalar_or_list(
                    getattr(controller, "command_input_limits", None)
                ),
                "command_output_limits": self._to_python_scalar_or_list(
                    getattr(controller, "command_output_limits", None)
                ),
                "dof_idx": self._to_python_scalar_or_list(getattr(controller, "dof_idx", None)),
            }
        return snapshot

    def _log_controller_snapshot_once(self, base_slice: slice, torso_slice: slice) -> None:
        if not self._debug_action_semantics or self._action_semantics_snapshot_logged:
            return
        snapshot = self._build_controller_snapshot(base_slice, torso_slice)
        dyn = snapshot["controller_action_idx"]
        if dyn.get("base") != snapshot["static_bridge_indices"]["base"]:
            snapshot["warning_base_slice_mismatch"] = {
                "static": snapshot["static_bridge_indices"]["base"],
                "dynamic": dyn.get("base"),
            }
        if dyn.get("trunk") != snapshot["static_bridge_indices"]["trunk"]:
            snapshot["warning_trunk_slice_mismatch"] = {
                "static": snapshot["static_bridge_indices"]["trunk"],
                "dynamic": dyn.get("trunk"),
            }
        print(
            "[eval_skill_sgnav_http][action_semantics_snapshot] "
            + json.dumps(snapshot, ensure_ascii=False),
            flush=True,
        )
        self._action_semantics_snapshot_logged = True

    def _compute_action_semantics_diag(
        self,
        *,
        step: int,
        cmd: int,
        action,
        base_slice: slice,
        torso_slice: slice,
    ) -> dict:
        dynamic_slice_map = self._controller_slice_map()
        noop_action, qpos_action = self._build_noop_reference_action()
        diag = {
            "step": int(step),
            "cmd": int(cmd),
            "base_action_mode": str(self._base_action_mode),
            "controller_order": [str(x) for x in self.robot.controller_order],
            "static_bridge_indices": {
                "base": self._indices_list(base_slice),
                "trunk": self._indices_list(torso_slice),
            },
            "dynamic_indices": {
                "base": self._indices_list(self.robot.controller_action_idx.get("base", [])),
                "trunk": self._indices_list(self.robot.controller_action_idx.get("trunk", [])),
            },
            "by_controller": summarize_action_by_slices(
                action,
                dynamic_slice_map,
                reference_action=noop_action,
            ),
        }
        if diag["static_bridge_indices"]["base"] != diag["dynamic_indices"]["base"]:
            diag["warning_base_slice_mismatch"] = True
        if diag["static_bridge_indices"]["trunk"] != diag["dynamic_indices"]["trunk"]:
            diag["warning_trunk_slice_mismatch"] = True
        if qpos_action is not None:
            diag["qpos_reference_by_controller"] = summarize_action_by_slices(
                action,
                dynamic_slice_map,
                reference_action=qpos_action,
            )
        if self._base_action_mode == "position":
            base_controller, _, _ = self._get_controller_runtime("base")
            base_indices = self._indices_list(self.robot.controller_action_idx.get("base", []))
            action_np = np.asarray(self._to_python_scalar_or_list(action), dtype=np.float64).reshape(-1)
            base_raw = action_np[base_indices] if base_indices else np.zeros(0, dtype=np.float64)
            diag["base_position_debug"] = {
                "motor_type": getattr(base_controller, "motor_type", None),
                "command_input_limits": self._to_python_scalar_or_list(
                    getattr(base_controller, "command_input_limits", None)
                ) if base_controller is not None else None,
                "command_output_limits": self._to_python_scalar_or_list(
                    getattr(base_controller, "command_output_limits", None)
                ) if base_controller is not None else None,
                "forward_step_m": float(self._forward_step_size_m),
                "turn_angle_rad": float(np.deg2rad(self._turn_angle_deg)),
                "raw_base_command": [float(v) for v in base_raw.tolist()],
                "preprocessed_base_command": (
                    self._preview_preprocess_command(base_controller, base_raw)
                    if base_controller is not None
                    else None
                ),
                "expected_units": (
                    "Raw action should encode local-frame [dx, dy, drz] in meters/radians per env.step "
                    "before HolonomicBaseJointController preprocessing."
                ),
                "controller_semantics": self._build_position_controller_truth(),
            }
        return diag

    def _compute_position_forward_substep_diag(
        self,
        *,
        step: int,
        substep_idx: int,
        before_pose: dict,
        after_pose: dict,
        expected_forward_step_m: float,
        cumulative_planar_m: float,
        consecutive_contacts: int,
        collision_info: dict,
    ) -> dict:
        before_pos = np.asarray(before_pose["pos_m"], dtype=np.float64)
        after_pos = np.asarray(after_pose["pos_m"], dtype=np.float64)
        before_rpy = np.asarray(before_pose["rpy_deg"], dtype=np.float64)
        after_rpy = np.asarray(after_pose["rpy_deg"], dtype=np.float64)
        dx_world = float(after_pos[0] - before_pos[0])
        dy_world = float(after_pos[1] - before_pos[1])
        dz_world = float(after_pos[2] - before_pos[2])
        planar_delta_m = float(math.hypot(dx_world, dy_world))
        yaw_before_rad = math.radians(float(before_rpy[2]))
        local_forward_m = float(
            math.cos(yaw_before_rad) * dx_world + math.sin(yaw_before_rad) * dy_world
        )
        local_lateral_m = float(
            -math.sin(yaw_before_rad) * dx_world + math.cos(yaw_before_rad) * dy_world
        )
        motion_heading_deg = float(
            math.degrees(math.atan2(dy_world, dx_world))
        ) if planar_delta_m > 1e-9 else float(before_rpy[2])
        heading_error_deg = (
            abs(self._wrap_angle_deg(motion_heading_deg - float(before_rpy[2])))
            if planar_delta_m > 1e-9
            else 0.0
        )
        base_ctrl, base_ctrl_idx, _ = self._get_controller_runtime("base")
        base_goal = None
        if base_ctrl is not None and base_ctrl_idx is not None:
            try:
                base_goal = {
                    str(k): self._to_python_scalar_or_list(v) for k, v in base_ctrl.get_goal(base_ctrl_idx).items()
                }
            except Exception:
                base_goal = None
        elif base_ctrl is not None and getattr(base_ctrl, "goal", None) is not None:
            base_goal = self._to_python_scalar_or_list(base_ctrl.goal)
        return {
            "step": int(step),
            "substep_idx": int(substep_idx),
            "expected_forward_step_m": float(expected_forward_step_m),
            "actual_planar_delta_m": float(planar_delta_m),
            "actual_world_delta_m": [dx_world, dy_world, dz_world],
            "actual_local_delta_m": [local_forward_m, local_lateral_m, dz_world],
            "cumulative_planar_m": float(cumulative_planar_m),
            "motion_heading_deg": float(motion_heading_deg),
            "heading_before_deg": float(before_rpy[2]),
            "heading_after_deg": float(after_rpy[2]),
            "heading_error_deg": float(heading_error_deg),
            "before_rpy_deg": [float(v) for v in before_rpy.tolist()],
            "after_rpy_deg": [float(v) for v in after_rpy.tolist()],
            "roll_pitch_abs_max_deg": float(
                max(
                    abs(float(after_rpy[0])),
                    abs(float(after_rpy[1])),
                    abs(float(before_rpy[0])),
                    abs(float(before_rpy[1])),
                )
            ),
            "base_contact": bool(collision_info.get("base_contact", False)),
            "n_contacts": int(collision_info.get("n_contacts", 0)),
            "max_impulse": float(collision_info.get("max_impulse", 0.0)),
            "consecutive_contacts": int(consecutive_contacts),
            "base_goal": base_goal,
        }

    @staticmethod
    def _to_python_scalar_or_list(x):
        if x is None:
            return None
        try:
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return float(x.item())
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [SGNavHTTPRunner._to_python_scalar_or_list(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x

    @staticmethod
    def _derived_intrinsic_from_aperture(
        width: int | None,
        height: int | None,
        horizontal_aperture_mm: float | None,
        focal_length_mm: float | None,
    ):
        if (
            width is None
            or height is None
            or horizontal_aperture_mm is None
            or focal_length_mm is None
            or width <= 0
            or height <= 0
            or float(horizontal_aperture_mm) <= 1e-9
            or float(focal_length_mm) <= 1e-9
        ):
            return None
        fx = float(focal_length_mm) / float(horizontal_aperture_mm) * float(width)
        # Keep square-pixel assumption for debug print; vertical aperture is usually tied by aspect ratio.
        fy = fx
        cx = (float(width) - 1.0) / 2.0
        cy = (float(height) - 1.0) / 2.0
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    def _resolve_head_sensor(self):
        return resolve_head_sensor_or_raise(self.robot, self._head_key)

    def _log_og_runtime_camera_params(
        self,
        requested_vs_kwargs: dict,
        *,
        stage: str = "runtime",
        include_camera_params: bool = True,
    ) -> None:
        k, sensor = self._resolve_head_sensor()

        req_w = requested_vs_kwargs.get("image_width")
        req_h = requested_vs_kwargs.get("image_height")
        req_ha = requested_vs_kwargs.get("horizontal_aperture")
        req_f = requested_vs_kwargs.get("focal_length")
        act_w = int(sensor.image_width)
        act_h = int(sensor.image_height)
        act_ha = float(sensor.horizontal_aperture)
        act_f = float(sensor.focal_length)
        act_clip = self._to_python_scalar_or_list(sensor.clipping_range)

        req_hfov_deg = None
        if req_ha is not None and req_f is not None and float(req_f) > 1e-9:
            try:
                req_hfov_deg = float(np.degrees(2.0 * np.arctan(float(req_ha) / (2.0 * float(req_f)))))
            except Exception:
                req_hfov_deg = None
        hfov_deg = None
        if act_ha is not None and act_f is not None and act_f > 1e-9:
            try:
                hfov_deg = float(np.degrees(2.0 * np.arctan(float(act_ha) / (2.0 * float(act_f)))))
            except Exception:
                hfov_deg = None

        intrinsic = None
        cam_params = None
        if include_camera_params:
            intrinsic = self._to_python_scalar_or_list(sensor.intrinsic_matrix)
            cp = sensor.camera_parameters
            cam_params = {
                "renderProductResolution": self._to_python_scalar_or_list(cp.get("renderProductResolution")),
                "cameraNearFar": self._to_python_scalar_or_list(cp.get("cameraNearFar")),
                "cameraFocalLength": self._to_python_scalar_or_list(cp.get("cameraFocalLength")),
                "cameraAperture": self._to_python_scalar_or_list(cp.get("cameraAperture")),
            }

        modalities = sorted(list(getattr(sensor, "modalities", [])))

        print(
            f"[eval_skill_sgnav_http][og_cam_runtime][{stage}] "
            f"head_key={self._head_key} resolved_sensor_key={k} sensor_class={sensor.__class__.__name__} "
            f"modalities={modalities}",
            flush=True,
        )
        print(
            f"[eval_skill_sgnav_http][og_cam_runtime][{stage}] "
            f"requested: image_width={req_w} image_height={req_h} horizontal_aperture={req_ha} "
            f"focal_length={req_f} hfov_deg_from_request={req_hfov_deg}",
            flush=True,
        )
        print(
            f"[eval_skill_sgnav_http][og_cam_runtime][{stage}] "
            f"actual: image_width={act_w} image_height={act_h} horizontal_aperture={act_ha} "
            f"focal_length={act_f} clipping_range={act_clip} hfov_deg_from_aperture={hfov_deg}",
            flush=True,
        )
        if intrinsic is not None:
            print(
                f"[eval_skill_sgnav_http][og_cam_runtime][{stage}] "
                f"intrinsic_matrix={intrinsic}",
                flush=True,
            )
        derived_k = self._derived_intrinsic_from_aperture(act_w, act_h, act_ha, act_f)
        if derived_k is not None:
            print(
                f"[eval_skill_sgnav_http][og_cam_runtime][{stage}] "
                f"derived_intrinsic_from_aperture={derived_k}",
                flush=True,
            )
        if cam_params is not None:
            print(
                f"[eval_skill_sgnav_http][og_cam_runtime][{stage}] "
                f"camera_params={cam_params}",
                flush=True,
            )
            try:
                rp = cam_params.get("renderProductResolution")
                if isinstance(rp, list) and len(rp) >= 2 and int(rp[0]) == 0 and int(rp[1]) == 0:
                    print(
                        f"[eval_skill_sgnav_http][og_cam_runtime][{stage}][warn] "
                        "camera_params still uninitialized (renderProductResolution=[0,0]); "
                        "this stage may be too early for intrinsic diagnostics.",
                        flush=True,
                    )
            except Exception:
                pass

        def _warn_mismatch(name: str, req, act, tol: float = 1e-6):
            if req is None or act is None:
                return
            try:
                diff = abs(float(act) - float(req))
                if diff > tol:
                    print(
                        f"[eval_skill_sgnav_http][og_cam_runtime][{stage}][warn] "
                        f"mismatch field={name} requested={req} actual={act} abs_diff={diff}",
                        flush=True,
                    )
            except Exception:
                pass

        _warn_mismatch("image_width", req_w, act_w, tol=0.0)
        _warn_mismatch("image_height", req_h, act_h, tol=0.0)
        _warn_mismatch("horizontal_aperture", req_ha, act_ha, tol=1e-4)
        _warn_mismatch("focal_length", req_f, act_f, tol=1e-4)
        _warn_mismatch("hfov_deg", req_hfov_deg, hfov_deg, tol=0.2)

    def _collect_camera_pose_world_payload(self) -> dict | None:
        used = str(self._camera_pose_source)
        source_name = ""
        if used == "sensor_world":
            try:
                pose_world, sensor_key = collect_head_sensor_pose_world(self.robot, self._head_key)
                pose = dict(pose_world)
                source_name = str(sensor_key)
            except Exception as e:
                print(
                    "[eval_skill_sgnav_http][camera_pose_world][warn] "
                    f"sensor_world failed ({type(e).__name__}: {e}); "
                    "fallback=head_link_extrinsic",
                    flush=True,
                )
                pose_world, link_name = collect_head_camera_pose_world_from_link(
                    self.robot,
                    configured_link_name=self._head_link_name,
                )
                pose = dict(pose_world)
                used = "head_link_extrinsic"
                source_name = str(link_name)
        else:
            pose_world, link_name = collect_head_camera_pose_world_from_link(
                self.robot,
                configured_link_name=self._head_link_name,
            )
            pose = dict(pose_world)
            source_name = str(link_name)
        print(
            "[eval_skill_sgnav_http][camera_pose_world] "
            f"used={used} "
            f"source_name={source_name} "
            f"position={pose['position']} "
            f"orientation_xyzw={pose['quaternion']}",
            flush=True,
        )
        if not self._camera_pose_align_logged:
            self._camera_pose_align_logged = True
            print(
                "[eval_skill_sgnav_http][camera_pose_source] "
                f"requested={self._camera_pose_source_requested} "
                f"enforced={self._camera_pose_source} used={used} "
                f"source_name={source_name}",
                flush=True,
            )
        should_log_source = (
            used != self._camera_pose_source
            or used != self._last_camera_pose_used
            or self._camera_pose_source_requested != self._camera_pose_source
        )
        if should_log_source:
            print(
                "[eval_skill_sgnav_http][camera_pose_source_step] "
                f"requested={self._camera_pose_source_requested} "
                f"enforced={self._camera_pose_source} used={used} "
                f"source_name={source_name}",
                flush=True,
            )
        self._last_camera_pose_used = used
        pose["source"] = str(used)
        return pose

    def _reapply_tro_after_reset(self, tro_state: dict) -> None:
        for tro_key, tro_data in tro_state.items():
            if tro_key == "robot_poses":
                if self.robot.model_name in tro_data:
                    rp = tro_data[self.robot.model_name][0]
                    self.robot.set_position_orientation(rp["position"], rp["orientation"])
            elif tro_key in self.env.task.object_scope:
                self.env.task.object_scope[tro_key].load_state(tro_data, serialized=False)
        if "robot_poses" in tro_state:
            self.env.scene.write_task_metadata(key="robot_poses", data=tro_state["robot_poses"])

    def _build_current_tro_state_snapshot(self) -> dict:
        tro_state: dict = {}
        tro_state["robot_poses"] = {
            str(self.robot.model_name): [
                {
                    "position": list(self._to_python_scalar_or_list(self.robot.get_position_orientation()[0])),
                    "orientation": list(self._to_python_scalar_or_list(self.robot.get_position_orientation()[1])),
                }
            ]
        }
        for tro_key, entity in self.env.task.object_scope.items():
            if getattr(entity, "is_system", False) or not getattr(entity, "exists", False):
                continue
            tro_state[tro_key] = entity.dump_state(serialized=False)
        return tro_state

    def _maybe_export_current_tro_state(self, task_name: str, instance_id: int) -> str | None:
        if self._save_tro_state_to is None:
            return None
        out_path = _resolve_tro_export_path(self._save_tro_state_to, task_name, instance_id)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tro_state = self._build_current_tro_state_snapshot()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tro_state, f, cls=TorchEncoder, indent=2)
        self._current_tro_export_path = out_path
        print(
            "[eval_skill_sgnav_http][tro_export] "
            f"task={task_name} instance_id={instance_id} "
            f"source={self._current_tro_source_path or '<snapshot>'} "
            f"saved_to={out_path} "
            f"reuse_arg=--tro_state_path {out_path}",
            flush=True,
        )
        return out_path

    def load_env(self, task_name: str, max_steps: int):
        if self.env is not None:
            og.sim.stop()
        self._camera_pose_align_logged = False
        self._current_tro_source_path = None
        self._current_tro_export_path = None
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        available_tasks = load_available_tasks()
        assert task_name in available_tasks, f"Invalid task: {task_name}"
        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        cfg["env"]["flatten_obs_space"] = True
        if self.partial_scene_load and not self.online_sampling:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms
        cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
        self._apply_base_controller_mode_to_robot_cfg(cfg["robots"][0])
        obs_mods = ["proprio", "rgb", "depth_linear"]
        vs_kwargs: dict = {
            "image_height": int(self.obs_height),
            "image_width": int(self.obs_width),
        }
        if self._vision_horizontal_aperture is not None:
            vs_kwargs["horizontal_aperture"] = float(self._vision_horizontal_aperture)
        elif self._vision_hfov_deg is not None:
            # OmniGibson VisionSensor expects aperture (mm), while SG-Nav yaml usually stores HFOV in degrees.
            f_mm = float(self._vision_focal_length_mm)
            hfov_rad = math.radians(float(self._vision_hfov_deg))
            vs_kwargs["focal_length"] = f_mm
            vs_kwargs["horizontal_aperture"] = float(2.0 * f_mm * math.tan(hfov_rad / 2.0))
        print(
            "[eval_skill_sgnav_http][cam_cfg] "
            f"base_action_mode={self._base_action_mode} "
            f"head_key={self._head_key} "
            f"image_width={int(vs_kwargs['image_width'])} image_height={int(vs_kwargs['image_height'])} "
            f"horizontal_aperture={vs_kwargs.get('horizontal_aperture')} "
            f"vision_hfov_deg={self._vision_hfov_deg} focal_length={vs_kwargs.get('focal_length')}",
            flush=True,
        )
        self._last_vs_kwargs = dict(vs_kwargs)
        sensor_cfg = {
            "VisionSensor": {
                "sensor_kwargs": vs_kwargs,
            },
        }
        if self.use_og_occupancy:
            obs_mods.extend(["scan", "occupancy_grid"])
            sensor_cfg["ScanSensor"] = {
                "modalities": ["scan", "occupancy_grid"],
                "sensor_kwargs": {
                    "occupancy_grid_resolution": self._og_occupancy_resolution,
                    "occupancy_grid_range": self._og_occupancy_range_m,
                    "horizontal_fov": 360.0,
                    "horizontal_resolution": 1.0,
                    "vertical_fov": 1.0,
                    "vertical_resolution": 1.0,
                },
            }
        cfg["robots"][0]["obs_modalities"] = obs_mods
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        cfg["robots"][0]["sensor_config"] = sensor_cfg
        cfg["task"]["termination_config"]["max_steps"] = max_steps
        cfg["task"]["include_obs"] = False
        if self.online_sampling:
            cfg["task"]["online_object_sampling"] = True
            cfg["task"]["use_presampled_robot_pose"] = False
            cfg["task"]["activity_instance_id"] = 0
        else:
            cfg["task"]["activity_instance_id"] = 0
            if self.rft_style_tro:
                cfg["task"]["use_presampled_robot_pose"] = False

        env = og.Environment(configs=cfg)
        if self.env_wrapper_cfg is not None:
            env = instantiate(self.env_wrapper_cfg, env=env)
        self.env = env
        self.robot = env.scene.object_registry("name", ROBOT_NAME)
        with og.sim.stopped():
            self.robot.base_footprint_link.mass = 250
        self.current_task_name = task_name
        env.load_observation_space()
        if self.robot is None:
            self.robot = env.scene.object_registry("name", "robot_r1")
        if self.robot is None:
            raise RuntimeError(
                f"Robot not found as name={ROBOT_NAME!r} or robot_r1; "
                "check generate_robot_config / scene registry."
            )
        self._last_asset_hash_mismatches = []
        if not self.online_sampling:
            self._last_asset_hash_mismatches = _collect_scene_asset_hash_mismatches(env.scene)
            if self._last_asset_hash_mismatches:
                preview = [
                    {
                        "name": item["name"],
                        "category": item["category"],
                        "model": item["model"],
                        "expected_hash": item["expected_hash"],
                        "current_hash": item["current_hash"],
                    }
                    for item in self._last_asset_hash_mismatches[:8]
                ]
                print(
                    "[eval_skill_sgnav_http][asset_hash_mismatch] "
                    f"count={len(self._last_asset_hash_mismatches)} "
                    f"scene_model={cfg['scene'].get('scene_model')} "
                    f"task={task_name} "
                    "offline tro_state replay may not match the original sampled instance. "
                    "Use the same OMNIGIBSON_DATA_PATH asset bundle that generated the task instances, "
                    "or regenerate tro_state with your current assets. "
                    f"preview={json.dumps(preview, ensure_ascii=False)}",
                    flush=True,
                )
                if self._fail_on_asset_hash_mismatch:
                    raise RuntimeError(
                        "Asset hash mismatch detected while loading an offline task instance scene; "
                        "set OMNIGIBSON_DATA_PATH to a matching asset bundle or regenerate tro_state."
                    )
        self._refresh_base_velocity_caps_from_controller()
        if self._base_action_mode == "position":
            print(
                "[eval_skill_sgnav_http][position_controller_truth] "
                + json.dumps(self._build_position_controller_truth(), ensure_ascii=False),
                flush=True,
            )
        self._log_controller_snapshot_once(
            ACTION_QPOS_INDICES["R1Pro"]["base"],
            ACTION_QPOS_INDICES["R1Pro"]["torso"],
        )
        self._log_og_runtime_camera_params(
            vs_kwargs,
            stage="post_load",
            include_camera_params=False,
        )
        if self.use_og_occupancy:
            try:
                sensor_names = list(getattr(self.robot, "sensors", {}).keys())
                obs_mods = sorted(list(getattr(self.robot, "obs_modalities", [])))
                sensor_mods = {}
                for n, s in getattr(self.robot, "sensors", {}).items():
                    try:
                        sensor_mods[n] = sorted(list(getattr(s, "modalities", [])))
                    except Exception:
                        sensor_mods[n] = []
                print(
                    "[eval_skill_sgnav_http][og_diag] "
                    f"obs_modalities={obs_mods} "
                    f"sensor_names={sensor_names} "
                    f"sensor_modalities={sensor_mods}",
                    flush=True,
                )
            except Exception as e:
                print(f"[eval_skill_sgnav_http][og_diag] failed: {e}", flush=True)
        return env

    def load_task_instance(self, instance_id: int):
        if self.online_sampling:
            self._cached_tro_state = None
            self._current_tro_source_path = None
            return True
        if self._tro_state_path is not None:
            tro_file_path = self._tro_state_path
            tro_label = "explicit_path"
        else:
            scene_model = self.env.task.scene_name
            tro_filename = self.env.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=self.env.task.activity_name,
                activity_definition_id=self.env.task.activity_definition_id,
                activity_instance_id=instance_id,
            )
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
            tro_label = "instance_lookup"
        if not os.path.exists(tro_file_path):
            self._cached_tro_state = None
            self._current_tro_source_path = os.path.abspath(tro_file_path)
            return False
        self._current_tro_source_path = os.path.abspath(tro_file_path)
        print(
            "[eval_skill_sgnav_http][tro] "
            f"source={tro_label} "
            f"path={self._current_tro_source_path} "
            f"task={self.env.task.activity_name} "
            f"instance_id={instance_id}",
            flush=True,
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        self._cached_tro_state = tro_state

        for tro_key, tro_data in tro_state.items():
            if tro_key == "robot_poses":
                if self.robot.model_name not in tro_data:
                    self._cached_tro_state = None
                    return False
                robot_pos = tro_data[self.robot.model_name][0]["position"]
                robot_quat = tro_data[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                self.env.scene.write_task_metadata(key=tro_key, data=tro_data)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_data, serialized=False)

        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()

        if self.rft_style_tro:
            self.env.reset()
            self._reapply_tro_after_reset(tro_state)
            for _ in range(POST_RESET_PHYSICS_STEPS):
                og.sim.step_physics()
                for entity in self.env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()
            setattr(self.env, "_eval_cached_tro_state", tro_state)
            return True

        self.env.scene.reset()
        return True

    def run_segment(
        self,
        seg: dict,
        max_steps: int,
        *,
        distance_threshold: float = MOVE_TO_SUCCESS_DISTANCE_THRESHOLD,
        min_steps_before_reach_early_stop: int | None = None,
        early_stop_on_reach_distance: bool = True,
        ignore_bddl_termination: bool | None = None,
    ) -> dict:
        if min_steps_before_reach_early_stop is None:
            min_steps_before_reach_early_stop = min(
                MOVE_TO_MIN_STEPS_BEFORE_REACH_EARLY_STOP,
                max(1, max_steps // 2),
            )
        if ignore_bddl_termination is None:
            ignore_bddl_termination = self._ignore_bddl_termination
        oid = str(seg.get("object_id", ""))
        instance_id = _resolve_segment_instance_id(seg, self._instance_id_override)

        print(
            "[eval_skill_sgnav_http][segment] "
            f"task={seg.get('task_name')} "
            f"object_id={oid or '<none>'} "
            f"instance_id={instance_id}",
            flush=True,
        )

        if not self.load_task_instance(instance_id):
            return {
                "success": False,
                "error": "tro_not_found",
                "instance_id": int(instance_id),
                "tro_source_path": self._current_tro_source_path,
                "steps": 0,
                "min_distance": float("inf"),
                "initial_distance": float("inf"),
                "final_distance": float("inf"),
            }

        obs, _ = self.env.reset()
        if self.rft_style_tro and self._cached_tro_state is not None:
            self._reapply_tro_after_reset(self._cached_tro_state)
            for _ in range(POST_RESET_PHYSICS_STEPS):
                og.sim.step_physics()
                for entity in self.env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()
        self._maybe_export_current_tro_state(str(seg.get("task_name") or self.current_task_name), instance_id)
        self._log_og_runtime_camera_params(
            self._last_vs_kwargs or {
                "image_width": int(self.obs_width),
                "image_height": int(self.obs_height),
            },
            stage="post_reset",
            include_camera_params=True,
        )

        if self.use_annotation_object_lookup and oid:
            tgt = ef._find_target_object_annotation(self.env, oid)
            dn = ef._annotation_display_name(oid) or oid
            all_matching = [(tgt, dn)] if tgt is not None else []
        else:
            dn = None
            all_matching = ef._find_all_matching_objects(
                self.env.scene, oid, task=getattr(self.env, "task", None)
            ) if oid else []
            if all_matching:
                dn = all_matching[0][1]

        self.client.reset(
            behavior_goal=oid or seg.get("prompt", "chair"),
            object_category_sg=_build_goal_sg_hint(oid, dn),
        )
        self._nav_prev_gps = None
        self._nav_prev_compass = None
        self._nav_prev_action = None
        self._nav_prev_collision_info = None

        target_obj = all_matching[0][0] if all_matching else None
        self.target = target_obj

        min_distance = float("inf")
        initial_distance = float("inf")
        final_distance = float("inf")
        base_slice = ACTION_QPOS_INDICES["R1Pro"]["base"]
        torso_slice = ACTION_QPOS_INDICES["R1Pro"]["torso"]
        steps_done = 0
        nav_dbg = self._nav_debug_path
        exit_reason: str | None = None
        last_step_info: dict | None = None

        for step in range(max_steps):
            current_step_min = float("inf")
            for obj, _dn in all_matching:
                obj_pos = ef._get_object_position(obj)
                dist = ef._min_distance_arms_to_point(self.robot, obj_pos)
                min_distance = min(min_distance, dist)
                current_step_min = min(current_step_min, dist)
                if step == 0:
                    initial_distance = min(initial_distance, dist)
            final_distance = current_step_min

            if (
                early_stop_on_reach_distance
                and min_distance < distance_threshold
                and step >= min_steps_before_reach_early_stop
            ):
                exit_reason = "early_reach_distance"
                steps_done = step
                break

            rgb, depth, gps, compass = build_sgnav_step_tensors(
                obs,
                self.robot,
                head_key=self._head_key,
                map_size_cm=self.map_size_cm,
                flatten_obs_dict=flatten_obs_dict,
                obs_width=self.obs_width,
                obs_height=self.obs_height,
                max_depth_m=self.max_depth_m,
            )
            og_occ = None
            og_occ_keys = []
            if self.use_og_occupancy:
                flat_o = flatten_obs_dict(obs)
                og_occ_keys = [
                    k for k in flat_o
                    if isinstance(k, str) and ("occupancy_grid" in k or k.endswith("::scan"))
                ]
                og_occ = extract_omnigibson_occupancy_grid(flat_o)
                if og_occ is None and step < 5:
                    print(
                        "[eval_skill_sgnav_http][warn] use_og_occupancy enabled but no occupancy_grid found "
                        f"at step={step}. keys_like_scan_or_occupancy={og_occ_keys[:8]}",
                        flush=True,
                    )
            target_world_xy = None
            if target_obj is not None:
                try:
                    op = ef._get_object_position(target_obj)
                    pa = op.detach().cpu().numpy().reshape(-1)
                    if pa.size >= 2 and np.isfinite(pa[0]) and np.isfinite(pa[1]):
                        target_world_xy = [float(pa[0]), float(pa[1])]
                except Exception:
                    pass
            cam_pose = None
            if self._send_camera_pose_world:
                cam_pose = self._collect_camera_pose_world_payload()
            prev_collision_info = (
                dict(self._nav_prev_collision_info)
                if isinstance(self._nav_prev_collision_info, dict)
                else None
            )
            action_idx = self.client.step(
                rgb,
                depth,
                gps,
                compass,
                og_occupancy=og_occ,
                og_occupancy_meta=self._og_occupancy_meta if og_occ is not None else None,
                target_world_xy=target_world_xy,
                camera_pose_world=cam_pose,
                collision_info=prev_collision_info,
            )
            nav_row = None
            if nav_dbg:
                d1 = depth.reshape(-1)
                fin = d1[np.isfinite(d1) & (d1 > 0)]
                med = float(np.median(fin)) if fin.size else float("nan")
                gps_delta_m = float("nan")
                prev_action = self._nav_prev_action
                compass_delta_rad = float("nan")
                compass_delta_deg = float("nan")
                if self._nav_prev_gps is not None:
                    gps_delta_m = float(np.linalg.norm(gps - self._nav_prev_gps))
                self._nav_prev_gps = gps.astype(np.float64).copy()
                if self._nav_prev_compass is not None:
                    dc = float(compass) - float(self._nav_prev_compass)
                    while dc > math.pi:
                        dc -= 2.0 * math.pi
                    while dc < -math.pi:
                        dc += 2.0 * math.pi
                    compass_delta_rad = float(dc)
                    compass_delta_deg = float(math.degrees(dc))
                world_xy = [float("nan"), float("nan")]
                try:
                    pos, _q = self.robot.get_position_orientation()
                    p3 = pos.detach().cpu().numpy().reshape(3)
                    world_xy = [float(p3[0]), float(p3[1])]
                except Exception:
                    pass
                nav_row = {
                    "step": step,
                    "action": int(action_idx),
                    "base_action_mode": str(self._base_action_mode),
                    "rgb_shape": list(rgb.shape),
                    "rgb_dtype": str(rgb.dtype),
                    "rgb_mean": float(np.mean(rgb)) if rgb.size else float("nan"),
                    "depth_min": float(np.nanmin(d1)) if d1.size else float("nan"),
                    "depth_max": float(np.nanmax(d1)) if d1.size else float("nan"),
                    "depth_median_pos": med,
                    "gps": [float(gps[0]), float(gps[1])],
                    "world_xy_m": world_xy,
                    "compass_rad": float(compass),
                    "compass_deg": float(math.degrees(compass)),
                    "prev_action": None if prev_action is None else int(prev_action),
                    "prev_action_compass_delta_rad": compass_delta_rad,
                    "prev_action_compass_delta_deg": compass_delta_deg,
                    "prev_action_turn_delta_deg": (
                        abs(float(compass_delta_deg))
                        if prev_action in (2, 3, 6) and math.isfinite(compass_delta_deg)
                        else float("nan")
                    ),
                    "map_size_cm": float(self.map_size_cm),
                    "gps_delta_m": gps_delta_m,
                    "og_occ_enabled": bool(self.use_og_occupancy),
                    "og_occ_present": bool(og_occ is not None),
                    "og_occ_key_count": int(len(og_occ_keys)),
                    "og_occ_keys_head": [str(k) for k in og_occ_keys[:4]],
                    "prev_action_base_contact": bool(
                        prev_collision_info.get("base_contact", False)
                    ) if isinstance(prev_collision_info, dict) else False,
                    "prev_action_collision_n_contacts": int(
                        prev_collision_info.get("n_contacts", 0)
                    ) if isinstance(prev_collision_info, dict) else 0,
                    "prev_action_collision_max_impulse": float(
                        prev_collision_info.get("max_impulse", 0.0)
                    ) if isinstance(prev_collision_info, dict) else 0.0,
                    "prev_action_collision_bodies": (
                        [str(x) for x in prev_collision_info.get("contact_bodies", [])[:8]]
                        if isinstance(prev_collision_info, dict)
                        else []
                    ),
                }
                if og_occ is not None:
                    o1 = np.asarray(og_occ, dtype=np.float32).reshape(-1)
                    of = o1[np.isfinite(o1)]
                    nav_row["og_occ_shape"] = list(np.asarray(og_occ).shape)
                    nav_row["og_occ_min"] = float(np.min(of)) if of.size else float("nan")
                    nav_row["og_occ_max"] = float(np.max(of)) if of.size else float("nan")
                    nav_row["og_occ_mean"] = float(np.mean(of)) if of.size else float("nan")
            if int(action_idx) == SGNAV_STOP_ACTION and target_world_xy is not None:
                exit_reason = "sgnav_stop_target_nav"
                self._nav_prev_compass = float(compass)
                self._nav_prev_action = int(action_idx)
                if nav_row is not None:
                    nav_row["sgnav_stop_target_nav"] = True
                    nav_row["exit_reason"] = exit_reason
                    nav_row["current_action_base_contact"] = False
                    nav_row["current_action_collision_n_contacts"] = 0
                    nav_row["current_action_collision_max_impulse"] = 0.0
                    nav_row["current_action_collision_bodies"] = []
                    print(
                        "[eval_skill_sgnav_http][motion_scale] ",
                        nav_row,
                        flush=True,
                    )
                    with open(nav_dbg, "a", encoding="utf-8") as df:
                        df.write(json.dumps(nav_row) + "\n")
                steps_done = step
                break
            self._nav_prev_compass = float(compass)
            self._nav_prev_action = int(action_idx)
            sim_step_hz = float("nan")
            try:
                dt = float(og.sim.get_sim_step_dt())
                if np.isfinite(dt) and dt > 1e-9:
                    sim_step_hz = 1.0 / dt
            except Exception:
                pass
            if not np.isfinite(sim_step_hz) or sim_step_hz <= 1e-6:
                sim_step_hz = float(getattr(gm, "DEFAULT_SIM_STEP_FREQ", 10.0) or 10.0)

            if self._base_action_mode == "position":
                if action_idx == 1:
                    n_phys = self._position_forward_physics_steps_per_nav_step
                elif action_idx in (2, 3, 6):
                    n_phys = self._position_turn_physics_steps_per_nav_step
                else:
                    n_phys = 1
                forward_step_sub = float(self._forward_step_size_m)
                turn_angle_sub_deg = float(self._turn_angle_deg)
                if step % int(self._motion_scale_log_every) == 0:
                    print(
                        "[eval_skill_sgnav_http][motion_scale] "
                        f"step={step} action={int(action_idx)} n_phys={int(n_phys)} "
                        f"repeat_mode=repeated_absolute_cmd base_action_mode=position "
                        f"repeat_cfg_forward={int(self._position_forward_physics_steps_per_nav_step)} "
                        f"repeat_cfg_turn={int(self._position_turn_physics_steps_per_nav_step)} "
                        f"forward_step_size_m={float(self._forward_step_size_m):.3f} "
                        f"turn_angle_deg={float(self._turn_angle_deg):.3f} "
                        f"forward_step_per_substep_m={forward_step_sub:.4f} "
                        f"turn_angle_per_substep_deg={turn_angle_sub_deg:.4f}",
                        flush=True,
                    )
                action_seed = self._build_controller_noop_action()
                act = discrete_sgnav_to_base_action(
                    action_idx,
                    self.robot.action_dim,
                    base_slice,
                    base_action_mode=self._base_action_mode,
                    forward_step_m=forward_step_sub,
                    turn_angle_rad=float(np.deg2rad(turn_angle_sub_deg)),
                    torso_slice=torso_slice,
                    robot=self.robot,
                    look_pitch_step_rad=self._physical_look_pitch_step_rad,
                    seed_action=action_seed,
                )
            else:
                # Repeat the same discrete action using multiple env.step calls (not og.sim.step_physics).
                # Forward / turn actions auto-repeat enough env.step calls so the achievable base
                # motion matches SG-Nav's discrete forward-step / turn-angle semantics instead of
                # moving only one or two physics frames and under-shooting the configured step size.
                if action_idx == 1:
                    forward_cap_per_step_m = float(self._base_cmd_max_linear_m_s) / max(
                        sim_step_hz, 1e-6
                    )
                    if np.isfinite(forward_cap_per_step_m) and forward_cap_per_step_m > 1e-9:
                        n_phys = max(
                            1,
                            int(
                                math.ceil(
                                    float(self._forward_step_size_m) / forward_cap_per_step_m
                                    - 1e-9
                                )
                            ),
                        )
                    else:
                        n_phys = 1
                elif action_idx in (2, 3, 6):
                    turn_rate_cap_rad_s = min(
                        abs(float(self.turn_vel)),
                        float(self._base_cmd_max_angular_rad_s),
                    )
                    turn_cap_per_step_rad = turn_rate_cap_rad_s / max(sim_step_hz, 1e-6)
                    if np.isfinite(turn_cap_per_step_rad) and turn_cap_per_step_rad > 1e-9:
                        n_phys = max(
                            int(self._physics_steps_per_nav_step),
                            int(
                                math.ceil(
                                    float(np.deg2rad(self._turn_angle_deg))
                                    / turn_cap_per_step_rad
                                    - 1e-9
                                )
                            ),
                        )
                    else:
                        n_phys = int(self._physics_steps_per_nav_step)
                else:
                    n_phys = 1
                dt_nav = max(float(n_phys) / max(sim_step_hz, 1e-6), 1e-6)
                v_cap = float(self._forward_step_size_m) / dt_nav
                w_cap = float(np.deg2rad(self._turn_angle_deg)) / dt_nav
                forward_vel_used = math.copysign(
                    min(abs(float(self.forward_vel)), max(v_cap, 0.0)),
                    float(self.forward_vel),
                )
                turn_vel_used = math.copysign(
                    min(abs(float(self.turn_vel)), max(w_cap, 0.0)),
                    float(self.turn_vel),
                )
                capped = (
                    abs(forward_vel_used - float(self.forward_vel)) > 1e-9
                    or abs(turn_vel_used - float(self.turn_vel)) > 1e-9
                )
                if capped or step % int(self._motion_scale_log_every) == 0:
                    print(
                        "[eval_skill_sgnav_http][motion_scale] "
                        f"step={step} action={int(action_idx)} n_phys={int(n_phys)} "
                        f"repeat_mode=env_step base_action_mode=velocity "
                        f"sim_hz={sim_step_hz:.3f} dt_nav={dt_nav:.4f} "
                        f"v_req={float(self.forward_vel):.3f} v_cap={v_cap:.3f} v_used={forward_vel_used:.3f} "
                        f"w_req={float(self.turn_vel):.3f} w_cap={w_cap:.3f} w_used={turn_vel_used:.3f} "
                        f"base_cmd_max_lin={float(self._base_cmd_max_linear_m_s):.3f} "
                        f"base_cmd_max_ang={float(self._base_cmd_max_angular_rad_s):.3f} "
                        f"forward_step_size_m={float(self._forward_step_size_m):.3f} "
                        f"turn_angle_deg={float(self._turn_angle_deg):.3f} capped={int(capped)}",
                        flush=True,
                    )
                action_seed = self._build_controller_noop_action()
                act = discrete_sgnav_to_base_action(
                    action_idx,
                    self.robot.action_dim,
                    base_slice,
                    base_action_mode=self._base_action_mode,
                    forward_vel=forward_vel_used,
                    turn_vel=turn_vel_used,
                    max_linear_m_s=float(self._base_cmd_max_linear_m_s),
                    max_angular_rad_s=float(self._base_cmd_max_angular_rad_s),
                    torso_slice=torso_slice,
                    robot=self.robot,
                    look_pitch_step_rad=self._look_pitch_step_rad,
                    seed_action=action_seed,
                )
            if self._debug_action_semantics and int(action_idx) in {0, 1, 2, 3, 4, 5, 6}:
                action_semantics_diag = self._compute_action_semantics_diag(
                    step=step,
                    cmd=int(action_idx),
                    action=act,
                    base_slice=base_slice,
                    torso_slice=torso_slice,
                )
                print(
                    "[eval_skill_sgnav_http][action_semantics] "
                    + json.dumps(action_semantics_diag, ensure_ascii=False),
                    flush=True,
                )
                if nav_row is not None:
                    nav_row["action_semantics"] = action_semantics_diag
            terminated = False
            truncated = False
            step_info = {}
            forward_substep_diags = []
            forward_guard_event = None
            forward_contact_streak = 0
            forward_cumulative_planar_m = 0.0
            for _k in range(n_phys):
                forward_before_pose = None
                do_forward_guard = (
                    self._base_action_mode == "position" and int(action_idx) == 1
                )
                if do_forward_guard:
                    forward_before_pose = self._get_robot_pose_diag()
                obs, _r, terminated, truncated, step_info = self.env.step(
                    {ROBOT_NAME: act}, n_render_iterations=1
                )
                if do_forward_guard:
                    sub_collision_info = _collect_base_collision_info(self.robot)
                    if bool(sub_collision_info.get("base_contact", False)):
                        forward_contact_streak += 1
                    else:
                        forward_contact_streak = 0
                    forward_after_pose = self._get_robot_pose_diag()
                    sub_diag = self._compute_position_forward_substep_diag(
                        step=step,
                        substep_idx=_k,
                        before_pose=forward_before_pose,
                        after_pose=forward_after_pose,
                        expected_forward_step_m=float(self._forward_step_size_m),
                        cumulative_planar_m=float(forward_cumulative_planar_m),
                        consecutive_contacts=int(forward_contact_streak),
                        collision_info=sub_collision_info,
                    )
                    forward_cumulative_planar_m += float(sub_diag["actual_planar_delta_m"])
                    sub_diag["cumulative_planar_m"] = float(forward_cumulative_planar_m)
                    guard_reason = None
                    if (
                        float(sub_diag["actual_planar_delta_m"])
                        > float(self._position_forward_guard_step_delta_m)
                    ):
                        guard_reason = "step_delta_exceeded"
                    elif (
                        float(sub_diag["roll_pitch_abs_max_deg"])
                        > float(self._position_forward_guard_roll_pitch_deg)
                    ):
                        guard_reason = "roll_pitch_exceeded"
                    elif (
                        int(forward_contact_streak)
                        >= int(self._position_forward_guard_consecutive_contacts)
                    ):
                        guard_reason = "consecutive_contacts_exceeded"
                    elif (
                        float(sub_diag["actual_planar_delta_m"]) > 0.03
                        and float(sub_diag["heading_error_deg"])
                        > float(self._position_forward_guard_heading_error_deg)
                    ):
                        guard_reason = "heading_error_exceeded"
                    if guard_reason is not None:
                        sub_diag["guard_reason"] = str(guard_reason)
                    forward_substep_diags.append(sub_diag)
                    if (
                        step % int(self._motion_scale_log_every) == 0
                        or guard_reason is not None
                    ):
                        print(
                            "[eval_skill_sgnav_http][position_forward_substep] "
                            + json.dumps(sub_diag, ensure_ascii=False),
                            flush=True,
                        )
                    if guard_reason is not None and self._position_forward_guard_enable:
                        forward_guard_event = {
                            "triggered": True,
                            "reason": str(guard_reason),
                            "substep_idx": int(_k),
                            "thresholds": {
                                "step_delta_m": float(self._position_forward_guard_step_delta_m),
                                "roll_pitch_deg": float(
                                    self._position_forward_guard_roll_pitch_deg
                                ),
                                "consecutive_contacts": int(
                                    self._position_forward_guard_consecutive_contacts
                                ),
                                "heading_error_deg": float(
                                    self._position_forward_guard_heading_error_deg
                                ),
                            },
                            "diag": sub_diag,
                        }
                        print(
                            "[eval_skill_sgnav_http][position_forward_guard] "
                            + json.dumps(
                                {
                                    "step": int(step),
                                    "substep_idx": int(_k),
                                    "reason": str(guard_reason),
                                    "diag": sub_diag,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        break
                if truncated or terminated:
                    break
            collision_info = _collect_base_collision_info(self.robot)
            settle_result = self._settle_after_action(
                step=step,
                action_idx=int(action_idx),
                had_contact=bool(collision_info.get("base_contact", False)),
            )
            if settle_result is not None:
                if settle_result.get("obs") is not None:
                    obs = settle_result["obs"]
                terminated = bool(terminated or settle_result.get("terminated", False))
                truncated = bool(truncated or settle_result.get("truncated", False))
                if settle_result.get("step_info"):
                    step_info = settle_result["step_info"]
                collision_info = self._merge_collision_infos(
                    collision_info, settle_result.get("collision_info")
                )
            collision_info["prev_action"] = int(action_idx)
            self._nav_prev_collision_info = collision_info
            if bool(collision_info.get("base_contact", False)):
                print(
                    "[eval_skill_sgnav_http][collision] "
                    f"step={step} action={int(action_idx)} "
                    f"n_contacts={int(collision_info.get('n_contacts', 0))} "
                    f"max_impulse={float(collision_info.get('max_impulse', 0.0)):.4f} "
                    f"bodies={collision_info.get('contact_bodies', [])}",
                    flush=True,
                )
            if nav_row is not None:
                nav_row["current_action_base_contact"] = bool(
                    collision_info.get("base_contact", False)
                )
                nav_row["current_action_collision_n_contacts"] = int(
                    collision_info.get("n_contacts", 0)
                )
                nav_row["current_action_collision_max_impulse"] = float(
                    collision_info.get("max_impulse", 0.0)
                )
                nav_row["current_action_collision_bodies"] = [
                    str(x) for x in collision_info.get("contact_bodies", [])[:8]
                ]
                if self._base_action_mode == "position" and int(action_idx) == 1:
                    nav_row["position_forward_guard_enabled"] = bool(
                        self._position_forward_guard_enable
                    )
                    nav_row["position_forward_guard_thresholds"] = {
                        "step_delta_m": float(self._position_forward_guard_step_delta_m),
                        "roll_pitch_deg": float(self._position_forward_guard_roll_pitch_deg),
                        "consecutive_contacts": int(
                            self._position_forward_guard_consecutive_contacts
                        ),
                        "heading_error_deg": float(
                            self._position_forward_guard_heading_error_deg
                        ),
                    }
                    nav_row["position_forward_substeps"] = forward_substep_diags
                    nav_row["position_forward_guard"] = (
                        forward_guard_event
                        if forward_guard_event is not None
                        else {"triggered": False}
                    )
                if settle_result is not None:
                    nav_row["action_settle"] = {
                        "enabled": True,
                        "n_steps": int(len(settle_result.get("samples", []))),
                        "final": (
                            settle_result["samples"][-1]
                            if settle_result.get("samples")
                            else None
                        ),
                    }
                print(
                    "[eval_skill_sgnav_http][motion_scale] ",
                    nav_row,
                    flush=True,
                )
                with open(nav_dbg, "a", encoding="utf-8") as df:
                    df.write(json.dumps(nav_row) + "\n")
            last_step_info = step_info
            steps_done = step + 1
            if truncated:
                exit_reason = "env_timeout"
                break
            if terminated:
                if ignore_bddl_termination:
                    # Navigation eval: ignore BDDL / predicate success or other terminal signals;
                    # only distance early-stop, truncated timeout, or max_steps ends the loop.
                    continue
                tc = step_info.get("done", {}).get("termination_conditions", {})
                if tc.get("predicate", {}).get("done"):
                    exit_reason = "bddl_goal_satisfied"
                else:
                    exit_reason = "env_terminated_non_predicate"
                break
        else:
            exit_reason = exit_reason or "max_steps"

        if exit_reason is None:
            exit_reason = "max_steps"

        bddl_task_success = None
        if last_step_info is not None:
            bddl_task_success = last_step_info.get("done", {}).get("success")

        if ignore_bddl_termination:
            success = min_distance < distance_threshold
        else:
            success = min_distance < distance_threshold and final_distance < initial_distance
        return {
            "success": bool(success),
            "instance_id": int(instance_id),
            "tro_source_path": self._current_tro_source_path,
            "tro_export_path": self._current_tro_export_path,
            "min_distance": float(min_distance),
            "final_distance": float(final_distance),
            "initial_distance": float(initial_distance),
            "steps": steps_done,
            "exit_reason": exit_reason,
            "bddl_task_success": bddl_task_success,
            "nav_only": bool(ignore_bddl_termination),
        }


def main():
    parser = argparse.ArgumentParser(description="Move-to eval via SG-Nav HTTP (isolated env)")
    parser.add_argument(
        "--sgnav_url",
        type=str,
        default="http://127.0.0.1:8765",
        help="SG-Nav behavior/sgnav_http_server.py base URL",
    )
    parser.add_argument("--task", type=str, nargs="+", default=None)
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="Force all evaluated segments to use this BEHAVIOR activity_instance_id. "
        "If omitted, use each segment's own instance_id when available, else 0. "
        "This is only effective in offline tro_state replay mode, e.g. with --no_online_sampling "
        "or --tro_state_path. In default online sampling mode, BehaviorTask ignores activity_instance_id.",
    )
    parser.add_argument(
        "--tro_state_path",
        type=str,
        default=None,
        help="Load this exact tro_state JSON file instead of resolving one from task + instance_id. "
        "Useful for reusing a previously exported tro snapshot.",
    )
    parser.add_argument(
        "--save_tro_state_to",
        type=str,
        default=None,
        help="Export the current effective tro_state after reset. If this is a directory, the script writes "
        "<task>_instance_<id>_tro_state.json inside it. The saved file can be reused later with --tro_state_path.",
    )
    parser.add_argument(
        "--fail_on_asset_hash_mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In offline tro_state mode, fail immediately if the loaded scene assets' USD hashes do not match "
        "the expected hashes recorded in the task instance scene. Useful when you need exact instance replay.",
    )
    parser.add_argument("--total_segments", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--online_sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="BDDL online sampling (default: true unless --move_to_segments_json)",
    )
    parser.add_argument("--partial_scene_load", action="store_true")
    parser.add_argument(
        "--full_scene_load",
        action="store_true",
        help="With --move_to_segments_json: load full scene (default partial rooms)",
    )
    parser.add_argument("--move_to_segments_json", type=str, default=None)
    parser.add_argument("--skills_flat", type=str, default=None)
    parser.add_argument(
        "--bddl_segments",
        action="store_true",
        help="Same as default segment source (BDDL move-to); kept for parity with eval_skill_flat.py.",
    )
    parser.add_argument(
        "--move_to_only",
        action="store_true",
        help="No-op flag (this script only evaluates move-to); matches eval_skill_flat.py CLI.",
    )
    parser.add_argument(
        "--no_online_sampling",
        action="store_true",
        help="Use cached tro_state JSON instead of online BDDL sampling (reproducible poses).",
    )
    parser.add_argument(
        "--max_objects_per_activity",
        type=int,
        default=10,
        help="Cap objects per BDDL activity when building segments (like eval_skill_flat --bddl_segments).",
    )
    parser.add_argument(
        "--object_id_contains",
        type=str,
        default=None,
        help="Keep segments whose object_id contains this substring (case-insensitive), e.g. radio for turning_on_radio.",
    )
    parser.add_argument(
        "--env_wrapper",
        action="store_true",
        help="Wrap env with RGBLowResWrapper (changes head resolution); default off for stable 640×480.",
    )
    parser.add_argument("--url_health", action="store_true", help="Only GET /health and exit")
    parser.add_argument(
        "--sgnav_config",
        type=str,
        default=None,
        help="SG-Nav OmegaConf yaml (e.g. SG-Nav/configs/sgnav_minimal.rgbd.yaml) for eval defaults: "
        "SIMULATOR RGB/DEPTH/TILT_ANGLE and optional SGNAV_RUNTIME behavior_* keys. "
        "If omitted, uses sibling SG-Nav/configs/sgnav_minimal.rgbd.yaml when that file exists.",
    )
    parser.add_argument(
        "--no_sgnav_yaml",
        action="store_true",
        help="Do not load any yaml; use only CLI flags and hardcoded fallbacks.",
    )
    parser.add_argument(
        "--map_size_cm",
        type=float,
        default=None,
        help="Habitat-style gps grid size (cm); must match SG_Nav_Agent.map_size_cm. "
        "Default from yaml SGNAV_RUNTIME.map_size_cm if set, else 4000.",
    )
    parser.add_argument(
        "--base_action_mode",
        type=str,
        choices=list(VALID_BASE_ACTION_MODES),
        default=None,
        help="How SG-Nav base commands are applied on the OmniGibson side. "
        "'velocity' keeps the old m/s,rad/s command path; 'position' switches the base "
        "controller to absolute local-frame [dx, dy, drz] commands per env.step. "
        "Default from yaml SGNAV_RUNTIME.behavior_base_action_mode if set, else velocity.",
    )
    parser.add_argument(
        "--forward_vel",
        type=float,
        default=None,
        help="Target forward speed in m/s (mapped to OG holonomic base cmd ∈[-1,1] via /1.5). "
        "Default from yaml SGNAV_RUNTIME.behavior_forward_vel if set, else 0.75.",
    )
    parser.add_argument(
        "--turn_vel",
        type=float,
        default=None,
        help="Target yaw rate in rad/s for discrete turns (mapped to cmd ∈[-1,1] via /π). "
        "Values above π still saturate at max angular velocity. "
        "Default from yaml SGNAV_RUNTIME.behavior_turn_vel if set, else 1.25.",
    )
    parser.add_argument(
        "--forward_step_size_m",
        type=float,
        default=None,
        help="Per SG-Nav navigation step target forward displacement (m) for velocity capping. "
        "Default from SIMULATOR.FORWARD_STEP_SIZE if set, else 0.1.",
    )
    parser.add_argument(
        "--turn_angle_deg",
        type=float,
        default=None,
        help="Per SG-Nav navigation step target turn angle (deg) for angular velocity capping. "
        "Default from SIMULATOR.TURN_ANGLE if set, else 30.",
    )
    parser.add_argument(
        "--max_depth_m",
        type=float,
        default=None,
        help="Clip depth_linear for SG-Nav (m). Default from SIMULATOR.DEPTH_SENSOR.MAX_DEPTH if set, else 10.",
    )
    parser.add_argument(
        "--obs_width",
        type=int,
        default=None,
        help="Resize RGB/depth width for OG VisionSensor + HTTP payload. Default from SIMULATOR.RGB_SENSOR.WIDTH if set, else 640.",
    )
    parser.add_argument(
        "--obs_height",
        type=int,
        default=None,
        help="Resize height. Default from SIMULATOR.RGB_SENSOR.HEIGHT if set, else 480.",
    )
    parser.add_argument(
        "--vision_horizontal_aperture",
        type=float,
        default=None,
        help="OmniGibson VisionSensor horizontal_aperture (native units). "
        "Default from yaml SGNAV_RUNTIME.behavior_vision_horizontal_aperture if set; else OG default.",
    )
    parser.add_argument(
        "--vision_hfov_deg",
        type=float,
        default=None,
        help="Target camera HFOV in degrees. If horizontal_aperture is not set, convert HFOV + focal_length "
        "to VisionSensor horizontal_aperture. Default from SIMULATOR.RGB_SENSOR.HFOV if present.",
    )
    parser.add_argument(
        "--vision_focal_length_mm",
        type=float,
        default=None,
        help="VisionSensor focal length (mm) used when converting --vision_hfov_deg to horizontal_aperture. "
        "Default 17.0, or yaml SGNAV_RUNTIME.behavior_vision_focal_length_mm if set.",
    )
    parser.add_argument(
        "--nav_debug_path",
        type=str,
        default=None,
        help="If set, append one JSON object per step (rgb/depth stats, gps, compass, action). "
        "May be a .jsonl file or an existing directory (uses sgnav_nav_debug.jsonl inside it).",
    )
    parser.add_argument(
        "--debug_action_semantics",
        action="store_true",
        help="Print one-time controller snapshots and per-step action vs no-op slice diffs to diagnose "
        "position-mode action semantics without changing control behavior.",
    )
    parser.add_argument(
        "--look_pitch_step_deg",
        type=float,
        default=None,
        help="LOOK_UP/LOOK_DOWN delta (deg) on torso_joint4. Default from SIMULATOR.TILT_ANGLE if set, else 30.",
    )
    parser.add_argument(
        "--physical_look_pitch_step_deg",
        type=float,
        default=None,
        help="Actual trunk pitch delta (deg) used to approximate LOOK_UP/LOOK_DOWN in OmniGibson. "
        "Defaults to yaml SGNAV_RUNTIME.behavior_physical_look_pitch_step_deg if set, else "
        "min(--look_pitch_step_deg, 8deg) to reduce body-motion side effects.",
    )
    parser.add_argument(
        "--ignore_bddl_termination",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default on: do not end the segment when the Behavior BDDL goal is satisfied; "
        "stop only on --distance_threshold (early stop), env timeout (truncated), or --max_steps. "
        "Success is min distance to target object vs threshold, not task completion.",
    )
    parser.add_argument(
        "--use_og_occupancy",
        action="store_true",
        help="Attach OmniGibson ScanSensor (scan+occupancy_grid), send local OG to SG-Nav HTTP /step as og_occupancy "
        "to fuse into collision_map (requires robot USD with a range sensor prim).",
    )
    parser.add_argument(
        "--og_occupancy_resolution",
        type=int,
        default=256,
        help="ScanSensor occupancy_grid_resolution (must match meta sent to SG-Nav).",
    )
    parser.add_argument(
        "--og_occupancy_range_m",
        type=float,
        default=5,
        help="ScanSensor occupancy_grid_range in meters (local grid extent).",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=MOVE_TO_SUCCESS_DISTANCE_THRESHOLD,
        help="Meters: success and early-stop when min arm-to-target distance falls below this.",
    )
    parser.add_argument(
        "--physics_steps_per_nav_step",
        type=int,
        default=None,
        help="OmniGibson env.step count per SG-Nav discrete action (same velocity command each time). "
        "If unset, read branch-specific defaults from yaml, else fall back to 15. "
        "e.g. 8 ≈ 8× physics integration per nav decision → larger motion per HTTP /step. "
        "LOOK_UP/DOWN (4/5) always use 1 to avoid K× torso pitch delta. In "
        "--base_action_mode=position this repeats the configured absolute base step that many times "
        "per nav action, effectively amplifying total motion.",
    )
    parser.add_argument(
        "--send-camera-pose-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), POST /step includes camera_pose_world so SG-Nav maps "
        "depth with measured pitch/height. By default this uses the exact head VisionSensor world pose; "
        "fallback / override via --camera_pose_source. Tune mapping via server yaml "
        "SGNAV_RUNTIME.camera_* keys.",
    )
    parser.add_argument(
        "--camera_pose_source",
        type=str,
        choices=["sensor_world", "head_link_extrinsic"],
        default="sensor_world",
        help="Source of camera_pose_world sent to SG-Nav. "
        "'sensor_world' uses the exact VisionSensor world pose (recommended). "
        "'head_link_extrinsic' composes the parent head-link world pose with the fixed local head-camera extrinsics.",
    )
    parser.add_argument(
        "--head-link-name",
        type=str,
        default="zed_link",
        help="Expected parent link name for the head camera pose composition. Must match the robot-type mapping "
        "used by OmniGibson teleop (R1Pro default: zed_link), otherwise the script raises.",
    )
    args = parser.parse_args()

    cfg_path: str | None = None
    if args.no_sgnav_yaml:
        cfg_path = None
    elif args.sgnav_config:
        cfg_path = args.sgnav_config
    else:
        cfg_path = _default_sgnav_yaml_path()

    yaml_defaults: dict = {}
    if cfg_path:
        yp = Path(cfg_path)
        if yp.is_file():
            yaml_defaults = eval_defaults_from_sgnav_yaml(yp)
            logger.info("Eval bridge defaults from %s: %s", yp, yaml_defaults or "(empty)")
        else:
            logger.warning("sgnav_config not found (ignored): %s", yp)

    merged = _merge_eval_yaml_and_args(yaml_defaults, args)
    logger.info(
        "Eval merged cfg: base_action_mode=%s obs=%sx%s max_depth_m=%s vision_horizontal_aperture=%s vision_hfov_deg=%s vision_focal_length_mm=%s",
        merged["base_action_mode"],
        int(merged["obs_width"]),
        int(merged["obs_height"]),
        float(merged["max_depth_m"]),
        merged.get("vision_horizontal_aperture"),
        merged.get("vision_hfov_deg"),
        merged.get("vision_focal_length_mm"),
    )

    client = SGNavHTTPClient(args.sgnav_url)
    if args.url_health:
        print(client.health())
        return

    gm.HEADLESS = True

    task_filter = [t.lower().replace(" ", "_") for t in args.task] if args.task else None

    online_sampling = True
    if args.no_online_sampling:
        online_sampling = False
    partial_scene_load = args.partial_scene_load
    rft_style_tro = False
    use_annotation_object_lookup = False

    if args.tro_state_path is not None and online_sampling:
        online_sampling = False
        logger.info(
            "Forcing offline tro_state mode because --tro_state_path=%s was provided.",
            os.path.abspath(str(args.tro_state_path)),
        )

    if args.move_to_segments_json:
        online_sampling = False
        partial_scene_load = not args.full_scene_load
        rft_style_tro = True
        use_annotation_object_lookup = True
        segments = ef._load_move_to_segments_json(Path(args.move_to_segments_json))
        ef._finalize_move_to_segments_for_eval(segments)
        if task_filter:
            ts = set(task_filter)
            segments = [s for s in segments if s["task_name"].lower().replace(" ", "_") in ts]
    elif args.skills_flat:
        if args.online_sampling is not None and not args.no_online_sampling:
            online_sampling = args.online_sampling
        segments = ef._load_segments_from_skills_flat_for_bddl(
            Path(args.skills_flat),
            ef.TASK_MAPPING_PATH,
            tasks=task_filter,
            use_skill_prompt_format=False,
            offline=not online_sampling,
        )
    else:
        if args.online_sampling is not None and not args.no_online_sampling:
            online_sampling = args.online_sampling
        available_tasks = load_available_tasks()
        segments = ef.extract_segments_from_bddl(
            available_tasks,
            tasks=task_filter,
            max_objects_per_activity=args.max_objects_per_activity,
        )

    if args.object_id_contains:
        sub = args.object_id_contains.lower()
        segments = [s for s in segments if sub in str(s.get("object_id", "")).lower()]
        if not segments:
            logger.error("No segments left after --object_id_contains %r", args.object_id_contains)
            sys.exit(1)

    if args.instance_id is not None:
        logger.info("Overriding all segments to instance_id=%d", int(args.instance_id))
        for seg in segments:
            seg["instance_id"] = int(args.instance_id)
        if online_sampling:
            logger.warning(
                "--instance_id=%d was provided, but online_sampling=True. "
                "BehaviorTask ignores activity_instance_id in online sampling mode, so this will not replay a fixed tro_state. "
                "Use --no_online_sampling or --tro_state_path for exact instance reuse.",
                int(args.instance_id),
            )

    logger.info(
        "Eval sampling mode: online_sampling=%s partial_scene_load=%s move_to_segments_json=%s",
        bool(online_sampling),
        bool(partial_scene_load),
        bool(args.move_to_segments_json),
    )

    segments = segments[: args.total_segments]
    if not segments:
        logger.error("No segments")
        sys.exit(1)

    env_wrapper_cfg = None
    if args.env_wrapper:
        try:
            from omegaconf import OmegaConf

            env_wrapper_cfg = OmegaConf.create(
                {"_target_": "omnigibson.learning.wrappers.RGBLowResWrapper"}
            )
        except Exception as e:
            logger.warning(f"env_wrapper requested but unavailable: {e}")

    nav_debug_path = args.nav_debug_path
    if nav_debug_path:
        nav_debug_path = os.path.abspath(nav_debug_path)
        if os.path.isdir(nav_debug_path):
            nav_debug_path = os.path.join(nav_debug_path, "sgnav_nav_debug.jsonl")

    runner = SGNavHTTPRunner(
        args.sgnav_url,
        env_wrapper_cfg=env_wrapper_cfg,
        online_sampling=online_sampling,
        partial_scene_load=partial_scene_load,
        rft_style_tro=rft_style_tro,
        use_annotation_object_lookup=use_annotation_object_lookup,
        obs_width=int(merged["obs_width"]),
        obs_height=int(merged["obs_height"]),
        map_size_cm=float(merged["map_size_cm"]),
        forward_vel=float(merged["forward_vel"]),
        turn_vel=float(merged["turn_vel"]),
        max_depth_m=float(merged["max_depth_m"]),
        nav_debug_path=nav_debug_path,
        look_pitch_step_rad=math.radians(float(merged["look_pitch_step_deg"])),
        physical_look_pitch_step_rad=(
            None
            if merged.get("physical_look_pitch_step_deg") is None
            else math.radians(float(merged["physical_look_pitch_step_deg"]))
        ),
        ignore_bddl_termination=args.ignore_bddl_termination,
        use_og_occupancy=args.use_og_occupancy,
        og_occupancy_resolution=args.og_occupancy_resolution,
        og_occupancy_range_m=args.og_occupancy_range_m,
        vision_horizontal_aperture=merged.get("vision_horizontal_aperture"),
        vision_hfov_deg=merged.get("vision_hfov_deg"),
        vision_focal_length_mm=float(merged.get("vision_focal_length_mm", 17.0)),
        physics_steps_per_nav_step=int(merged["physics_steps_per_nav_step"]),
        position_forward_physics_steps_per_nav_step=int(
            merged["position_forward_physics_steps_per_nav_step"]
        ),
        position_turn_physics_steps_per_nav_step=int(
            merged["position_turn_physics_steps_per_nav_step"]
        ),
        position_forward_guard_enable=bool(merged["position_forward_guard_enable"]),
        position_forward_guard_step_delta_m=float(
            merged["position_forward_guard_step_delta_m"]
        ),
        position_forward_guard_roll_pitch_deg=float(
            merged["position_forward_guard_roll_pitch_deg"]
        ),
        position_forward_guard_consecutive_contacts=int(
            merged["position_forward_guard_consecutive_contacts"]
        ),
        position_forward_guard_heading_error_deg=float(
            merged["position_forward_guard_heading_error_deg"]
        ),
        send_camera_pose_world=bool(args.send_camera_pose_world),
        head_link_name=str(args.head_link_name),
        camera_pose_source=str(args.camera_pose_source),
        base_action_mode=str(merged["base_action_mode"]),
        forward_step_size_m=float(merged["forward_step_size_m"]),
        turn_angle_deg=float(merged["turn_angle_deg"]),
        debug_action_semantics=bool(args.debug_action_semantics),
        instance_id_override=args.instance_id,
        fail_on_asset_hash_mismatch=bool(args.fail_on_asset_hash_mismatch),
        tro_state_path=args.tro_state_path,
        save_tro_state_to=args.save_tro_state_to,
    )

    for i, seg in enumerate(segments):
        tn = seg["task_name"]
        instance_id = _resolve_segment_instance_id(seg, args.instance_id)
        logger.info(
            f"=== Segment {i+1}/{len(segments)} task={tn} object={seg.get('object_id')} "
            f"instance_id={instance_id} ==="
        )
        runner.load_env(tn, max_steps=args.max_steps + 100)
        res = runner.run_segment(
            seg,
            max_steps=args.max_steps,
            distance_threshold=args.distance_threshold,
        )
        logger.info(f"result: {res}")
        # Always echo to stdout so nohup/terminals without INFO still show why the segment ended.
        print(f"[eval_skill_sgnav_http] segment {i + 1}/{len(segments)} result: {res}", flush=True)
        if runner.env is not None:
            runner.env.close()
            runner.env = None
    og.shutdown()


if __name__ == "__main__":
    main()
