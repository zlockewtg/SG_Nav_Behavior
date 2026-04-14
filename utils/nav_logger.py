"""Unified per-step JSON logger for SG-Nav navigation.

Replaces 50+ scattered print() calls with a single JSONL file.
Terminal output is limited to one ``[SG-Nav][stage]`` line per step.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    try:
        import numpy as np
        if isinstance(value, np.floating):
            fv = float(value)
            return None if not math.isfinite(fv) else fv
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return value


ACTION_NAMES = ("STOP", "FWD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "TURN")


def action_name(action_id: int) -> str:
    if 0 <= action_id < len(ACTION_NAMES):
        return ACTION_NAMES[action_id]
    return str(action_id)


class NavLogger:
    """Append-only JSONL logger.  One record per navigation step."""

    def __init__(self, log_path: str):
        self._path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._current: dict[str, Any] = {}

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------

    def begin_step(self, step: int, episode: int = 0):
        self._current = {
            "step": step,
            "episode": episode,
            "timestamp_ms": int(time.time() * 1000),
        }

    def set(self, key: str, value: Any):
        self._current[key] = value

    def set_nested(self, section: str, key: str, value: Any):
        if section not in self._current:
            self._current[section] = {}
        self._current[section][key] = value

    def add_flag(self, flag: str):
        if "flags" not in self._current:
            self._current["flags"] = []
        self._current["flags"].append(flag)

    def flush_step(self):
        """Write current record to JSONL and print stage to terminal."""
        record = _json_safe(self._current)
        if not record:
            return
        if "flags" not in record:
            record["flags"] = []

        stage = record.get("stage", {})
        act = record.get("action", {})
        stage_code = stage.get("code", "unknown") if isinstance(stage, dict) else "unknown"
        act_id = act.get("id", "?") if isinstance(act, dict) else "?"
        act_nm = act.get("name", "?") if isinstance(act, dict) else "?"
        step = record.get("step", "?")
        print(
            f"[SG-Nav][stage] step={step} stage={stage_code} "
            f"action={act_id}({act_nm})",
            flush=True,
        )

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._current = {}

    # ------------------------------------------------------------------
    # Special event records (not per-step)
    # ------------------------------------------------------------------

    def log_event(self, event: str, **kwargs):
        record = _json_safe({
            "event": event,
            "timestamp_ms": int(time.time() * 1000),
            **kwargs,
        })
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Convenience setters for common sections
    # ------------------------------------------------------------------

    def set_stage(
        self,
        code: str,
        label: str,
        goal_map_src: str,
        found_goal: bool,
        found_possible_goal: bool,
        using_random_goal: bool,
        active_gt_world_nav: bool,
    ):
        self._current["stage"] = {
            "code": code,
            "label": label,
            "goal_map_src": goal_map_src,
            "found_goal": found_goal,
            "found_possible_goal": found_possible_goal,
            "using_random_goal": using_random_goal,
            "active_gt_world_nav": active_gt_world_nav,
        }

    def set_action(self, action_id: int, raw_id: int | None = None, forward_guard_blocked: bool = False):
        self._current["action"] = {
            "id": action_id,
            "name": action_name(action_id),
            "raw_id": raw_id if raw_id is not None else action_id,
            "forward_guard_blocked": forward_guard_blocked,
        }

    def set_pose(self, full_pose_m_deg, gps, compass_rad: float, compass_offset_applied: bool = True):
        self._current["pose"] = {
            "full_pose_m_deg": [round(float(x), 3) for x in full_pose_m_deg[:3]],
            "gps": [round(float(x), 4) for x in gps[:2]],
            "compass_rad": round(float(compass_rad), 4),
            "compass_offset_applied": compass_offset_applied,
        }

    def set_goal(self, goal_rc, goal_drift, goal_gps=None, distance_m=None, heading_error_deg=None):
        self._current["goal"] = {
            "goal_rc": list(goal_rc) if goal_rc is not None else None,
            "goal_drift": goal_drift,
            "goal_gps": [round(float(x), 4) for x in goal_gps[:2]] if goal_gps is not None else None,
            "goal_distance_m": round(float(distance_m), 3) if distance_m is not None else None,
            "heading_error_deg": round(float(heading_error_deg), 2) if heading_error_deg is not None else None,
        }

    def set_detection(self, ran: bool, goal_bbox_count: int = 0, goal_votes_total: int = 0, found_this_step: bool = False):
        self._current["detection"] = {
            "ran_this_step": ran,
            "goal_bbox_count": goal_bbox_count,
            "goal_votes_total": goal_votes_total,
            "found_goal_this_step": found_this_step,
        }

    def set_mapping(
        self,
        depth_min: float,
        depth_max: float,
        depth_median: float,
        depth_zero_frac: float,
        full_map_occ: float,
        collision_occ: float,
        free_map_occ: float,
        traversible_free: float,
        suspend_active: bool = False,
        suspend_countdown: int = 0,
    ):
        self._current["mapping"] = {
            "depth_stats": {
                "min": round(depth_min, 3),
                "max": round(depth_max, 3),
                "median_positive": round(depth_median, 3),
                "zero_fraction": round(depth_zero_frac, 3),
            },
            "full_map_occ_ratio": round(full_map_occ, 4),
            "collision_occ_ratio": round(collision_occ, 4),
            "free_map_occ_ratio": round(free_map_occ, 4),
            "traversible_free_ratio": round(traversible_free, 4),
            "suspend_active": suspend_active,
            "suspend_countdown": suspend_countdown,
        }

    def set_camera(
        self,
        active: bool,
        source: str = "unknown",
        tilt_source: str = "",
        pitch_deg_used: float = 0.0,
        height_cm_used: float = 0.0,
        roll_deg: float = 0.0,
        rejected: bool = False,
        fall_suspected: bool = False,
    ):
        self._current["camera"] = {
            "active": active,
            "source": source,
            "tilt_source": tilt_source,
            "pitch_deg_used": round(pitch_deg_used, 2),
            "height_cm_used": round(height_cm_used, 1),
            "roll_deg": round(roll_deg, 2),
            "rejected": rejected,
            "fall_suspected": fall_suspected,
        }

    def set_planner(
        self,
        stg_rc=None,
        start_rc=None,
        start_o_deg: float = 0.0,
        relative_angle_deg: float = 0.0,
        align_thr_deg: float = 0.0,
        observed_turn_deg: float | None = None,
        collision_painted: bool = False,
        former_collide: int = 0,
    ):
        self._current["planner"] = {
            "stg_rc": list(stg_rc) if stg_rc is not None else None,
            "start_rc": list(start_rc) if start_rc is not None else None,
            "start_o_deg": round(start_o_deg, 2),
            "relative_angle_deg": round(relative_angle_deg, 2),
            "align_thr_deg": round(align_thr_deg, 1),
            "observed_turn_deg": round(observed_turn_deg, 2) if observed_turn_deg is not None else None,
            "collision_painted": collision_painted,
            "former_collide": former_collide,
        }

    def set_fbe(self, ran: bool, frontier_count=None, eligible_count=None, selected_goal_rc=None, score_max=None):
        self._current["fbe"] = {
            "ran_this_step": ran,
            "frontier_count": frontier_count,
            "eligible_count": eligible_count,
            "selected_goal_rc": list(selected_goal_rc) if selected_goal_rc is not None else None,
            "score_max": round(float(score_max), 3) if score_max is not None else None,
        }

    def set_scenegraph(self, updated: bool, perception_ran: bool, node_count: int = 0, edge_count: int = 0):
        self._current["scenegraph"] = {
            "updated_this_step": updated,
            "perception_ran": perception_ran,
            "node_count": node_count,
            "edge_count": edge_count,
        }
