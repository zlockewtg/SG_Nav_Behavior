"""Visualization helpers for SG-Nav.

Extracted from SG_Nav_Agent to keep the main navigation class focused
on planning. Functions here build the occupancy panel, detection overlays,
scene-graph text, and write video / JSON snapshots.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from matplotlib import colors

from utils.geometry import apply_flipud
from utils.image_process import (
    add_resized_image,
    add_resized_image_contain,
    add_rectangle,
    add_text,
    add_text_list,
)


# ── Occupancy panel ──────────────────────────────────────────────────

_CROP_RATIO = 0.30  # fraction of map canvas kept after zoom

_UNKNOWN_RGB = np.array(colors.to_rgb("#FFFFFF"), dtype=np.float64)
_FREE_RGB = np.array(colors.to_rgb("#E7E7E7"), dtype=np.float64)
_DEFAULT_FREE_RGB = np.array(colors.to_rgb("#D6D6D6"), dtype=np.float64)
_OBSTACLE_RGB = np.array(colors.to_rgb("#A2A2A2"), dtype=np.float64)


def build_occupancy_panel(
    full_map_np: np.ndarray,
    fbe_free_map_np: np.ndarray,
    collision_map_np: np.ndarray,
    default_free_map_np: np.ndarray | None,
    occ_threshold: float,
    free_threshold: float,
    robot_rc: tuple[int, int],
    robot_radius_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3-colour occupancy image (uint8 HxWx3).

    Returns ``(obs_mask, free_mask, panel_hw3_uint8)``.
    """
    fm = apply_flipud(full_map_np)
    fb = apply_flipud(fbe_free_map_np)
    coll = apply_flipud(collision_map_np)

    obs_mask = np.logical_or(fm > occ_threshold, coll > 0.5)
    free_mask_raw = fb > free_threshold
    shown_free = np.logical_and(free_mask_raw, ~obs_mask)

    h, w = fm.shape[:2]
    shown_default_free = np.zeros((h, w), dtype=bool)
    if default_free_map_np is not None:
        df = apply_flipud(default_free_map_np)
        shown_default_free = np.logical_and(df > 0.5, ~obs_mask)
    shown_free_total = np.logical_or(shown_free, shown_default_free)
    unknown_mask = np.logical_and(~obs_mask, ~shown_free_total)

    panel = np.tile(_UNKNOWN_RGB, (h, w, 1))
    panel[shown_free] = _FREE_RGB
    panel[shown_default_free] = _DEFAULT_FREE_RGB
    panel[obs_mask] = _OBSTACLE_RGB
    return obs_mask, shown_free_total, (panel * 255).astype(np.uint8)


# ── Agent / goal overlay on map tensor ───────────────────────────────

def paint_agent_and_goal(
    map_tensor_3hw: torch.Tensor,
    history_pose: list,
    goal_map: np.ndarray | None,
    planner_stg: tuple[int, int] | None,
    map_size_cm: int,
    resolution: int,
) -> tuple[int, int] | None:
    """Draw trajectory + goal + STG on occupancy panel tensor.  Returns STG rc or None."""
    h, w = int(map_tensor_3hw.shape[1]), int(map_tensor_3hw.shape[2])

    def _paint(cr, cc, size, rgb, alpha=1.0):
        y0, y1 = max(0, cr - size), min(h, cr + size)
        x0, x1 = max(0, cc - size), min(w, cc + size)
        if y1 <= y0 or x1 <= x0:
            return
        ori = map_tensor_3hw[:, y0:y1, x0:x1]
        new = torch.zeros_like(ori)
        for ch, v in enumerate(rgb):
            new[ch] = float(v)
        map_tensor_3hw[:, y0:y1, x0:x1] = alpha * new + (1 - alpha) * ori

    for idx, pose in enumerate(history_pose):
        draw_step_num = 30
        alpha = max(0.0, 1.0 - (len(history_pose) - idx) / draw_step_num)
        sz = 2 if idx == len(history_pose) - 1 else 1
        px = float(pose[0].item()) if hasattr(pose[0], "item") else float(pose[0])
        py = float(pose[1].item()) if hasattr(pose[1], "item") else float(pose[1])
        rx = int(np.clip(int(px * 100.0 / resolution), 0, w - 1))
        ry = int(np.clip(int((map_size_cm / 100.0 - py) * 100.0 / resolution), 0, h - 1))
        _paint(ry, rx, sz, (1.0, 0.0, 0.0), alpha)

    if goal_map is not None:
        gv = apply_flipud(np.asarray(goal_map))
        ys, xs = np.where(gv == 1)
        if len(ys) > 0:
            _paint(int(ys[0]), int(xs[0]), 2, (0.0, 1.0, 0.0), 1.0)

    stg_plan_rc = None
    if planner_stg is not None:
        sr = int(np.clip(planner_stg[0], 0, h - 1))
        sc = int(np.clip(planner_stg[1], 0, w - 1))
        _paint(sr, sc, 2, (1.0, 1.0, 0.0), 1.0)
        stg_plan_rc = (sr, sc)
    return stg_plan_rc


# ── Detection overlay ────────────────────────────────────────────────

def render_detection_overlay(
    rgb_img: np.ndarray,
    predictions,
    obj_goal: str,
    label_match_fn,
) -> np.ndarray:
    """RGB image with green (other) / red (navigation) GLIP boxes."""
    vis = np.asarray(rgb_img).copy()
    if predictions is None or len(predictions) == 0:
        return vis

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    labels = []
    if hasattr(predictions, "get_field"):
        try:
            labels = list(predictions.get_field("labels"))
        except Exception:
            pass
    scores = None
    if hasattr(predictions, "get_field"):
        try:
            scores = predictions.get_field("scores")
        except Exception:
            pass

    boxes = predictions.bbox if hasattr(predictions, "bbox") else []
    n = min(len(boxes), len(labels) if labels else len(boxes))
    nav_count = 0
    for i in range(n):
        try:
            box = boxes[i].to(torch.int64)
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        except Exception:
            continue
        x1 = max(0, min(vis_bgr.shape[1] - 1, x1))
        y1 = max(0, min(vis_bgr.shape[0] - 1, y1))
        x2 = max(0, min(vis_bgr.shape[1] - 1, x2))
        y2 = max(0, min(vis_bgr.shape[0] - 1, y2))

        raw_lab = labels[i] if i < len(labels) else "obj"
        is_nav = label_match_fn(raw_lab)
        if is_nav:
            nav_count += 1
        color = (0, 0, 255) if is_nav else (0, 255, 0)
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), color, 2)
        lab = str(raw_lab)
        if scores is not None:
            try:
                lab = f"{lab}:{float(scores[i]):.2f}"
            except Exception:
                pass
        if is_nav:
            lab = "[nav] " + lab
        cv2.putText(vis_bgr, lab, (x1, max(12, y1 - 4)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    h, w = vis_bgr.shape[:2]
    hint = f"RED=nav({nav_count}) GREEN=other | obj_goal={obj_goal!r}"
    cv2.putText(vis_bgr, hint, (4, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


# ── Scene graph text for visualization panel ─────────────────────────

def build_scenegraph_text(scenegraph, show_frame_only: bool):
    """Return (text_node, text_edge, explanation)."""
    nodes = getattr(scenegraph, "nodes", [])
    global_caps = [n.caption for n in nodes if getattr(n, "caption", None)]

    frame_caps = []
    seg = getattr(scenegraph, "segment2d_results", [])
    if len(seg) > 0 and isinstance(seg[-1], dict):
        frame_caps = [str(c) for c in seg[-1].get("caption", []) if str(c).strip()]

    if show_frame_only:
        src = frame_caps if frame_caps else global_caps
        text_node = ", ".join(src)
    else:
        text_node = f"[frame] {', '.join(frame_caps[:24]) or '-'} | [global] {', '.join(global_caps[:24]) or '-'}"

    edge_lines, seen = [], set()
    for node in nodes:
        for edge in getattr(node, "edges", []):
            rel = getattr(edge, "relation", None)
            n1 = getattr(edge.node1, "caption", None)
            n2 = getattr(edge.node2, "caption", None)
            if not n1 or not n2 or not rel:
                continue
            key = tuple(sorted([n1, n2]) + [rel])
            if key in seen:
                continue
            seen.add(key)
            edge_lines.append(f"({n1}, {rel}, {n2})")
    text_edge = "; ".join(edge_lines)

    reason = getattr(scenegraph, "reason_visualization", "")
    if reason:
        explanation = reason
    elif text_edge:
        explanation = "Edge relations are derived from VLM/LLM proposals."
    elif text_node:
        skip = False
        rc = getattr(scenegraph, "runtime_cfg", None)
        if rc and rc.get("skip_edge_llm", False):
            skip = True
        tail = " | edges off (skip_edge_llm)" if skip else ""
        explanation = f"{text_node[:300]}{tail}"
    else:
        explanation = "Scene graph is empty for the current frame."

    return text_node, text_edge, explanation


# ── Composite frame builder ──────────────────────────────────────────

def compose_visualization_frame(
    *,
    occ_panel: np.ndarray,
    det_rgb: np.ndarray,
    robot_rc: tuple[int, int],
    stg_plan_rc: tuple[int, int] | None,
    obj_goal: str,
    goal_distance_str: str,
    text_node: str,
    text_edge: str,
    explanation: str,
    goal_map_src: str,
) -> np.ndarray:
    """Build the full 360x800 visualization image."""
    ph, pw = occ_panel.shape[:2]
    acy, acx = robot_rc
    acx = max(0, min(pw - 1, acx))
    acy = max(0, min(ph - 1, acy))

    cv2.circle(occ_panel, (acx, acy), 4, (255, 0, 0), -1)

    occupancy_map = np.full_like(occ_panel, 255)
    cx, cy = pw // 2, ph // 2
    sx, sy = cx - acx, cy - acy
    src_l, src_t = max(0, -sx), max(0, -sy)
    dst_l, dst_t = max(0, sx), max(0, sy)
    cw = min(pw - src_l, pw - dst_l)
    ch = min(ph - src_t, ph - dst_t)
    if cw > 0 and ch > 0:
        occupancy_map[dst_t:dst_t + ch, dst_l:dst_l + cw] = occ_panel[src_t:src_t + ch, src_l:src_l + cw]

    if stg_plan_rc is not None:
        stg_x_s = int(stg_plan_rc[1]) + sx
        stg_y_s = int(stg_plan_rc[0]) + sy
        if 0 <= stg_x_s < pw and 0 <= stg_y_s < ph:
            cv2.drawMarker(occupancy_map, (stg_x_s, stg_y_s), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=11, thickness=2, line_type=cv2.LINE_AA)

    crop_side = max(41, int(round(float(min(ph, pw)) * _CROP_RATIO)))
    cr = min(crop_side // 2, cx, cy, pw - 1 - cx, ph - 1 - cy)
    occ_zoom = occupancy_map[max(0, cy - cr):min(ph, cy + cr + 1), max(0, cx - cr):min(pw, cx + cr + 1)]

    panel_top, panel_bottom = 45, 285
    vis = np.full((360, 800, 3), 255, dtype=np.uint8)
    vis = add_resized_image(vis, det_rgb, (10, panel_top), (320, 240))
    vis = add_resized_image_contain(vis, occ_zoom, (340, panel_top), (180, 240),
                                     interpolation_up=cv2.INTER_NEAREST, interpolation_down=cv2.INTER_AREA)
    vis = add_rectangle(vis, (10, panel_top), (330, panel_bottom), (128, 128, 128), thickness=1)
    vis = add_rectangle(vis, (340, panel_top), (520, panel_bottom), (128, 128, 128), thickness=1)
    vis = add_rectangle(vis, (540, panel_top), (790, 160), (128, 128, 128), thickness=1)
    vis = add_rectangle(vis, (540, 170), (790, panel_bottom), (128, 128, 128), thickness=1)
    vis = add_rectangle(vis, (10, 295), (790, 350), (128, 128, 128), thickness=1)

    vis = add_text(vis, f"Observation (Goal: {obj_goal},  dist={goal_distance_str})", (50, 36), font_scale=0.5, thickness=1)
    vis = add_text(vis, "Occupancy (unknown/free/obs)", (360, 36), font_scale=0.5, thickness=1)
    vis = add_text(vis, "Scene Graph Nodes", (580, 36), font_scale=0.5, thickness=1)
    vis = add_text(vis, "Scene Graph Edges", (580, 162), font_scale=0.5, thickness=1)
    vis = add_text(vis, "LLM Explanation", (330, 286), font_scale=0.5, thickness=1)
    from utils.image_process import line_list
    vis = add_text_list(vis, line_list(text_node, 40), (550, 64), font_scale=0.3, thickness=1)
    vis = add_text_list(vis, line_list(text_edge, 40), (550, 190), font_scale=0.3, thickness=1)
    vis = add_text_list(vis, line_list(explanation, 150), (20, 314), font_scale=0.3, thickness=1)
    return vis


# ── Video save ───────────────────────────────────────────────────────

def save_video(frames: list[np.ndarray], save_dir: str, episode_idx: int):
    """Write MP4 with optional ffmpeg H.264 transcode."""
    if not frames:
        return
    os.makedirs(save_dir, exist_ok=True)
    base = f"vid_{episode_idx:06d}"
    final = os.path.join(save_dir, f"{base}.mp4")
    tmp = os.path.join(save_dir, f"{base}.partial.mp4")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(tmp, fourcc, 4.0, (w, h))
    if not video.isOpened():
        return
    try:
        for f in frames:
            video.write(f)
    finally:
        video.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264 = os.path.join(save_dir, f"{base}.h264.tmp.mp4")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp, "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
               "-movflags", "+faststart", h264]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode == 0 and os.path.isfile(h264) and os.path.getsize(h264) > 0:
                os.remove(tmp)
                os.replace(h264, final)
                return
        except OSError:
            pass
        if os.path.isfile(h264):
            try:
                os.remove(h264)
            except OSError:
                pass
    os.replace(tmp, final)


# ── Scene graph JSON snapshot ────────────────────────────────────────

def save_scenegraph_json(scenegraph, step: int, navigate_step: int,
                          obj_goal: str, obj_goal_sg: str,
                          found_goal: bool, found_possible_goal: bool,
                          json_path: str, step_dir: str):
    nodes = list(getattr(scenegraph, "nodes", []))
    nidx = {id(n): i for i, n in enumerate(nodes)}
    jn = []
    for i, n in enumerate(nodes):
        center = getattr(n, "center", None)
        room = getattr(n, "room_node", None)
        jn.append({
            "id": i,
            "caption": str(getattr(n, "caption", "") or ""),
            "center": [int(center[0]), int(center[1])] if center is not None else None,
            "room": str(getattr(room, "caption", "") or ""),
            "is_goal_node": bool(getattr(n, "is_goal_node", False)),
        })
    seen, je = set(), []
    for n in nodes:
        for e in getattr(n, "edges", []):
            eid = id(e)
            if eid in seen:
                continue
            seen.add(eid)
            n1 = getattr(e, "node1", None)
            n2 = getattr(e, "node2", None)
            if n1 is None or n2 is None:
                continue
            je.append({
                "node1_id": nidx.get(id(n1)),
                "node1_caption": str(getattr(n1, "caption", "") or ""),
                "node2_id": nidx.get(id(n2)),
                "node2_caption": str(getattr(n2, "caption", "") or ""),
                "relation": getattr(e, "relation", None),
            })

    payload = {
        "step": step, "navigate_step": navigate_step,
        "obj_goal": obj_goal, "obj_goal_sg": obj_goal_sg,
        "found_goal": found_goal, "found_possible_goal": found_possible_goal,
        "nodes": jn, "edges": je,
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(step_dir, exist_ok=True)
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)
    sp = os.path.join(step_dir, f"scenegraph_{step:06d}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
