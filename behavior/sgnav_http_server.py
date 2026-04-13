#!/usr/bin/env python3
"""
Run SG-Nav in an isolated conda/env behind a small HTTP API.

Start from SG-Nav repo root (e.g. conda activate SG_Nav):

  python behavior/sgnav_http_server.py --host 0.0.0.0 --port 8765

Endpoints (JSON):
  GET  /health  -> {"status":"ok"}
  POST /reset  -> {"behavior_goal": str} OR {"object_category": str, "object_category_sg": str?}
                  Calls SG_Nav_Agent.reset(...)
  POST /step   -> ObservationMsg (see _parse_step_body) -> {"action": int, "total_steps": int}
                 Optional: ``camera_pose_world``: { "position": [x,y,z] m, "quaternion": [x,y,z,w] } (e.g. zed_link).
                 When set, SG_Nav syncs depth→map pitch/height before update_map (see SGNAV_RUNTIME.camera_*).
                 Set SGNAV_RUNTIME.log_camera_pose_world: true to print received pose to stderr each step.
                 Optional: og_occupancy (same base64 array layout as rgb), og_occupancy_meta: {resolution, range_m}
                 fuses OmniGibson ScanSensor-style local OG into SG-Nav collision_map.
                 Optional ``target_world_xy``: [x_m, y_m] (same world frame as client ``habitat_gps_compass``).
                 If ``SGNAV_RUNTIME.gt_pathplan_when_target_on_map: true`` and the cell is explored on the map,
                 the agent plans toward that point via FMM.

See ../openpi-comet/scripts/eval_skill_sgnav_http.py for the OmniGibson client.

Runtime options are configured in the yaml config file under ``SGNAV_RUNTIME``:

  detect_interval: 4
  scenegraph_update_interval: 1
  planner_align_angle_deg: 26.0
  planner_face_goal_max_error_deg: 15.0   # -1 = wide cone only; >=0 = face STG within this deg before FORWARD
  panorama_spin_until_step: 22
  goal_distance_threshold: 8.0
  show_frame_nodes_only: true
  glip_append_goal_sg: true
  glip_extra_captions: "radio,radio receiver"
  log_pose_align_trace: false   # true → stderr [SG-Nav][pose_align] (compass vs STG bearing)
  scenegraph:
    skip_edge_llm: true
    edge_max_per_step: 24
    edge_llm_batch_size: 8
    edge_llm_timeout_s: 20.0
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


def _configure_live_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


_configure_live_stdio()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _AgentHolder:
    lock = threading.Lock()
    agent = None
    args = None
    config_path: Optional[str] = None
    warmed_up = False
    warmup_thread: Optional[threading.Thread] = None


def _flush_episode_video_if_needed(agent) -> None:
    """SG-Nav writes mp4 in save_video(); HTTP mode never calls update_metrics, so we flush explicitly."""
    if agent is None:
        return
    if getattr(agent, "args", None) is None or not agent.args.visualize:
        return
    if len(agent.visualize_image_list) == 0:
        return
    try:
        agent.save_video()
        agent.visualize_image_list = []
    except Exception as e:
        sys.stderr.write(f"[sgnav_http_server] save_video failed: {e}\n")


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    ln = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(ln) if ln else b"{}"
    return json.loads(raw.decode("utf-8"))


def _decode_array_blob(obj: dict) -> np.ndarray:
    """Decode {"dtype": "uint8|float32", "shape": [H,W,...], "data": base64}."""
    dtype = np.dtype(obj["dtype"])
    shape = tuple(int(x) for x in obj["shape"])
    data = base64.standard_b64decode(obj["data"])
    arr = np.frombuffer(data, dtype=dtype)
    return arr.reshape(shape)


def _parse_step_body(data: dict) -> dict:
    """Build observation dict for SG_Nav_Agent.act."""
    rgb = _decode_array_blob(data["rgb"])
    depth = _decode_array_blob(data["depth"])
    if depth.ndim == 2:
        depth = depth[..., np.newaxis]
    # GLIP bbox centers are in RGB pixel space; depth must match H×W or indexing throws (e.g. y=851 vs H=800).
    rh, rw = int(rgb.shape[0]), int(rgb.shape[1])
    dh, dw = int(depth.shape[0]), int(depth.shape[1])
    if rh != dh or rw != dw:
        d2 = cv2.resize(
            depth[..., 0].astype(np.float32),
            (rw, rh),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = d2[..., np.newaxis]
    gps = np.asarray(data["gps"], dtype=np.float64).reshape(2)
    comp = np.asarray(data["compass"], dtype=np.float64).reshape(-1)
    if comp.size == 0:
        comp = np.array([0.0], dtype=np.float64)
    out = {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "gps": gps,
        "compass": comp,
    }
    og = data.get("og_occupancy")
    if isinstance(og, dict) and "data" in og:
        out["og_occupancy"] = _decode_array_blob(og).astype(np.float32, copy=False)
        meta = data.get("og_occupancy_meta")
        if isinstance(meta, dict):
            out["og_occupancy_meta"] = meta
    tw = data.get("target_world_xy")
    if isinstance(tw, (list, tuple)) and len(tw) >= 2:
        try:
            out["target_world_xy"] = [float(tw[0]), float(tw[1])]
        except (TypeError, ValueError):
            pass
    cp = data.get("camera_pose_world")
    if isinstance(cp, dict):
        pos = cp.get("position")
        quat = cp.get("quaternion")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3 and isinstance(quat, (list, tuple)) and len(quat) >= 4:
            try:
                out_cp = {
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                }
                src = cp.get("source")
                if isinstance(src, str) and src:
                    out_cp["source"] = src
                out["camera_pose_world"] = out_cp
            except (TypeError, ValueError):
                pass
    ci = data.get("collision_info")
    if isinstance(ci, dict):
        try:
            out["collision_info"] = {
                "base_contact": bool(ci.get("base_contact", False)),
                "n_contacts": int(ci.get("n_contacts", 0)),
                "contact_bodies": [str(x) for x in list(ci.get("contact_bodies", []))[:8]],
                "max_impulse": float(ci.get("max_impulse", 0.0)),
                "prev_action": (
                    None if ci.get("prev_action") is None else int(ci.get("prev_action"))
                ),
            }
        except (TypeError, ValueError):
            pass
    return out


def _ensure_agent(cfg_path: str, nav_args) -> None:
    if _AgentHolder.agent is not None:
        return
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    from omegaconf import OmegaConf
    from SG_Nav import SG_Nav_Agent

    cfg = OmegaConf.load(cfg_path)
    _AgentHolder.agent = SG_Nav_Agent(cfg, nav_args)
    _AgentHolder.agent.simulator = None


def _maybe_warmup_agent(force: bool = False, runs: int = 2):
    agent = _AgentHolder.agent
    if agent is None:
        return None
    if _AgentHolder.warmed_up and not force:
        return {"status": "already_warmed"}
    result = agent.warmup(runs=runs)
    _AgentHolder.warmed_up = True
    return result


def _start_background_warmup(cfg_path: str, nav_args, runs: int) -> bool:
    thr = _AgentHolder.warmup_thread
    if thr is not None and thr.is_alive():
        return False

    def _run():
        print(
            f"startup warmup begin: runs={max(1, int(runs))}",
            flush=True,
        )
        try:
            with _AgentHolder.lock:
                _ensure_agent(cfg_path, nav_args)
                result = _maybe_warmup_agent(force=True, runs=runs)
            print(f"startup warmup complete: {result}", flush=True)
        except Exception as e:
            print(
                f"startup warmup failed: {type(e).__name__}: {e}",
                flush=True,
            )

    thr = threading.Thread(target=_run, name="sgnav-startup-warmup", daemon=True)
    _AgentHolder.warmup_thread = thr
    thr.start()
    return True


class SGNavHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cfg_path: str = ""
    nav_args = None

    def log_message(self, format, *log_args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % log_args))

    def do_GET(self):
        if self.path.rstrip("/").endswith("/health") or self.path == "/health":
            _json_response(self, 200, {"status": "ok"})
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        if self.path.rstrip("/").endswith("/reset") or self.path == "/reset":
            self._handle_reset()
            return
        if self.path.rstrip("/").endswith("/step") or self.path == "/step":
            self._handle_step()
            return
        if self.path.rstrip("/").endswith("/warmup") or self.path == "/warmup":
            self._handle_warmup()
            return
        _json_response(self, 404, {"error": "not_found"})

    def _handle_warmup(self):
        try:
            data = _read_json_body(self)
        except json.JSONDecodeError as e:
            _json_response(self, 400, {"error": "invalid_json", "detail": str(e)})
            return
        runs = data.get("runs", getattr(self.nav_args, "warmup_runs", 2))
        try:
            runs = max(1, int(runs))
        except (TypeError, ValueError):
            _json_response(self, 400, {"error": "invalid_runs"})
            return
        force = bool(data.get("force", False))
        with _AgentHolder.lock:
            try:
                _ensure_agent(self.cfg_path, self.nav_args)
                result = _maybe_warmup_agent(force=force, runs=runs)
            except Exception as e:
                _json_response(self, 500, {"error": str(e), "type": type(e).__name__})
                return
        _json_response(self, 200, {"status": "warmed", "result": result})

    def _handle_reset(self):
        try:
            data = _read_json_body(self)
        except json.JSONDecodeError as e:
            _json_response(self, 400, {"error": "invalid_json", "detail": str(e)})
            return
        with _AgentHolder.lock:
            try:
                _ensure_agent(self.cfg_path, self.nav_args)
                agent = _AgentHolder.agent
                _flush_episode_video_if_needed(agent)
                from utils.behavior_taxonomy import behavior_goal_to_mp3d

                if "behavior_goal" in data and data["behavior_goal"]:
                    mp3d, sg = behavior_goal_to_mp3d(str(data["behavior_goal"]))
                    agent.reset(object_category=mp3d, object_category_sg=sg)
                elif "object_category" in data and data["object_category"]:
                    oc = str(data["object_category"])
                    osg = data.get("object_category_sg")
                    osg = str(osg) if osg is not None else None
                    agent.reset(object_category=oc, object_category_sg=osg)
                else:
                    _json_response(
                        self,
                        400,
                        {"error": "need behavior_goal or object_category"},
                    )
                    return
            except Exception as e:
                _json_response(self, 500, {"error": str(e), "type": type(e).__name__})
                return
        _json_response(self, 200, {"status": "reset", "obj_goal": agent.obj_goal})

    def _handle_step(self):
        try:
            data = _read_json_body(self)
        except json.JSONDecodeError as e:
            _json_response(self, 400, {"error": "invalid_json", "detail": str(e)})
            return
        with _AgentHolder.lock:
            if _AgentHolder.agent is None:
                _json_response(self, 400, {"error": "call /reset first"})
                return
            try:
                obs = _parse_step_body(data)
                out = _AgentHolder.agent.act(obs)
                action = int(out["action"])
                agent = _AgentHolder.agent
                total = int(agent.total_steps)
                if agent.args.visualize and agent.total_steps >= 500:
                    _flush_episode_video_if_needed(agent)
            except Exception as e:
                _json_response(self, 500, {"error": str(e), "type": type(e).__name__})
                return
        _json_response(self, 200, {"action": action, "total_steps": total})


def main():
    parser = argparse.ArgumentParser(description="SG-Nav HTTP server (isolated env)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs/sgnav_minimal.rgbd.yaml"),
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=2)
    args = parser.parse_args()

    sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)
    nav_args = argparse.Namespace(
        visualize=bool(args.visualize),
        split_l=-1,
        split_r=-1,
        warmup=bool(args.warmup),
        warmup_runs=max(1, int(args.warmup_runs)),
    )
    _AgentHolder.config_path = args.config
    SGNavHTTPHandler.cfg_path = args.config
    SGNavHTTPHandler.nav_args = nav_args
    server = HTTPServer((args.host, args.port), SGNavHTTPHandler)
    print(f"SG-Nav HTTP server on http://{args.host}:{args.port}", flush=True)
    print("POST /step   body: ObservationMsg (arrays as base64)", flush=True)
    print("POST /warmup body: {\"runs\": 2, \"force\": false}  # optional explicit CUDA/model warmup", flush=True)
    if args.visualize:
        print(
            "  --visualize: writes data/visualization/experiment_0/video/current_frame.jpg each act step; "
            "vid_XXXXXX.mp4 on next /reset, at 500 steps, or server shutdown.",
            flush=True,
        )
    if args.warmup:
        started = _start_background_warmup(args.config, nav_args, nav_args.warmup_runs)
        if started:
            print(
                "startup warmup scheduled in background; /health is available immediately, "
                "and /step will wait on the agent lock if warmup is still running.",
                flush=True,
            )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutdown", flush=True)
    finally:
        with _AgentHolder.lock:
            _flush_episode_video_if_needed(_AgentHolder.agent)
        server.server_close()


if __name__ == "__main__":
    main()
