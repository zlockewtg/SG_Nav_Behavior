from __future__ import annotations

import copy
import math
import os
import re
import sys

import cv2
import numpy as np
import skimage
import torch


def _configure_live_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


_configure_live_stdio()

from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from scenegraph import SceneGraph

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner
from utils.utils_fmm.mapping import Semantic_Mapping
from utils.utils_glip import *
from utils.nav_logger import NavLogger, action_name
from utils.geometry import (
    quat_xyzw_to_rpy, quat_xyzw_rotate_vec,
    wrap_angle_rad, wrap_angle_deg,
    horiz_angle_from_pixel_u, depth_m_at_xy,
    goal_gps_to_map_rc, map_rc_to_goal_gps, world_xy_to_grid_rc,
    get_goal_gps as _get_goal_gps_impl,
    get_relative_goal_gps as _get_relative_goal_gps_impl,
    get_goal_stop_status as _get_goal_stop_status_impl,
    apply_flipud, transform_rc_flipud, inverse_transform_rc_flipud,
    disk_mask, robot_map_rc,
)
from utils.camera_pose import (
    client_camera_pose_active, sync_camera_extrinsics,
    check_fall_suspected, maybe_suspend_mapping, consume_mapping_suspend,
    compute_health_flags,
)
from utils.nav_visualization import (
    build_occupancy_panel, paint_agent_and_goal,
    render_detection_overlay, build_scenegraph_text,
    compose_visualization_frame,
    save_video as _save_video_impl,
    save_scenegraph_json as _save_scenegraph_json_impl,
)


class SG_Nav_Agent():
    def __init__(self, task_config, args=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            # SG-Nav runs fixed-size image tensors after HTTP resize, so cudnn benchmark
            # can cache faster kernels after the first warmup/inference.
            torch.backends.cudnn.benchmark = True
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self._face_goal_pending = False
        self._face_goal_gps = None
        self._face_goal_turn_steps = 0
        self._face_goal_max_turn_steps = 20
        # Goal confirmation distance in meters. Keep env-tunable for different camera/depth setups.
        runtime_cfg = {}
        if hasattr(task_config, "get"):
            runtime_cfg = task_config.get("SGNAV_RUNTIME", {}) or {}
        elif hasattr(task_config, "SGNAV_RUNTIME"):
            runtime_cfg = task_config.SGNAV_RUNTIME

        def _runtime_get(key, default):
            if runtime_cfg is None:
                return default
            if hasattr(runtime_cfg, "get"):
                return runtime_cfg.get(key, default)
            return getattr(runtime_cfg, key, default)

        self.runtime_cfg = runtime_cfg
        self.distance_threshold = float(_runtime_get("goal_distance_threshold", 5.0))
        # HTTP / OmniGibson: fix map “forward” vs robot forward (Semantic_Mapping uses heading−90°).
        self._compass_offset_rad = float(
            np.deg2rad(float(_runtime_get("compass_offset_deg", 0.0)))
        )
        self._gps_negate_y = bool(_runtime_get("gps_negate_y", False))
        # Visualization-only: when the occupancy panel is almost all unknown, legacy code could
        # paint planner traversible as free-space. That also brings along visited path / robot-local
        # bootstrap cells and can look like a duplicated "ghost" map in another corner.
        self._visualize_use_traversible_fallback = bool(
            _runtime_get("visualize_use_traversible_fallback", False)
        )
        # Visualization-only centered viewport size as a fraction of the square global map canvas.
        # Smaller values zoom in more; fixed ratio avoids frame-to-frame size pumping.
        self._visualize_centered_crop_ratio = float(
            _runtime_get("visualize_centered_crop_ratio", 0.50)
        )
        # One line per step: high-level navigation stage in both code and Chinese.
        self._log_nav_stage = bool(_runtime_get("log_nav_stage", True))
        # HTTP / OmniGibson: if ``target_world_xy`` is on the explored occupancy map, plan straight there (FMM).
        self._gt_pathplan_when_target_on_map = bool(
            _runtime_get("gt_pathplan_when_target_on_map", False)
        )
        # OmniGibson HTTP ``og_occupancy`` → collision_map (see _fuse_og_occupancy_into_collision).
        self._og_occ_reseed = bool(_runtime_get("og_occ_reseed_collision_from_depth", True))
        self._og_occ_obs_max = float(_runtime_get("og_occ_obstacle_max", 0.12))
        self._og_occ_free_min = float(_runtime_get("og_occ_free_min", 0.85))
        self._og_occ_fuse_obs = bool(_runtime_get("og_occ_fuse_obstacles", False))
        self._active_gt_world_nav = False
        self._last_goal_map_src_effective = None
        self._prev_goal_rc = None
        self._last_goal_bbox_n = 0
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.last_segment_num = 0
        self.goal_merge_threshold = 0.8
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        _bert_path = _runtime_get("bert_base_uncased_path", "") or ""
        if isinstance(_bert_path, str) and _bert_path.strip():
            _bp = os.path.abspath(os.path.expanduser(_bert_path.strip()))
            if os.path.isdir(_bp):
                os.environ["SGNAV_BERT_BASE_UNCASED_PATH"] = _bp
                print(f"[SG-Nav][bert] local pretrained: {_bp}", flush=True)
            else:
                print(
                    f"[SG-Nav][bert] bert_base_uncased_path is not a directory ({_bp}); "
                    "using Hugging Face Hub",
                    flush=True,
                )

        ### ------ init glip model ------ ###
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )

        self.map_size_cm = int(round(float(_runtime_get("map_size_cm", 6000.0))))
        self.map_size_cm = max(1000, min(12000, self.map_size_cm))
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        ds = task_config.SIMULATOR.DEPTH_SENSOR
        self.sensor_width = int(ds.WIDTH)
        self.sensor_height = int(ds.HEIGHT)
        self.hfov_deg = float(ds.HFOV)
        self.depth_invalid_value = float(getattr(ds, "MIN_DEPTH", 0.5))
        self.depth_max_value = float(getattr(ds, "MAX_DEPTH", 10.0))
        
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix
        print(
            "[SG-Nav][proj_cfg] "
            f"sensor_wh=({self.sensor_width},{self.sensor_height}) hfov_deg={self.hfov_deg:.2f} "
            f"camera_matrix_f={float(self.camera_matrix.f):.3f} "
            f"xc={float(self.camera_matrix.xc):.3f} zc={float(self.camera_matrix.zc):.3f}",
            flush=True,
        )
        
        # Use fixed MP3D order (obj.npy rows). Do not use ``categories_21.index`` — GLIP lists may omit classes.
        self.goal_idx = {name: i for i, name in enumerate(CANONICAL_MP3D_GOAL_ORDER)}
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self)

        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        self.visualization_dir = f'data/visualization/{self.experiment_name}/'
        self.current_frame_path = os.path.join(self.visualization_dir, "video", "current_frame.jpg")
        self.current_frame_det_path = os.path.join(self.visualization_dir, "video", "current_frame_det.jpg")
        self.current_frame_step_dir = os.path.join(self.visualization_dir, "video", "frames")
        self.scenegraph_json_path = os.path.join(self.visualization_dir, "video", "current_scenegraph.json")
        self.scenegraph_json_step_dir = os.path.join(self.visualization_dir, "video", "scenegraph")

        self.glip_object_caption = object_captions
        self._glip_goal_sg_match_tokens: list[str] = []
        # Keep object-goal updates alive after the initial panorama scan.
        # detect_interval=1 means every step; larger values trade quality for speed.
        self.detect_interval = max(1, int(_runtime_get("detect_interval", 4)))
        # Scene graph construction is heavier than object detection; allow it to run less often.
        # scenegraph_update_interval=1 keeps the legacy per-step behavior.
        self.scenegraph_update_interval = max(
            1, int(_runtime_get("scenegraph_update_interval", 1))
        )
        # 1: show only latest-frame captions in "Scene Graph Nodes"; 0: show global accumulated nodes.
        self.show_frame_nodes_only = bool(_runtime_get("show_frame_nodes_only", True))
        self.glip_append_goal_sg = bool(_runtime_get("glip_append_goal_sg", True))
        self.glip_extra_captions = str(_runtime_get("glip_extra_captions", "") or "")
        self.save_scenegraph_json = bool(_runtime_get("save_scenegraph_json", True))
        # Local FMM policy: only FORWARD when |heading − stg| <= this (deg). Legacy was 16° — too tight
        # with noisy compass / moving STG → endless micro-turns (feels like "still panning").
        self._planner_align_angle_deg = float(_runtime_get("planner_align_angle_deg", 26.0))
        # After a turn, require |heading error| <= (align - hysteresis) before FORWARD — reduces left-right-left jitter
        # when compass/STG noise toggles the sign of relative_angle around the threshold.
        self._planner_align_hysteresis_deg = float(_runtime_get("planner_align_hysteresis_deg", 10.0))
        # When >= 0: FMM step only issues FORWARD if |bearing(STG) − heading| ≤ this (deg); else turn in place
        # (body faces the short-term goal before translating). -1 keeps the wide cone from planner_align_* above.
        self._planner_face_goal_max_error_deg = float(
            _runtime_get("planner_face_goal_max_error_deg", -1.0)
        )
        # Estimate the real per-step turn angle from compass deltas after TURN actions.
        # This keeps local planning aligned with the robot's actual executed motion instead of
        # assuming yaml TURN_ANGLE matches the low-level controller.
        self._planner_use_observed_turn = bool(
            _runtime_get("planner_use_observed_turn", True)
        )
        self._planner_observed_turn_ema_alpha = float(
            _runtime_get("planner_observed_turn_ema_alpha", 0.35)
        )
        self._planner_min_effective_turn_deg = float(
            _runtime_get("planner_min_effective_turn_deg", 1.0)
        )
        self._planner_forward_error_turn_ratio = float(
            _runtime_get("planner_forward_error_turn_ratio", 0.5)
        )
        self._planner_observed_turn_deg = None
        self._planner_last_turn_delta_deg = None
        self._prev_obs_compass_rad = None
        self._diag_prev_full_map_pose_rc = None
        self._diag_prev_full_map_pose_deg = None
        # Steps 1..N: while no goal hint, keep returning TURN (6) for panorama. Default 22 matches original SG-Nav.
        self._panorama_spin_until_step = max(1, int(_runtime_get("panorama_spin_until_step", 22)))
        # FMM local short-term goal window (grid cells); larger → coarser STG, less twitchy. See utils_fmm/fmm_planner.py.
        self._fmm_step_size = max(1, int(_runtime_get("fmm_step_size", 5)))
        # Subgoal “close enough” threshold inside get_short_term_goal (before decrease_stop_cond scaling).
        self._fmm_stop_cond = float(_runtime_get("fmm_stop_cond", 0.5))
        # Goal-approach stop policy once the target has been confirmed on the map:
        # use a separate distance threshold for "stop near target" instead of the generic exploration stop_cond.
        self._goal_stop_distance_threshold_m = float(
            _runtime_get("goal_stop_distance_threshold_m", self._fmm_stop_cond)
        )
        self._goal_stop_face_threshold_deg = float(
            _runtime_get("goal_stop_face_threshold_deg", 15.0)
        )
        # ``found_possible_goal`` is only a hint, not task completion. If local
        # planning keeps returning STOP here, keep observing briefly, then fall
        # back to exploration instead of ending the episode.
        self._possible_goal_stop_escape_steps = max(
            1, int(_runtime_get("possible_goal_stop_escape_steps", 6))
        )
        self._possible_goal_stop_escape_cooldown_steps = max(
            0, int(_runtime_get("possible_goal_stop_escape_cooldown_steps", 12))
        )
        self._possible_goal_stop_streak = 0
        self._possible_goal_escape_cooldown = 0
        # ``torch.cuda.empty_cache()`` every step usually hurts steady-state latency because
        # the next step needs to re-request memory from the driver. Keep it opt-in for debugging.
        self._cuda_empty_cache_each_step = bool(
            _runtime_get("cuda_empty_cache_each_step", False)
        )
        # Forward-motion collision heuristic: GPS delta below this (m) while driving → paint obstacle ahead.
        self.collision_threshold = float(_runtime_get("collision_threshold_m", 0.08))
        # Hard safety guard before issuing FORWARD:
        # if near lookahead cells are obstacle/non-traversible, convert FWD -> turn.
        self._forward_guard_enable = bool(_runtime_get("forward_guard_enable", True))
        self._forward_guard_lookahead_m = float(_runtime_get("forward_guard_lookahead_m", 0.50))
        self._forward_guard_samples = max(1, int(_runtime_get("forward_guard_samples", 6)))
        self._forward_guard_cone_half_angle_deg = float(
            _runtime_get("forward_guard_cone_half_angle_deg", 25.0)
        )
        self._random_goal_min_distance_m = float(
            _runtime_get("random_goal_min_distance_m", 0.60)
        )
        self._random_goal_obstacle_clearance_m = float(
            _runtime_get("random_goal_obstacle_clearance_m", 0.35)
        )
        self._random_goal_min_distance_cells = max(
            0,
            min(
                256,
                int(
                    round(
                        max(0.0, self._random_goal_min_distance_m)
                        * 100.0
                        / float(self.map_resolution)
                    )
                ),
            ),
        )
        self._random_goal_obstacle_clearance_cells = max(
            0,
            min(
                256,
                int(
                    round(
                        max(0.0, self._random_goal_obstacle_clearance_m)
                        * 100.0
                        / float(self.map_resolution)
                    )
                ),
            ),
        )
        # Restore local "robot-nearby should not be obstacle" behavior:
        # clear a small disk around the current robot cell in collision/traversible construction.
        self._traversible_clear_robot_obs_enable = bool(
            _runtime_get("traversible_clear_robot_obs_enable", True)
        )
        self._traversible_clear_robot_obs_radius_m = float(
            _runtime_get("traversible_clear_robot_obs_radius_m", 0.30)
        )
        self._traversible_clear_robot_obs_radius_cells = max(
            0,
            min(
                64,
                int(
                    round(
                        max(0.0, self._traversible_clear_robot_obs_radius_m)
                        * 100.0
                        / float(self.map_resolution)
                    )
                ),
            ),
        )
        # Obstacle inflation radius for ``get_traversible`` (meters first, then legacy cells fallback).
        # Example at 5cm/cell: 0.2m -> 4 cells.
        self._traversible_obstacle_inflation_m = float(
            _runtime_get("traversible_obstacle_inflation_m", -1.0)
        )
        self._traversible_obstacle_disk_radius = max(
            0, min(16, int(_runtime_get("traversible_obstacle_disk_radius", 2)))
        )
        if self._traversible_obstacle_inflation_m >= 0.0:
            self._traversible_obstacle_inflation_cells = max(
                0,
                min(
                    64,
                    int(
                        round(
                            self._traversible_obstacle_inflation_m
                            * 100.0
                            / float(self.map_resolution)
                        )
                    ),
                ),
            )
        else:
            self._traversible_obstacle_inflation_cells = int(
                self._traversible_obstacle_disk_radius
            )
        # Depth-map obstacle classification threshold on ``full_map`` values (0..1).
        # Lower catches thinner/noisier obstacles; higher is conservative.
        self._traversible_occ_from_depth_min = float(
            _runtime_get("traversible_occ_from_depth_min", 0.35)
        )
        # Clear map-border obstacle artifacts (cells) before traversible construction.
        # Helps when the robot is close to the global map edge and local obstacle ratio is spuriously high.
        self._traversible_clear_border_obs_cells = max(
            0, min(16, int(_runtime_get("traversible_clear_border_obs_cells", 2)))
        )
        # Unknown-space policy for traversible generation:
        # when true, only ``fbe_free_map``-observed free space is traversible.
        self._traversible_require_explored = bool(
            _runtime_get("traversible_require_explored", False)
        )
        self._traversible_free_map_min = float(
            _runtime_get("traversible_free_map_min", 0.45)
        )
        # Map frame auto-selection for traversible construction.
        # Picks the transform with highest local known evidence around current start.
        self._map_frame_auto_select = bool(_runtime_get("map_frame_auto_select", False))
        self._map_frame_auto_select_win_cells = max(
            5, int(_runtime_get("map_frame_auto_select_win_cells", 40))
        )
        self._map_frame_fixed_mode = str(
            _runtime_get("map_frame_fixed_mode", "flipud")
        ).strip().lower()
        # Legacy flipud start-cell mirror heuristic can move the local center to the wrong half-map
        # (e.g. start_pose_plan row ~770 mirrored to ~30), making occupancy panel appear blank.
        # Keep disabled by default; only enable when explicitly debugging frame alignment.
        self._start_plan_flipud_mirror_enable = bool(
            _runtime_get("start_plan_flipud_mirror_enable", False)
        )
        # If mirror is enabled, require a meaningful local-known improvement before switching.
        self._start_plan_flipud_mirror_min_delta = float(
            _runtime_get("start_plan_flipud_mirror_min_delta", 0.05)
        )
        # If pose-derived local-known is already decent, do not mirror even when alt is slightly better.
        self._start_plan_flipud_pose_known_min = float(
            _runtime_get("start_plan_flipud_pose_known_min", 0.35)
        )
        # Edge diagnostics: warn when start is near border and local occupancy is much denser than global.
        self._map_edge_warn_margin_cells = max(
            0, int(_runtime_get("map_edge_warn_margin_cells", 30))
        )
        self._map_edge_warn_local_occ_min = float(
            _runtime_get("map_edge_warn_local_occ_min", 0.45)
        )
        self._map_edge_warn_global_occ_max = float(
            _runtime_get("map_edge_warn_global_occ_max", 0.08)
        )
        self._map_edge_warn_local_free_max = float(
            _runtime_get("map_edge_warn_local_free_max", 0.02)
        )
        self._last_map_frame_transform = "flipud"
        self._last_obs_dilated_core = None
        self._last_free_from_depth_core = None
        self._last_traversible_core = None
        self._last_traversible_start = None
        self._last_traversible_frame_mode = "id"
        # Mapping density thresholds (voxel count normalization inside utils_fmm/mapping.py).
        # Lower values make obstacle/free evidence appear easier; 10 is often too strict for OG depth.
        self._sem_map_pred_threshold = float(_runtime_get("sem_map_pred_threshold", 1.0))
        self._sem_exp_pred_threshold = float(_runtime_get("sem_exp_pred_threshold", 1.0))
        self._free_map_pred_threshold = float(_runtime_get("free_map_pred_threshold", 0.5))
        self._free_exp_pred_threshold = float(_runtime_get("free_exp_pred_threshold", 0.5))
        self._diag_freeze_camera_extrinsics = bool(
            _runtime_get("diag_freeze_camera_extrinsics", False)
        )
        self._diag_disable_pose_bridge = bool(
            _runtime_get("diag_disable_pose_bridge", False)
        )
        self._diag_sem_threshold_profile = str(
            _runtime_get("diag_sem_threshold_profile", "current")
        ).strip().lower()
        if self._diag_sem_threshold_profile not in ("current", "legacy_strict"):
            self._diag_sem_threshold_profile = "current"
        self._diag_map_warp_mode = str(
            _runtime_get("diag_map_warp_mode", "nearest")
        ).strip().lower()
        if self._diag_map_warp_mode not in ("nearest", "bilinear"):
            self._diag_map_warp_mode = "nearest"
        self._diag_full_map_any_toggle = bool(
            self._diag_freeze_camera_extrinsics
            or self._diag_disable_pose_bridge
            or self._diag_sem_threshold_profile != "current"
            or self._diag_map_warp_mode != "nearest"
        )
        # Bootstrap for explored-only traversible: if free map is too sparse at startup,
        # open a small local non-obstacle region around the agent to avoid deadlock.
        self._traversible_bootstrap_enable = bool(
            _runtime_get("traversible_bootstrap_enable", True)
        )
        self._traversible_bootstrap_min_free_ratio = float(
            _runtime_get("traversible_bootstrap_min_free_ratio", 0.002)
        )
        self._traversible_bootstrap_radius_m = float(
            _runtime_get("traversible_bootstrap_radius_m", 0.8)
        )
        self._traversible_bootstrap_radius_cells = max(
            0,
            min(
                128,
                int(
                    round(
                        self._traversible_bootstrap_radius_m
                        * 100.0
                        / float(self.map_resolution)
                    )
                ),
            ),
        )
        # HTTP / OmniGibson: optional ``camera_pose_world`` (zed_link position + xyzw quaternion) per step.
        # When present, depth→map uses that pitch (via set_view_angles) and optional world Z→sensor height (cm).
        self._camera_extrinsic_use_client_pose = bool(
            _runtime_get("camera_extrinsic_use_client_pose", True)
        )
        self._camera_height_from_world_z = bool(_runtime_get("camera_height_from_world_z", True))
        # Fallback when client camera_pose_world is not available: use a fixed sensor height (cm) for depth projection.
        # Keep <=0 to disable and fall back to cfg AGENT_0.HEIGHT*100.
        self._camera_height_fallback_cm = float(
            _runtime_get("camera_height_fallback_cm", -1.0)
        )
        self._camera_height_z_offset_m = float(_runtime_get("camera_height_z_offset_m", 0.0))
        # Tilt source for depth->map projection from camera_pose_world quaternion.
        # Options:
        # - pitch / roll / yaw: Euler components
        # - cam_z_down: downward pitch derived from rotated camera +Z vector (stable for sensor-frame offsets)
        # - cam_y_down: downward pitch derived from rotated camera +Y vector
        self._camera_tilt_source = str(
            _runtime_get("camera_tilt_source", "pitch")
        ).strip().lower()
        # When ``camera_tilt_source=roll`` and Euler branch flips across +/-180, mapped pitch can jump.
        # If enabled, fallback to vector-derived ``cam_z_down_pitch_deg`` when roll-mapped pitch is outlier.
        self._camera_tilt_fallback_to_cam_z_down_on_outlier = bool(
            _runtime_get("camera_tilt_fallback_to_cam_z_down_on_outlier", True)
        )
        self._camera_pitch_mapping_scale = float(_runtime_get("camera_pitch_mapping_scale", 1.0))
        self._camera_pitch_mapping_offset_deg = float(
            _runtime_get("camera_pitch_mapping_offset_deg", 0.0)
        )
        # Mapping vertical bands (cm):
        # semantic obstacle map uses [map_obstacle_min_z_cm, map_obstacle_max_z_cm] where max can track camera height.
        self._map_obstacle_min_z_cm = float(_runtime_get("map_obstacle_min_z_cm", 0.0))
        self._map_obstacle_max_z_cm = float(_runtime_get("map_obstacle_max_z_cm", -1.0))
        self._map_obstacle_above_camera_cm = float(
            _runtime_get("map_obstacle_above_camera_cm", 50.0)
        )
        # free map keeps near-floor support region.
        self._map_free_min_z_cm = float(_runtime_get("map_free_min_z_cm", -150.0))
        self._map_free_max_z_cm = float(_runtime_get("map_free_max_z_cm", 25.0))
        # Clamp client camera extrinsics to keep mapping stable when link pose spikes (e.g. torso singular states).
        self._camera_pitch_min_deg = float(_runtime_get("camera_pitch_min_deg", -45.0))
        self._camera_pitch_max_deg = float(_runtime_get("camera_pitch_max_deg", 45.0))
        self._camera_height_min_cm = float(_runtime_get("camera_height_min_cm", 80.0))
        self._camera_height_max_cm = float(_runtime_get("camera_height_max_cm", 220.0))
        # Reject obvious camera extrinsic outliers (e.g. occasional bad pose packets) and reuse last good values.
        self._camera_extrinsic_reject_outlier = bool(
            _runtime_get("camera_extrinsic_reject_outlier", True)
        )
        self._camera_height_reject_min_cm = float(
            _runtime_get("camera_height_reject_min_cm", 110.0)
        )
        self._camera_height_reject_max_cm = float(
            _runtime_get("camera_height_reject_max_cm", 210.0)
        )
        self._camera_pitch_reject_abs_deg = float(
            _runtime_get("camera_pitch_reject_abs_deg", 55.0)
        )
        # Additional guard for motion-induced camera pose jitter: during locomotion, reject sudden
        # pitch / height jumps even if the absolute values are still within the global valid range.
        # This keeps depth->map stable when the simulated head sensor bobs while the robot moves.
        self._camera_extrinsic_reject_motion_delta = bool(
            _runtime_get("camera_extrinsic_reject_motion_delta", False)
        )
        self._camera_pitch_reject_delta_deg = float(
            _runtime_get("camera_pitch_reject_delta_deg", 6.0)
        )
        self._camera_height_reject_delta_cm = float(
            _runtime_get("camera_height_reject_delta_cm", 4.0)
        )
        self._last_good_camera_height_cm = None
        self._last_good_camera_pitch_deg = None
        # Suspend depth->map fusion for a few steps when collision / fall likely invalidates camera extrinsics.
        self._map_suspend_on_fall = bool(_runtime_get("map_suspend_on_fall", True))
        self._map_suspend_steps_on_fall = max(
            0, int(_runtime_get("map_suspend_steps_on_fall", 4))
        )
        self._map_suspend_roll_deg = float(
            _runtime_get("map_suspend_roll_deg", 70.0)
        )
        self._map_suspend_pitch_deg = float(
            _runtime_get("map_suspend_pitch_deg", 50.0)
        )
        self._map_suspend_height_cm = float(
            _runtime_get("map_suspend_height_cm", 95.0)
        )
        self._map_suspend_height_delta_cm = float(
            _runtime_get("map_suspend_height_delta_cm", 20.0)
        )
        # Apply mapping thresholds to modules.
        if self._diag_sem_threshold_profile == "legacy_strict":
            self._diag_effective_sem_map_pred_threshold = 10.0
            self._diag_effective_sem_exp_threshold = 10.0
        else:
            self._diag_effective_sem_map_pred_threshold = float(self._sem_map_pred_threshold)
            self._diag_effective_sem_exp_threshold = float(self._sem_exp_pred_threshold)
        self.sem_map_module.map_pred_threshold = float(
            self._diag_effective_sem_map_pred_threshold
        )
        self.sem_map_module.exp_pred_threshold = float(
            self._diag_effective_sem_exp_threshold
        )
        self.free_map_module.map_pred_threshold = float(self._free_map_pred_threshold)
        self.free_map_module.exp_pred_threshold = float(self._free_exp_pred_threshold)
        self.sem_map_module._diag_warp_mode = str(self._diag_map_warp_mode)
        self.sem_map_module._diag_collect_fusion_stats = bool(
            True or self._diag_full_map_any_toggle
        )
        self.free_map_module._diag_warp_mode = "nearest"
        self.free_map_module._diag_collect_fusion_stats = False
        self.room_map_module._diag_warp_mode = "nearest"
        self.room_map_module._diag_collect_fusion_stats = False
        self._last_camera_diag = None
        print('scene graph module init finish!!!')

    def _sync_cuda_for_timing(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def warmup(self, runs: int = 2):
        """Prime CUDA kernels / model weights so the first real request is faster."""
        runs = max(1, int(runs))
        dummy_rgb = np.full(
            (self.sensor_height, self.sensor_width, 3), 127, dtype=np.uint8
        )
        dummy_depth = np.full(
            (self.sensor_height, self.sensor_width, 1),
            min(max(self.depth_invalid_value + 0.5, 1.5), max(self.depth_max_value - 0.5, 1.5)),
            dtype=np.float32,
        )
        pose_center_m = float(self.map_size_cm) / 100.0 / 2.0
        dummy_pose = torch.tensor(
            [pose_center_m, pose_center_m, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        score_vec = torch.zeros((9), device=self.device)
        type_mask = torch.zeros(
            (9, self.sensor_height, self.sensor_width),
            dtype=torch.float32,
            device=self.device,
        )
        depth_t = torch.from_numpy(dummy_depth[..., 0]).to(self.device)

        import time

        timings_ms = []
        self._sync_cuda_for_timing()
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.sem_map_module(depth_t, dummy_pose, torch.zeros_like(self.full_map))
                _ = self.free_map_module(depth_t, dummy_pose, torch.zeros_like(self.fbe_free_map))
                _ = self.room_map_module(
                    depth_t,
                    dummy_pose,
                    torch.zeros_like(self.room_map),
                    type_mask,
                    score_vec,
                )
            _ = self.glip_demo.inference(dummy_rgb[:, :, [2, 1, 0]], object_captions)
            _ = self.glip_demo.inference(dummy_rgb[:, :, [2, 1, 0]], rooms_captions)
            self._sync_cuda_for_timing()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        print(
            "[SG-Nav][warmup] "
            f"runs={runs} device={self.device} "
            f"iter_ms={[round(x, 1) for x in timings_ms]}",
            flush=True,
        )
        return {
            "runs": runs,
            "device": str(self.device),
            "iter_ms": [round(float(x), 3) for x in timings_ms],
        }


    def _set_full_pose_from_arrays(self, gps_arr, compass_arr):
        gps_t = torch.from_numpy(np.asarray(gps_arr, dtype=np.float64)).to(self.device)
        comp_t = torch.from_numpy(np.asarray(compass_arr, dtype=np.float64)).to(self.device)
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0 + gps_t[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0 - gps_t[1]
        self.full_pose[2:] = comp_t * 57.29577951308232
        self._clamp_full_pose_xy_to_map_meters()

    def _maybe_suspend_mapping_due_to_fall(self):
        if not bool(getattr(self, "_map_suspend_on_fall", False)):
            return
        self._mapping_suspend_countdown = maybe_suspend_mapping(
            self._last_camera_diag,
            self._last_good_camera_height_cm,
            int(getattr(self, "_mapping_suspend_countdown", 0)),
            getattr(self, "_client_collision_prev", None),
        )

    def _consume_mapping_suspend(self, stage: str) -> bool:
        skip, new_cd = consume_mapping_suspend(int(getattr(self, "_mapping_suspend_countdown", 0)))
        self._mapping_suspend_countdown = new_cd
        return skip

    def _sync_glip_object_caption(self):
        """
        Extend GLIP phrase list with open-vocab tokens (e.g. radio) from obj_goal_sg and/or env.

        glip_extra_captions: comma/semicolon/pipe-separated phrases, always appended.
        glip_append_goal_sg: set false to disable auto tokens from obj_goal_sg.
        """
        base = object_captions
        extras: list[str] = []
        env_extra = self.glip_extra_captions.strip()
        if env_extra:
            for part in re.split(r"[,;|]", env_extra):
                p = part.strip()
                if p:
                    extras.append(p)
        if self.glip_append_goal_sg:
            sg = getattr(self, "obj_goal_sg", None) or ""
            extras.extend(extract_tokens_for_glip_from_goal_sg(sg))
        self.glip_object_caption = compose_glip_object_caption(base, extras)
        seen_tok: set[str] = set()
        self._glip_goal_sg_match_tokens = []
        for p in extras:
            low = p.lower()
            if low not in seen_tok:
                seen_tok.add(low)
                self._glip_goal_sg_match_tokens.append(low)

        if getattr(self, "scenegraph", None) is not None:
            ns0 = getattr(self.scenegraph, "_node_space_default", self.scenegraph.node_space)
            if extras:
                bits: list[str] = []
                seen_ns: set[str] = set()
                for x in extras:
                    x = str(x).strip()
                    if not x:
                        continue
                    k = x.lower()
                    if k in seen_ns:
                        continue
                    seen_ns.add(k)
                    bits.append(x.rstrip(".") + ".")
                self.scenegraph.node_space = (ns0 + " " + " ".join(bits)).strip()
            else:
                self.scenegraph.node_space = ns0

    def _pix_i(self, v):
        return int(v.item()) if hasattr(v, "item") else int(v)

    def _refresh_co_occurrence_priors(self):
        """Fill prob_array_* for PSL; uniform prior if goal is outside MP3D training categories."""
        idx = self.goal_idx.get(self.obj_goal)
        if idx is not None:
            self.prob_array_room = self.co_occur_room_mtx[idx]
            self.prob_array_obj = self.co_occur_mtx[idx]
        else:
            n_o = self.co_occur_room_mtx.shape[1]
            n_p = self.co_occur_mtx.shape[1]
            self.prob_array_room = np.ones(n_o, dtype=np.float64) / n_o
            self.prob_array_obj = np.ones(n_p, dtype=np.float64) / n_p


    def _update_observed_turn_from_compass(self, observations) -> None:
        comp = np.asarray(observations.get("compass", [0.0]), dtype=np.float64).reshape(-1)
        if comp.size <= 0 or not math.isfinite(float(comp[0])):
            return
        cur_compass_rad = float(comp[0])
        prev_compass_rad = self._prev_obs_compass_rad
        self._prev_obs_compass_rad = cur_compass_rad
        if prev_compass_rad is None:
            return
        if int(getattr(self, "prev_action", 0)) not in (2, 3, 6):
            return

        delta_rad = wrap_angle_rad(cur_compass_rad - float(prev_compass_rad))
        delta_deg = abs(math.degrees(delta_rad))
        self._planner_last_turn_delta_deg = float(delta_deg)
        if not math.isfinite(delta_deg):
            return
        if self._planner_observed_turn_deg is None:
            self._planner_observed_turn_deg = float(delta_deg)
            return
        alpha = float(np.clip(self._planner_observed_turn_ema_alpha, 0.0, 1.0))
        self._planner_observed_turn_deg = float(
            (1.0 - alpha) * float(self._planner_observed_turn_deg) + alpha * float(delta_deg)
        )



    def _sync_camera_extrinsics(self, observations):
        """Delegate to utils/camera_pose.sync_camera_extrinsics."""
        modules = (self.sem_map_module, self.free_map_module, self.room_map_module)
        result = sync_camera_extrinsics(
            observations, modules,
            camera_tilt_source=self._camera_tilt_source,
            camera_height_fallback_cm=self._camera_height_fallback_cm,
            camera_pitch_min_deg=self._camera_pitch_min_deg,
            camera_pitch_max_deg=self._camera_pitch_max_deg,
            camera_height_min_cm=self._camera_height_min_cm,
            camera_height_max_cm=self._camera_height_max_cm,
            map_obstacle_min_z_cm=self._map_obstacle_min_z_cm,
            last_good_height_cm=self._last_good_camera_height_cm,
            last_good_pitch_deg=self._last_good_camera_pitch_deg,
        )
        self._last_camera_diag = result["diag"]
        self._last_good_camera_height_cm = result["last_good_height_cm"]
        self._last_good_camera_pitch_deg = result["last_good_pitch_deg"]

    def reset(self, object_category=None, object_category_sg=None):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.origins[:] = np.nan
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = True
        self._active_gt_world_nav = False
        self._prev_goal_rc = None  # (row, col) on goal_map for drift diagnostics
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_possible_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.last_segment_num = 0
        self._planner_observed_turn_deg = None
        self._planner_last_turn_delta_deg = None
        self._prev_obs_compass_rad = None
        self._diag_prev_full_map_pose_rc = None
        self._diag_prev_full_map_pose_deg = None
        self._client_collision_prev = None
        self._mapping_suspend_countdown = 0
        self._mapping_suspend_last_step = -1
        self._last_goal_map_src_effective = None
        self._possible_goal_stop_streak = 0
        self._possible_goal_escape_cooldown = 0
        self._face_goal_pending = False
        self._face_goal_gps = None
        self._face_goal_turn_steps = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        if object_category is not None:
            self.obj_goal = object_category
            self.obj_goal_sg = (
                object_category_sg if object_category_sg is not None else object_category
            )
        elif getattr(self, "simulator", None) is not None and hasattr(
            self.simulator, "_env"
        ) and getattr(self.simulator._env, "current_episode", None) is not None:
            self.obj_goal = self.simulator._env.current_episode.object_category
            self.obj_goal_sg = self.simulator._env.current_episode.object_category
        else:
            raise ValueError(
                "Reset requires object_category=... unless agent.simulator is a habitat "
                "Benchmark with _env.current_episode (set agent.simulator before reset)."
            )
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'
        self._sync_glip_object_caption()
        self.current_obj_predictions = []
        self.obj_locations = [[] for _ in range(len(CANONICAL_MP3D_GOAL_ORDER))]
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self.goal_distance_for_vis = None

        self.scenegraph.reset()
        self._last_goal_bbox_n = 0
        self._last_good_camera_height_cm = None
        self._last_good_camera_pitch_deg = None
        self._goal_map_src_for_visualization = None
        self._active_goal_gps = None

    def _act_action_name(self, a):
        names = ("STOP", "FWD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "TURN")
        try:
            ai = int(a)
        except (TypeError, ValueError):
            return str(a)
        return names[ai] if 0 <= ai < len(names) else str(a)

    def _emit_spin_diagnosis(self, **_kw):
        """Spin diagnosis now goes to NavLogger; terminal output suppressed."""
        pass

    def _resolve_nav_stage(
        self,
        *,
        goal_map_src=None,
        early=None,
        panorama=False,
        stuck_abort=False,
        max_steps_stop=False,
    ):
        """
        Return a stable high-level stage label for logs.

        stage_code is machine-friendly; stage_zh is for quick human scanning.
        """
        if max_steps_stop:
            return "episode_stop", "步数上限停止"
        if stuck_abort:
            return "stuck_abort", "卡死停止"
        if early is not None or panorama:
            return "panorama_scan", "全景扫描"

        src = "" if goal_map_src is None else str(goal_map_src)
        effective_src = src
        if src == "keep_previous_goal_map":
            last_src = getattr(self, "_last_goal_map_src_effective", None)
            if last_src:
                effective_src = str(last_src)
        if self.found_goal or effective_src == "found_goal":
            return "goal_driven_nav", "目标驱动导航"
        if effective_src == "gt_world_on_map" or getattr(self, "_active_gt_world_nav", False):
            return "goal_driven_nav", "目标驱动导航"
        if effective_src == "face_goal_pending" or getattr(self, "_face_goal_pending", False):
            return "goal_driven_nav", "目标朝向对齐"
        if self.found_possible_goal or effective_src == "possible_goal":
            return "possible_goal_nav", "疑似目标导航"
        if "frontier" in effective_src:
            return "frontier_explore", "frontier探索"
        if "random" in effective_src or getattr(self, "using_random_goal", False):
            return "random_recovery", "随机脱困"
        if effective_src == "keep_previous_goal_map":
            if self.found_goal:
                return "goal_driven_nav", "目标驱动导航"
            if self.found_possible_goal:
                return "possible_goal_nav", "疑似目标导航"
        return "unknown", "未知阶段"

    def _emit_nav_stage(
        self,
        *,
        goal_map_src=None,
        number_action=None,
        early=None,
        panorama=False,
        stuck_abort=False,
        max_steps_stop=False,
    ):
        if not getattr(self, "_log_nav_stage", True):
            return
        stage_code, stage_zh = self._resolve_nav_stage(
            goal_map_src=goal_map_src,
            early=early,
            panorama=panorama,
            stuck_abort=stuck_abort,
            max_steps_stop=max_steps_stop,
        )
        action_name = self._act_action_name(number_action) if number_action is not None else "n/a"
        src = "none" if goal_map_src is None else str(goal_map_src)
        extra = []
        if early is not None:
            extra.append(f"early={early}")
        if panorama:
            extra.append("panorama=1")
        if max_steps_stop:
            extra.append("max_steps_stop=1")
        if stuck_abort:
            extra.append("stuck_abort=1")
        extra_txt = ""
        if extra:
            extra_txt = " " + " ".join(extra)
        print(
            f"[SG-Nav][stage] step={self.total_steps} "
            f"stage={stage_code}({stage_zh}) "
            f"action={number_action}({action_name}) "
            f"src={src}{extra_txt}",
            flush=True,
        )

    def _ingest_client_collision_info(self, observations):
        info = observations.get("collision_info")
        if not isinstance(info, dict):
            self._client_collision_prev = None
            return
        parsed = {
            "base_contact": bool(info.get("base_contact", False)),
            "n_contacts": max(0, int(info.get("n_contacts", 0))),
            "contact_bodies": [str(x) for x in list(info.get("contact_bodies", []))[:8]],
            "max_impulse": float(info.get("max_impulse", 0.0)),
            "prev_action": None if info.get("prev_action") is None else int(info.get("prev_action")),
        }
        self._client_collision_prev = parsed

    def _clear_possible_goal_nav_state(self):
        self.found_possible_goal = False
        self.found_goal_times = 0
        self._possible_goal_stop_streak = 0
        self.possible_goal_temp_gps = np.array([0.0, 0.0])
        if isinstance(getattr(self, "goal_gps_map", None), np.ndarray):
            self.goal_gps_map.fill(0.0)

    def _navigation_goal_label_match(self, label) -> bool:
        """Same rules as ``goal_bbox`` in ``detect_objects``: drives map / navigation."""
        lab_l = str(label).lower()
        if self.obj_goal in lab_l:
            return True
        if self.obj_goal == "gym_equipment" and label in ("treadmill", "exercise machine"):
            return True
        if self._glip_goal_sg_match_tokens and any(tok in lab_l for tok in self._glip_goal_sg_match_tokens):
            return True
        return False

    def detect_objects(self, observations):
        self.current_obj_predictions = self.glip_demo.inference(
            observations["rgb"][:, :, [2, 1, 0]],
            self.glip_object_caption,
        )
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)

        
        shortest_distance = 120
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        for j, label in enumerate(obj_labels):
            if self._navigation_goal_label_match(label):
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        self._last_goal_bbox_n = len(goal_bbox)
        possible_goal_allowed = (
            int(getattr(self, "_possible_goal_escape_cooldown", 0)) <= 0
        )
        # Match ``goal_gps_to_map_rc`` in utils/geometry.py (half map in cells).
        half_cells = float(self.map_size_cm) / 10.0

        for j, label in enumerate(obj_labels):
            if label in CANONICAL_MP3D_GOAL_ORDER:
                confidence = self.current_obj_predictions.get_field("scores")[j]
                bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = horiz_angle_from_pixel_u(center_point[0], self.sensor_width, self.hfov_deg)
                temp_distance = depth_m_at_xy(self.depth, center_point[0], center_point[1])
                if temp_distance >= self.distance_threshold:
                    continue
                obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(half_cells - obj_gps[1] * 100 / self.resolution)
                y = int(half_cells + obj_gps[0] * 100 / self.resolution)
                self.obj_locations[CANONICAL_MP3D_GOAL_ORDER.index(label)].append([confidence, x, y])
        
        if self.scenegraph.obj_goal in self.scenegraph.small_objects:
            self.segment_num = len(self.scenegraph.segment2d_results)
            goal_mask = []
            if self.segment_num > self.last_segment_num:
                self.last_segment_num = self.segment_num
                segment2d_result = self.scenegraph.segment2d_results[-1]
                indices = []
                for index, element in enumerate(segment2d_result['caption']):
                    if self.obj_goal_sg in element.split(' '):
                        for node in self.scenegraph.nodes:
                            if node.is_goal_node and node.object['image_idx'][-1] == len(self.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                                indices.append(index)
                goal_mask = [segment2d_result['mask'][index] for index in indices]
            if len(goal_mask) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                for mask in goal_mask:
                    center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
                    center_point = torch.tensor([center_point[1], center_point[0]])
                    temp_direction = horiz_angle_from_pixel_u(center_point[0], self.sensor_width, self.hfov_deg)
                    temp_distance = depth_m_at_xy(self.depth, center_point[0], center_point[1])
                    k = 0
                    pos_neg = 1
                    dh, dw = int(self.depth.shape[0]), int(self.depth.shape[1])
                    while temp_distance >= 100 and 0 < self._pix_i(center_point[1]) + int(pos_neg * k) < dh - 1 and 0 < self._pix_i(center_point[0]) + int(pos_neg * k) < dw - 1:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(
                            depth_m_at_xy(self.depth, center_point[0], center_point[1] + int(pos_neg * k)),
                            depth_m_at_xy(self.depth, center_point[0] + int(pos_neg * k), center_point[1]),
                        )
                        
                    if temp_distance >= self.distance_threshold:
                        if possible_goal_allowed:
                            self.found_possible_goal = True
                    else:
                        if self.found_goal:
                            if temp_distance < self.distance_threshold:
                                self.found_goal_times = self.found_goal_times + 1
                        self.found_goal = True
                        self.found_possible_goal = False
                    
                    ## select the closest goal
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                if self.found_goal:
                    self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
                elif not possible_goal_detected_before and possible_goal_allowed:
                    # if detected a long goal before, then don't change it until see a goal within 5 meters
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            else:
                if self.found_goal:
                    self.found_goal = False
                    self.found_goal_times = 0
            self.goal_distance_for_vis = float(shortest_distance) if shortest_distance < 120 else None
            return
        else:
            if len(goal_bbox) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                goal_prediction.bbox = torch.stack(goal_bbox)
                _vote_boxes_far = 0
                _vote_boxes_near = 0
                _vote_boxes_near_oob = 0
                for box in goal_prediction.bbox:
                    box = box.to(torch.int64)
                    center_point = (box[:2] + box[2:]) // 2
                    temp_direction = horiz_angle_from_pixel_u(center_point[0], self.sensor_width, self.hfov_deg)
                    temp_distance = depth_m_at_xy(self.depth, center_point[0], center_point[1])
                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    k = 0
                    pos_neg = 1
                    dh, dw = int(self.depth.shape[0]), int(self.depth.shape[1])
                    while temp_distance >= 100 and 0 < self._pix_i(center_point[1]) + int(pos_neg * k) < dh - 1 and 0 < self._pix_i(center_point[0]) + int(pos_neg * k) < dw - 1:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(
                            depth_m_at_xy(self.depth, center_point[0], center_point[1] + int(pos_neg * k)),
                            depth_m_at_xy(self.depth, center_point[0] + int(pos_neg * k), center_point[1]),
                        )
                        
                    if temp_distance >= self.distance_threshold:
                        if possible_goal_allowed:
                            self.found_possible_goal = True
                        _vote_boxes_far += 1
                    else:
                        thres = int(self.goal_merge_threshold * 100 / self.map_resolution)
                        goal_r, goal_c = self._goal_gps_to_raw_goal_map_rc(goal_gps)
                        if 0 <= goal_r < self.map_size and 0 <= goal_c < self.map_size:
                            goal_gps_map_local = self.goal_gps_map[
                                max(goal_r - thres, 0):min(goal_r + thres, self.map_size - 1),
                                max(goal_c - thres, 0):min(goal_c + thres, self.map_size - 1),
                            ]
                            if goal_gps_map_local.max() > 0:
                                goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] = goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] + 1
                            else:
                                self.goal_gps_map[goal_r, goal_c] = 1
                            _vote_boxes_near += 1
                        else:
                            _vote_boxes_near_oob += 1
                        # Do NOT clear found_possible_goal here. Legacy code set False on every *near* box,
                        # which blocked ``act()``'s ``elif self.found_possible_goal`` goal_map and kept the
                        # panorama spin (``not found_goal and not found_possible_goal``) even with red boxes
                        # while votes were still accumulating toward ``found_goal``.
                    
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                self.found_goal_times = self.goal_gps_map.max()
                if self.found_goal_times >= self.scenegraph.cfg.obj_min_detections:
                    self.found_goal = True
                # Until ``found_goal``, keep possible-goal mode whenever we have a usable hint (far, near
                # vote, or any in-frame distance estimate) so ``act`` steers toward ``possible_goal_temp_gps``.
                if not self.found_goal:
                    if possible_goal_allowed and (
                        _vote_boxes_far > 0
                        or _vote_boxes_near > 0
                        or int(self.found_goal_times) > 0
                        or shortest_distance < 120
                    ):
                        self.found_possible_goal = True
                if self.found_goal:
                    goal_rows, goal_cols = np.where(
                        self.goal_gps_map == self.goal_gps_map.max()
                    )
                    self.goal_gps = self._raw_goal_map_rc_to_goal_gps(
                        goal_rows[0], goal_cols[0]
                    )
                elif possible_goal_allowed and shortest_distance < 120:
                    # Keep a moving far-goal hint so agent starts approaching distant detections
                    # instead of oscillating in exploration mode.
                    self.possible_goal_temp_gps = self.get_goal_gps(
                        observations, shortest_distance_angle, shortest_distance
                    )
            self.goal_distance_for_vis = float(shortest_distance) if shortest_distance < 120 else None
            return
                        
    def act(self, observations):
        if self.total_steps >= 500:
            self._emit_nav_stage(max_steps_stop=True, number_action=0)
            self._emit_spin_diagnosis(max_steps_stop=True, number_action=0)
            return {"action": 0}
        
        self.total_steps += 1
        if self._possible_goal_escape_cooldown > 0:
            self._possible_goal_escape_cooldown -= 1
        self._detect_objects_done_this_act = False
        if self.navigate_steps == 0:
            self._refresh_co_occurrence_priors()

        depth_arr = np.asarray(observations["depth"], dtype=np.float32)
        invalid_depth = ~np.isfinite(depth_arr)
        invalid_depth |= depth_arr <= (float(self.depth_invalid_value) + 1e-4)
        depth_arr[invalid_depth] = 100.0
        observations["depth"] = depth_arr
        # GLIP / room type_mask / Semantic_Mapping assume DEPTH_SENSOR.{HEIGHT,WIDTH}. OmniGibson (or any
        # client) may send larger frames → bbox y/x can exceed cfg dims (e.g. 882 vs 800) and depth
        # intrinsics no longer match the depth buffer.
        H_cfg, W_cfg = int(self.sensor_height), int(self.sensor_width)
        rgb_u8 = observations["rgb"]
        depth_obs = observations["depth"]
        if int(rgb_u8.shape[0]) != H_cfg or int(rgb_u8.shape[1]) != W_cfg:
            observations["rgb"] = cv2.resize(
                rgb_u8, (W_cfg, H_cfg), interpolation=cv2.INTER_LINEAR
            )
            d_plane = depth_obs[..., 0] if depth_obs.ndim == 3 else depth_obs
            d_resized = cv2.resize(
                d_plane.astype(np.float32), (W_cfg, H_cfg), interpolation=cv2.INTER_NEAREST
            )
            observations["depth"] = (
                d_resized[..., np.newaxis] if depth_obs.ndim == 3 else d_resized
            )
        # Bridge calibration: Habitat-style gps/compass may not match OG world + SG-Nav pose_transform.
        comp_client = np.asarray(observations["compass"], dtype=np.float64).reshape(-1).copy()
        comp = comp_client.copy()
        comp[0] += self._compass_offset_rad
        observations["compass_client"] = comp_client.copy()
        observations["compass"] = comp
        self._update_observed_turn_from_compass(observations)
        self._ingest_client_collision_info(observations)
        gps_client = np.asarray(observations["gps"], dtype=np.float64).reshape(-1).copy()
        gps = gps_client.copy()
        if self._gps_negate_y:
            gps[1] = -gps[1]
        observations["gps_client"] = gps_client.copy()
        observations["gps_raw"] = gps.copy()
        observations["gps"] = gps
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        self.rgb_visualization = observations["rgb"]

        self.scenegraph.set_agent(self)
        self.scenegraph.set_navigate_steps(self.navigate_steps)
        self.scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg)
        self.scenegraph.set_room_map(self.room_map)
        self.scenegraph.set_fbe_free_map(self.fbe_free_map)
        self.scenegraph.set_observations(observations)
        self.scenegraph.set_full_map(self.full_map)
        self.scenegraph.set_full_pose(self.full_pose)
        run_scenegraph_update = (self.total_steps % self.scenegraph_update_interval) == 0
        if run_scenegraph_update:
            self.scenegraph.update_scenegraph()

        self._sync_camera_extrinsics(observations)
        self._maybe_suspend_mapping_due_to_fall()
        self.update_map(observations)
        self.update_free_map(observations)
        # Snapshot after depth->map update and before planning-only transforms.

        # Run GLIP + room map before scripted camera returns. Legacy code returned TURN for steps 2–14
        # *before* any detection, so found_possible_goal stayed false and the agent never left panorama.
        run_periodic_detect = (self.total_steps % self.detect_interval) == 0
        spin_cap = self._panorama_spin_until_step
        if self.total_steps > 1:
            self.detect_objects(observations)
            self._detect_objects_done_this_act = True
            if run_periodic_detect or (self.total_steps <= spin_cap):
                room_detection_result = self.glip_demo.inference(
                    observations["rgb"][:, :, [2, 1, 0]], rooms_captions
                )
                self.update_room_map(observations, room_detection_result)

        use_script_cam_tilt = not client_camera_pose_active(observations)
        if self.total_steps == 1:
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(30)
                self.free_map_module.set_view_angles(30)
            self._emit_nav_stage(early="step1_lookdown", number_action=5)
            self._emit_spin_diagnosis(early="step1_lookdown", number_action=5)
            return {"action": 5}
        elif self.total_steps <= 7 and not (self.found_goal or self.found_possible_goal):
            self._emit_nav_stage(early="step2_7_turn", number_action=6)
            self._emit_spin_diagnosis(early="step2_7_turn", number_action=6)
            return {"action": 6}
        elif self.total_steps == 8:
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(60)
                self.free_map_module.set_view_angles(60)
            self._emit_nav_stage(early="step8_lookdown", number_action=5)
            self._emit_spin_diagnosis(early="step8_lookdown", number_action=5)
            return {"action": 5}
        elif self.total_steps <= 14 and not (self.found_goal or self.found_possible_goal):
            self._emit_nav_stage(early="step9_14_turn", number_action=6)
            self._emit_spin_diagnosis(early="step9_14_turn", number_action=6)
            return {"action": 6}
        elif self.total_steps <= 15 and not (self.found_goal or self.found_possible_goal):
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(30)
                self.free_map_module.set_view_angles(30)
            self._emit_nav_stage(early="step15_look_pitch", number_action=4)
            self._emit_spin_diagnosis(early="step15_look_pitch", number_action=4)
            return {"action": 4}
        elif self.total_steps <= 16 and not (self.found_goal or self.found_possible_goal):
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(0)
                self.free_map_module.set_view_angles(0)
            self._emit_nav_stage(early="step16_look_pitch", number_action=4)
            self._emit_spin_diagnosis(early="step16_look_pitch", number_action=4)
            return {"action": 4}
        # Panorama buffers + optional extra spin only while no goal hint
        if (self.total_steps <= spin_cap and not self.found_goal) or run_periodic_detect:
            self.panoramic.append(observations["rgb"][:, :, [2, 1, 0]])
            self.panoramic_depth.append(observations["depth"])
            if self.total_steps <= spin_cap and (not self.found_goal and not self.found_possible_goal):
                self._emit_nav_stage(panorama=True, number_action=6)
                self._emit_spin_diagnosis(panorama=True, number_action=6)
                return {"action": 6}
                    
        if np.linalg.norm(observations["gps"] - self.last_gps) >= 0.05:
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            
        self.last_gps = observations["gps"]
        
        if run_scenegraph_update:
            self.scenegraph.perception()
        if run_scenegraph_update and self.save_scenegraph_json:
            self.save_scenegraph_json_snapshot()
          
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        # Keep y in the same map-meter convention as ``full_pose``; frame transform is handled in get_traversible.
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        # Planner/traversible now uses a fixed plan frame (e.g. flipud) internally.
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0], input_pose)
        
        goal_map_src = "keep_previous_goal_map"
        if self.found_goal:
            self._active_gt_world_nav = False
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            goal_r, goal_c = self._goal_gps_to_raw_goal_map_rc(self.goal_gps)
            self.goal_map[goal_r, goal_c] = 1
            goal_map_src = "found_goal"
        elif self._try_apply_gt_world_target_goal_map(observations):
            goal_map_src = "gt_world_on_map"
        elif self._face_goal_pending and self._face_goal_gps is not None:
            self._active_gt_world_nav = False
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            goal_r, goal_c = self._goal_gps_to_raw_goal_map_rc(self._face_goal_gps)
            self.goal_map[goal_r, goal_c] = 1
            goal_map_src = "face_goal_pending"
        elif self.found_possible_goal:
            self._active_gt_world_nav = False
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            goal_r_raw, goal_c_raw = self._goal_gps_to_raw_goal_map_rc(
                self.possible_goal_temp_gps
            )
            projected_goal = self._project_raw_goal_rc_to_possible_goal_safe_cell(
                goal_r_raw, goal_c_raw
            )
            if projected_goal is not None:
                goal_r, goal_c, project_dist_cells = projected_goal
                self.goal_map[goal_r, goal_c] = 1
                goal_map_src = "possible_goal"
            else:
                self._clear_possible_goal_nav_state()
                self.goal_loc = self.fbe(traversible, cur_start)
                if self.goal_loc is None:
                    self.random_this_ex += 1
                    self.goal_map = self.set_random_goal()
                    self.using_random_goal = True
                    goal_map_src = "possible_goal_fallback_random"
                else:
                    self.fronter_this_ex += 1
                    gr = int(np.clip(int(self.goal_loc[0]), 0, self.map_size - 1))
                    gc = int(np.clip(int(self.goal_loc[1]), 0, self.map_size - 1))
                    self.goal_map[gr, gc] = 1
                    goal_map_src = "possible_goal_fallback_frontier"
        elif self.first_fbe:
            self._active_gt_world_nav = False
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = False
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
                goal_map_src = "first_fbe_random"
            else:
                self.fronter_this_ex += 1
                gr = int(np.clip(int(self.goal_loc[0]), 0, self.map_size - 1))
                gc = int(np.clip(int(self.goal_loc[1]), 0, self.map_size - 1))
                self.goal_map[gr, gc] = 1
                goal_map_src = "first_fbe_frontier"
        
        # local policy (goal map must share frame with traversible).
        goal_map_plan = self._apply_map_frame_transform(
            self.goal_map, getattr(self, "_last_map_frame_transform", "id")
        )
        traversible_plan = traversible
        if goal_map_src == "possible_goal":
            possible_goal_traversible = self._build_possible_goal_traversible()
            if possible_goal_traversible is not None:
                traversible_plan = possible_goal_traversible
        stg_y, stg_x, replan, number_action = self._plan(
            traversible_plan, goal_map_plan, self.full_pose, cur_start, cur_start_o, self.found_goal
        )
        # Keep possible-goal mode alive across brief local planner "stop" outputs.
        # This avoids bouncing back to exploration before getting close enough.
        
        # reach long-term goal and fbe
        # Exploration: if FMM says stop at short-term subgoal (action 0), we immediately pick a NEW frontier
        # via fbe() — goal can jump every such step → "never finishes turning toward" the old frontier.
        replan_fbe_hit = False
        if (
            (
                not self.found_goal
                and not self.found_possible_goal
                and not self._active_gt_world_nav
                and not self._face_goal_pending
                and number_action == 0
            )
            or (self.using_random_goal and self.move_since_random > 20)
        ):
            replan_fbe_hit = True
            self._active_gt_world_nav = False
            if (self.using_random_goal and self.move_since_random > 20):
                goal_x, goal_y = np.where(self.goal_map == 1)
                x_0 = max(goal_x[0] - 8, 0)
                y_0 = max(goal_y[0] - 8, 0)
                x_1 = min(goal_x[0] + 8, self.map_size)
                y_1 = min(goal_y[0] + 8, self.map_size)
                # fbe_free_map is [1,1,H,W]; clear on the 2D map plane.
                self.fbe_free_map[0, 0, x_0:x_1, y_0:y_1] = 0
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
                goal_map_src = "replan_fbe_random"
            else:
                self.fronter_this_ex += 1
                gr = int(np.clip(int(self.goal_loc[0]), 0, self.map_size - 1))
                gc = int(np.clip(int(self.goal_loc[1]), 0, self.map_size - 1))
                self.goal_map[gr, gc] = 1
                goal_map_src = "replan_fbe_frontier"
            goal_map_plan = self._apply_map_frame_transform(
                self.goal_map, getattr(self, "_last_map_frame_transform", "id")
            )
            stg_y, stg_x, replan, number_action = self._plan(
                traversible, goal_map_plan, self.full_pose, cur_start, cur_start_o, self.found_goal
            )
        
        # Treat a planner STOP/TURN while we still have a semantic target as goal-side behavior,
        # not as exploration deadlock. Without this guard, brief alignment / near-goal stops
        # can accumulate ``not_move_steps`` and incorrectly trigger random recovery.
        if (
            (self.found_goal and goal_map_src == "found_goal")
            or (self._active_gt_world_nav and goal_map_src == "gt_world_on_map")
            or self._face_goal_pending
        ):
            self.not_move_steps = 0

        possible_goal_escape_hit = False
        if (
            self.found_possible_goal
            and not self.found_goal
            and goal_map_src == "possible_goal"
        ):
            if number_action == 0:
                self._possible_goal_stop_streak += 1
                if (
                    self._possible_goal_stop_streak
                    < int(self._possible_goal_stop_escape_steps)
                ):
                    # Possible-goal STOP is not success; keep scanning instead of
                    # propagating STOP to the outer episode loop.
                    number_action = 6
                    self.not_move_steps = 0
                else:
                    possible_goal_escape_hit = True
                    self._possible_goal_escape_cooldown = int(
                        self._possible_goal_stop_escape_cooldown_steps
                    )
                    self._clear_possible_goal_nav_state()
                    self._active_gt_world_nav = False
                    self.not_use_random_goal()
                    self.goal_loc = self.fbe(traversible, cur_start)
                    self.goal_map = np.zeros(self.full_map.shape[-2:])
                    if self.goal_loc is None:
                        self.random_this_ex += 1
                        self.goal_map = self.set_random_goal()
                        self.using_random_goal = True
                        goal_map_src = "possible_goal_escape_random"
                    else:
                        self.fronter_this_ex += 1
                        gr = int(np.clip(int(self.goal_loc[0]), 0, self.map_size - 1))
                        gc = int(np.clip(int(self.goal_loc[1]), 0, self.map_size - 1))
                        self.goal_map[gr, gc] = 1
                        goal_map_src = "possible_goal_escape_frontier"
                    goal_map_plan = self._apply_map_frame_transform(
                        self.goal_map, getattr(self, "_last_map_frame_transform", "id")
                    )
                    stg_y, stg_x, replan, number_action = self._plan(
                        traversible,
                        goal_map_plan,
                        self.full_pose,
                        cur_start,
                        cur_start_o,
                        self.found_goal,
                    )
                    self.not_move_steps = 0
            else:
                self._possible_goal_stop_streak = 0
        else:
            self._possible_goal_stop_streak = 0

        allow_exploration_stuck_reset = (
            not self.found_goal
            and not self.found_possible_goal
            and not self._active_gt_world_nav
            and not self._face_goal_pending
        )
        self.loop_time = 0
        stuck_reset_hit = False
        while (
            (allow_exploration_stuck_reset and number_action == 0)
            or (allow_exploration_stuck_reset and self.not_move_steps >= 7)
        ):
            stuck_reset_hit = True
            if self.not_move_steps >= 7:
                self.found_goal = False
                self.found_possible_goal = False
                self._active_gt_world_nav = False
            self.loop_time += 1
            self.random_this_ex += 1
            if self.loop_time > 20:
                self._emit_nav_stage(
                    stuck_abort=True,
                    goal_map_src=goal_map_src,
                    number_action=0,
                )
                self._emit_spin_diagnosis(
                    stuck_abort=True,
                    goal_map_src=goal_map_src,
                    number_action=0,
                )
                return {"action": 0}
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            goal_map_src = "stuck_cleared_random"
            goal_map_plan = self._apply_map_frame_transform(
                self.goal_map, getattr(self, "_last_map_frame_transform", "id")
            )
            stg_y, stg_x, replan, number_action = self._plan(
                traversible, goal_map_plan, self.full_pose, cur_start, cur_start_o, self.found_goal
            )
        
        self._goal_map_src_for_visualization = str(goal_map_src)
        if self.args.visualize:
            self.visualize(traversible, observations, number_action)

        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        goal_rc = None
        goal_drift = None
        gys, gxs = np.where(self.goal_map != 0)
        if gys.size > 0:
            goal_rc = (int(gys[0]), int(gxs[0]))
            if self._prev_goal_rc is not None:
                goal_drift = abs(goal_rc[0] - self._prev_goal_rc[0]) + abs(
                    goal_rc[1] - self._prev_goal_rc[1]
                )
            self._prev_goal_rc = goal_rc
        else:
            self._prev_goal_rc = None

        self.navigate_steps += 1
        if self._cuda_empty_cache_each_step and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if goal_map_src != "keep_previous_goal_map":
            self._last_goal_map_src_effective = str(goal_map_src)
        self._emit_spin_diagnosis(
            goal_map_src=goal_map_src,
            number_action=number_action,
            replan_fbe_hit=replan_fbe_hit,
            stuck_reset_hit=stuck_reset_hit,
        )
        self._emit_nav_stage(
            goal_map_src=goal_map_src,
            number_action=number_action,
        )

        return {"action": number_action}
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    def fbe(self, traversible, start):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        # Frontier extraction should only dilate committed obstacle cells, not arbitrary non-zero
        # semantic-map noise. Otherwise tiny residual values in ``full_map`` spread into thick walls.
        fbe_obs = self.full_map[0,0].cpu().numpy() > float(self._traversible_occ_from_depth_min)
        fbe_map[
            skimage.morphology.binary_dilation(fbe_obs, skimage.morphology.disk(4))
        ] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map_raw = (diff == 1).cpu().numpy()
        frame_mode = str(getattr(self, "_last_map_frame_transform", "id"))
        frontier_map_plan = self._apply_map_frame_transform(frontier_map_raw, frame_mode)
        frontier_locations_plan = np.argwhere(frontier_map_plan)
        num_frontiers = int(frontier_locations_plan.shape[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance in planner frame
        planner = FMMPlanner(traversible, None)
        state = [int(start[0]) + 1, int(start[1]) + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist
        frontier_locations_plan_b = frontier_locations_plan + 1
        distances = fmm_dist[
            frontier_locations_plan_b[:, 0], frontier_locations_plan_b[:, 1]
        ] / 20

        # Keep scenegraph scoring in raw map coordinates (legacy semantics).
        h, w = int(frontier_map_plan.shape[0]), int(frontier_map_plan.shape[1])
        frontier_locations_raw = np.zeros_like(frontier_locations_plan, dtype=np.int32)
        for i, loc in enumerate(frontier_locations_plan):
            rr, cc = self._inverse_transform_rc_by_frame_mode(
                int(loc[0]), int(loc[1]), h, w, frame_mode
            )
            frontier_locations_raw[i, 0] = int(rr)
            frontier_locations_raw[i, 1] = int(cc)
        frontier_locations_raw_b = frontier_locations_raw + 1
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16_raw_b = frontier_locations_raw_b[idx_16]
        self.frontier_locations = frontier_locations_raw_b
        self.frontier_locations_16 = frontier_locations_16_raw_b
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        scores = self.scenegraph.score(frontier_locations_16_raw_b, num_16_frontiers)
                
        scores += 2 * distances_16_inverse
        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations_raw_b[idx_16_max] - 1
        goal = (
            int(np.clip(int(goal[0]), 0, self.map_size - 1)),
            int(np.clip(int(goal[1]), 0, self.map_size - 1)),
        )
        self.scores = scores
        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        return _get_goal_gps_impl(observations['gps'], observations['compass'], angle, distance)

    def _goal_gps_to_raw_goal_map_rc(self, goal_gps):
        return goal_gps_to_map_rc(goal_gps, self.map_size_cm, self.resolution, self.map_size)

    def _raw_goal_map_rc_to_goal_gps(self, r: int, c: int):
        return map_rc_to_goal_gps(r, c, self.map_size_cm, self.resolution)

    def _possible_goal_safe_mask_plan(self):
        free_core = getattr(self, "_last_free_from_depth_core", None)
        obs_core = getattr(self, "_last_obs_dilated_core", None)
        if free_core is None or obs_core is None:
            return None
        free_core = np.asarray(free_core, dtype=bool)
        obs_core = np.asarray(obs_core, dtype=bool)
        if free_core.ndim != 2 or obs_core.ndim != 2 or free_core.shape != obs_core.shape:
            return None

        known_free = free_core
        visited = getattr(self, "visited", None)
        if visited is not None:
            visited = np.asarray(visited == 1, dtype=bool)
            if visited.ndim == 2 and visited.shape == free_core.shape:
                visited = self._apply_map_frame_transform(
                    visited, getattr(self, "_last_traversible_frame_mode", "id")
                )
                known_free = np.logical_or(known_free, visited)
        return np.logical_and(known_free, np.logical_not(obs_core))

    def _project_raw_goal_rc_to_possible_goal_safe_cell(self, r_raw: int, c_raw: int):
        safe_plan = self._possible_goal_safe_mask_plan()
        if safe_plan is None:
            return None
        h, w = int(safe_plan.shape[0]), int(safe_plan.shape[1])
        frame_mode = str(getattr(self, "_last_traversible_frame_mode", "id"))
        r_plan, c_plan = self._transform_rc_by_frame_mode(
            int(r_raw), int(c_raw), h, w, frame_mode
        )
        if safe_plan[r_plan, c_plan]:
            return int(r_raw), int(c_raw), 0.0
        candidate_idx = np.argwhere(safe_plan)
        if candidate_idx.shape[0] <= 0:
            return None
        d2 = (candidate_idx[:, 0] - int(r_plan)) ** 2 + (candidate_idx[:, 1] - int(c_plan)) ** 2
        best = candidate_idx[int(np.argmin(d2))]
        rr_raw, cc_raw = self._inverse_transform_rc_by_frame_mode(
            int(best[0]), int(best[1]), h, w, frame_mode
        )
        return int(rr_raw), int(cc_raw), float(np.sqrt(float(np.min(d2))))

    def _build_possible_goal_traversible(self):
        safe_plan = self._possible_goal_safe_mask_plan()
        if safe_plan is None:
            return None
        traversible = np.asarray(safe_plan, dtype=np.float32).copy()
        start_rc = getattr(self, "_last_traversible_start", None)
        if start_rc is not None and len(start_rc) >= 2:
            sy = int(np.clip(int(start_rc[0]), 0, traversible.shape[0] - 1))
            sx = int(np.clip(int(start_rc[1]), 0, traversible.shape[1] - 1))
            traversible[sy, sx] = 1.0
        h, w = traversible.shape
        bounded = np.ones((h + 2, w + 2), dtype=np.float32)
        bounded[1:h + 1, 1:w + 1] = traversible
        return bounded

    def _get_active_goal_map_rc(self):
        goal_map = getattr(self, "goal_map", None)
        if goal_map is None:
            return None
        gm = np.asarray(goal_map)
        if gm.ndim != 2:
            return None
        ys, xs = np.where(gm > 0)
        if ys.size <= 0:
            return None
        return int(ys[0]), int(xs[0])

    def _get_active_goal_gps(self, observations=None):
        """
        Return the current navigation target in gps/map-centered coordinates.
        Prefer reconstructing it from ``goal_map`` so debug outputs match the green
        long-term goal marker exactly across found-goal / possible-goal / gt-world modes.
        """
        goal_rc = self._get_active_goal_map_rc()
        if goal_rc is not None:
            return self._raw_goal_map_rc_to_goal_gps(goal_rc[0], goal_rc[1])

        if bool(getattr(self, "found_possible_goal", False)):
            goal = getattr(self, "possible_goal_temp_gps", None)
            if goal is not None and len(goal) >= 2:
                return np.asarray(goal, dtype=np.float32).reshape(2)

        if bool(getattr(self, "_active_gt_world_nav", False)) and isinstance(observations, dict):
            tw = observations.get("target_world_xy")
            if tw is not None and len(tw) >= 2:
                try:
                    tx = float(tw[0])
                    ty = float(tw[1])
                    half_map_m = float(self.map_size_cm) / 200.0
                    return np.array(
                        [tx - half_map_m, half_map_m - ty], dtype=np.float32
                    )
                except (TypeError, ValueError):
                    pass

        goal = getattr(self, "goal_gps", None)
        if goal is not None and len(goal) >= 2:
            return np.asarray(goal, dtype=np.float32).reshape(2)
        return None

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self._get_active_goal_gps(observations)
        return _get_relative_goal_gps_impl(observations['gps'], observations['compass'], goal_gps)

    def _get_goal_stop_status(self, agent_pose, goal_gps=None):
        if goal_gps is None:
            goal_gps = getattr(self, "goal_gps", None)
        return _get_goal_stop_status_impl(agent_pose, goal_gps, self.map_size_cm)

    def _get_face_goal_stop_state(self, agent_pose):
        """Resolve the best available goal GPS for face-goal override.

        Priority: active goal from goal_map > saved face-goal GPS from
        a previous turn step > self.goal_gps.  Returns the stop-status
        dict with an extra ``_goal_gps`` key so the caller can persist it.
        """
        goal_gps = self._get_active_goal_gps()
        if goal_gps is None:
            goal_gps = getattr(self, "_face_goal_gps", None)
        if goal_gps is None:
            goal_gps = getattr(self, "goal_gps", None)
        result = _get_goal_stop_status_impl(agent_pose, goal_gps, self.map_size_cm)
        if result is not None and goal_gps is not None:
            result["_goal_gps"] = np.asarray(goal_gps, dtype=np.float32).copy()
        return result
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.fbe_free_map_vis = copy.deepcopy(self.full_map).to(self.device)
        self.default_free_map = np.zeros((full_w, full_h), dtype=np.float32)
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        # Episode-local GPS origin. Filled from the first observation of each reset so the
        # map stays centered on the agent's start pose instead of world coordinate (0, 0).
        self.origins = np.full((2,), np.nan, dtype=np.float32)
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def _clamp_full_pose_xy_to_map_meters(self):
        """Clamp world x,y so grid indices stay in [0, map_size-1] (avoids IndexError when GPS drifts past map edge)."""
        cell = float(self.map_resolution) / 100.0
        half = 0.5 * cell
        max_xy = (float(self.map_size) - 0.5) * cell
        self.full_pose[0] = torch.clamp(self.full_pose[0], half, max_xy)
        self.full_pose[1] = torch.clamp(self.full_pose[1], half, max_xy)

    def _rebase_gps_to_episode_origin(self, gps):
        gps_arr = np.asarray(gps, dtype=np.float64).reshape(-1).copy()
        if gps_arr.shape[0] < 2:
            return gps_arr
        if not np.all(np.isfinite(self.origins[:2])):
            self.origins = gps_arr[:2].astype(np.float32).copy()
        gps_arr[:2] -= self.origins[:2]
        return gps_arr

    def _get_update_map_pose_inputs(self, observations):
        if bool(getattr(self, "_diag_disable_pose_bridge", False)):
            gps_src = observations.get("gps_client", observations.get("gps"))
            compass_src = observations.get("compass_client", observations.get("compass"))
            pose_source = "client_raw"
        else:
            gps_src = observations.get("gps")
            compass_src = observations.get("compass")
            pose_source = "bridge"
        gps_arr = np.asarray(gps_src, dtype=np.float64).reshape(-1).copy()
        compass_arr = np.asarray(compass_src, dtype=np.float64).reshape(-1).copy()
        return gps_arr, compass_arr, pose_source

    def update_map(self, observations):
        map_gps, map_compass, map_pose_source = self._get_update_map_pose_inputs(observations)
        self._set_full_pose_from_arrays(map_gps, map_compass)
        pose_h, pose_w = int(self.full_map.shape[-2]), int(self.full_map.shape[-1])
        pose_rc = self._robot_map_rc_from_full_pose(pose_h, pose_w)
        prev_pose_rc = getattr(self, "_diag_prev_full_map_pose_rc", None)
        if prev_pose_rc is None:
            pose_delta_r = 0
            pose_delta_c = 0
            pose_delta_l2 = 0.0
        else:
            pose_delta_r = int(pose_rc[0] - prev_pose_rc[0])
            pose_delta_c = int(pose_rc[1] - prev_pose_rc[1])
            pose_delta_l2 = float(
                np.hypot(float(pose_delta_r), float(pose_delta_c))
            )
        self._diag_prev_full_map_pose_rc = (int(pose_rc[0]), int(pose_rc[1]))
        self._diag_prev_full_map_pose_deg = float(self.full_pose[2].detach().cpu().item())
        if self._consume_mapping_suspend("update_map"):
            return
        collect_fusion = bool(getattr(self.sem_map_module, "_diag_collect_fusion_stats", False))
        fm_prev = (
            self.full_map[0, 0].detach().cpu().numpy() if collect_fusion else None
        )
        with torch.no_grad():
            self.full_map = self.sem_map_module(
                torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device),
                self.full_pose,
                self.full_map,
            )
        fusion_diag = getattr(self.sem_map_module, "_last_fusion_diag", None)
        if fm_prev is not None:
            fm_now = self.full_map[0, 0].detach().cpu().numpy()
            occ_thr = float(self._traversible_occ_from_depth_min)
            prev_occ = fm_prev > occ_thr
            now_occ = fm_now > occ_thr
            add_occ = np.logical_and(np.logical_not(prev_occ), now_occ)
            del_occ = np.logical_and(prev_occ, np.logical_not(now_occ))
            add_ratio = float(np.mean(add_occ))
            del_ratio = float(np.mean(del_occ))

            h, w = int(now_occ.shape[0]), int(now_occ.shape[1])
            px = float(self.full_pose[0].detach().cpu().item())
            py = float(self.full_pose[1].detach().cpu().item())
            sy = int(round((float(self.map_size_cm) / 100.0 - py) * 100.0 / float(self.map_resolution)))
            sx = int(round(px * 100.0 / float(self.map_resolution)))
            sy = max(0, min(h - 1, sy))
            sx = max(0, min(w - 1, sx))
            win = 40
            y0 = max(0, sy - win)
            y1 = min(h, sy + win + 1)
            x0 = max(0, sx - win)
            x1 = min(w, sx + win + 1)
            if y1 > y0 and x1 > x0:
                add_local = float(np.mean(add_occ[y0:y1, x0:x1]))
                occ_local = float(np.mean(now_occ[y0:y1, x0:x1]))
                add_local_count = int(np.sum(add_occ[y0:y1, x0:x1]))
                prev_local_count = int(np.sum(prev_occ[y0:y1, x0:x1]))
                now_local_count = int(np.sum(now_occ[y0:y1, x0:x1]))
            else:
                add_local = float("nan")
                occ_local = float("nan")
                add_local_count = 0
                prev_local_count = 0
                now_local_count = 0
            yy, xx = np.ogrid[:h, :w]
            rr2 = (yy - sy) ** 2 + (xx - sx) ** 2
            near = rr2 <= (12 ** 2)
            ring = np.logical_and(rr2 >= (4 ** 2), rr2 <= (10 ** 2))
            add_near = float(np.mean(np.logical_and(add_occ, near)))
            add_ring = float(np.mean(np.logical_and(add_occ, ring)))
        # collision_map was a one-time numpy slice at init; full_map is a new tensor each step.
        # Reseed from current depth map first (depth-only path), then optionally fuse OG occupancy.
        if self._og_occ_reseed:
            fm = self.full_map[0, 0].detach().cpu().numpy()
            self.collision_map = (fm > 0.5).astype(np.float32).copy()
        self._fuse_og_occupancy_into_collision(observations)
        self._clear_collision_near_robot()

    def _clear_collision_near_robot(self):
        # Robot-local prior free support is handled via ``default_free_map`` / ``fbe_free_map``.
        # Do not clear current obstacle evidence here: observed depth/collision must override the prior.
        return

    def _robot_map_rc_from_full_pose(self, h: int, w: int) -> tuple[int, int]:
        return robot_map_rc(self.full_pose, self.map_size_cm, self.resolution, h, w)

    @staticmethod
    def _disk_mask_for_center(h: int, w: int, cy: int, cx: int, radius_cells: int):
        return disk_mask(h, w, cy, cx, radius_cells)

    def _fuse_og_occupancy_into_collision(self, observations):
        """
        Optional: merge OmniGibson ScanSensor local occupancy (HTTP ``og_occupancy``) into collision_map.

        ``og_occupancy`` is H×W or H×W×1 float32; values ~0 obstacle, ~0.5 unknown, ~1 free (see OccupancyGridState).
        ``og_occupancy_meta`` may include ``resolution`` (int, default H) and ``range_m`` (float, local grid extent).
        """
        occ = observations.get("og_occupancy")
        if occ is None:
            return
        meta = observations.get("og_occupancy_meta") or {}
        occ = np.asarray(occ, dtype=np.float32)
        if occ.ndim == 3:
            occ = occ[..., 0]
        h, w = occ.shape
        range_m = float(meta.get("range_m", 5.0))
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        # Match OG rasterization: meters per cell from actual grid side (meta ``resolution`` can disagree with H×W).
        cell_m = range_m / float(max(h, w, 1))
        # Match utils_fmm/mapping.py pose_transform: heading_deg -= 90 before cos/sin (depth → map).
        # Using raw compass here rotated OG occupancy 90° vs depth-built free/semantic maps.
        yaw = float(np.asarray(observations["compass"]).reshape(-1)[0]) - (np.pi / 2.0)
        px = float(self.full_pose[0].cpu().item())
        py = float(self.full_pose[1].cpu().item())
        jj, ii = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        lx = (jj - cx) * cell_m
        ly = -(ii - cy) * cell_m
        c, s = np.cos(yaw), np.sin(yaw)
        mx = px + lx * c - ly * s
        my = py + lx * s + ly * c
        gi = np.rint(mx * 100.0 / float(self.map_resolution)).astype(np.int32)
        gj = np.rint(my * 100.0 / float(self.map_resolution)).astype(np.int32)
        ms = int(self.map_size)
        valid = (gi >= 0) & (gi < ms) & (gj >= 0) & (gj < ms)
        obs_hi = float(self._og_occ_obs_max)
        free_lo = float(self._og_occ_free_min)
        # OG discrete classes are ~0 / 0.5 / 1; keep obstacle band strict so unknown (0.5) is never painted.
        obs_mask = valid & (occ < obs_hi) if self._og_occ_fuse_obs else np.zeros_like(valid, dtype=bool)
        free_mask = valid & (occ > free_lo)
        if np.any(obs_mask):
            gi_o = np.clip(gi[obs_mask], 0, ms - 1)
            gj_o = np.clip(gj[obs_mask], 0, ms - 1)
            self.collision_map[gi_o, gj_o] = 1.0
        if np.any(free_mask):
            gi_f = np.clip(gi[free_mask], 0, ms - 1)
            gj_f = np.clip(gj[free_mask], 0, ms - 1)
            self.collision_map[gi_f, gj_f] = 0.0

    def update_free_map(self, observations):
        self._set_full_pose_from_arrays(observations["gps"], observations["compass"])
        if self._consume_mapping_suspend("update_free_map"):
            return
        with torch.no_grad():
            self.fbe_free_map = self.free_map_module(
                torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device),
                self.full_pose,
                self.fbe_free_map,
            )
        h, w = int(self.fbe_free_map.shape[-2]), int(self.fbe_free_map.shape[-1])
        sy, sx = self._robot_map_rc_from_full_pose(h, w)
        radius_cells = int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0))
        obs_map = self.full_map[0, 0].detach().cpu().numpy() > float(self._traversible_occ_from_depth_min)
        coll_map = np.asarray(self.collision_map, dtype=np.float32) > 0.5
        obs_or_coll = np.logical_or(obs_map, coll_map)
        if np.any(obs_or_coll):
            obs_or_coll_t = torch.from_numpy(obs_or_coll).to(self.device)
            self.fbe_free_map[0, 0][obs_or_coll_t] = 0.0
        # Keep a visualization-only copy before adding synthetic robot-local free prior.
        self.fbe_free_map_vis = self.fbe_free_map.clone()
        default_free = self._disk_mask_for_center(h, w, sy, sx, radius_cells)
        default_free = np.logical_and(default_free, np.logical_not(obs_or_coll))
        self.default_free_map.fill(0.0)
        self.default_free_map[default_free] = 1.0
        if np.any(default_free):
            default_free_t = torch.from_numpy(default_free).to(self.device)
            self.fbe_free_map[0, 0][default_free_t] = 1.0
    
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        with torch.no_grad():
            self.room_map = self.room_map_module(
                torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device),
                self.full_pose,
                self.room_map,
                torch.from_numpy(type_mask).to(self.device).type(torch.float32),
                score_vec,
            )

    def _apply_map_frame_transform(self, arr, mode: str = "flipud"):
        return apply_flipud(arr)

    def _transform_rc_by_frame_mode(self, r: int, c: int, h: int, w: int, mode: str = "flipud"):
        return transform_rc_flipud(r, c, h, w)

    def _inverse_transform_rc_by_frame_mode(self, r: int, c: int, h: int, w: int, mode: str = "flipud"):
        return inverse_transform_rc_flipud(r, c, h, w)

    def get_traversible(self, map_pred, pose_pred):
        grid = np.asarray(map_pred, dtype=np.float32)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start_raw_pose = [int(r * 100 / self.map_resolution - gy1), int(c * 100 / self.map_resolution - gx1)]
        start_raw_pose = pu.threshold_poses(start_raw_pose, grid.shape)
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        occ_map = grid[y1:y2, x1:x2]
        coll_map = np.asarray(
            self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2], dtype=np.float32
        )
        free_map = (
            self.fbe_free_map.detach().cpu().numpy()[0, 0][gy1:gy2, gx1:gx2][y1:y2, x1:x2]
        )

        obs_from_depth_raw = occ_map > float(self._traversible_occ_from_depth_min)
        obs_from_collision_raw = coll_map > 0.5
        free_thr = float(self._traversible_free_map_min)
        free_from_depth_raw = free_map > free_thr
        frame_mode = str(getattr(self, "_map_frame_fixed_mode", "flipud"))
        if frame_mode not in (
            "id",
            "flipud",
            "fliplr",
            "rot180",
            "rot90",
            "rot270",
            "transpose",
            "transpose_flipud",
        ):
            frame_mode = "flipud"
        frame_scores = None
        self._last_map_frame_transform = frame_mode
        occ_map_plan = self._apply_map_frame_transform(occ_map, frame_mode)
        coll_map_plan = self._apply_map_frame_transform(coll_map, frame_mode)
        obs_from_depth = self._apply_map_frame_transform(obs_from_depth_raw, frame_mode)
        obs_from_collision = self._apply_map_frame_transform(obs_from_collision_raw, frame_mode)
        free_map_plan = self._apply_map_frame_transform(free_map, frame_mode)
        free_from_depth = free_map_plan > free_thr
        h, w = int(obs_from_depth.shape[0]), int(obs_from_depth.shape[1])
        sy_raw_pose = int(np.clip(int(start_raw_pose[0]), 0, h - 1))
        sx_raw_pose = int(np.clip(int(start_raw_pose[1]), 0, w - 1))
        sy_plan_pose, sx_plan_pose = self._transform_rc_by_frame_mode(
            sy_raw_pose, sx_raw_pose, h, w, frame_mode
        )
        sy_plan, sx_plan = int(sy_plan_pose), int(sx_plan_pose)
        start_pick_mode = "pose_transform"
        start_pick_score = float("nan")
        start_pick_score_alt = float("nan")
        if frame_mode == "flipud" and bool(
            getattr(self, "_start_plan_flipud_mirror_enable", False)
        ):
            known_plan = np.logical_or(np.logical_or(obs_from_depth, obs_from_collision), free_from_depth)

            def _start_known_score(cy: int, cx: int, win: int = 24) -> float:
                y0 = max(0, int(cy) - win)
                y1 = min(h, int(cy) + win + 1)
                x0 = max(0, int(cx) - win)
                x1 = min(w, int(cx) + win + 1)
                if y1 <= y0 or x1 <= x0:
                    return 0.0
                return float(np.mean(known_plan[y0:y1, x0:x1]))

            sy_plan_alt = int(np.clip((h - 1) - sy_plan_pose, 0, h - 1))
            sx_plan_alt = int(sx_plan_pose)
            score_pose = _start_known_score(sy_plan_pose, sx_plan_pose)
            score_alt = _start_known_score(sy_plan_alt, sx_plan_alt)
            start_pick_score = float(score_pose)
            start_pick_score_alt = float(score_alt)
            min_delta = float(getattr(self, "_start_plan_flipud_mirror_min_delta", 0.05))
            pose_known_min = float(
                getattr(self, "_start_plan_flipud_pose_known_min", 0.35)
            )
            if (
                (score_alt - score_pose) >= min_delta
                and score_pose < pose_known_min
            ):
                sy_plan, sx_plan = sy_plan_alt, sx_plan_alt
                start_pick_mode = "flipud_mirror_y"
            else:
                start_pick_mode = "pose_transform_keep"
        elif frame_mode == "flipud":
            start_pick_mode = "pose_transform_mirror_disabled"
        sy_raw, sx_raw = self._inverse_transform_rc_by_frame_mode(
            int(sy_plan), int(sx_plan), h, w, frame_mode
        )
        start_plan = [int(sy_plan), int(sx_plan)]
        self._last_start_raw = (int(sy_raw), int(sx_raw))
        self._last_start_plan = (int(sy_plan), int(sx_plan))
        self._last_start_pick_mode = str(start_pick_mode)

        vy0 = max(0, int(sy_raw) - 2)
        vy1 = min(int(grid.shape[0]), int(sy_raw) + 3)
        vx0 = max(0, int(sx_raw) - 2)
        vx1 = min(int(grid.shape[1]), int(sx_raw) + 3)
        if vy1 > vy0 and vx1 > vx0:
            self.visited[gy1:gy2, gx1:gx2][vy0:vy1, vx0:vx1] = 1

        inter = float(np.sum(np.logical_and(obs_from_depth, obs_from_collision)))
        union = float(np.sum(np.logical_or(obs_from_depth, obs_from_collision)))
        depth_n = float(np.sum(obs_from_depth))
        coll_n = float(np.sum(obs_from_collision))
        occ_iou = (inter / union) if union > 0.0 else float("nan")
        occ_precision = (inter / coll_n) if coll_n > 0.0 else float("nan")
        occ_recall = (inter / depth_n) if depth_n > 0.0 else float("nan")
        robot_local = None
        if (
            bool(getattr(self, "_traversible_clear_robot_obs_enable", False))
            and int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0)) > 0
        ):
            default_free_raw = np.asarray(
                self.default_free_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2], dtype=np.float32
            ) > 0.5
            robot_local = self._apply_map_frame_transform(default_free_raw, frame_mode)
            free_from_depth = np.logical_or(free_from_depth, robot_local)
        border_cells = int(getattr(self, "_traversible_clear_border_obs_cells", 0))
        if border_cells > 0:
            b = int(min(border_cells, (h - 1) // 2, (w - 1) // 2))
            if b > 0:
                obs_from_depth[:b, :] = False
                obs_from_depth[-b:, :] = False
                obs_from_depth[:, :b] = False
                obs_from_depth[:, -b:] = False
                obs_from_collision[:b, :] = False
                obs_from_collision[-b:, :] = False
                obs_from_collision[:, :b] = False
                obs_from_collision[:, -b:] = False
        selem = skimage.morphology.disk(int(self._traversible_obstacle_inflation_cells))
        obs_core = np.logical_or(obs_from_depth, obs_from_collision)
        obs_dilated = skimage.morphology.binary_dilation(obs_core, selem)

        if self._traversible_require_explored:
            traversible = np.logical_and(free_from_depth, np.logical_not(obs_dilated))
        else:
            traversible = np.logical_not(obs_dilated)

        # Startup bootstrap: explored-only policy can produce all-blocked maps before free_map warms up.
        if self._traversible_require_explored and self._traversible_bootstrap_enable:
            tr_ratio = float(np.mean(traversible))
            if (
                tr_ratio < float(self._traversible_bootstrap_min_free_ratio)
                and int(self._traversible_bootstrap_radius_cells) > 0
            ):
                yy, xx = np.ogrid[: traversible.shape[0], : traversible.shape[1]]
                rr2 = (yy - sy_plan) ** 2 + (xx - sx_plan) ** 2
                local = rr2 <= int(self._traversible_bootstrap_radius_cells) ** 2
                local_non_obs = np.logical_and(local, np.logical_not(obs_dilated))
                traversible = np.logical_or(traversible, local_non_obs)

        sy = int(sy_plan - y1)
        sx = int(sx_plan - x1)
        y0s = max(0, sy - 1)
        y1s = min(traversible.shape[0], sy + 2)
        x0s = max(0, sx - 1)
        x1s = min(traversible.shape[1], sx + 2)
        if y1s > y0s and x1s > x0s:
            start_patch = np.zeros_like(traversible, dtype=bool)
            start_patch[y0s:y1s, x0s:x1s] = True
            # Never open cells already considered obstacles.
            traversible[np.logical_and(start_patch, np.logical_not(obs_dilated))] = 1
        # Keep the exact agent cell traversible to avoid planner deadlock on transient map noise.
        traversible[sy, sx] = 1
        traversible = traversible * 1.

        visited_mask_raw = self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1
        visited_mask = self._apply_map_frame_transform(visited_mask_raw, frame_mode)
        # Visited should not override current obstacle evidence.
        traversible[np.logical_and(visited_mask, np.logical_not(obs_dilated))] = 1
        # Cache core grids for action-level forward guard in ``_plan``.
        self._last_obs_dilated_core = np.asarray(obs_dilated, dtype=bool).copy()
        self._last_free_from_depth_core = np.asarray(free_from_depth, dtype=bool).copy()
        self._last_traversible_core = np.asarray(traversible > 0.5, dtype=bool).copy()
        self._last_traversible_start = (int(sy_plan), int(sx_plan))
        self._last_traversible_frame_mode = str(frame_mode)
        traversible = add_boundary(traversible)
        return traversible, start_plan, start_o

    def _world_xy_m_to_grid_rc_for_nav(self, tx: float, ty: float) -> tuple[int, int]:
        return world_xy_to_grid_rc(tx, ty, self.resolution, self.map_size)

    def _target_world_cell_visible_on_map(self, r: int, c: int, radius: int = 3) -> bool:
        """True if visited, free-space, or semantic map has evidence near (r, c)."""
        h, w = int(self.visited.shape[0]), int(self.visited.shape[1])
        r0 = max(0, r - radius)
        r1 = min(h, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(w, c + radius + 1)
        if r0 < r1 and c0 < c1 and np.max(self.visited[r0:r1, c0:c1]) > 0:
            return True
        fb = self.fbe_free_map.detach().cpu().numpy()[0, 0]
        if fb.shape[0] >= r1 and fb.shape[1] >= c1 and r0 < r1 and c0 < c1:
            if np.max(fb[r0:r1, c0:c1]) > 0.45:
                return True
        fm = self.full_map.detach().cpu().numpy()[0, 0]
        if fm.shape[0] >= r1 and fm.shape[1] >= c1 and r0 < r1 and c0 < c1:
            if np.max(np.abs(fm[r0:r1, c0:c1])) > 0.02:
                return True
        return False

    def _try_apply_gt_world_target_goal_map(self, observations: dict) -> bool:
        """
        If runtime allows and ``target_world_xy`` lies on explored occupancy, set ``goal_map`` to that cell
        and use existing FMM local policy (``goal_found`` stays false — no GLIP block_goal path).
        """
        if not self._gt_pathplan_when_target_on_map:
            return False
        tw = observations.get("target_world_xy")
        if tw is None or len(tw) < 2:
            return False
        try:
            tx, ty = float(tw[0]), float(tw[1])
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(tx) and math.isfinite(ty)):
            return False
        r_abs, c_abs = self._world_xy_m_to_grid_rc_for_nav(tx, ty)
        visible_abs = self._target_world_cell_visible_on_map(r_abs, c_abs)

        if not visible_abs:
            return False
        g = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        g[r_abs, c_abs] = 1.0
        self.goal_map = g
        self.not_use_random_goal()
        self.first_fbe = False
        self._active_gt_world_nav = True
        return True

    def _is_forward_blocked(self, start, start_o):
        if not bool(getattr(self, "_forward_guard_enable", False)):
            return False, None
        obs = getattr(self, "_last_obs_dilated_core", None)
        tr = getattr(self, "_last_traversible_core", None)
        if obs is None or tr is None:
            return False, None
        obs = np.asarray(obs, dtype=bool)
        tr = np.asarray(tr, dtype=bool)
        if obs.ndim != 2 or tr.ndim != 2 or obs.shape != tr.shape:
            return False, None

        sy = int(np.clip(int(start[0]), 0, obs.shape[0] - 1))
        sx = int(np.clip(int(start[1]), 0, obs.shape[1] - 1))
        lookahead_cells = max(
            1,
            int(
                round(
                    max(0.01, float(self._forward_guard_lookahead_m))
                    * 100.0
                    / float(self.map_resolution)
                )
            ),
        )
        samples = max(1, int(self._forward_guard_samples))
        theta = math.radians(float(start_o))
        clear_radius_cells = int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0))
        cone_half_deg = float(getattr(self, "_forward_guard_cone_half_angle_deg", 25.0))

        # Check a fan of rays spanning [-cone_half_deg, +cone_half_deg] around heading.
        n_rays = max(3, int(cone_half_deg / 5.0) * 2 + 1)
        ray_angles = [
            theta + math.radians(cone_half_deg * (2.0 * i / max(n_rays - 1, 1) - 1.0))
            for i in range(n_rays)
        ]

        for k in range(1, samples + 1):
            dist_cells = max(1, int(round(float(k) * lookahead_cells / float(samples))))
            if dist_cells <= clear_radius_cells:
                continue
            for ray_idx, ray_theta in enumerate(ray_angles):
                dr = math.sin(ray_theta)
                dc = math.cos(ray_theta)
                rr = int(round(sy + dr * dist_cells))
                cc = int(round(sx + dc * dist_cells))
                if rr < 0 or rr >= obs.shape[0] or cc < 0 or cc >= obs.shape[1]:
                    info = {
                        "block_rc": (rr, cc),
                        "block_type": "oob",
                        "sample_idx": int(k),
                        "ray_idx": int(ray_idx),
                        "n_rays": int(n_rays),
                        "cone_half_deg": float(cone_half_deg),
                        "dist_cells": int(dist_cells),
                        "lookahead_cells": int(lookahead_cells),
                        "samples": int(samples),
                        "frame_mode": str(getattr(self, "_last_traversible_frame_mode", "id")),
                    }
                    return True, info
                hit_obs = bool(obs[rr, cc])
                hit_non_tr = not bool(tr[rr, cc])
                if hit_obs or hit_non_tr:
                    if hit_obs and hit_non_tr:
                        btype = "obs+non_traversible"
                    elif hit_obs:
                        btype = "obs"
                    else:
                        btype = "non_traversible"
                    info = {
                        "block_rc": (int(rr), int(cc)),
                        "block_type": btype,
                        "sample_idx": int(k),
                        "ray_idx": int(ray_idx),
                        "n_rays": int(n_rays),
                        "cone_half_deg": float(cone_half_deg),
                        "dist_cells": int(dist_cells),
                        "lookahead_cells": int(lookahead_cells),
                        "samples": int(samples),
                        "frame_mode": str(getattr(self, "_last_traversible_frame_mode", "id")),
                    }
                    return True, info
        return False, None

    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold
            client_collision = getattr(self, "_client_collision_prev", None) or {}
            client_contact = bool(client_collision.get("base_contact", False))
            collision_reasons = []
            if dist < col_threshold:
                collision_reasons.append("low_gps_delta")
            if client_contact:
                collision_reasons.append("client_contact")

            if collision_reasons: # Collision
                self.former_collide += 1
                painted = []
                newly_marked = 0
                lateral_width = 2
                cos_t = np.cos(np.deg2rad(t1))
                sin_t = np.sin(np.deg2rad(t1))
                perp_cos = -sin_t
                perp_sin = cos_t
                for i in range(length):
                    for lat in range(-lateral_width, lateral_width + 1):
                        wx = x1 + 0.05 * ((i + buf) * cos_t + lat * perp_cos)
                        wy = y1 + 0.05 * ((i + buf) * sin_t + lat * perp_sin)
                        r, c = wy, wx
                        r = int(round(r * 100 / self.map_resolution))
                        c = int(round(c * 100 / self.map_resolution))
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        if self.collision_map[r, c] <= 0.5:
                            newly_marked += 1
                        painted.append((int(r), int(c)))
                        self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        should_face_goal = (
            goal_found
            or bool(getattr(self, "_active_gt_world_nav", False))
            or bool(getattr(self, "_face_goal_pending", False))
        )
        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found, should_face_goal)
        goal_stop_state = self._get_face_goal_stop_state(agent_pose) if should_face_goal else None
        goal_stop_override = False

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg
        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            align_thr = self._planner_align_angle_deg
            hys = max(0.0, float(getattr(self, "_planner_align_hysteresis_deg", 0.0)))
            forward_exit = max(0.0, align_thr - hys)
            face_eps = float(getattr(self, "_planner_face_goal_max_error_deg", -1.0))
            use_strict_face_goal = face_eps >= 0.0
            observed_turn_deg = None
            if bool(getattr(self, "_planner_use_observed_turn", False)):
                obs_turn = getattr(self, "_planner_observed_turn_deg", None)
                if obs_turn is not None and math.isfinite(float(obs_turn)):
                    observed_turn_deg = max(
                        float(getattr(self, "_planner_min_effective_turn_deg", 1.0)),
                        abs(float(obs_turn)),
                    )
            align_thr_cfg = float(align_thr)
            forward_exit_cfg = float(forward_exit)
            face_eps_cfg = float(face_eps)
            if observed_turn_deg is not None:
                ratio = max(
                    0.0,
                    float(getattr(self, "_planner_forward_error_turn_ratio", 0.5)),
                )
                align_thr = min(float(align_thr), float(observed_turn_deg))
                forward_exit = min(
                    float(forward_exit),
                    max(
                        float(getattr(self, "_planner_min_effective_turn_deg", 1.0)),
                        float(observed_turn_deg) * ratio,
                    ),
                )
                if use_strict_face_goal:
                    face_eps = min(
                        float(face_eps),
                        max(
                            float(getattr(self, "_planner_min_effective_turn_deg", 1.0)),
                            float(observed_turn_deg) * ratio,
                        ),
                    )

            if use_strict_face_goal:
                # Face STG before driving: forward only inside a tight angular window.
                if self.former_collide < 10:
                    if abs(relative_angle) <= face_eps:
                        action = 1
                    elif relative_angle > 0:
                        action = 3
                    else:
                        action = 2
                elif self.prev_action == 1:
                    if relative_angle > 0:
                        action = 3
                    else:
                        action = 2
                else:
                    action = 1
                if self.former_collide >= 10 and self.prev_action != 1:
                    self.former_collide = 0
                if stg_y == start[0] and stg_x == start[1]:
                    action = 0 if replan else 1
            else:
                if self.former_collide < 10:
                    if self.prev_action in (3, 6):
                        if abs(relative_angle) <= forward_exit:
                            action = 1
                        elif relative_angle > 0:
                            action = 3
                        else:
                            action = 2
                    elif self.prev_action == 2:
                        if abs(relative_angle) <= forward_exit:
                            action = 1
                        elif relative_angle < 0:
                            action = 2
                        else:
                            action = 3
                    elif relative_angle > align_thr:
                        action = 3
                    elif relative_angle < -align_thr:
                        action = 2
                    else:
                        action = 1
                elif self.prev_action == 1:
                    if relative_angle > 0:
                        action = 3 # Right
                    else:
                        action = 2 # Left
                else:
                    action = 1
                if self.former_collide >= 10 and self.prev_action != 1:
                    self.former_collide  = 0
                if stg_y == start[0] and stg_x == start[1]:
                    action = 0 if replan else 1

            action_raw = int(action)
            fg_blocked = False
            fg_info = None
            if action_raw == 1 and bool(getattr(self, "_forward_guard_enable", False)):
                fg_blocked, fg_info = self._is_forward_blocked(start, start_o)
                if fg_blocked:
                    if abs(float(relative_angle)) > 1e-3:
                        action = 3 if float(relative_angle) > 0 else 2
                        turn_pick = "stg_relative_angle"
                    elif int(self.prev_action) in (2, 3, 6):
                        action = 2 if int(self.prev_action) == 2 else 3
                        turn_pick = "prev_turn_direction"
                    else:
                        action = 3
                        turn_pick = "default_right"

        if (
            should_face_goal
            and goal_stop_state is not None
            and goal_stop_state["distance_m"] <= float(self._goal_stop_distance_threshold_m)
        ):
            goal_stop_override = True
            if (
                goal_stop_state["distance_m"] <= 1e-3
                or abs(goal_stop_state["heading_error_deg"])
                <= float(self._goal_stop_face_threshold_deg)
                or self._face_goal_turn_steps >= self._face_goal_max_turn_steps
            ):
                action = 0
                self._face_goal_pending = False
                self._face_goal_gps = None
                self._face_goal_turn_steps = 0
            elif goal_stop_state["heading_error_deg"] > 0:
                action = 2
                self._face_goal_pending = True
                self._face_goal_gps = goal_stop_state.get("_goal_gps")
                self._face_goal_turn_steps += 1
            else:
                action = 3
                self._face_goal_pending = True
                self._face_goal_gps = goal_stop_state.get("_goal_gps")
                self._face_goal_turn_steps += 1

        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found, should_face_goal=False):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None, step_size=self._fmm_step_size)
        if goal_found or should_face_goal:
            self.planner.stop_cond = max(
                0.05, float(self._goal_stop_distance_threshold_m)
            )
        else:
            self.planner.stop_cond = self._fmm_stop_cond

        if self.dilation_deg!=0: 
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except Exception:
                goal = add_boundary(self.set_random_goal(), value=0)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop
    
    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0]
        goal = np.zeros_like(obstacle_map)
        np.random.seed(self.total_steps)
        goal_candidates_raw = None

        traversible_core = getattr(self, "_last_traversible_core", None)
        frame_mode = str(getattr(self, "_last_traversible_frame_mode", "id"))
        start_rc = getattr(self, "_last_traversible_start", None)
        if traversible_core is not None:
            traversible_core = np.asarray(traversible_core, dtype=bool)
            if traversible_core.ndim == 2 and traversible_core.shape == goal.shape:
                candidate_mask = traversible_core.copy()
                obs_core = getattr(self, "_last_obs_dilated_core", None)
                if obs_core is not None:
                    obs_core = np.asarray(obs_core, dtype=bool)
                    if obs_core.shape == candidate_mask.shape:
                        clear_cells = int(
                            max(
                                0,
                                getattr(self, "_random_goal_obstacle_clearance_cells", 0),
                            )
                        )
                        if clear_cells > 0:
                            selem = skimage.morphology.disk(clear_cells)
                            obs_keepout = skimage.morphology.binary_dilation(obs_core, selem)
                        else:
                            obs_keepout = obs_core
                        candidate_mask = np.logical_and(candidate_mask, np.logical_not(obs_keepout))
                if start_rc is not None and len(start_rc) >= 2:
                    sy = int(np.clip(int(start_rc[0]), 0, candidate_mask.shape[0] - 1))
                    sx = int(np.clip(int(start_rc[1]), 0, candidate_mask.shape[1] - 1))
                    min_dist_cells = int(getattr(self, "_random_goal_min_distance_cells", 0))
                    if min_dist_cells > 0:
                        yy, xx = np.ogrid[: candidate_mask.shape[0], : candidate_mask.shape[1]]
                        near_robot = ((yy - sy) ** 2 + (xx - sx) ** 2) < (min_dist_cells ** 2)
                        candidate_mask = np.logical_and(candidate_mask, np.logical_not(near_robot))
                candidate_idx = np.argwhere(candidate_mask)
                if candidate_idx.shape[0] == 0:
                    candidate_idx = np.argwhere(traversible_core)
                if candidate_idx.shape[0] > 0:
                    pick = candidate_idx[np.random.choice(candidate_idx.shape[0], 1)[0]]
                    h_goal, w_goal = self._inverse_transform_rc_by_frame_mode(
                        int(pick[0]), int(pick[1]), goal.shape[0], goal.shape[1], frame_mode
                    )
                    goal_candidates_raw = (int(h_goal), int(w_goal), int(candidate_idx.shape[0]))

        if goal_candidates_raw is None:
            goal_index = np.where((obstacle_map < 1))
            if len(goal_index[0]) != 0:
                i = np.random.choice(len(goal_index[0]), 1)[0]
                h_goal = goal_index[0][i]
                w_goal = goal_index[1][i]
            else:
                h_goal = np.random.choice(goal.shape[0], 1)[0]
                w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']

    def visualize(self, traversible, observations, number_action):
        if not (self.args and self.args.visualize):
            return
        self.refresh_scenegraph_text_for_visualization()
        fm = self.full_map[0, 0].detach().cpu().numpy()
        fb = getattr(self, "fbe_free_map_vis", self.fbe_free_map)[0, 0].detach().cpu().numpy()
        coll = np.asarray(self.collision_map, dtype=np.float32)
        h, w = fm.shape[:2]
        robot_rc = robot_map_rc(self.full_pose, self.map_size_cm, self.resolution, h, w)
        _, _, occ_panel = build_occupancy_panel(
            fm, fb, coll, None,
            float(self._traversible_occ_from_depth_min),
            float(self._traversible_free_map_min),
            robot_rc,
            int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0)),
        )
        panel_tensor = torch.from_numpy(occ_panel.astype(np.float64) / 255.0).permute(2, 0, 1)
        stg_plan_rc = paint_agent_and_goal(
            panel_tensor, self.history_pose,
            getattr(self, "goal_map", None), None,
            self.map_size_cm, self.resolution,
        )
        occ_panel = (panel_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        det_rgb = render_detection_overlay(
            self.rgb_visualization,
            getattr(self, "current_obj_predictions", None),
            self.obj_goal,
            self._navigation_goal_label_match,
        )
        goal_dist_str = "N/A" if self.goal_distance_for_vis is None else f"{self.goal_distance_for_vis:.2f}m"
        goal_map_src = str(getattr(self, "_last_goal_map_src_effective", "") or "")
        visualize_image = compose_visualization_frame(
            occ_panel=occ_panel, det_rgb=det_rgb,
            robot_rc=robot_rc, stg_plan_rc=stg_plan_rc,
            obj_goal=self.obj_goal, goal_distance_str=goal_dist_str,
            text_node=self.text_node, text_edge=self.text_edge,
            explanation=self.explanation, goal_map_src=goal_map_src,
        )
        visualize_image = visualize_image[:, :, ::-1]
        self.visualize_image_list.append(visualize_image)
        os.makedirs(os.path.dirname(self.current_frame_path), exist_ok=True)
        tmp_panel = self.current_frame_path + ".tmp.jpg"
        tmp_det = self.current_frame_det_path + ".tmp.jpg"
        cv2.imwrite(tmp_panel, visualize_image)
        cv2.imwrite(tmp_det, det_rgb[:, :, ::-1])
        os.replace(tmp_panel, self.current_frame_path)
        os.replace(tmp_det, self.current_frame_det_path)
        os.makedirs(self.current_frame_step_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.current_frame_step_dir, f"current_frame_{int(self.total_steps):06d}.jpg"), visualize_image)
        cv2.imwrite(os.path.join(self.current_frame_step_dir, f"current_frame_det_{int(self.total_steps):06d}.jpg"), det_rgb[:, :, ::-1])

    def refresh_scenegraph_text_for_visualization(self):
        self.text_node, self.text_edge, self.explanation = build_scenegraph_text(
            self.scenegraph, self.show_frame_nodes_only,
        )

    def save_video(self):
        _save_video_impl(
            self.visualize_image_list,
            os.path.join(self.visualization_dir, "video"),
            self.count_episodes,
        )

    def save_scenegraph_json_snapshot(self):
        sg = getattr(self, "scenegraph", None)
        if sg is None:
            return
        _save_scenegraph_json_impl(
            sg, int(self.total_steps), int(self.navigate_steps),
            str(getattr(self, "obj_goal", "")),
            str(getattr(self, "obj_goal_sg", "")),
            bool(getattr(self, "found_goal", False)),
            bool(getattr(self, "found_possible_goal", False)),
            self.scenegraph_json_path, self.scenegraph_json_step_dir,
        )


