import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
from matplotlib import colors
import cv2
import numpy as np
import pandas
import skimage
import torch
import habitat

from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from scenegraph import SceneGraph

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner    
from utils.utils_fmm.mapping import Semantic_Mapping
from utils.utils_glip import *
from utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    compute_crop_rect,
    draw_agent,
    draw_goal,
    line_list,
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
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
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
        # Print goal_gps_map vote totals when GLIP sees the target (e.g. radio); disable with log_goal_votes: false in yaml.
        self._log_goal_votes = bool(_runtime_get("log_goal_votes", True))
        # One line per step: goal_map source, flags, planner action — set nav_debug_trace: true in yaml to locate stalls.
        self._nav_debug_trace = bool(_runtime_get("nav_debug_trace", False))
        # GPS/compass vs STG heading: stderr [SG-Nav][pose_align] (see yaml comment).
        self._log_pose_align_trace = bool(_runtime_get("log_pose_align_trace", False))
        # HTTP / OmniGibson: fix map “forward” vs robot forward (Semantic_Mapping uses heading−90°).
        self._compass_offset_rad = float(
            np.deg2rad(float(_runtime_get("compass_offset_deg", 0.0)))
        )
        self._gps_negate_y = bool(_runtime_get("gps_negate_y", False))
        # Lightweight: print goal grid cell + Manhattan drift vs previous step (detect frontier / GPS goal hopping).
        self._log_goal_cell_drift = bool(_runtime_get("log_goal_cell_drift", False))
        # One line per step: why the agent is turning / not driving forward (see [SG-Nav][spin]).
        self._log_spin_diagnosis = bool(_runtime_get("log_spin_diagnosis", True))
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

        self.map_size_cm = int(round(float(_runtime_get("map_size_cm", 4000.0))))
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
        # Forward-motion collision heuristic: GPS delta below this (m) while driving → paint obstacle ahead.
        self.collision_threshold = float(_runtime_get("collision_threshold_m", 0.08))
        # Hard safety guard before issuing FORWARD:
        # if near lookahead cells are obstacle/non-traversible, convert FWD -> turn.
        self._forward_guard_enable = bool(_runtime_get("forward_guard_enable", True))
        self._forward_guard_lookahead_m = float(_runtime_get("forward_guard_lookahead_m", 0.30))
        self._forward_guard_samples = max(1, int(_runtime_get("forward_guard_samples", 4)))
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
        self._log_map_frame_select = bool(_runtime_get("log_map_frame_select", True))
        self._log_free_map_distribution = bool(
            _runtime_get("log_free_map_distribution", True)
        )
        self._log_obstacle_map_distribution = bool(
            _runtime_get("log_obstacle_map_distribution", True)
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
        self._last_traversible_core = None
        self._last_traversible_start = None
        self._last_traversible_frame_mode = "id"
        # Mapping density thresholds (voxel count normalization inside utils_fmm/mapping.py).
        # Lower values make obstacle/free evidence appear easier; 10 is often too strict for OG depth.
        self._sem_map_pred_threshold = float(_runtime_get("sem_map_pred_threshold", 1.0))
        self._sem_exp_pred_threshold = float(_runtime_get("sem_exp_pred_threshold", 1.0))
        self._free_map_pred_threshold = float(_runtime_get("free_map_pred_threshold", 0.5))
        self._free_exp_pred_threshold = float(_runtime_get("free_exp_pred_threshold", 0.5))
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
        self._log_mapping_height_band = bool(_runtime_get("log_mapping_height_band", True))
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
        self._last_good_camera_height_cm = None
        self._last_good_camera_pitch_deg = None
        self._log_camera_extrinsic_clamp = bool(_runtime_get("log_camera_extrinsic_clamp", True))
        # stderr: one line per step when camera_pose_world is applied (see _sync_camera_extrinsics).
        self._log_camera_pose_world = bool(_runtime_get("log_camera_pose_world", False))
        # Print coordinate-frame assumptions once per reset; keep true while validating planner/panel alignment.
        self._log_coordinate_assumptions = bool(_runtime_get("log_coordinate_assumptions", True))
        # Per-step depth/map/traversible stats for diagnosing "all light gray" panel cases.
        self._log_mapping_stats = bool(_runtime_get("log_mapping_stats", True))
        # Map write diagnostics:
        # - full_map_delta: how many obstacle cells are newly added/removed each step.
        # - collision_paint: cells painted by forward-collision heuristic in _plan.
        self._log_full_map_delta = bool(_runtime_get("log_full_map_delta", True))
        self._log_collision_paint = bool(_runtime_get("log_collision_paint", True))
        # Structured health checks for depth/map/camera; parse [SG-Nav][check] lines while running.
        self._log_runtime_checks = bool(_runtime_get("log_runtime_checks", True))
        self._check_traversible_free_warn = float(_runtime_get("check_traversible_free_warn", 0.95))
        self._check_depth_zero_warn = float(_runtime_get("check_depth_zero_warn", 0.25))
        self._check_depth_median_min_m = float(_runtime_get("check_depth_median_min_m", 0.08))
        self._check_depth_median_max_m = float(_runtime_get("check_depth_median_max_m", 8.0))
        self._check_camera_pitch_abs_warn_deg = float(
            _runtime_get("check_camera_pitch_abs_warn_deg", 35.0)
        )
        self._check_camera_roll_abs_warn_deg = float(
            _runtime_get("check_camera_roll_abs_warn_deg", 45.0)
        )
        self._check_camera_height_min_warn_cm = float(
            _runtime_get("check_camera_height_min_warn_cm", 70.0)
        )
        self._check_camera_height_max_warn_cm = float(
            _runtime_get("check_camera_height_max_warn_cm", 210.0)
        )
        # Apply mapping thresholds to modules.
        self.sem_map_module.map_pred_threshold = float(self._sem_map_pred_threshold)
        self.sem_map_module.exp_pred_threshold = float(self._sem_exp_pred_threshold)
        self.free_map_module.map_pred_threshold = float(self._free_map_pred_threshold)
        self.free_map_module.exp_pred_threshold = float(self._free_exp_pred_threshold)
        if self._log_mapping_stats:
            print(
                "[SG-Nav][map_module_thresholds] "
                f"sem_map_pred={self.sem_map_module.map_pred_threshold:.2f} "
                f"sem_exp_pred={self.sem_map_module.exp_pred_threshold:.2f} "
                f"free_map_pred={self.free_map_module.map_pred_threshold:.2f} "
                f"free_exp_pred={self.free_map_module.exp_pred_threshold:.2f}",
                flush=True,
            )
        self._last_camera_diag = None
        print('scene graph module init finish!!!')

    def _sync_mapping_height_band(self):
        """Update Semantic_Mapping z bands to match current projection height and runtime policy."""
        h_cm = float(self.sem_map_module._camera_height_cm_for_projection())
        obs_min = float(self._map_obstacle_min_z_cm)
        if float(self._map_obstacle_max_z_cm) > 0.0:
            obs_max = float(self._map_obstacle_max_z_cm)
        else:
            obs_max = h_cm + float(self._map_obstacle_above_camera_cm)
        obs_max = max(obs_max, obs_min + 5.0)

        self.sem_map_module.min_z_consider = obs_min
        self.sem_map_module.max_z_consider = obs_max
        self.free_map_module.min_z_consider = float(self._map_free_min_z_cm)
        self.free_map_module.max_z_consider = float(self._map_free_max_z_cm)

        if self._log_mapping_height_band:
            print(
                "[SG-Nav][height_band] "
                f"step={self.total_steps} h_cm={h_cm:.2f} "
                f"sem_z=[{obs_min:.1f},{obs_max:.1f}] "
                f"free_z=[{float(self._map_free_min_z_cm):.1f},{float(self._map_free_max_z_cm):.1f}]",
                flush=True,
            )

    def _emit_coordinate_assumptions_once(self):
        if not getattr(self, "_log_coordinate_assumptions", False):
            return
        print(
            "[SG-Nav][coord_assumption] "
            f"A1: planner_grid_frame_fixed={self._map_frame_fixed_mode}; "
            "A2: start_row_col is derived from full_pose(x,y) map-meter coords before frame transform; "
            "A3: goal_map/collision_map/visited share same row-col frame as full_map[0,0]; "
            "A4: occupancy panel renders unknown/free/obstacle from full_map + fbe_free_map + collision_map. "
            "If A1-A4 mismatch runtime observations, revisit get_traversible/fbe/set_random_goal.",
            flush=True,
        )

    def _log_mapping_snapshot(self, observations, traversible=None):
        if not getattr(self, "_log_mapping_stats", False):
            return
        d = np.asarray(observations.get("depth"))
        if d.ndim == 3:
            d = d[..., 0]
        d1 = d.reshape(-1) if d.size else np.asarray([], dtype=np.float32)
        dfin = d1[np.isfinite(d1)] if d1.size else d1
        dpos = dfin[dfin > 0] if dfin.size else dfin
        dmin = float(np.min(dfin)) if dfin.size else float("nan")
        dmax = float(np.max(dfin)) if dfin.size else float("nan")
        dmed = float(np.median(dpos)) if dpos.size else float("nan")
        dnear_min = float(np.mean(dfin <= 1e-6)) if dfin.size else float("nan")

        fm = self.full_map[0, 0].detach().cpu().numpy()
        full_occ = float(np.mean(fm > 0.5))
        full_nonzero = float(np.mean(np.abs(fm) > 1e-6))
        coll = np.asarray(self.collision_map)
        coll_occ = float(np.mean(coll > 0.5)) if coll.size else float("nan")
        fb = self.fbe_free_map[0, 0].detach().cpu().numpy()
        free_map_occ = float(np.mean(fb > float(self._traversible_free_map_min)))
        free_map_nonzero = float(np.mean(np.abs(fb) > 1e-6))

        tr_free = float("nan")
        tr_obs = float("nan")
        if traversible is not None:
            tr = np.asarray(traversible)
            if tr.ndim == 2 and tr.shape[0] >= 3 and tr.shape[1] >= 3:
                tr = tr[1:-1, 1:-1]
            if tr.size:
                tr_free = float(np.mean(tr > 0.5))
                tr_obs = float(np.mean(tr <= 0.5))

        print(
            "[SG-Nav][map_stats] "
            f"step={self.total_steps} "
            f"depth[min,max,med_pos]=[{dmin:.3f},{dmax:.3f},{dmed:.3f}] "
            f"depth_zero_frac={dnear_min:.3f} "
            f"full_map_occ={full_occ:.3f} full_map_nonzero={full_nonzero:.3f} "
            f"collision_occ={coll_occ:.3f} "
            f"free_map_occ={free_map_occ:.3f} free_map_nonzero={free_map_nonzero:.3f} "
            f"traversible_free={tr_free:.3f} traversible_obs={tr_obs:.3f}",
            flush=True,
        )
        self._log_runtime_check_flags(
            dmin=dmin,
            dmax=dmax,
            dmed=dmed,
            dnear_min=dnear_min,
            full_occ=full_occ,
            coll_occ=coll_occ,
            tr_free=tr_free,
            tr_obs=tr_obs,
        )

    def _log_runtime_check_flags(
        self, *, dmin, dmax, dmed, dnear_min, full_occ, coll_occ, tr_free, tr_obs
    ):
        if not getattr(self, "_log_runtime_checks", False):
            return
        flags = []
        cam = self._last_camera_diag if isinstance(self._last_camera_diag, dict) else None

        if math.isfinite(tr_free) and tr_free >= self._check_traversible_free_warn:
            flags.append("TRAVERSIBLE_TOO_FREE")
        if math.isfinite(dnear_min) and dnear_min >= self._check_depth_zero_warn:
            flags.append("DEPTH_TOO_MANY_ZEROS")
        if math.isfinite(dmed) and (
            dmed < self._check_depth_median_min_m
            or dmed > self._check_depth_median_max_m
        ):
            flags.append("DEPTH_MEDIAN_OUT_OF_RANGE")
        if cam and cam.get("active", False):
            roll_abs = abs(float(cam.get("roll_deg", 0.0)))
            pitch_abs = abs(float(cam.get("pitch_deg_used", 0.0)))
            h_used = cam.get("height_cm_used", None)
            tilt_src_for_map = str(cam.get("tilt_source", "") or "")
            tilt_fallback = cam.get("tilt_fallback", None)
            # Large roll is often expected for fixed camera-frame offsets (e.g. sensor mounted with ~90deg roll).
            # Only warn when mapping actually relies on raw roll (without fallback).
            if (
                tilt_src_for_map == "roll"
                and tilt_fallback is None
                and roll_abs >= self._check_camera_roll_abs_warn_deg
            ):
                flags.append("CAMERA_ROLL_LARGE")
            if pitch_abs >= self._check_camera_pitch_abs_warn_deg:
                flags.append("CAMERA_PITCH_LARGE")
            if h_used is not None and (
                float(h_used) < self._check_camera_height_min_warn_cm
                or float(h_used) > self._check_camera_height_max_warn_cm
            ):
                flags.append("CAMERA_HEIGHT_OUTLIER")
            # Strong fall suspicion: severe tilt + very low camera height.
            if h_used is not None and float(h_used) < 90.0 and (
                roll_abs > 60.0 or pitch_abs > 45.0
            ):
                flags.append("FALL_SUSPECTED")

        flags_txt = "OK" if not flags else "|".join(flags)
        cam_txt = "inactive"
        if cam and cam.get("active", False):
            cam_txt = (
                f"src={cam.get('pose_source', 'unknown')} "
                f"roll={float(cam.get('roll_deg', float('nan'))):.1f} "
                f"pitch_used={float(cam.get('pitch_deg_used', float('nan'))):.1f} "
                f"h_used_cm={cam.get('height_cm_used', None)!r}"
            )

        print(
            "[SG-Nav][check] "
            f"step={self.total_steps} flags={flags_txt} "
            f"depth[min,max,med]=[{dmin:.3f},{dmax:.3f},{dmed:.3f}] "
            f"zero_frac={dnear_min:.3f} map_occ={full_occ:.3f} coll_occ={coll_occ:.3f} "
            f"tr_free={tr_free:.3f} tr_obs={tr_obs:.3f} cam={cam_txt}",
            flush=True,
        )

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

    def _horiz_angle_from_pixel_u(self, u_pixel):
        """Horizontal viewing angle in degrees, linear pinhole model (matches legacy 640/79)."""
        cx = (self.sensor_width - 1) / 2.0
        return (float(u_pixel) - cx) * self.hfov_deg / float(self.sensor_width)

    def _pix_i(self, v):
        return int(v.item()) if hasattr(v, "item") else int(v)

    def _depth_m_at_xy(self, x_col, y_row):
        """Depth (meters) at pixel (column x, row y); clips to the depth buffer."""
        h, w = int(self.depth.shape[0]), int(self.depth.shape[1])
        xc = int(np.clip(self._pix_i(x_col), 0, w - 1))
        yr = int(np.clip(self._pix_i(y_row), 0, h - 1))
        return self.depth[yr, xc, 0]

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

    @staticmethod
    def _quat_xyzw_to_rpy(qx, qy, qz, qw):
        """Roll/pitch/yaw (rad); quaternion order x,y,z,w (OmniGibson-style)."""
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(n) or n < 1e-9:
            return 0.0, 0.0, 0.0
        qx, qy, qz, qw = (qx / n, qy / n, qz / n, qw / n)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _wrap_angle_rad_pm_pi(angle_rad: float) -> float:
        x = float(angle_rad)
        while x > math.pi:
            x -= 2.0 * math.pi
        while x < -math.pi:
            x += 2.0 * math.pi
        return x

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

        delta_rad = self._wrap_angle_rad_pm_pi(cur_compass_rad - float(prev_compass_rad))
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

    @staticmethod
    def _quat_xyzw_rotate_vec(qx, qy, qz, qw, vx, vy, vz):
        """Rotate vector by quaternion (xyzw)."""
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(n) or n < 1e-9:
            return float(vx), float(vy), float(vz)
        x, y, z, w = (qx / n, qy / n, qz / n, qw / n)
        # R(q) for xyzw
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - z * w)
        r02 = 2.0 * (x * z + y * w)
        r10 = 2.0 * (x * y + z * w)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - x * w)
        r20 = 2.0 * (x * z - y * w)
        r21 = 2.0 * (y * z + x * w)
        r22 = 1.0 - 2.0 * (x * x + y * y)
        wx = r00 * vx + r01 * vy + r02 * vz
        wy = r10 * vx + r11 * vy + r12 * vz
        wz = r20 * vx + r21 * vy + r22 * vz
        return float(wx), float(wy), float(wz)

    def _client_camera_pose_active(self, observations):
        if not self._camera_extrinsic_use_client_pose:
            return False
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

    def _sync_camera_extrinsics(self, observations):
        """Align Semantic_Mapping pitch/height with client-reported camera link before update_map."""
        modules = (self.sem_map_module, self.free_map_module, self.room_map_module)
        if not self._client_camera_pose_active(observations):
            h_fb = None
            if self._camera_height_fallback_cm > 0.0:
                h_fb = float(
                    np.clip(
                        self._camera_height_fallback_cm,
                        self._camera_height_min_cm,
                        self._camera_height_max_cm,
                    )
                )
            for m in modules:
                m._extrinsic_height_cm = h_fb
            self._sync_mapping_height_band()
            self._last_camera_diag = {"active": False}
            if self._log_camera_pose_world and h_fb is not None:
                print(
                    "[SG-Nav][camera_pose] "
                    f"step={self.total_steps} mode=fallback_fixed_height "
                    f"extrinsic_height_cm={h_fb:.3f}",
                    flush=True,
                )
            return
        cpp = observations["camera_pose_world"]
        px = float(cpp["position"][0])
        py = float(cpp["position"][1])
        pz = float(cpp["position"][2])
        cam_source = str(cpp.get("source", "unknown"))
        qx, qy, qz, qw = (float(cpp["quaternion"][i]) for i in range(4))
        roll, pitch, yaw = self._quat_xyzw_to_rpy(qx, qy, qz, qw)
        # Vector-based tilt diagnostics (independent from Euler branch choice):
        # depth_utils camera frame uses +Y as forward(depth axis), +Z as image-up.
        # Note: ``Semantic_Mapping.set_view_angles`` expects a "look-down positive" command and
        # internally negates it before applying the x-axis rotation. The vector-derived tilt we
        # log below follows the direct world-vector sign, so we convert signs only at the final
        # handoff into the mapping module.
        z_w = self._quat_xyzw_rotate_vec(qx, qy, qz, qw, 0.0, 0.0, 1.0)
        y_w = self._quat_xyzw_rotate_vec(qx, qy, qz, qw, 0.0, 1.0, 0.0)
        z_h = math.hypot(z_w[0], z_w[1])
        y_h = math.hypot(y_w[0], y_w[1])
        cam_z_down_pitch_deg = -math.degrees(math.atan2(z_w[2], max(z_h, 1e-9)))
        cam_y_down_pitch_deg = -math.degrees(math.atan2(y_w[2], max(y_h, 1e-9)))
        tilt_source = self._camera_tilt_source
        tilt_fallback = None
        if tilt_source == "roll":
            tilt_src_deg = math.degrees(roll)
        elif tilt_source == "yaw":
            tilt_src_deg = math.degrees(yaw)
        elif tilt_source == "cam_z_down":
            tilt_src_deg = float(cam_z_down_pitch_deg)
        elif tilt_source == "cam_y_down":
            tilt_src_deg = float(cam_y_down_pitch_deg)
        else:
            tilt_source = "pitch"
            tilt_src_deg = math.degrees(pitch)
        pitch_deg_raw = (
            float(tilt_src_deg) * self._camera_pitch_mapping_scale
            + self._camera_pitch_mapping_offset_deg
        )
        if (
            tilt_source == "roll"
            and bool(getattr(self, "_camera_tilt_fallback_to_cam_z_down_on_outlier", True))
            and math.isfinite(float(cam_z_down_pitch_deg))
            and abs(float(pitch_deg_raw)) > float(self._camera_pitch_reject_abs_deg)
        ):
            # roll(+offset) can jump by ~180 due Euler branch ambiguity; use vector-derived tilt instead.
            pitch_deg_raw = float(cam_z_down_pitch_deg)
            tilt_fallback = "roll_outlier_to_cam_z_down"
        # roll / yaw are periodic in 360deg; unwrap mapped pitch around last good value
        # to avoid branch-cut jumps (e.g. +179 -> -179) polluting depth projection.
        if (
            tilt_source in ("roll", "yaw")
            and self._last_good_camera_pitch_deg is not None
            and math.isfinite(float(pitch_deg_raw))
        ):
            period = abs(float(self._camera_pitch_mapping_scale)) * 360.0
            if period > 1e-6:
                ref = float(self._last_good_camera_pitch_deg)
                k = round((float(pitch_deg_raw) - ref) / period)
                pitch_deg_raw = float(pitch_deg_raw) - float(k) * period
        pitch_deg = float(np.clip(pitch_deg_raw, self._camera_pitch_min_deg, self._camera_pitch_max_deg))
        h_cm_raw = None
        h_cm = None
        reject_reason = None
        if self._camera_height_from_world_z:
            z_m = pz + self._camera_height_z_offset_m
            h_cm_raw = z_m * 100.0
            h_cm = float(np.clip(h_cm_raw, self._camera_height_min_cm, self._camera_height_max_cm))

        if self._camera_extrinsic_reject_outlier:
            pitch_bad = (
                (not np.isfinite(pitch_deg_raw))
                or abs(float(pitch_deg_raw)) > float(self._camera_pitch_reject_abs_deg)
            )
            height_bad = False
            if self._camera_height_from_world_z:
                height_bad = (
                    (not np.isfinite(h_cm_raw))
                    or float(h_cm_raw) < float(self._camera_height_reject_min_cm)
                    or float(h_cm_raw) > float(self._camera_height_reject_max_cm)
                )
            if pitch_bad or height_bad:
                reasons = []
                if pitch_bad:
                    reasons.append("pitch_outlier")
                if height_bad:
                    reasons.append("height_outlier")
                reject_reason = ",".join(reasons)
                if self._last_good_camera_pitch_deg is not None:
                    pitch_deg = float(self._last_good_camera_pitch_deg)
                else:
                    pitch_deg = float(np.clip(0.0, self._camera_pitch_min_deg, self._camera_pitch_max_deg))
                if self._camera_height_from_world_z:
                    if self._last_good_camera_height_cm is not None:
                        h_cm = float(self._last_good_camera_height_cm)
                    elif self._camera_height_fallback_cm > 0.0:
                        h_cm = float(
                            np.clip(
                                self._camera_height_fallback_cm,
                                self._camera_height_min_cm,
                                self._camera_height_max_cm,
                            )
                        )
                    else:
                        h_cm = float(
                            np.clip(
                                self.config.SIMULATOR.AGENT_0.HEIGHT * 100.0,
                                self._camera_height_min_cm,
                                self._camera_height_max_cm,
                            )
                        )
            else:
                self._last_good_camera_pitch_deg = float(pitch_deg)
                if self._camera_height_from_world_z and h_cm is not None:
                    self._last_good_camera_height_cm = float(h_cm)

        view_angle_cmd_deg = float(-pitch_deg)
        for m in modules:
            m.set_view_angles(view_angle_cmd_deg)
            if self._camera_height_from_world_z:
                m._extrinsic_height_cm = h_cm
            else:
                m._extrinsic_height_cm = None
        self._sync_mapping_height_band()
        self._last_camera_diag = {
            "active": True,
            "roll_deg": float(math.degrees(roll)),
            "pitch_deg_raw": float(pitch_deg_raw),
            "pitch_deg_used": float(pitch_deg),
            "view_angle_cmd_deg": float(view_angle_cmd_deg),
            "tilt_source": tilt_source,
            "pose_source": cam_source,
            "tilt_source_raw_deg": float(tilt_src_deg),
            "tilt_fallback": tilt_fallback,
            "cam_z_down_pitch_deg": float(cam_z_down_pitch_deg),
            "cam_y_down_pitch_deg": float(cam_y_down_pitch_deg),
            "height_cm_raw": None if h_cm_raw is None else float(h_cm_raw),
            "height_cm_used": None if h_cm is None else float(h_cm),
            "quat_xyzw": [float(qx), float(qy), float(qz), float(qw)],
            "pos_m": [float(px), float(py), float(pz)],
        }
        if self._log_camera_extrinsic_clamp and (
            abs(pitch_deg - pitch_deg_raw) > 1e-6
            or (h_cm_raw is not None and h_cm is not None and abs(h_cm - h_cm_raw) > 1e-6)
        ):
            print(
                "[SG-Nav][camera_extrinsic_clamp] "
                f"step={self.total_steps} "
                f"pitch_raw={pitch_deg_raw:.3f} pitch_used={pitch_deg:.3f} "
                f"height_raw_cm={h_cm_raw!r} height_used_cm={h_cm!r}"
                + (f" reject={reject_reason}" if reject_reason else ""),
                flush=True,
            )
        if self._log_camera_pose_world:
            h_cm_log = (
                self.sem_map_module._extrinsic_height_cm
                if self._camera_height_from_world_z
                else None
            )
            print(
                "[SG-Nav][camera_pose] "
                f"step={self.total_steps} "
                f"pos_m=[{px:.5f}, {py:.5f}, {pz:.5f}] "
                f"quat_xyzw=[{qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}] "
                f"rpy_deg=[{math.degrees(roll):.3f}, {math.degrees(pitch):.3f}, {math.degrees(yaw):.3f}] "
                f"pose_source={cam_source} "
                f"tilt_source={tilt_source} tilt_source_raw_deg={tilt_src_deg:.3f} "
                f"tilt_fallback={tilt_fallback} "
                f"cam_z_down_pitch_deg={cam_z_down_pitch_deg:.3f} "
                f"cam_y_down_pitch_deg={cam_y_down_pitch_deg:.3f} "
                f"map_pitch_deg_raw={pitch_deg_raw:.3f} map_pitch_deg_used={pitch_deg:.3f} "
                f"view_angle_cmd_deg={view_angle_cmd_deg:.3f} "
                f"extrinsic_height_cm={h_cm_log!r}",
                flush=True,
            )

    def add_predicates(self, model):
        predicate = Predicate('IsNearObj', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('ObjCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('IsNearRoom', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('RoomCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
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
        self._emit_coordinate_assumptions_once()
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

    def _act_action_name(self, a):
        names = ("STOP", "FWD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "TURN")
        try:
            ai = int(a)
        except (TypeError, ValueError):
            return str(a)
        return names[ai] if 0 <= ai < len(names) else str(a)

    def _emit_spin_diagnosis(
        self,
        *,
        early=None,
        panorama=False,
        goal_map_src=None,
        number_action=None,
        replan_fbe_hit=False,
        stuck_reset_hit=False,
        stuck_abort=False,
        max_steps_stop=False,
    ):
        """stderr: classify why motion is mostly turn / exploration (not necessarily bug)."""
        if not getattr(self, "_log_spin_diagnosis", True):
            return
        bbox_n = int(getattr(self, "_last_goal_bbox_n", 0))
        fg, fpg = self.found_goal, self.found_possible_goal
        an = self._act_action_name(number_action) if number_action is not None else "n/a"

        if max_steps_stop:
            cause = "episode_cap:total_steps>=500_emits_stop"
        elif early:
            cause = f"early_camera:{early}"
        elif stuck_abort:
            cause = "stuck_loop_abort:too_many_stuck_iterations_emits_stop"
        elif panorama:
            cause = "panorama_spin:step<=panorama_spin_until_step_and_no_fg_and_no_fpg"
            if bbox_n == 0:
                cause += "|detail=no_glip_goal_bbox"
            else:
                cause += "|detail=have_bbox_but_nav_flags_not_set_check_vote_depth"
        elif goal_map_src is not None and number_action is not None:
            na = int(number_action)
            parts = []
            if not fg and not fpg:
                if bbox_n == 0:
                    parts.append("spin_why=no_glip_match_for_obj_goal")
                else:
                    parts.append("spin_why=bbox_present_but_not_in_goal_nav_mode")
            else:
                parts.append("spin_why=heading_align_or_subgoal_in_object_nav")
            gms = goal_map_src
            if gms.startswith("replan_fbe") or gms.startswith("first_fbe"):
                parts.append(f"explore={gms}")
            elif "random" in gms:
                parts.append(f"explore={gms}")
            elif gms in ("found_goal", "possible_goal"):
                parts.append("target=object_on_map")
            elif gms == "gt_world_on_map":
                parts.append("target=sim_world_xy_on_occupancy")
            elif gms == "keep_previous_goal_map":
                parts.append("map=unchanged_from_prior_step")
            if replan_fbe_hit:
                parts.append("replan_fbe=1")
            if stuck_reset_hit:
                parts.append("stuck_reset=1")
            if na in (2, 3, 6):
                parts.append("motion=turn_not_forward")
            elif na == 1:
                parts.append("motion=forward")
            elif na == 0:
                parts.append("motion=stop")
            cause = "planner|" + ";".join(parts)
        else:
            cause = "unknown"

        gms_s = f" goal_map_src={goal_map_src}" if goal_map_src is not None else ""
        print(
            f"[SG-Nav][spin] step={self.total_steps} cause={cause}{gms_s} "
            f"bbox_n={bbox_n} fg={fg} fpg={fpg} not_move={self.not_move_steps} "
            f"action={number_action}({an})",
            flush=True,
        )

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

        if (
            self._log_goal_votes
            and self.scenegraph.obj_goal not in self.scenegraph.small_objects
            and len(goal_bbox) == 0
        ):
            n = len(obj_labels) if obj_labels is not None else 0
            sample = [str(obj_labels[j]) for j in range(min(8, n))] if n else []
            print(
                "[SG-Nav][goal_vote] "
                f"step={self.total_steps} goal={self.obj_goal!r} match_tokens={self._glip_goal_sg_match_tokens!r} "
                f"bbox=0 n_glip={n} labels={sample}",
                flush=True,
            )
        
        for j, label in enumerate(obj_labels):
            if label in CANONICAL_MP3D_GOAL_ORDER:
                confidence = self.current_obj_predictions.get_field("scores")[j]
                bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = self._horiz_angle_from_pixel_u(center_point[0])
                temp_distance = self._depth_m_at_xy(center_point[0], center_point[1])
                if temp_distance >= self.distance_threshold:
                    continue
                obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
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
                    temp_direction = self._horiz_angle_from_pixel_u(center_point[0])
                    temp_distance = self._depth_m_at_xy(center_point[0], center_point[1])
                    k = 0
                    pos_neg = 1
                    dh, dw = int(self.depth.shape[0]), int(self.depth.shape[1])
                    while temp_distance >= 100 and 0 < self._pix_i(center_point[1]) + int(pos_neg * k) < dh - 1 and 0 < self._pix_i(center_point[0]) + int(pos_neg * k) < dw - 1:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(
                            self._depth_m_at_xy(center_point[0], center_point[1] + int(pos_neg * k)),
                            self._depth_m_at_xy(center_point[0] + int(pos_neg * k), center_point[1]),
                        )
                        
                    if temp_distance >= self.distance_threshold:
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
                elif not possible_goal_detected_before:
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
                    temp_direction = self._horiz_angle_from_pixel_u(center_point[0])
                    temp_distance = self._depth_m_at_xy(center_point[0], center_point[1])
                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    k = 0
                    pos_neg = 1
                    dh, dw = int(self.depth.shape[0]), int(self.depth.shape[1])
                    while temp_distance >= 100 and 0 < self._pix_i(center_point[1]) + int(pos_neg * k) < dh - 1 and 0 < self._pix_i(center_point[0]) + int(pos_neg * k) < dw - 1:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(
                            self._depth_m_at_xy(center_point[0], center_point[1] + int(pos_neg * k)),
                            self._depth_m_at_xy(center_point[0] + int(pos_neg * k), center_point[1]),
                        )
                        
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                        _vote_boxes_far += 1
                    else:
                        thres = int(self.goal_merge_threshold * 100 / self.map_resolution)
                        if 0 <= int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) < self.map_size and 0 <= int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) < self.map_size:
                            goal_gps_map_local = self.goal_gps_map[max(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) - thres, 0):min(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) + thres, self.map_size - 1), max(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) - thres, 0):min(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) + thres, self.map_size - 1)]
                            if goal_gps_map_local.max() > 0:
                                goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] = goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] + 1
                            else:
                                self.goal_gps_map[
                                    min(
                                        max(
                                            int(
                                                self.map_size_cm / 10
                                                + goal_gps[1] * 100 / self.resolution
                                            ),
                                            0,
                                        ),
                                        self.map_size - 1,
                                    ),
                                    min(
                                        max(
                                            int(
                                                self.map_size_cm / 10
                                                + goal_gps[0] * 100 / self.resolution
                                            ),
                                            0,
                                        ),
                                        self.map_size - 1,
                                    ),
                                ] = 1
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
                    if (
                        _vote_boxes_far > 0
                        or _vote_boxes_near > 0
                        or int(self.found_goal_times) > 0
                        or shortest_distance < 120
                    ):
                        self.found_possible_goal = True
                if self._log_goal_votes:
                    print(
                        "[SG-Nav][goal_vote] "
                        f"step={self.total_steps} goal={self.obj_goal!r} "
                        f"bbox={len(goal_bbox)} far_ge_{self.distance_threshold}m={_vote_boxes_far} "
                        f"near_map_ok={_vote_boxes_near} near_map_oob={_vote_boxes_near_oob} "
                        f"vote_max={int(self.found_goal_times)}/{int(self.scenegraph.cfg.obj_min_detections)} "
                        f"found_goal={self.found_goal} found_possible_goal={self.found_possible_goal}",
                        flush=True,
                    )

                if self.found_goal:
                    self.goal_gps = np.flip(np.array(np.where(self.goal_gps_map == self.goal_gps_map.max()))[:, 0])
                    self.goal_gps = (self.goal_gps - self.map_size_cm / 10) / 100 * self.resolution
                elif shortest_distance < 120:
                    # Keep a moving far-goal hint so agent starts approaching distant detections
                    # instead of oscillating in exploration mode.
                    self.possible_goal_temp_gps = self.get_goal_gps(
                        observations, shortest_distance_angle, shortest_distance
                    )
            self.goal_distance_for_vis = float(shortest_distance) if shortest_distance < 120 else None
            return
                        
    def act(self, observations):
        if self.total_steps >= 500:
            self._emit_spin_diagnosis(max_steps_stop=True, number_action=0)
            return {"action": 0}
        
        self.total_steps += 1
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
        comp = np.asarray(observations["compass"], dtype=np.float64).reshape(-1).copy()
        comp[0] += self._compass_offset_rad
        observations["compass"] = comp
        self._update_observed_turn_from_compass(observations)
        gps = np.asarray(observations["gps"], dtype=np.float64).reshape(-1).copy()
        if self._gps_negate_y:
            gps[1] = -gps[1]
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
        self.scenegraph.update_scenegraph()

        self._sync_camera_extrinsics(observations)
        self.update_map(observations)
        self.update_free_map(observations)
        # Snapshot after depth->map update and before planning-only transforms.
        self._log_mapping_snapshot(observations, traversible=None)

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

        use_script_cam_tilt = not self._client_camera_pose_active(observations)
        if self.total_steps == 1:
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(30)
                self.free_map_module.set_view_angles(30)
            self._emit_spin_diagnosis(early="step1_lookdown", number_action=5)
            return {"action": 5}
        elif self.total_steps <= 7 and not (self.found_goal or self.found_possible_goal):
            self._emit_spin_diagnosis(early="step2_7_turn", number_action=6)
            return {"action": 6}
        elif self.total_steps == 8:
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(60)
                self.free_map_module.set_view_angles(60)
            self._emit_spin_diagnosis(early="step8_lookdown", number_action=5)
            return {"action": 5}
        elif self.total_steps <= 14 and not (self.found_goal or self.found_possible_goal):
            self._emit_spin_diagnosis(early="step9_14_turn", number_action=6)
            return {"action": 6}
        elif self.total_steps <= 15 and not (self.found_goal or self.found_possible_goal):
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(30)
                self.free_map_module.set_view_angles(30)
            self._emit_spin_diagnosis(early="step15_look_pitch", number_action=4)
            return {"action": 4}
        elif self.total_steps <= 16 and not (self.found_goal or self.found_possible_goal):
            if use_script_cam_tilt:
                self.sem_map_module.set_view_angles(0)
                self.free_map_module.set_view_angles(0)
            self._emit_spin_diagnosis(early="step16_look_pitch", number_action=4)
            return {"action": 4}
        # Panorama buffers + optional extra spin only while no goal hint
        if (self.total_steps <= spin_cap and not self.found_goal) or run_periodic_detect:
            self.panoramic.append(observations["rgb"][:, :, [2, 1, 0]])
            self.panoramic_depth.append(observations["depth"])
            if self.total_steps <= spin_cap and (not self.found_goal and not self.found_possible_goal):
                if self._nav_debug_trace:
                    print(
                        f"[SG-Nav][trace] step={self.total_steps} phase=panorama_spin_until_goal "
                        f"found_goal={self.found_goal} found_possible_goal={self.found_possible_goal} action=6(TURN)",
                        flush=True,
                    )
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
        
        self.scenegraph.perception()
        if self.save_scenegraph_json:
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
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
            goal_map_src = "found_goal"
        elif self._try_apply_gt_world_target_goal_map(observations):
            goal_map_src = "gt_world_on_map"
        elif self.found_possible_goal:
            self._active_gt_world_nav = False
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[0]*100/self.resolution)))] = 1
            goal_map_src = "possible_goal"
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
        stg_y, stg_x, replan, number_action = self._plan(
            traversible, goal_map_plan, self.full_pose, cur_start, cur_start_o, self.found_goal
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
        
        self.loop_time = 0
        stuck_reset_hit = False
        while (
            (
                not self.found_goal
                and not self.found_possible_goal
                and not self._active_gt_world_nav
                and number_action == 0
            )
            or self.not_move_steps >= 7
        ):
            stuck_reset_hit = True
            if self.not_move_steps >= 7:
                self.found_goal = False
                self.found_possible_goal = False
                self._active_gt_world_nav = False
            self.loop_time += 1
            self.random_this_ex += 1
            if self.loop_time > 20:
                if self._nav_debug_trace:
                    print(
                        f"[SG-Nav][trace] step={self.total_steps} phase=stuck_loop_abort "
                        f"goal_map_src={goal_map_src} fg={self.found_goal} fpg={self.found_possible_goal}",
                        flush=True,
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
        
        if getattr(self, "_log_pose_align_trace", False):
            dbg = getattr(self, "_last_planner_debug", None) or {}
            cr = float(np.asarray(observations["compass"]).reshape(-1)[0])
            g0 = float(observations["gps"][0])
            g1 = float(observations["gps"][1])
            fp = self.full_pose.detach().cpu().numpy()
            parts = [
                f"step={self.total_steps}",
                f"gps=[{g0:.3f},{g1:.3f}]",
                f"compass_rad={cr:.4f}",
                f"full_pose_m_deg=[{fp[0]:.3f},{fp[1]:.3f},{fp[2]:.2f}]",
                f"map_size_cm={self.map_size_cm}",
            ]
            if dbg.get("stop"):
                parts.append("planner_stop=1")
            elif "relative_angle_deg" in dbg:
                parts.append(f"rel_deg={dbg['relative_angle_deg']:.2f}")
                parts.append(f"agent_deg={dbg['angle_agent_deg']:.2f}")
                parts.append(f"stg_bearing_deg={dbg['angle_st_goal_deg']:.2f}")
                parts.append(
                    f"align_thr={dbg['align_thr_deg']:.1f} fwd_exit={dbg['forward_exit_deg']:.1f}"
                )
                if dbg.get("align_thr_cfg_deg") is not None:
                    parts.append(
                        f"align_cfg={dbg['align_thr_cfg_deg']:.1f} fwd_exit_cfg={dbg['forward_exit_cfg_deg']:.1f}"
                    )
                if dbg.get("face_goal_only_deg", -1.0) >= 0.0:
                    parts.append(
                        f"face_eps={dbg['face_goal_only_deg']:.1f} face_eps_cfg={dbg['face_goal_only_cfg_deg']:.1f}"
                    )
            if dbg.get("observed_turn_deg") is not None:
                parts.append(f"obs_turn_deg={dbg['observed_turn_deg']:.2f}")
            if dbg.get("last_turn_delta_deg") is not None:
                parts.append(f"last_turn_deg={dbg['last_turn_delta_deg']:.2f}")
            parts.append(f"stg_ij={dbg.get('stg')}")
            parts.append(
                f"start_ij={dbg.get('start_ij')} start_o_deg={dbg.get('start_o_deg', float('nan')):.2f}"
            )
            parts.append(f"action={number_action}")
            print("[SG-Nav][pose_align] " + " ".join(str(p) for p in parts), flush=True)

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
        torch.cuda.empty_cache()

        if self._log_goal_cell_drift or self._nav_debug_trace:
            act_names = ("STOP", "FWD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "TURN")
            an = act_names[int(number_action)] if 0 <= int(number_action) < len(act_names) else str(number_action)
            drift_s = f" drift={goal_drift}" if goal_drift is not None else ""
            rc_s = f" goal_rc={goal_rc}" if goal_rc is not None else " goal_rc=None"
            if self._log_goal_cell_drift and not self._nav_debug_trace:
                print(
                    "[SG-Nav][goal_cell] "
                    f"step={self.total_steps}{rc_s}{drift_s} src={goal_map_src} "
                    f"replan_fbe={replan_fbe_hit} action={int(number_action)}({an})",
                    flush=True,
                )
            elif self._nav_debug_trace:
                print(
                    "[SG-Nav][trace] "
                    f"step={self.total_steps} goal_map_src={goal_map_src} "
                    f"found_goal={self.found_goal} found_possible_goal={self.found_possible_goal} "
                    f"replan_fbe={replan_fbe_hit} stuck_reset={stuck_reset_hit} "
                    f"not_move_steps={self.not_move_steps} random_goal={self.using_random_goal} "
                    f"{rc_s}{drift_s} "
                    f"action={int(number_action)}({an})",
                    flush=True,
                )

        self._emit_spin_diagnosis(
            goal_map_src=goal_map_src,
            number_action=number_action,
            replan_fbe_hit=replan_fbe_hit,
            stuck_reset_hit=stuck_reset_hit,
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
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

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
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)

    @staticmethod
    def _wrap_angle_deg(angle_deg):
        angle_deg = float(angle_deg)
        angle_deg = (angle_deg + 180.0) % 360.0 - 180.0
        return angle_deg

    def _get_goal_stop_status(self, agent_pose, goal_gps=None):
        if goal_gps is None:
            goal_gps = getattr(self, "goal_gps", None)
        if goal_gps is None or len(goal_gps) < 2:
            return None
        try:
            goal_x = float(goal_gps[0])
            goal_y = float(goal_gps[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(goal_x) and math.isfinite(goal_y)):
            return None

        half_map_m = float(self.map_size_cm) / 200.0
        agent_x = float(agent_pose[0]) - half_map_m
        agent_y = half_map_m - float(agent_pose[1])
        dx = goal_x - agent_x
        dy = goal_y - agent_y
        rho = math.hypot(dx, dy)
        bearing_deg = math.degrees(math.atan2(dy, dx))
        heading_deg = float(agent_pose[2])
        heading_error_deg = self._wrap_angle_deg(bearing_deg - heading_deg)
        return {
            "distance_m": float(rho),
            "bearing_deg": float(bearing_deg),
            "heading_deg": float(heading_deg),
            "heading_error_deg": float(heading_error_deg),
        }
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.default_free_map = np.zeros((full_w, full_h), dtype=np.float32)
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
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

    def update_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self._clamp_full_pose_xy_to_map_meters()
        fm_prev = None
        if getattr(self, "_log_full_map_delta", False):
            fm_prev = self.full_map[0, 0].detach().cpu().numpy().copy()
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
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
            else:
                add_local = float("nan")
                occ_local = float("nan")
            yy, xx = np.ogrid[:h, :w]
            rr2 = (yy - sy) ** 2 + (xx - sx) ** 2
            near = rr2 <= (12 ** 2)
            ring = np.logical_and(rr2 >= (4 ** 2), rr2 <= (10 ** 2))
            add_near = float(np.mean(np.logical_and(add_occ, near)))
            add_ring = float(np.mean(np.logical_and(add_occ, ring)))
            print(
                "[SG-Nav][full_map_delta] "
                f"step={self.total_steps} occ_thr={occ_thr:.3f} "
                f"prev_occ={float(np.mean(prev_occ)):.3f} now_occ={float(np.mean(now_occ)):.3f} "
                f"add_occ={add_ratio:.4f} del_occ={del_ratio:.4f} "
                f"local_occ={occ_local:.3f} local_add={add_local:.4f} "
                f"add_near={add_near:.4f} add_ring={add_ring:.4f} "
                f"start_ij=({sy},{sx})",
                flush=True,
            )
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
        px = float(self.full_pose[0].detach().cpu().item())
        py = float(self.full_pose[1].detach().cpu().item())
        sy = int(round((float(self.map_size_cm) / 100.0 - py) * 100.0 / float(self.map_resolution)))
        sx = int(round(px * 100.0 / float(self.map_resolution)))
        sy = max(0, min(int(h) - 1, sy))
        sx = max(0, min(int(w) - 1, sx))
        return sy, sx

    @staticmethod
    def _disk_mask_for_center(h: int, w: int, cy: int, cx: int, radius_cells: int):
        if int(radius_cells) <= 0:
            return np.zeros((int(h), int(w)), dtype=bool)
        yy, xx = np.ogrid[: int(h), : int(w)]
        return ((yy - int(cy)) ** 2 + (xx - int(cx)) ** 2) <= (int(radius_cells) ** 2)

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
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self._clamp_full_pose_xy_to_map_meters()
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        h, w = int(self.fbe_free_map.shape[-2]), int(self.fbe_free_map.shape[-1])
        sy, sx = self._robot_map_rc_from_full_pose(h, w)
        radius_cells = int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0))
        obs_map = self.full_map[0, 0].detach().cpu().numpy() > float(self._traversible_occ_from_depth_min)
        coll_map = np.asarray(self.collision_map, dtype=np.float32) > 0.5
        obs_or_coll = np.logical_or(obs_map, coll_map)
        if np.any(obs_or_coll):
            obs_or_coll_t = torch.from_numpy(obs_or_coll).to(self.device)
            self.fbe_free_map[0, 0][obs_or_coll_t] = 0.0
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
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)

    def _apply_map_frame_transform(self, arr, mode: str):
        a = np.asarray(arr)
        if mode == "id":
            return a
        if mode == "flipud":
            return np.flipud(a)
        if mode == "fliplr":
            return np.fliplr(a)
        if mode == "rot180":
            return np.rot90(a, 2)
        if mode == "rot90":
            return np.rot90(a, 1)
        if mode == "rot270":
            return np.rot90(a, 3)
        if mode == "transpose":
            return np.transpose(a)
        if mode == "transpose_flipud":
            return np.flipud(np.transpose(a))
        return a

    def _transform_rc_by_frame_mode(self, r: int, c: int, h: int, w: int, mode: str):
        rr, cc = int(r), int(c)
        if mode == "id":
            pass
        elif mode == "flipud":
            rr = h - 1 - rr
        elif mode == "fliplr":
            cc = w - 1 - cc
        elif mode == "rot180":
            rr = h - 1 - rr
            cc = w - 1 - cc
        elif mode == "rot90":
            rr, cc = h - 1 - cc, rr
        elif mode == "rot270":
            rr, cc = cc, w - 1 - rr
        elif mode == "transpose":
            rr, cc = cc, rr
        elif mode == "transpose_flipud":
            rr, cc = h - 1 - cc, rr
        rr = max(0, min(h - 1, rr))
        cc = max(0, min(w - 1, cc))
        return rr, cc

    def _inverse_transform_rc_by_frame_mode(self, r: int, c: int, h: int, w: int, mode: str):
        rr, cc = int(r), int(c)
        if mode == "id":
            pass
        elif mode == "flipud":
            rr = h - 1 - rr
        elif mode == "fliplr":
            cc = w - 1 - cc
        elif mode == "rot180":
            rr = h - 1 - rr
            cc = w - 1 - cc
        elif mode == "rot90":
            rr, cc = cc, h - 1 - rr
        elif mode == "rot270":
            rr, cc = w - 1 - cc, rr
        elif mode == "transpose":
            rr, cc = cc, rr
        elif mode == "transpose_flipud":
            rr, cc = cc, h - 1 - rr
        rr = max(0, min(h - 1, rr))
        cc = max(0, min(w - 1, cc))
        return rr, cc

    def _auto_select_map_frame_transform(self, obs_from_depth, obs_from_collision, free_from_depth, start):
        # Baseline SG_Nav_0 used only vertical flip ([::-1]) before traversible.
        # Restrict candidates to avoid accidental 90deg frame mis-selection on symmetric local patterns.
        modes = ["id", "flipud"]
        win = int(getattr(self, "_map_frame_auto_select_win_cells", 40))
        sy = int(start[0])
        sx = int(start[1])
        scores = {}
        best_mode = "id"
        best_score = -1e9
        prev_mode = str(getattr(self, "_last_map_frame_transform", "id"))
        for mode in modes:
            od = self._apply_map_frame_transform(obs_from_depth, mode)
            oc = self._apply_map_frame_transform(obs_from_collision, mode)
            fr = self._apply_map_frame_transform(free_from_depth, mode)
            y0 = max(0, sy - win)
            y1 = min(od.shape[0], sy + win + 1)
            x0 = max(0, sx - win)
            x1 = min(od.shape[1], sx + win + 1)
            if y1 <= y0 or x1 <= x0:
                local_known = 0.0
                local_obs = 0.0
            else:
                known = np.logical_or(np.logical_or(od[y0:y1, x0:x1], oc[y0:y1, x0:x1]), fr[y0:y1, x0:x1])
                local_known = float(np.mean(known))
                local_obs = float(np.mean(np.logical_or(od[y0:y1, x0:x1], oc[y0:y1, x0:x1])))
            score = local_known + 0.25 * local_obs
            if mode == prev_mode:
                score += 1e-3  # tiny hysteresis to avoid mode flapping on ties
            scores[mode] = score
            if score > best_score:
                best_score = score
                best_mode = mode
        return best_mode, scores
    
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
        if frame_mode == "flipud":
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
            if score_alt > (score_pose + 1e-4):
                sy_plan, sx_plan = sy_plan_alt, sx_plan_alt
                start_pick_mode = "flipud_mirror_y"
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

        if getattr(self, "_log_mapping_stats", False):
            def _local_ratio(arr, cy, cx, win=40):
                y0 = max(0, int(cy) - win)
                y1 = min(arr.shape[0], int(cy) + win + 1)
                x0 = max(0, int(cx) - win)
                x1 = min(arr.shape[1], int(cx) + win + 1)
                if y1 <= y0 or x1 <= x0:
                    return float("nan")
                return float(np.mean(arr[y0:y1, x0:x1]))

            def _q(v, p):
                if v.size <= 0:
                    return float("nan")
                return float(np.quantile(v, p))

            def _local_vec(arr, cy, cx, win=40):
                y0 = max(0, int(cy) - win)
                y1 = min(arr.shape[0], int(cy) + win + 1)
                x0 = max(0, int(cx) - win)
                x1 = min(arr.shape[1], int(cx) + win + 1)
                if y1 <= y0 or x1 <= x0:
                    return np.asarray([], dtype=np.float32)
                return np.asarray(arr[y0:y1, x0:x1], dtype=np.float32).reshape(-1)

            obs_union_raw = np.logical_or(obs_from_depth_raw, obs_from_collision_raw)
            obs_union_plan = np.logical_or(obs_from_depth, obs_from_collision)
            free_on_obs = float(np.mean(np.logical_and(free_from_depth_raw, obs_union_raw)))
            free_non_obs = float(np.mean(np.logical_and(free_from_depth_raw, np.logical_not(obs_union_raw))))
            free_total = float(np.mean(free_from_depth_raw))
            free_conflict_ratio = (
                free_on_obs / max(free_total, 1e-6) if math.isfinite(free_total) else float("nan")
            )

            full_l_plan = _local_ratio(obs_from_depth, sy_plan, sx_plan)
            coll_l_plan = _local_ratio(obs_from_collision, sy_plan, sx_plan)
            free_l_plan = _local_ratio(free_from_depth, sy_plan, sx_plan)
            known_l_plan = _local_ratio(np.logical_or(obs_union_plan, free_from_depth), sy_plan, sx_plan)
            full_l_raw_ref = _local_ratio(obs_from_depth_raw, sy_raw, sx_raw)
            coll_l_raw_ref = _local_ratio(obs_from_collision_raw, sy_raw, sx_raw)
            free_l_raw_ref = _local_ratio(free_from_depth_raw, sy_raw, sx_raw)
            known_l_raw_ref = _local_ratio(np.logical_or(obs_union_raw, free_from_depth_raw), sy_raw, sx_raw)
            edge_margin_plan = int(min(sy_plan, h - 1 - sy_plan, sx_plan, w - 1 - sx_plan))
            nearest_occ_plan_d = float("nan")
            nearest_occ_plan_rc = None
            occ_rc = np.argwhere(obs_union_plan)
            if occ_rc.shape[0] > 0:
                d2 = (occ_rc[:, 0] - int(sy_plan)) ** 2 + (occ_rc[:, 1] - int(sx_plan)) ** 2
                j = int(np.argmin(d2))
                nearest_occ_plan_d = float(np.sqrt(float(d2[j])))
                nearest_occ_plan_rc = (int(occ_rc[j, 0]), int(occ_rc[j, 1]))
            print(
                "[SG-Nav][map_diag] "
                f"step={self.total_steps} frame={frame_mode} "
                f"global_plan[full,coll,free]=[{float(np.mean(obs_from_depth)):.3f},"
                f"{float(np.mean(obs_from_collision)):.3f},{float(np.mean(free_from_depth)):.3f}] "
                f"local_plan[full,coll,free,known]=[{full_l_plan:.3f},{coll_l_plan:.3f},{free_l_plan:.3f},{known_l_plan:.3f}] "
                f"local_raw_ref[full,coll,free,known]=[{full_l_raw_ref:.3f},{coll_l_raw_ref:.3f},{free_l_raw_ref:.3f},{known_l_raw_ref:.3f}] "
                f"free_raw[non_obs,conflict,conflict_ratio]=[{free_non_obs:.3f},{free_on_obs:.3f},{free_conflict_ratio:.3f}] "
                f"start_plan=({sy_plan},{sx_plan}) start_raw_ref=({sy_raw},{sx_raw}) "
                f"start_pick={start_pick_mode} pick_score={start_pick_score:.3f} pick_score_alt={start_pick_score_alt:.3f} "
                f"start_pose_raw=({sy_raw_pose},{sx_raw_pose}) start_pose_plan=({int(sy_plan_pose)},{int(sx_plan_pose)}) "
                f"edge_margin_plan={edge_margin_plan} "
                f"nearest_occ_plan_d={nearest_occ_plan_d:.1f} nearest_occ_plan_rc={nearest_occ_plan_rc}",
                flush=True,
            )
            if (
                edge_margin_plan <= int(getattr(self, "_map_edge_warn_margin_cells", 0))
                and full_l_plan >= float(getattr(self, "_map_edge_warn_local_occ_min", 1.0))
                and float(np.mean(obs_from_depth)) <= float(getattr(self, "_map_edge_warn_global_occ_max", 0.0))
                and free_l_plan <= float(getattr(self, "_map_edge_warn_local_free_max", 0.0))
            ):
                print(
                    "[SG-Nav][map_edge_warn] "
                    f"step={self.total_steps} frame={frame_mode} "
                    f"edge_margin={edge_margin_plan} "
                    f"local_occ={full_l_plan:.3f} local_free={free_l_plan:.3f} "
                    f"global_occ={float(np.mean(obs_from_depth)):.3f} "
                    f"hint=increase_map_size_cm_or_check_depth_projection",
                    flush=True,
                )

            if bool(getattr(self, "_log_free_map_distribution", True)):
                fg_raw = np.asarray(free_map, dtype=np.float32).reshape(-1)
                fg_plan = np.asarray(free_map_plan, dtype=np.float32).reshape(-1)
                fl_raw = _local_vec(free_map, sy_raw, sx_raw)
                fl_plan = _local_vec(free_map_plan, sy_plan, sx_plan)
                raw_gt0 = float(np.mean(fg_raw > 1e-6)) if fg_raw.size > 0 else float("nan")
                raw_ge_thr = float(np.mean(fg_raw > free_thr)) if fg_raw.size > 0 else float("nan")
                plan_gt0 = float(np.mean(fg_plan > 1e-6)) if fg_plan.size > 0 else float("nan")
                plan_ge_thr = float(np.mean(fg_plan > free_thr)) if fg_plan.size > 0 else float("nan")
                lraw_gt0 = float(np.mean(fl_raw > 1e-6)) if fl_raw.size > 0 else float("nan")
                lraw_ge_thr = float(np.mean(fl_raw > free_thr)) if fl_raw.size > 0 else float("nan")
                lplan_gt0 = float(np.mean(fl_plan > 1e-6)) if fl_plan.size > 0 else float("nan")
                lplan_ge_thr = float(np.mean(fl_plan > free_thr)) if fl_plan.size > 0 else float("nan")
                print(
                    "[SG-Nav][free_diag] "
                    f"step={self.total_steps} frame={frame_mode} thr={free_thr:.3f} "
                    f"global_raw[p50,p90,p99,max]=[{_q(fg_raw,0.50):.3f},{_q(fg_raw,0.90):.3f},{_q(fg_raw,0.99):.3f},{_q(fg_raw,1.00):.3f}] "
                    f"global_plan[p50,p90,p99,max]=[{_q(fg_plan,0.50):.3f},{_q(fg_plan,0.90):.3f},{_q(fg_plan,0.99):.3f},{_q(fg_plan,1.00):.3f}] "
                    f"global_pass_raw[gt0,ge_thr]=[{raw_gt0:.4f},{raw_ge_thr:.4f}] "
                    f"global_pass_plan[gt0,ge_thr]=[{plan_gt0:.4f},{plan_ge_thr:.4f}] "
                    f"local_raw_ref[p50,p90,p99,max]=[{_q(fl_raw,0.50):.3f},{_q(fl_raw,0.90):.3f},{_q(fl_raw,0.99):.3f},{_q(fl_raw,1.00):.3f}] "
                    f"local_plan[p50,p90,p99,max]=[{_q(fl_plan,0.50):.3f},{_q(fl_plan,0.90):.3f},{_q(fl_plan,0.99):.3f},{_q(fl_plan,1.00):.3f}] "
                    f"local_pass_raw[gt0,ge_thr]=[{lraw_gt0:.4f},{lraw_ge_thr:.4f}] "
                    f"local_pass_plan[gt0,ge_thr]=[{lplan_gt0:.4f},{lplan_ge_thr:.4f}]",
                    flush=True,
                )
            if bool(getattr(self, "_log_obstacle_map_distribution", True)):
                occ_raw_v = np.asarray(occ_map, dtype=np.float32).reshape(-1)
                occ_plan_v = np.asarray(occ_map_plan, dtype=np.float32).reshape(-1)
                coll_raw_v = np.asarray(coll_map, dtype=np.float32).reshape(-1)
                coll_plan_v = np.asarray(coll_map_plan, dtype=np.float32).reshape(-1)
                occ_l_raw_v = _local_vec(occ_map, sy_raw, sx_raw)
                occ_l_plan_v = _local_vec(occ_map_plan, sy_plan, sx_plan)
                coll_l_raw_v = _local_vec(coll_map, sy_raw, sx_raw)
                coll_l_plan_v = _local_vec(coll_map_plan, sy_plan, sx_plan)
                occ_thr = float(self._traversible_occ_from_depth_min)
                print(
                    "[SG-Nav][obs_diag] "
                    f"step={self.total_steps} frame={frame_mode} occ_thr={occ_thr:.3f} coll_thr=0.500 "
                    f"occ_global_raw[p50,p90,p99,max]=[{_q(occ_raw_v,0.50):.3f},{_q(occ_raw_v,0.90):.3f},{_q(occ_raw_v,0.99):.3f},{_q(occ_raw_v,1.00):.3f}] "
                    f"occ_global_plan[p50,p90,p99,max]=[{_q(occ_plan_v,0.50):.3f},{_q(occ_plan_v,0.90):.3f},{_q(occ_plan_v,0.99):.3f},{_q(occ_plan_v,1.00):.3f}] "
                    f"occ_global_pass_raw[ge_occ_thr,ge_0.5]=[{float(np.mean(occ_raw_v >= occ_thr)):.4f},{float(np.mean(occ_raw_v >= 0.5)):.4f}] "
                    f"occ_global_pass_plan[ge_occ_thr,ge_0.5]=[{float(np.mean(occ_plan_v >= occ_thr)):.4f},{float(np.mean(occ_plan_v >= 0.5)):.4f}] "
                    f"occ_local_raw_ref[p50,p90,p99,max]=[{_q(occ_l_raw_v,0.50):.3f},{_q(occ_l_raw_v,0.90):.3f},{_q(occ_l_raw_v,0.99):.3f},{_q(occ_l_raw_v,1.00):.3f}] "
                    f"occ_local_plan[p50,p90,p99,max]=[{_q(occ_l_plan_v,0.50):.3f},{_q(occ_l_plan_v,0.90):.3f},{_q(occ_l_plan_v,0.99):.3f},{_q(occ_l_plan_v,1.00):.3f}] "
                    f"occ_local_pass_raw[ge_occ_thr,ge_0.5]=[{float(np.mean(occ_l_raw_v >= occ_thr)):.4f},{float(np.mean(occ_l_raw_v >= 0.5)):.4f}] "
                    f"occ_local_pass_plan[ge_occ_thr,ge_0.5]=[{float(np.mean(occ_l_plan_v >= occ_thr)):.4f},{float(np.mean(occ_l_plan_v >= 0.5)):.4f}] "
                    f"coll_global_raw[p50,p90,p99,max]=[{_q(coll_raw_v,0.50):.3f},{_q(coll_raw_v,0.90):.3f},{_q(coll_raw_v,0.99):.3f},{_q(coll_raw_v,1.00):.3f}] "
                    f"coll_global_plan[p50,p90,p99,max]=[{_q(coll_plan_v,0.50):.3f},{_q(coll_plan_v,0.90):.3f},{_q(coll_plan_v,0.99):.3f},{_q(coll_plan_v,1.00):.3f}] "
                    f"coll_local_raw_ref[p50,p90,p99,max]=[{_q(coll_l_raw_v,0.50):.3f},{_q(coll_l_raw_v,0.90):.3f},{_q(coll_l_raw_v,0.99):.3f},{_q(coll_l_raw_v,1.00):.3f}] "
                    f"coll_local_plan[p50,p90,p99,max]=[{_q(coll_l_plan_v,0.50):.3f},{_q(coll_l_plan_v,0.90):.3f},{_q(coll_l_plan_v,0.99):.3f},{_q(coll_l_plan_v,1.00):.3f}]",
                    flush=True,
                )

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
                if getattr(self, "_log_mapping_stats", False):
                    print(
                        "[SG-Nav][border_clear] "
                        f"step={self.total_steps} frame={frame_mode} cells={b}",
                        flush=True,
                    )
        if getattr(self, "_log_mapping_stats", False):
            win = 40
            y0 = max(0, sy_plan - win)
            y1w = min(obs_from_depth.shape[0], sy_plan + win + 1)
            x0 = max(0, sx_plan - win)
            x1w = min(obs_from_depth.shape[1], sx_plan + win + 1)
            if y1w > y0 and x1w > x0:
                occ_depth_local = float(np.mean(obs_from_depth[y0:y1w, x0:x1w]))
                occ_collision_local = float(np.mean(obs_from_collision[y0:y1w, x0:x1w]))
                free_depth_local = float(np.mean(free_from_depth[y0:y1w, x0:x1w]))
                inter_l = float(
                    np.sum(
                        np.logical_and(
                            obs_from_depth[y0:y1w, x0:x1w],
                            obs_from_collision[y0:y1w, x0:x1w],
                        )
                    )
                )
                union_l = float(
                    np.sum(
                        np.logical_or(
                            obs_from_depth[y0:y1w, x0:x1w],
                            obs_from_collision[y0:y1w, x0:x1w],
                        )
                    )
                )
                depth_l = float(np.sum(obs_from_depth[y0:y1w, x0:x1w]))
                coll_l = float(np.sum(obs_from_collision[y0:y1w, x0:x1w]))
                occ_iou_local = (inter_l / union_l) if union_l > 0.0 else float("nan")
                occ_precision_local = (inter_l / coll_l) if coll_l > 0.0 else float("nan")
                occ_recall_local = (inter_l / depth_l) if depth_l > 0.0 else float("nan")
            else:
                occ_depth_local = float("nan")
                occ_collision_local = float("nan")
                free_depth_local = float("nan")
                occ_iou_local = float("nan")
                occ_precision_local = float("nan")
                occ_recall_local = float("nan")
            print(
                "[SG-Nav][occ_align] "
                f"step={self.total_steps} frame={frame_mode} "
                f"iou={occ_iou:.3f} prec={occ_precision:.3f} rec={occ_recall:.3f} "
                f"iou_local={occ_iou_local:.3f} prec_local={occ_precision_local:.3f} "
                f"rec_local={occ_recall_local:.3f}",
                flush=True,
            )
            print(
                "[SG-Nav][occ_diag] "
                f"step={self.total_steps} occ_depth={float(np.mean(obs_from_depth)):.3f} "
                f"occ_collision={float(np.mean(obs_from_collision)):.3f} "
                f"free_depth={float(np.mean(free_from_depth)):.3f} "
                f"occ_depth_local={occ_depth_local:.3f} "
                f"occ_collision_local={occ_collision_local:.3f} "
                f"free_depth_local={free_depth_local:.3f} "
                f"frame={frame_mode}",
                flush=True,
            )
            if self._log_map_frame_select:
                if frame_scores is None:
                    score_text = "fixed"
                else:
                    score_text = " ".join(f"{k}:{v:.3f}" for k, v in frame_scores.items())
                print(
                    "[SG-Nav][frame_select] "
                    f"step={self.total_steps} selected={frame_mode} scores={score_text}",
                    flush=True,
                )
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
                print(
                    "[SG-Nav][traversible_bootstrap] "
                    f"step={self.total_steps} tr_ratio={tr_ratio:.4f} "
                    f"radius_cells={int(self._traversible_bootstrap_radius_cells)}",
                    flush=True,
                )

        if not traversible[sy_plan, sx_plan]:
            print("Not traversible, step is  ", self.navigate_steps)
        
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
        self._last_traversible_core = np.asarray(traversible > 0.5, dtype=bool).copy()
        self._last_traversible_start = (int(sy_plan), int(sx_plan))
        self._last_traversible_frame_mode = str(frame_mode)
        traversible = add_boundary(traversible)
        self._log_mapping_snapshot({"depth": self.depth}, traversible=traversible)
        return traversible, start_plan, start_o

    def _world_xy_m_to_grid_rc_for_nav(self, tx: float, ty: float) -> tuple[int, int]:
        """Map grid (row, col) from world meters; same convention as ``get_traversible`` start (gx1=gy1=0)."""
        start_y = float(self.map_size_cm) / 100.0 - float(ty)
        start_x = float(tx)
        r = int(round(start_y * 100.0 / float(self.map_resolution)))
        c = int(round(start_x * 100.0 / float(self.map_resolution)))
        return (
            max(0, min(self.map_size - 1, r)),
            max(0, min(self.map_size - 1, c)),
        )

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
        r, c = self._world_xy_m_to_grid_rc_for_nav(tx, ty)
        if not self._target_world_cell_visible_on_map(r, c):
            return False
        g = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        g[r, c] = 1.0
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
        dr = math.sin(theta)
        dc = math.cos(theta)
        clear_radius_cells = int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0))

        for k in range(1, samples + 1):
            dist_cells = max(1, int(round(float(k) * lookahead_cells / float(samples))))
            if dist_cells <= clear_radius_cells:
                continue
            rr = int(round(sy + dr * dist_cells))
            cc = int(round(sx + dc * dist_cells))
            if rr < 0 or rr >= obs.shape[0] or cc < 0 or cc >= obs.shape[1]:
                info = {
                    "block_rc": (rr, cc),
                    "block_type": "oob",
                    "sample_idx": int(k),
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

            if dist < col_threshold: # Collision
                self.former_collide += 1
                painted = []
                newly_marked = 0
                for i in range(length):
                    wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1)))
                    wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1)))
                    r, c = wy, wx
                    r = int(round(r * 100 / self.map_resolution))
                    c = int(round(c * 100 / self.map_resolution))
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    if self.collision_map[r, c] <= 0.5:
                        newly_marked += 1
                    painted.append((int(r), int(c)))
                    self.collision_map[r,c] = 1
                if getattr(self, "_log_collision_paint", False):
                    uniq = sorted(set(painted))
                    if len(uniq) > 0:
                        rs = [p[0] for p in uniq]
                        cs = [p[1] for p in uniq]
                        span = f"r=[{min(rs)},{max(rs)}] c=[{min(cs)},{max(cs)}]"
                    else:
                        span = "r=[nan,nan] c=[nan,nan]"
                    print(
                        "[SG-Nav][collision_paint] "
                        f"step={self.total_steps} dist={float(dist):.4f} thr={float(col_threshold):.4f} "
                        f"prev_action={int(self.prev_action)} former_collide={int(self.former_collide)} "
                        f"n_samples={int(length)} n_unique={len(uniq)} n_new={int(newly_marked)} "
                        f"{span} heading_deg={float(t1):.2f}",
                        flush=True,
                    )
            else:
                self.former_collide = 0

        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)
        goal_stop_state = self._get_goal_stop_status(agent_pose) if goal_found else None
        goal_stop_override = False

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg
            if getattr(self, "_log_pose_align_trace", False):
                self._last_planner_debug = {
                    "stop": True,
                    "stg": (int(stg_y), int(stg_x)),
                    "start_ij": (int(start[0]), int(start[1])),
                    "start_o_deg": float(start_o),
                    "observed_turn_deg": None if getattr(self, "_planner_observed_turn_deg", None) is None else float(self._planner_observed_turn_deg),
                    "last_turn_delta_deg": None if getattr(self, "_planner_last_turn_delta_deg", None) is None else float(self._planner_last_turn_delta_deg),
                    "replan": bool(replan),
                    "action": int(action),
                }

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
                    act_names = ("STOP", "FWD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "TURN")
                    an_raw = act_names[action_raw] if 0 <= action_raw < len(act_names) else str(action_raw)
                    an_used = act_names[int(action)] if 0 <= int(action) < len(act_names) else str(int(action))
                    br, bc = fg_info.get("block_rc", (None, None))
                    btype = fg_info.get("block_type", "unknown")
                    print(
                        "[SG-Nav][forward_guard] "
                        f"step={self.total_steps} "
                        f"action_raw={action_raw}({an_raw})->action_used={int(action)}({an_used}) "
                        f"start_ij=({int(start[0])},{int(start[1])}) start_o_deg={float(start_o):.2f} "
                        f"rel_deg={float(relative_angle):.2f} "
                        f"block_rc=({br},{bc}) block_type={btype} "
                        f"sample_idx={fg_info.get('sample_idx')} dist_cells={fg_info.get('dist_cells')} "
                        f"lookahead_cells={fg_info.get('lookahead_cells')} samples={fg_info.get('samples')} "
                        f"frame={fg_info.get('frame_mode')} turn_pick={turn_pick}",
                        flush=True,
                    )

            if getattr(self, "_log_pose_align_trace", False):
                self._last_planner_debug = {
                    "stop": False,
                    "stg": (int(stg_y), int(stg_x)),
                    "start_ij": (int(start[0]), int(start[1])),
                    "start_o_deg": float(start_o),
                    "angle_st_goal_deg": float(angle_st_goal),
                    "angle_agent_deg": float(angle_agent),
                    "relative_angle_deg": float(relative_angle),
                    "align_thr_deg": float(align_thr),
                    "align_thr_cfg_deg": float(align_thr_cfg),
                    "forward_exit_deg": float(forward_exit),
                    "forward_exit_cfg_deg": float(forward_exit_cfg),
                    "face_goal_only_deg": float(face_eps) if use_strict_face_goal else -1.0,
                    "face_goal_only_cfg_deg": float(face_eps_cfg) if use_strict_face_goal else -1.0,
                    "observed_turn_deg": None if observed_turn_deg is None else float(observed_turn_deg),
                    "last_turn_delta_deg": None if getattr(self, "_planner_last_turn_delta_deg", None) is None else float(self._planner_last_turn_delta_deg),
                    "prev_action": int(self.prev_action),
                    "former_collide": int(self.former_collide),
                    "replan": bool(replan),
                    "action_raw": int(action_raw),
                    "forward_guard_blocked": bool(fg_blocked),
                    "action": int(action),
                }
                if fg_info is not None:
                    self._last_planner_debug["forward_guard_info"] = fg_info

        if (
            goal_found
            and goal_stop_state is not None
            and goal_stop_state["distance_m"] <= float(self._goal_stop_distance_threshold_m)
        ):
            goal_stop_override = True
            if (
                goal_stop_state["distance_m"] <= 1e-3
                or abs(goal_stop_state["heading_error_deg"])
                <= float(self._goal_stop_face_threshold_deg)
            ):
                action = 0
            elif goal_stop_state["heading_error_deg"] > 0:
                action = 2
            else:
                action = 3

            if getattr(self, "_log_pose_align_trace", False):
                if not isinstance(getattr(self, "_last_planner_debug", None), dict):
                    self._last_planner_debug = {}
                self._last_planner_debug.update(
                    {
                        "goal_stop_override": True,
                        "goal_stop_distance_m": float(goal_stop_state["distance_m"]),
                        "goal_stop_heading_error_deg": float(
                            goal_stop_state["heading_error_deg"]
                        ),
                        "goal_stop_distance_thr_m": float(
                            self._goal_stop_distance_threshold_m
                        ),
                        "goal_stop_face_thr_deg": float(
                            self._goal_stop_face_threshold_deg
                        ),
                        "action": int(action),
                    }
                )

        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
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
        if goal_found:
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
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
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
        if self.args and self.args.visualize:
            ep_done = self.total_steps == 500
            sim = getattr(self, "simulator", None)
            if sim is not None and hasattr(sim, "_env") and hasattr(sim._env, "episode_over"):
                ep_done = ep_done or sim._env.episode_over
            if ep_done:
                self.save_video()

    def visualize(self, traversible, observations, number_action):
        if self.args.visualize:
            self.refresh_scenegraph_text_for_visualization()
            # 3-color occupancy panel:
            # unknown=white, free=light gray, obstacle=dark gray.
            fm = self.full_map[0, 0].detach().cpu().numpy()
            fb = self.fbe_free_map[0, 0].detach().cpu().numpy()
            coll = np.asarray(self.collision_map, dtype=np.float32)
            frame_mode = str(getattr(self, "_last_map_frame_transform", "id"))
            fm = self._apply_map_frame_transform(fm, frame_mode)
            fb = self._apply_map_frame_transform(fb, frame_mode)
            coll = self._apply_map_frame_transform(coll, frame_mode)
            obs_mask_0 = np.logical_or(
                fm > float(self._traversible_occ_from_depth_min), coll > 0.5
            )
            free_mask_0 = fb > float(self._traversible_free_map_min)
            # Occupancy panel should always show geometric semantics:
            # obstacle from depth/collision, free from free-map evidence, rest unknown.
            # (Do not paint unknown as free even if traversible policy allows unknown.)
            shown_free_0 = np.logical_and(free_mask_0, ~obs_mask_0)

            h, w = int(fm.shape[0]), int(fm.shape[1])
            cx_pose_raw = int(self.history_pose[-1][0].item() * 100.0 / self.resolution)
            cy_pose_raw = int(
                (self.map_size_cm / 100.0 - self.history_pose[-1][1].item()) * 100.0 / self.resolution
            )
            cy_pose_plan, cx_pose_plan = self._transform_rc_by_frame_mode(
                cy_pose_raw, cx_pose_raw, h, w, frame_mode
            )
            start_plan_cached = getattr(self, "_last_start_plan", None)
            start_raw_cached = getattr(self, "_last_start_raw", None)
            if (
                isinstance(start_plan_cached, (tuple, list))
                and len(start_plan_cached) == 2
                and np.isfinite(float(start_plan_cached[0]))
                and np.isfinite(float(start_plan_cached[1]))
            ):
                plan_cy = int(np.clip(int(start_plan_cached[0]), 0, h - 1))
                plan_cx = int(np.clip(int(start_plan_cached[1]), 0, w - 1))
                if (
                    isinstance(start_raw_cached, (tuple, list))
                    and len(start_raw_cached) == 2
                    and np.isfinite(float(start_raw_cached[0]))
                    and np.isfinite(float(start_raw_cached[1]))
                ):
                    raw_cy = int(np.clip(int(start_raw_cached[0]), 0, h - 1))
                    raw_cx = int(np.clip(int(start_raw_cached[1]), 0, w - 1))
                else:
                    raw_cy, raw_cx = self._inverse_transform_rc_by_frame_mode(
                        plan_cy, plan_cx, h, w, frame_mode
                    )
                center_src = "cached_start_plan"
            else:
                raw_cx = max(0, min(w - 1, int(cx_pose_raw)))
                raw_cy = max(0, min(h - 1, int(cy_pose_raw)))
                plan_cx = max(0, min(w - 1, int(cx_pose_plan)))
                plan_cy = max(0, min(h - 1, int(cy_pose_plan)))
                center_src = "pose_history_transform"
            obs_mask = obs_mask_0
            radius_cells = int(getattr(self, "_traversible_clear_robot_obs_radius_cells", 0))
            default_free_mask = self._disk_mask_for_center(h, w, plan_cy, plan_cx, radius_cells)
            default_free_mask = np.logical_and(default_free_mask, np.logical_not(obs_mask))
            shown_default_free = default_free_mask
            shown_free = np.logical_and(shown_free_0, np.logical_not(shown_default_free))
            shown_free_total = np.logical_or(shown_free, shown_default_free)
            unknown_mask = np.logical_and(~obs_mask, ~shown_free_total)

            def _known_score(cy, cx, win=20):
                y0 = max(0, int(cy) - win)
                y1 = min(h, int(cy) + win + 1)
                x0 = max(0, int(cx) - win)
                x1 = min(w, int(cx) + win + 1)
                if y1 <= y0 or x1 <= x0:
                    return 0.0
                return float(
                    np.mean(
                        np.logical_or(
                            obs_mask[y0:y1, x0:x1], shown_free_total[y0:y1, x0:x1]
                        )
                    )
                )

            known_raw_ref = _known_score(raw_cy, raw_cx)
            known_plan = _known_score(plan_cy, plan_cx)
            acx, acy = plan_cx, plan_cy
            crop_center_mode = "plan_frame"
            edge_margin = int(min(acy, h - 1 - acy, acx, w - 1 - acx))

            # Fallback: if the panel is almost all unknown, use planner traversible as free hint.
            if float(np.mean(unknown_mask)) > 0.985 and traversible is not None:
                tr = np.asarray(traversible)
                if tr.ndim == 2 and tr.shape[0] >= h + 2 and tr.shape[1] >= w + 2:
                    tr = tr[1:-1, 1:-1]
                if tr.shape == obs_mask.shape:
                    tr_free = tr > 0.5
                    shown_free = np.logical_or(shown_free, np.logical_and(tr_free, ~obs_mask))
                    shown_free_total = np.logical_or(shown_free, shown_default_free)
                    unknown_mask = np.logical_and(~obs_mask, ~shown_free_total)

            print(
                "[SG-Nav][viz_occ] "
                f"step={self.total_steps} frame={frame_mode} "
                f"obs={float(np.mean(obs_mask)):.3f} free={float(np.mean(shown_free_total)):.3f} "
                f"unk={float(np.mean(unknown_mask)):.3f} "
                f"crop_center={crop_center_mode} "
                f"known_plan={known_plan:.3f} known_raw_ref={known_raw_ref:.3f} "
                f"center_plan_rc=({acy},{acx}) edge_margin={edge_margin} "
                f"center_src={center_src}",
                flush=True,
            )

            unknown_rgb = np.array(colors.to_rgb("#FFFFFF"), dtype=np.float64)
            free_rgb = np.array(colors.to_rgb("#E7E7E7"), dtype=np.float64)
            default_free_rgb = np.array(colors.to_rgb("#D6D6D6"), dtype=np.float64)
            obstacle_rgb = np.array(colors.to_rgb("#A2A2A2"), dtype=np.float64)
            panel_hw3 = np.tile(unknown_rgb, (h, w, 1))
            panel_hw3[unknown_mask] = unknown_rgb
            panel_hw3[shown_free] = free_rgb
            panel_hw3[shown_default_free] = default_free_rgb
            panel_hw3[obs_mask] = obstacle_rgb
            paper_map_trans = torch.from_numpy(panel_hw3).permute(2, 0, 1).double()
            self.visualize_agent_and_goal(paper_map_trans)
            occ_panel = (paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            ph, pw = int(occ_panel.shape[0]), int(occ_panel.shape[1])
            acx = max(0, min(pw - 1, acx))
            acy = max(0, min(ph - 1, acy))
            crop_h, crop_w = 150, 200
            half_h, half_w = crop_h // 2, crop_w // 2
            src_top = int(acy - half_h)
            src_left = int(acx - half_w)
            src_bottom = src_top + crop_h
            src_right = src_left + crop_w
            # Keep the robot centered in the occupancy panel even near global-map borders.
            occupancy_map = np.full((crop_h, crop_w, 3), 255, dtype=np.uint8)
            copy_top = max(0, src_top)
            copy_left = max(0, src_left)
            copy_bottom = min(ph, src_bottom)
            copy_right = min(pw, src_right)
            if copy_bottom > copy_top and copy_right > copy_left:
                dst_top = copy_top - src_top
                dst_left = copy_left - src_left
                dst_bottom = dst_top + (copy_bottom - copy_top)
                dst_right = dst_left + (copy_right - copy_left)
                occupancy_map[dst_top:dst_bottom, dst_left:dst_right] = occ_panel[
                    copy_top:copy_bottom, copy_left:copy_right
                ]
            marker_x = int(half_w)
            marker_y = int(half_h)
            cv2.circle(occupancy_map, (marker_x, marker_y), 2, (255, 0, 0), -1)
            panel_top = 45
            panel_bottom = 285
            visualize_image = np.full((360, 800, 3), 255, dtype=np.uint8)
            det_rgb = self._render_detection_overlay(self.rgb_visualization)
            visualize_image = add_resized_image(visualize_image, det_rgb, (10, panel_top), (320, 240))
            visualize_image = add_resized_image(visualize_image, occupancy_map, (340, panel_top), (180, 240))
            visualize_image = add_rectangle(visualize_image, (10, panel_top), (330, panel_bottom), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (340, panel_top), (520, panel_bottom), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, panel_top), (790, 160), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 170), (790, panel_bottom), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (10, 295), (790, 350), (128, 128, 128), thickness=1)
            visualize_image = add_text(
                visualize_image,
                "Observation (Goal: {},  dist={})".format(
                    self.obj_goal,
                    "N/A" if self.goal_distance_for_vis is None else f"{self.goal_distance_for_vis:.2f}m",
                ),
                (50, 36),
                font_scale=0.5,
                thickness=1,
            )
            visualize_image = add_text(
                visualize_image,
                "Occupancy (unknown/free/obs)",
                (360, 36),
                font_scale=0.5,
                thickness=1,
            )
            visualize_image = add_text(visualize_image, "Scene Graph Nodes", (580, 36), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Edges", (580, 162), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "LLM Explanation", (330, 286), font_scale=0.5, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_node, 40), (550, 64), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_edge, 40), (550, 190), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.explanation, 150), (20, 314), font_scale=0.3, thickness=1)
            visualize_image = visualize_image[:, :, ::-1]
            self.visualize_image_list.append(visualize_image)
            os.makedirs(os.path.dirname(self.current_frame_path), exist_ok=True)
            # Write via temp + replace to avoid stale reads on editors that cache file handles.
            tmp_panel = self.current_frame_path + ".tmp.jpg"
            tmp_det = self.current_frame_det_path + ".tmp.jpg"
            cv2.imwrite(tmp_panel, visualize_image)
            cv2.imwrite(tmp_det, det_rgb[:, :, ::-1])
            os.replace(tmp_panel, self.current_frame_path)
            os.replace(tmp_det, self.current_frame_det_path)

            # Also keep per-step snapshots for debugging when IDE image tab does not auto-refresh.
            os.makedirs(self.current_frame_step_dir, exist_ok=True)
            step_name = f"current_frame_{int(self.total_steps):06d}.jpg"
            step_det_name = f"current_frame_det_{int(self.total_steps):06d}.jpg"
            cv2.imwrite(os.path.join(self.current_frame_step_dir, step_name), visualize_image)
            cv2.imwrite(os.path.join(self.current_frame_step_dir, step_det_name), det_rgb[:, :, ::-1])

    def _render_detection_overlay(self, rgb_img):
        """
        Return an RGB image with GLIP boxes overlaid.

        **Green**: any GLIP detection. **Red**: label matches ``obj_goal`` / match_tokens —
        only red boxes feed ``goal_bbox`` and navigation. If you see green around the
        object but no red, the class name does not contain the goal string (e.g. ``tv`` vs ``radio``).

        Drawing uses a temporary BGR image: OpenCV expects BGR ``color=``; applying BGR tuples to an
        RGB buffer swaps channels (nav red was showing as **blue** in RGB viewers).
        """
        vis = np.asarray(rgb_img).copy()
        preds = getattr(self, "current_obj_predictions", None)
        if preds is None or len(preds) == 0:
            return vis

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        labels = []
        if hasattr(preds, "get_field"):
            try:
                labels = list(preds.get_field("labels"))
            except Exception:
                labels = []
        scores = None
        if hasattr(preds, "get_field"):
            try:
                scores = preds.get_field("scores")
            except Exception:
                scores = None

        boxes = preds.bbox if hasattr(preds, "bbox") else []
        n = min(len(boxes), len(labels) if labels else len(boxes))
        nav_match_count = 0
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
            is_nav = self._navigation_goal_label_match(raw_lab)
            if is_nav:
                nav_match_count += 1
            # BGR: red = nav, green = non-nav GLIP
            color = (0, 0, 255) if is_nav else (0, 255, 0)
            cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), color, 2)
            lab = str(raw_lab)
            if scores is not None:
                try:
                    sc = float(scores[i])
                    lab = f"{lab}:{sc:.2f}"
                except Exception:
                    pass
            if is_nav:
                lab = "[nav] " + lab
            cv2.putText(
                vis_bgr,
                lab,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        h, w = vis_bgr.shape[:2]
        hint = f"RED=nav({nav_match_count}) GREEN=other | obj_goal={self.obj_goal!r}"
        cv2.putText(
            vis_bgr,
            hint,
            (4, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    def refresh_scenegraph_text_for_visualization(self):
        # Keep visualization text in sync with the latest scene graph state.
        nodes = getattr(self.scenegraph, "nodes", [])
        global_caps = [node.caption for node in nodes if getattr(node, "caption", None)]

        frame_caps = []
        seg = getattr(self.scenegraph, "segment2d_results", [])
        if len(seg) > 0 and isinstance(seg[-1], dict):
            frame_caps = [str(c) for c in seg[-1].get("caption", []) if str(c).strip()]

        if self.show_frame_nodes_only:
            src = frame_caps if len(frame_caps) > 0 else global_caps
            self.text_node = ", ".join(src)
        else:
            # Show both scopes for debugging consistency between current RGB and global graph memory.
            frame_text = ", ".join(frame_caps[:24]) if frame_caps else "-"
            global_text = ", ".join(global_caps[:24]) if global_caps else "-"
            self.text_node = f"[frame] {frame_text} | [global] {global_text}"

        edge_lines = []
        seen = set()
        for node in nodes:
            for edge in getattr(node, "edges", []):
                relation = getattr(edge, "relation", None)
                n1 = getattr(edge.node1, "caption", None)
                n2 = getattr(edge.node2, "caption", None)
                if not n1 or not n2 or not relation:
                    continue
                key = tuple(sorted([n1, n2]) + [relation])
                if key in seen:
                    continue
                seen.add(key)
                edge_lines.append(f"({n1}, {relation}, {n2})")
        self.text_edge = "; ".join(edge_lines)

        reason = getattr(self.scenegraph, "reason_visualization", "")
        if reason:
            self.explanation = reason
        elif self.text_edge:
            self.explanation = "Edge relations are derived from VLM/LLM proposals."
        elif self.text_node:
            # Prefer actual node / frame captions over a generic skip_edge_llm banner.
            tail = ""
            if getattr(self.scenegraph, "runtime_cfg", None) and self.scenegraph.runtime_cfg.get(
                "skip_edge_llm", False
            ):
                tail = " | edges off (skip_edge_llm)"
            self.explanation = f"{self.text_node[:300]}{tail}"
        elif getattr(self.scenegraph, "runtime_cfg", None) and self.scenegraph.runtime_cfg.get(
            "skip_edge_llm", False
        ):
            self.explanation = (
                "No scene-graph node text this frame. Edge/VLM is off (skip_edge_llm); "
                "enable it in yaml for relation-based explanations."
            )
        else:
            self.explanation = "Scene graph is empty for the current frame."

    def save_video(self):
        """Write episode visualization to MP4.

        OpenCV ``mp4v`` (MPEG-4 Part 2) is valid but many players (e.g. Windows Media Player,
        some browsers) handle H.264 better. When ``ffmpeg`` is on PATH, we transcode to
        libx264 + yuv420p and ``+faststart`` so files are broadly playable and the moov atom
        is near the start of the file.
        """
        save_video_dir = os.path.join(self.visualization_dir, "video")
        os.makedirs(save_video_dir, exist_ok=True)
        base = f"vid_{self.count_episodes:06d}"
        final_path = os.path.join(save_video_dir, f"{base}.mp4")
        tmp_path = os.path.join(save_video_dir, f"{base}.partial.mp4")

        if not self.visualize_image_list:
            return

        height, width = self.visualize_image_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(tmp_path, fourcc, 4.0, (width, height))
        opened = video.isOpened()
        try:
            if not opened:
                import sys

                sys.stderr.write(
                    "[SG_Nav] VideoWriter failed to open (codec/backend). "
                    f"path={tmp_path!r} size={width}x{height}\n"
                )
                return
            for visualize_image in self.visualize_image_list:
                video.write(visualize_image)
        finally:
            video.release()

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            h264_tmp = os.path.join(save_video_dir, f"{base}.h264.tmp.mp4")
            cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-i",
                tmp_path,
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                h264_tmp,
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if (
                    r.returncode == 0
                    and os.path.isfile(h264_tmp)
                    and os.path.getsize(h264_tmp) > 0
                ):
                    os.remove(tmp_path)
                    os.replace(h264_tmp, final_path)
                    return
            except OSError:
                pass
            if os.path.isfile(h264_tmp):
                try:
                    os.remove(h264_tmp)
                except OSError:
                    pass

        os.replace(tmp_path, final_path)

    def save_scenegraph_json_snapshot(self):
        scenegraph = getattr(self, "scenegraph", None)
        if scenegraph is None:
            return

        nodes = list(getattr(scenegraph, "nodes", []))
        node_idx = {id(node): i for i, node in enumerate(nodes)}
        json_nodes = []
        for i, node in enumerate(nodes):
            center = getattr(node, "center", None)
            room_node = getattr(node, "room_node", None)
            json_nodes.append(
                {
                    "id": i,
                    "caption": str(getattr(node, "caption", "") or ""),
                    "center": [int(center[0]), int(center[1])] if center is not None else None,
                    "room": str(getattr(room_node, "caption", "") or ""),
                    "is_goal_node": bool(getattr(node, "is_goal_node", False)),
                }
            )

        seen_edges = set()
        json_edges = []
        for node in nodes:
            for edge in getattr(node, "edges", []):
                eid = id(edge)
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                n1 = getattr(edge, "node1", None)
                n2 = getattr(edge, "node2", None)
                if n1 is None or n2 is None:
                    continue
                json_edges.append(
                    {
                        "node1_id": node_idx.get(id(n1)),
                        "node1_caption": str(getattr(n1, "caption", "") or ""),
                        "node2_id": node_idx.get(id(n2)),
                        "node2_caption": str(getattr(n2, "caption", "") or ""),
                        "relation": getattr(edge, "relation", None),
                    }
                )

        payload = {
            "step": int(self.total_steps),
            "navigate_step": int(self.navigate_steps),
            "obj_goal": str(getattr(self, "obj_goal", "")),
            "obj_goal_sg": str(getattr(self, "obj_goal_sg", "")),
            "found_goal": bool(getattr(self, "found_goal", False)),
            "found_possible_goal": bool(getattr(self, "found_possible_goal", False)),
            "nodes": json_nodes,
            "edges": json_edges,
        }

        os.makedirs(os.path.dirname(self.scenegraph_json_path), exist_ok=True)
        os.makedirs(self.scenegraph_json_step_dir, exist_ok=True)

        latest_tmp = self.scenegraph_json_path + ".tmp"
        with open(latest_tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(latest_tmp, self.scenegraph_json_path)

        step_path = os.path.join(self.scenegraph_json_step_dir, f"scenegraph_{int(self.total_steps):06d}.json")
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def visualize_agent_and_goal(self, map):
        frame_mode = str(getattr(self, "_last_map_frame_transform", "id"))
        h, w = int(map.shape[1]), int(map.shape[2])

        def _paint_square(center_r: int, center_c: int, size: int, color_index: int, alpha: float = 1.0):
            y0 = max(0, int(center_r) - size)
            y1 = min(h, int(center_r) + size)
            x0 = max(0, int(center_c) - size)
            x1 = min(w, int(center_c) + size)
            if y1 <= y0 or x1 <= x0:
                return
            color_ori = map[:, y0:y1, x0:x1]
            color_new = torch.zeros_like(color_ori)
            color_new[color_index] = 1
            map[:, y0:y1, x0:x1] = alpha * color_new + (1 - alpha) * color_ori

        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 2 if idx == len(self.history_pose) - 1 else 1
            use_cached_latest = (
                idx == len(self.history_pose) - 1
                and isinstance(getattr(self, "_last_start_plan", None), (tuple, list))
                and len(getattr(self, "_last_start_plan", None)) == 2
            )
            if use_cached_latest:
                ry = int(np.clip(int(self._last_start_plan[0]), 0, h - 1))
                rx = int(np.clip(int(self._last_start_plan[1]), 0, w - 1))
            else:
                px = float(pose[0].item() if hasattr(pose[0], "item") else pose[0])
                py = float(pose[1].item() if hasattr(pose[1], "item") else pose[1])
                cx = int(px * 100.0 / self.resolution)
                cy = int((self.map_size_cm / 100.0 - py) * 100.0 / self.resolution)
                ry, rx = self._transform_rc_by_frame_mode(cy, cx, h, w, frame_mode)
            _paint_square(ry, rx, agent_size, color_index=0, alpha=alpha)

        goal_map = getattr(self, "goal_map", None)
        if goal_map is not None:
            goal_map_vis = self._apply_map_frame_transform(goal_map, frame_mode)
            ys, xs = np.where(goal_map_vis == 1)
            if len(ys) > 0:
                _paint_square(int(ys[0]), int(xs[0]), size=2, color_index=1, alpha=1.0)
        return map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    args = parser.parse_args()
    os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd.yaml"
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = SG_Nav_Agent(task_config=config, args=args)

    challenge = habitat.Challenge(eval_remote=False, split_l=args.split_l, split_r=args.split_r)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
