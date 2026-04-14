# SG_Nav 精简方案

## 一、终端输出：仅保留 `[SG-Nav][stage]`

精简后终端**唯一输出**为每步一行的导航阶段信息（初始化信息除外）：

```
[SG-Nav][stage] step=42 stage=frontier_explore action=1(FWD)
[SG-Nav][stage] step=43 stage=goal_driven_nav action=3(RIGHT)
```

所有其他诊断、调试、统计信息**全部写入 JSON 日志文件**，不打印到终端。

---

## 二、JSON 日志文件设计

### 2.1 文件路径

```
data/visualization/{experiment}/nav_log.jsonl
```

每步追加一行 JSON（JSONL 格式），方便流式读取和 `grep/jq` 查询。

### 2.2 每步记录结构

```json
{
  "step": 42,
  "episode": 0,
  "timestamp_ms": 1718300000000,
  "stage": {
    "code": "frontier_explore",
    "label": "frontier探索",
    "goal_map_src": "replan_fbe_frontier",
    "found_goal": false,
    "found_possible_goal": false,
    "using_random_goal": false,
    "active_gt_world_nav": false
  },
  "action": {
    "id": 1,
    "name": "FWD",
    "raw_id": 1,
    "forward_guard_blocked": false
  },
  "pose": {
    "full_pose_m_deg": [15.3, 14.8, 90.2],
    "gps": [0.31, -0.22],
    "compass_rad": 1.571,
    "compass_offset_applied": true
  },
  "goal": {
    "goal_rc": [580, 612],
    "goal_drift": 3,
    "goal_gps": [0.45, -0.10],
    "goal_distance_m": 2.3,
    "heading_error_deg": 12.5
  },
  "detection": {
    "ran_this_step": true,
    "goal_bbox_count": 0,
    "goal_votes_total": 0,
    "found_goal_this_step": false
  },
  "mapping": {
    "depth_stats": {
      "min": 0.12,
      "max": 7.8,
      "median_positive": 3.2,
      "zero_fraction": 0.02
    },
    "full_map_occ_ratio": 0.015,
    "collision_occ_ratio": 0.003,
    "free_map_occ_ratio": 0.08,
    "traversible_free_ratio": 0.62,
    "suspend_active": false,
    "suspend_countdown": 0
  },
  "camera": {
    "active": true,
    "source": "camera_pose_world",
    "tilt_source": "cam_z_down",
    "pitch_deg_used": -5.2,
    "height_cm_used": 132.0,
    "roll_deg": 2.1,
    "rejected": false,
    "fall_suspected": false
  },
  "planner": {
    "stg_rc": [582, 615],
    "start_rc": [580, 610],
    "start_o_deg": 88.5,
    "relative_angle_deg": 12.3,
    "align_thr_deg": 26.0,
    "observed_turn_deg": 18.5,
    "collision_painted": false,
    "former_collide": 0
  },
  "fbe": {
    "ran_this_step": false,
    "frontier_count": null,
    "eligible_count": null,
    "selected_goal_rc": null,
    "score_max": null
  },
  "scenegraph": {
    "updated_this_step": true,
    "perception_ran": true,
    "node_count": 12,
    "edge_count": 8
  },
  "flags": []
}
```

### 2.3 `flags` 字段

运行时健康警告以字符串数组方式记录，替代之前的 `[SG-Nav][check]`：

```json
"flags": ["DEPTH_TOO_MANY_ZEROS", "CAMERA_PITCH_LARGE"]
```

可能的值：
- `TRAVERSIBLE_TOO_FREE` — traversible 几乎全 free（> 95%）
- `DEPTH_TOO_MANY_ZEROS` — 深度图零值过多（> 25%）
- `DEPTH_MEDIAN_OUT_OF_RANGE` — 深度中值异常
- `CAMERA_PITCH_LARGE` — 相机 pitch 过大
- `CAMERA_HEIGHT_OUTLIER` — 相机高度异常
- `FALL_SUSPECTED` — 疑似跌倒
- `COLLISION_PAINTED` — 本步标记了碰撞
- `FORWARD_GUARD_BLOCKED` — 前向保护触发
- `MAPPING_SUSPENDED` — 建图暂停中

### 2.4 特殊事件记录

Episode 重置和初始化也写入同一文件：

```json
{"event": "episode_reset", "episode": 1, "obj_goal": "radio", "map_size_cm": 6000, "timestamp_ms": ...}
{"event": "init_complete", "sensor_wh": [720, 720], "hfov_deg": 99, "map_resolution": 5, "timestamp_ms": ...}
```

---

## 三、保留的 YAML 参数（共 30 个）

### 3.1 传感器与硬件（必须匹配实际机器人/仿真器）

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `DEPTH_SENSOR.WIDTH` | 720 | 深度图宽度（像素） |
| `DEPTH_SENSOR.HEIGHT` | 720 | 深度图高度（像素） |
| `DEPTH_SENSOR.HFOV` | 99 | 水平视场角（度） |
| `DEPTH_SENSOR.MIN_DEPTH` | 0.1 | 深度最小有效值（米），低于此值视为无效 |
| `DEPTH_SENSOR.MAX_DEPTH` | 8 | 深度最大有效值（米） |
| `AGENT_0.HEIGHT` | 0.88 | Agent 高度（米），当无 camera_pose_world 时用于推算相机高度 |
| `AGENT_0.RADIUS` | 0.18 | Agent 碰撞半径（米），影响障碍物膨胀 |
| `SIMULATOR.TURN_ANGLE` | 20 | 每次转弯角度（度），供客户端使用 |
| `SIMULATOR.FORWARD_STEP_SIZE` | 0.25 | 每次前进步长（米），供客户端使用 |

### 3.2 全局地图

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `map_size_cm` | 6000 | 全局地图边长（厘米），决定可探索范围。范围 1000-12000。6000cm = 60m × 60m |

### 3.3 检测与场景图

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `detect_interval` | 5 | 每 N 步运行一次 GLIP 物体检测 + 房间检测。越小检测越频繁但越慢 |
| `scenegraph_update_interval` | 5 | 每 N 步更新一次场景图（GroundingDINO + SAM + 3D融合）。比检测更重，可设更大 |
| `goal_distance_threshold` | 5.0 | GLIP 检测到目标时，深度值（米）低于此阈值才确认为"找到目标" |
| `save_scenegraph_json` | false | 是否每步保存场景图 JSON 快照（开启会增加 I/O） |
| `scenegraph.skip_edge_llm` | false | 跳过 LLM 推理场景图边关系（关闭 = 更快但无空间关系） |

### 3.4 导航规划器

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `panorama_spin_until_step` | 22 | 初始全景扫描持续步数。前 N 步无目标线索时持续旋转扫描环境 |
| `planner_align_angle_deg` | 30 | 航向偏差（度）超过此值时转向、低于时前进。越大越激进 |
| `fmm_step_size` | 3 | FMM 短期子目标窗口大小（格子）。越大子目标越远，路径越平滑 |
| `collision_threshold_m` | 0.08 | 前进后 GPS 位移低于此值判定为碰撞，在前方标记障碍物 |
| `goal_stop_distance_threshold_m` | 0.3 | 确认目标后的停止距离（米）。到达此距离内开始面朝目标并 STOP |
| `goal_stop_face_threshold_deg` | 12.0 | 停止前面朝目标的角度容差（度） |

### 3.5 相机外参（HTTP/OmniGibson 专用，必须匹配机器人相机安装）

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `compass_offset_deg` | -90 | 罗盘偏移（度）。弥补机器人坐标系与地图"前方"的旋转差异 |
| `camera_tilt_source` | "cam_z_down" | 从 camera_pose_world 四元数提取俯仰角的方式。可选：`pitch`, `roll`, `cam_z_down`, `cam_y_down` |
| `camera_height_fallback_cm` | 100.0 | 无 camera_pose_world 时的固定相机高度（厘米） |
| `camera_pitch_min_deg` | -45 | 相机 pitch 有效范围下限（度），超出钳制 |
| `camera_pitch_max_deg` | 45 | 相机 pitch 有效范围上限（度） |
| `camera_height_min_cm` | 120 | 相机高度有效范围下限（厘米），低于此值钳制 |
| `camera_height_max_cm` | 220 | 相机高度有效范围上限（厘米） |

### 3.6 Traversible（可通行区域）

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `traversible_obstacle_inflation_m` | 0.15 | 障碍物膨胀半径（米）。在障碍边缘外扩此距离作为不可通行。越大越保守 |
| `traversible_occ_from_depth_min` | 0.45 | full_map 值超过此阈值的格子视为障碍物。越低越敏感 |
| `sem_map_pred_threshold` | 2.0 | 语义地图体素归一化阈值。越低越容易在 full_map 上产生障碍标记 |

---

## 四、硬编码的参数（从 YAML 中移除）

以下参数将在代码中硬编码为固定值，不再暴露给 YAML：

### 4.1 地图帧（硬编码 `flipud`）

删除所有帧变换选项，硬编码为 `flipud`：
- ~~`map_frame_fixed_mode`~~ → `"flipud"`
- ~~`map_frame_auto_select`~~ → 删除
- ~~`map_frame_auto_select_win_cells`~~ → 删除
- ~~`start_plan_flipud_mirror_enable`~~ → 删除
- ~~`start_plan_flipud_mirror_min_delta`~~ → 删除
- ~~`start_plan_flipud_pose_known_min`~~ → 删除

### 4.2 规划器细节（硬编码为当前默认值）

- ~~`planner_align_hysteresis_deg`~~ → `5.0`
- ~~`planner_face_goal_max_error_deg`~~ → `30.0`
- ~~`planner_use_observed_turn`~~ → `true`
- ~~`planner_observed_turn_ema_alpha`~~ → `0.35`
- ~~`planner_min_effective_turn_deg`~~ → `1.0`
- ~~`planner_forward_error_turn_ratio`~~ → `0.5`
- ~~`fmm_stop_cond`~~ → `1.0`
- ~~`possible_goal_stop_escape_steps`~~ → `6`
- ~~`possible_goal_stop_escape_cooldown_steps`~~ → `12`

### 4.3 前向保护（硬编码）

- ~~`forward_guard_enable`~~ → `true`
- ~~`forward_guard_lookahead_m`~~ → `0.50`
- ~~`forward_guard_samples`~~ → `6`
- ~~`forward_guard_cone_half_angle_deg`~~ → `25.0`

### 4.4 随机目标 & Bootstrap（硬编码）

- ~~`random_goal_min_distance_m`~~ → `0.60`
- ~~`random_goal_obstacle_clearance_m`~~ → `0.35`
- ~~`traversible_bootstrap_enable`~~ → `true`
- ~~`traversible_bootstrap_min_free_ratio`~~ → `0.002`
- ~~`traversible_bootstrap_radius_m`~~ → `0.9`
- ~~`traversible_clear_robot_obs_enable`~~ → `true`
- ~~`traversible_clear_robot_obs_radius_m`~~ → `0.30`
- ~~`traversible_clear_border_obs_cells`~~ → `2`
- ~~`traversible_require_explored`~~ → `false`
- ~~`traversible_free_map_min`~~ → `0.25`

### 4.5 地图暂停（简化为开关 + 固定阈值）

- ~~`map_suspend_on_fall`~~ → `true`（硬编码开启）
- ~~`map_suspend_steps_on_fall`~~ → `4`
- ~~`map_suspend_roll_deg`~~ → `85.0`
- ~~`map_suspend_pitch_deg`~~ → `50.0`
- ~~`map_suspend_height_cm`~~ → `95.0`
- ~~`map_suspend_height_delta_cm`~~ → `20.0`

### 4.6 相机外参细节（硬编码）

- ~~`camera_extrinsic_use_client_pose`~~ → `true`
- ~~`camera_height_from_world_z`~~ → `true`
- ~~`camera_height_z_offset_m`~~ → `0.0`
- ~~`camera_tilt_fallback_to_cam_z_down_on_outlier`~~ → `true`
- ~~`camera_pitch_mapping_scale`~~ → `1.0`
- ~~`camera_pitch_mapping_offset_deg`~~ → `0.0`
- ~~`camera_extrinsic_reject_outlier`~~ → `true`
- ~~`camera_height_reject_min_cm`~~ → `120.0`
- ~~`camera_height_reject_max_cm`~~ → `210.0`
- ~~`camera_pitch_reject_abs_deg`~~ → `55.0`
- ~~`camera_extrinsic_reject_motion_delta`~~ → `true`
- ~~`camera_pitch_reject_delta_deg`~~ → `6.0`
- ~~`camera_height_reject_delta_cm`~~ → `8.0`

### 4.7 深度→地图高度带（硬编码）

- ~~`map_obstacle_min_z_cm`~~ → `25.0`
- ~~`map_obstacle_max_z_cm`~~ → `-1.0`（= 相机高度 + 50cm）
- ~~`map_obstacle_above_camera_cm`~~ → `50.0`
- ~~`map_free_min_z_cm`~~ → `-150.0`
- ~~`map_free_max_z_cm`~~ → `10.0`

### 4.8 映射融合阈值（硬编码）

- ~~`sem_exp_pred_threshold`~~ → `2.5`
- ~~`free_map_pred_threshold`~~ → `0.5`
- ~~`free_exp_pred_threshold`~~ → `0.5`

### 4.9 可视化（硬编码）

- ~~`visualize_use_traversible_fallback`~~ → 删除（永远 false）
- ~~`visualize_centered_crop_ratio`~~ → `0.30`

### 4.10 GPS / 其他

- ~~`gps_negate_y`~~ → `false`（硬编码）
- ~~`gt_pathplan_when_target_on_map`~~ → `true`（硬编码）
- ~~`cuda_empty_cache_each_step`~~ → 删除

---

## 五、删除的代码路径

| 删除项 | 原始行数估计 | 理由 |
|--------|-------------|------|
| `SG_Nav_0.py` | 全文件 | 旧版备份，无引用 |
| `main()` + Habitat Challenge 入口 | ~30 行 | 仅走 HTTP |
| PSL 导入 + `add_predicates()` + `add_rules()` | ~25 行 | 从未执行 |
| `_auto_select_map_frame_transform()` | ~35 行 | 从未被调用 |
| 脚本化相机控制 (steps 1-16 LOOK/TURN) | ~30 行 | HTTP 模式下始终跳过 |
| `_emit_spin_diagnosis()` | ~75 行 | 替换为 JSON 日志 |
| `_log_mapping_snapshot()` | ~45 行 | 替换为 JSON 日志 |
| `_log_runtime_check_flags()` | ~60 行 | 替换为 JSON 日志 |
| `_dump_camera_map_sync_debug()` | ~50 行 | 替换为 JSON 日志 |
| `_dump_map_fusion_motion_debug()` | ~70 行 | 替换为 JSON 日志 |
| 25+ 个日志开关及其 `if` 分支 | ~300 行 | 全部替换为统一 JSON writer |
| 诊断模式 (`legacy_strict`, `bilinear`) | ~30 行 | 废弃选项 |
| OG occupancy 融合 | ~50 行 | 当前未使用 |
| Behavior bridge YAML 参数 | ~15 行 | 移到客户端 |
| `update_metrics()` | ~10 行 | Habitat 专用 |
| flipud mirror 启发法 | ~40 行 | 从未启用 |
| `_diag_full_map_any_toggle` 及相关 | ~20 行 | 简化诊断 |

**已完成：5023 行 → 2774 行（减少 45%），同时抽取 4 个 utils 模块。**

---

## 六、精简后 YAML 样例

```yaml
SIMULATOR:
  TURN_ANGLE: 20
  FORWARD_STEP_SIZE: 0.25
  AGENT_0:
    HEIGHT: 0.88
    RADIUS: 0.18
  DEPTH_SENSOR:
    WIDTH: 720
    HEIGHT: 720
    HFOV: 99
    MIN_DEPTH: 0.1
    MAX_DEPTH: 8

TASK:
  TYPE: ObjectNav-v1
  POSSIBLE_ACTIONS: ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
                      "LOOK_UP", "LOOK_DOWN", "TURN_RIGHT_2"]

SGNAV_RUNTIME:
  # --- 全局地图 ---
  map_size_cm: 6000

  # --- 检测与场景图 ---
  detect_interval: 5
  scenegraph_update_interval: 5
  goal_distance_threshold: 8.0
  save_scenegraph_json: false
  scenegraph:
    skip_edge_llm: false

  # --- 导航规划 ---
  panorama_spin_until_step: 22
  planner_align_angle_deg: 30
  fmm_step_size: 3
  collision_threshold_m: 0.08
  goal_stop_distance_threshold_m: 0.3
  goal_stop_face_threshold_deg: 12.0

  # --- 相机外参（匹配机器人） ---
  compass_offset_deg: -90.0
  camera_tilt_source: cam_z_down
  camera_height_fallback_cm: 100.0
  camera_pitch_min_deg: -45.0
  camera_pitch_max_deg: 45.0
  camera_height_min_cm: 120.0
  camera_height_max_cm: 220.0

  # --- 可通行区域 ---
  traversible_obstacle_inflation_m: 0.15
  traversible_occ_from_depth_min: 0.45
  sem_map_pred_threshold: 2.0
```

---

## 七、精简后代码结构预览

```
SG_Nav.py  (~3800 行，减少约 25%)
├── class SG_Nav_Agent
│   ├── __init__()           ~250 行 (原 625 行)
│   ├── _nav_log_writer      统一 JSON 日志写入器
│   ├── warmup()
│   ├── reset()
│   ├── act()                ~400 行 (无日志分支)
│   ├── detect_objects()
│   ├── fbe()
│   ├── update_map / free_map / room_map
│   ├── get_traversible()    ~250 行 (删除帧选项+镜像)
│   ├── _plan()              ~200 行 (删除日志分支)
│   ├── _get_stg()
│   ├── set_random_goal()
│   ├── visualize()
│   └── save_video()
└── (无 main 函数)
```

---

## 八、实施状态

- [x] 新建 `utils/nav_logger.py` — 统一 JSON 写入器
- [x] 新建 `utils/geometry.py` — 坐标变换、角度工具
- [x] 新建 `utils/camera_pose.py` — 相机外参同步
- [x] 新建 `utils/nav_visualization.py` — 可视化、视频保存
- [x] 删除 dead code — PSL、main()、_auto_select、debug dump 等
- [x] 精简 `__init__` — 移除 18 个死日志标志
- [x] 替换所有 print — 38 → 4（init 3 + `[SG-Nav][stage]` 1）
- [x] 精简 YAML 配置 — 移除死参数，分组注释
- [x] warmup 默认开启 — YAML `warmup: true, warmup_runs: 2`，server 自动读取
- [ ] 测试验证 — 用 HTTP 服务器端到端跑一轮确认功能正常
