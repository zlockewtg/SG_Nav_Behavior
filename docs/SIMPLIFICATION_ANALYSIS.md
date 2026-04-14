# SG_Nav 精简分析文档

## 一、代码整体架构

`SG_Nav.py` 共 **5023 行**，核心类 `SG_Nav_Agent`。系统有两条运行路径：

| 路径 | 入口 | 状态 |
|------|------|------|
| **Habitat ObjectNav 挑战** | `SG_Nav.py → main()` + `habitat.Challenge.submit()` | 原始设计，目前不活跃 |
| **HTTP/OmniGibson 服务** | `sgnav_http_server.py` → `SG_Nav_Agent` | **当前主要使用路径** |

辅助文件：
- `SG_Nav_0.py` — 旧版备份，无任何文件导入它
- `scenegraph.py` — 场景图构建（GroundingDINO + SAM + Ollama LLM）
- `utils/utils_glip.py` — GLIP 检测词表、MP3D 类别映射
- `utils/utils_fmm/mapping.py` — 深度→占用图融合（torch.nn.Module）
- `utils/utils_fmm/fmm_planner.py` — Fast Marching Method 路径规划
- `behavior/sgnav_http_server.py` — HTTP API 服务器
- `behavior/eval_skill_sgnav_http.py` — OmniGibson 评估客户端

---

## 二、冗余设计与可删除代码

### 2.1 可直接删除的代码/文件

| 项目 | 位置 | 理由 |
|------|------|------|
| **`SG_Nav_0.py`** | 根目录 | 旧版完整备份，无任何代码导入，可直接删除 |
| **`main()` 函数** | `SG_Nav.py:4991-5022` | Habitat Challenge 入口，当前走 HTTP 路径，不再需要 |
| **PSL 导入** | `SG_Nav.py:33-36` (`pslpython`) | `PSLModel`, `Partition`, `Predicate`, `Rule` 被导入但**从未在此文件执行推理**，`add_predicates` / `add_rules` 也从未被调用 |
| **`add_predicates()` / `add_rules()`** | `SG_Nav.py:1590-1610` | PSL 辅助方法，无调用者 |
| **`_auto_select_map_frame_transform()`** | `SG_Nav.py:3379-3412` | 自动选择地图帧方法，`_map_frame_auto_select` flag 存在但 `get_traversible` 从未调用此方法 |
| **`_map_frame_auto_select` 相关参数** | `__init__` | `map_frame_auto_select`, `map_frame_auto_select_win_cells` — dead code |
| **argparse 中的 `--split_l` / `--split_r`** | `main()` 及 `__init__` | 仅在 Habitat challenge split 模式使用 |
| **Habitat episode 目标读取** | `reset():1662-1676` | `simulator._env.current_episode` 分支，HTTP 模式永远传 `object_category` |
| **`update_metrics()`** | `SG_Nav.py:4387-4397` | Habitat SPL 指标计算，HTTP 模式不使用 |
| **大量 habitat-lab/ 目录** | `habitat-lab/` | 整个 Habitat 基线代码库，SG-Nav 仅用少数工具函数 |
| **旧 Habitat config** | `configs/ddppo_*.yaml`, `configs/challenge_*.yaml` | Habitat 挑战配置，HTTP 模式不使用 |

### 2.2 可大幅精简的设计

#### A. 脚本化相机控制 (Scripted Camera Tilt)

**位置**：`act()` 中 step 1-16 的 LOOK_UP / LOOK_DOWN / TURN 编排（约 100 行）

**现状**：仅当 `use_script_cam_tilt=True`（即没有 client camera pose）时生效。在 HTTP/OmniGibson 模式下，client 提供 `camera_pose_world`，此分支永远跳过。

**建议**：如果确认只走 HTTP 路径，可删除整个脚本化相机控制。

#### B. 地图帧变换 (Map Frame Transform) 复杂性

**位置**：`get_traversible()` 中约 400 行

**现状**：
- `_map_frame_fixed_mode` 支持 `id`, `flipud`, `fliplr`, `rot90`, `rot180`, `rot270`
- `_start_plan_flipud_mirror_enable` — 可选镜像启发法
- `_auto_select_map_frame_transform` — 从未调用的自动选择
- 实际使用中永远是 `flipud`

**建议**：硬编码 `flipud`，删除其他帧模式和镜像逻辑。

#### C. 诊断模式 (Diagnostic Toggles)

**位置**：`__init__` 中约 50 行，`_sync_camera_extrinsics` / `get_traversible` 中散布

**参数**：
- `diag_freeze_camera_extrinsics` — 冻结相机外参（调试用）
- `diag_disable_pose_bridge` — 禁用位姿桥（调试用）
- `diag_sem_threshold_profile` — `current` vs `legacy_strict`
- `diag_map_warp_mode` — `nearest` vs `bilinear`

**现状**：全部默认关闭，仅用于 A/B 测试特定建图问题。

**建议**：可以删除 `legacy_strict` profile 和 `bilinear` warp 模式。保留 `diag_freeze_camera_extrinsics` 和 `diag_disable_pose_bridge` 作为调试开关但简化实现。

#### D. OmniGibson Occupancy 融合

**位置**：`_fuse_og_occupancy_into_collision()` 约 50 行

**参数**：`og_occ_*` 系列（4 个参数）

**现状**：当 observations 中包含 `og_occupancy` 时触发，将 OmniGibson ScanSensor 格点占用地图融合进 collision_map。

**建议**：如果不再使用 OG ScanSensor 占用图，可删除。

#### E. 目标同义词重映射

**位置**：`reset()` 中的 `gym_equipment` → `exercise equipment` 等映射

**现状**：Habitat MP3D ObjectNav 的 21 类别中部分名称与常用词不同。如果目标不再限于 MP3D 类别，这些映射无意义。

#### F. Co-occurrence 先验矩阵

**位置**：`__init__` 加载 `tools/obj.npy` / `tools/room.npy`

**现状**：用于 `fbe()` 中 frontier scoring 的 PSL 先验 + `scenegraph.score()`。如果场景图自身已有足够的 scoring 逻辑，这些矩阵可能是冗余的。

---

## 三、当前所有可配置参数（按功能分类）

### 3.1 传感器与地图基础

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `DEPTH_SENSOR.WIDTH` / `HEIGHT` | 720 | 深度/RGB 传感器分辨率 | **保留** |
| `DEPTH_SENSOR.HFOV` | 99 | 水平视场角（度） | **保留** |
| `DEPTH_SENSOR.MIN_DEPTH` | 0.1 | 深度最小值（米） | **保留** |
| `DEPTH_SENSOR.MAX_DEPTH` | 8 | 深度最大值（米） | **保留** |
| `AGENT_0.HEIGHT` | 0.88 | Agent 高度（米），camera fallback 用 | **保留** |
| `AGENT_0.RADIUS` | 0.18 | Agent 半径 | **保留** |
| `SIMULATOR.TURN_ANGLE` | 20 | 转弯角度（度） | **保留** |
| `SIMULATOR.FORWARD_STEP_SIZE` | 0.25 | 前进步长（米） | **保留** |
| `map_size_cm` | 6000 | 全局地图尺寸（厘米），范围 1000-12000 | **保留** |

### 3.2 目标检测 (GLIP)

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `detect_interval` | 4 | 每 N 步运行一次物体检测 | **保留** |
| `glip_append_goal_sg` | true | 将目标关键词追加到 GLIP 检测词表 | **保留** |
| `glip_extra_captions` | "" | 额外的 GLIP 检测短语（逗号分隔） | **保留** |
| `goal_distance_threshold` | 5.0 | 目标确认的距离门限（米） | **保留** |
| **GLIP confidence** | 0.61 (硬编码) | GLIP 置信度阈值 | 考虑提取为参数 |
| **GLIP min_image_size** | 800 (硬编码) | GLIP 最小图像尺寸 | 考虑提取为参数 |

### 3.3 场景图 (Scene Graph)

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `scenegraph_update_interval` | 1 | 场景图更新频率（每 N 步） | **保留** |
| `show_frame_nodes_only` | true | 可视化中只显示当前帧节点 | **保留** |
| `save_scenegraph_json` | true | 保存场景图 JSON 快照 | **保留** |
| `scenegraph.skip_edge_llm` | false | 跳过 LLM 推理边关系 | **保留** |
| `scenegraph.edge_max_per_step` | 24 | 每步最大边推理数 | **保留** |
| `scenegraph.edge_llm_batch_size` | 8 | LLM 批量大小 | **保留** |
| `scenegraph.edge_llm_timeout_s` | 20.0 | LLM 超时（秒） | **保留** |

### 3.4 导航规划器 (FMM Planner)

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `planner_align_angle_deg` | 26.0 | 前进时允许的最大航向偏差（度），超出则转向 | **保留** |
| `planner_align_hysteresis_deg` | 10.0 | 转向→前进的回差（避免左右抖动） | **保留** |
| `planner_face_goal_max_error_deg` | -1.0 | 面朝目标后才前进；-1=禁用（用宽锥代替） | **保留** |
| `planner_use_observed_turn` | true | 从罗盘增量估计实际转弯角度 | **保留** |
| `planner_observed_turn_ema_alpha` | 0.35 | 转弯估计 EMA 平滑系数 | 可硬编码 |
| `planner_min_effective_turn_deg` | 1.0 | 最小有效转弯度数 | 可硬编码 |
| `planner_forward_error_turn_ratio` | 0.5 | 前进误差中转弯分量占比 | 可硬编码 |
| `panorama_spin_until_step` | 22 | 初始全景扫描持续步数 | **保留** |
| `fmm_step_size` | 5 | FMM 短期目标窗口大小（格子） | **保留** |
| `fmm_stop_cond` | 0.5 | FMM 子目标"足够近"阈值 | **保留** |
| `goal_stop_distance_threshold_m` | =fmm_stop_cond | 确认目标后的停止距离（米） | **保留** |
| `goal_stop_face_threshold_deg` | 15.0 | 停止时面朝目标的角度容差 | **保留** |
| `possible_goal_stop_escape_steps` | 6 | "可能目标"处连续 STOP 后的逃逸步数 | 可硬编码 |
| `possible_goal_stop_escape_cooldown_steps` | 12 | 逃逸后冷却步数 | 可硬编码 |

### 3.5 碰撞与前向保护

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `collision_threshold_m` | 0.08 | GPS 增量低于此值判定为碰撞 | **保留** |
| `forward_guard_enable` | true | 前进前锥形安全检查 | **保留** |
| `forward_guard_lookahead_m` | 0.50 | 前视距离 | 可硬编码 |
| `forward_guard_samples` | 6 | 锥形采样点数 | 可硬编码 |
| `forward_guard_cone_half_angle_deg` | 25.0 | 锥形半角 | 可硬编码 |

### 3.6 Traversible（可通行区域）构建

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `traversible_obstacle_inflation_m` | -1.0 | 障碍物膨胀半径（米）；<0 时用 disk_radius | **保留** |
| `traversible_obstacle_disk_radius` | 2 | 障碍物膨胀格子数（0-16） | **保留** |
| `traversible_occ_from_depth_min` | 0.35 | 深度→障碍判定阈值 | **保留** |
| `traversible_clear_border_obs_cells` | 2 | 清除地图边界伪障碍的格子数 | 可硬编码 |
| `traversible_require_explored` | false | 仅已探索区域可通行（vs 未知=可通行） | **保留** |
| `traversible_free_map_min` | 0.45 | free map 概率下限 | **保留** |
| `traversible_clear_robot_obs_enable` | true | 清除机器人周围的障碍标记 | 可硬编码 |
| `traversible_clear_robot_obs_radius_m` | 0.30 | 清除半径 | 可硬编码 |
| `traversible_bootstrap_enable` | true | 启动时如果 free 太少，注入局部可通行区域 | **保留** |
| `traversible_bootstrap_min_free_ratio` | 0.002 | 触发 bootstrap 的最小 free 比例 | 可硬编码 |
| `traversible_bootstrap_radius_m` | 0.8 | bootstrap 半径 | 可硬编码 |
| `random_goal_min_distance_m` | 0.60 | 随机目标最小距离 | 可硬编码 |
| `random_goal_obstacle_clearance_m` | 0.35 | 随机目标障碍物间距 | 可硬编码 |

### 3.7 地图帧变换

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `map_frame_fixed_mode` | "flipud" | 地图帧变换模式 | **硬编码为 flipud** |
| `map_frame_auto_select` | false | 自动选择帧模式（从未调用） | **删除** |
| `map_frame_auto_select_win_cells` | 40 | 自动选择窗口（从未调用） | **删除** |
| `start_plan_flipud_mirror_enable` | false | flipud 镜像启发法 | **删除** |
| `start_plan_flipud_mirror_min_delta` | 0.05 | 镜像触发增量 | **删除** |
| `start_plan_flipud_pose_known_min` | 0.35 | 镜像触发的最小已知度 | **删除** |

### 3.8 深度→地图投影 (Mapping Fusion)

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `sem_map_pred_threshold` | 1.0 | 语义地图体素归一化阈值（越低越敏感） | **保留** |
| `sem_exp_pred_threshold` | 1.0 | 语义探索阈值 | **保留** |
| `free_map_pred_threshold` | 0.5 | free map 预测阈值 | **保留** |
| `free_exp_pred_threshold` | 0.5 | free map 探索阈值 | **保留** |
| `map_obstacle_min_z_cm` | 0.0 | 障碍物检测最低高度（cm） | **保留** |
| `map_obstacle_max_z_cm` | -1.0 | 障碍物检测最高高度（<0=相机高度+offset） | 可硬编码逻辑 |
| `map_obstacle_above_camera_cm` | 50.0 | 相机上方多少cm仍算障碍 | 可硬编码 |
| `map_free_min_z_cm` | -150.0 | free space 最低高度 | 可硬编码 |
| `map_free_max_z_cm` | 25.0 | free space 最高高度 | 可硬编码 |

### 3.9 相机外参 (Camera Extrinsics) — HTTP/OG 专用

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `camera_extrinsic_use_client_pose` | true | 使用 HTTP 客户端的 camera_pose_world | **保留** |
| `camera_height_from_world_z` | true | 从世界坐标 Z 推导相机高度 | **保留** |
| `camera_height_fallback_cm` | -1.0 | 无客户端位姿时的固定高度 | **保留** |
| `camera_height_z_offset_m` | 0.0 | 高度 Z 偏移 | 可硬编码 |
| `camera_tilt_source` | "pitch" | 倾斜来源：pitch/roll/yaw/cam_z_down/cam_y_down | **保留** |
| `camera_tilt_fallback_to_cam_z_down_on_outlier` | true | roll 异常时回退到向量法 | 可硬编码 |
| `camera_pitch_mapping_scale` | 1.0 | pitch 缩放因子 | 可硬编码 |
| `camera_pitch_mapping_offset_deg` | 0.0 | pitch 偏移 | 可硬编码 |
| `camera_pitch_min_deg` / `max_deg` | -45 / 45 | pitch 钳制范围 | **保留** |
| `camera_height_min_cm` / `max_cm` | 80 / 220 | 高度钳制范围 | **保留** |
| `camera_extrinsic_reject_outlier` | true | 拒绝明显异常的外参 | **保留** |
| `camera_height_reject_min_cm` / `max_cm` | 110 / 210 | 拒绝带高度范围 | **保留** |
| `camera_pitch_reject_abs_deg` | 55.0 | 拒绝带 pitch 绝对值 | 可硬编码 |
| `camera_extrinsic_reject_motion_delta` | false | 帧间跳变拒绝 | **保留** |
| `camera_pitch_reject_delta_deg` | 6.0 | 帧间 pitch 跳变阈值 | **保留** |
| `camera_height_reject_delta_cm` | 4.0 | 帧间高度跳变阈值 | **保留** |

### 3.10 地图暂停 (Map Suspend on Fall)

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `map_suspend_on_fall` | true | 跌倒时暂停深度→地图融合 | **保留** |
| `map_suspend_steps_on_fall` | 4 | 暂停步数 | **保留** |
| `map_suspend_roll_deg` | 70.0 | 触发暂停的 roll 阈值 | 可硬编码 |
| `map_suspend_pitch_deg` | 50.0 | 触发暂停的 pitch 阈值 | 可硬编码 |
| `map_suspend_height_cm` | 95.0 | 触发暂停的高度阈值 | 可硬编码 |
| `map_suspend_height_delta_cm` | 20.0 | 触发暂停的高度跳变 | 可硬编码 |

### 3.11 GPS / 罗盘桥接

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `compass_offset_deg` | 0.0 | 罗盘偏移（地图"前方"vs 机器人前方） | **保留** |
| `gps_negate_y` | false | GPS Y 轴翻转 | **保留** |
| `gt_pathplan_when_target_on_map` | false | 当 target_world_xy 在已探索地图上时直接 FMM 规划 | **保留** |

### 3.12 OmniGibson Occupancy 融合

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `og_occ_reseed_collision_from_depth` | true | 融合 OG 占用前重新从深度生成 collision | 可删除整组 |
| `og_occ_obstacle_max` | 0.12 | OG 格点障碍物阈值 | 可删除整组 |
| `og_occ_free_min` | 0.85 | OG 格点 free 阈值 | 可删除整组 |
| `og_occ_fuse_obstacles` | false | 是否融合障碍通道 | 可删除整组 |

### 3.13 Behavior Bridge（HTTP 客户端侧运动参数）

这些参数仅被 `eval_skill_sgnav_http.py` 客户端读取，SG_Nav 本身不使用：

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `behavior_forward_vel` | 0.5 | velocity 模式前进速度 | 移到客户端配置 |
| `behavior_turn_vel` | 1.75 | velocity 模式转弯速度 | 移到客户端配置 |
| `behavior_velocity_forward_vel` | 0.5 | 同上（显式 velocity 分支） | 移到客户端配置 |
| `behavior_velocity_turn_vel` | 1.75 | 同上 | 移到客户端配置 |
| `behavior_physics_steps_per_nav_step` | 2 | 每导航步的物理步数 | 移到客户端配置 |
| `behavior_position_*` 系列 | 各异 | position 分支的步长/角度/物理步数 | 移到客户端配置 |
| `behavior_look_pitch_sign` | -1 | pitch 方向符号 | 移到客户端配置 |
| `behavior_physical_look_pitch_limit_margin_deg` | 10 | pitch 关节余量 | 移到客户端配置 |

### 3.14 日志与调试开关

当前有 **25+ 个日志开关**，严重增加代码复杂度：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `log_goal_votes` | true | 打印 GLIP goal-map 投票总数 |
| `nav_debug_trace` | false | 每步导航调试行 |
| `log_pose_align_trace` | false | 规划器与罗盘对齐日志 |
| `log_goal_cell_drift` | false | 目标格子漂移日志 |
| `log_spin_diagnosis` | true | 转圈/停滞诊断 |
| `log_nav_stage` | true | 高层导航阶段 |
| `log_map_frame_select` | true | 帧变换选择日志 |
| `log_free_map_distribution` | true | free map 直方图 |
| `log_obstacle_map_distribution` | true | obstacle map 直方图 |
| `log_mapping_height_band` | true | 高度带日志 |
| `log_camera_extrinsic_clamp` | true | 外参钳制日志 |
| `log_camera_pose_world` | false | 每步客户端位姿 |
| `log_coordinate_assumptions` | true | 坐标假设日志 |
| `log_mapping_stats` | true | 每步地图统计 |
| `log_full_map_delta` | true | full_map 增量变化 |
| `log_collision_paint` | true | 碰撞标记日志 |
| `log_runtime_checks` | true | 运行时健康检查 |
| `dump_motion_debug_json` | true | JSONL 运动/相机同步 |
| `check_traversible_free_warn` | 0.95 | traversible 警告阈值 |
| `check_depth_zero_warn` | 0.25 | 深度全零警告阈值 |
| `check_depth_median_*` | 0.08/8.0 | 深度中值范围 |
| `check_camera_pitch_abs_warn_deg` | 35.0 | 相机 pitch 警告 |
| `check_camera_roll_abs_warn_deg` | 45.0 | 相机 roll 警告 |
| `check_camera_height_*_warn_cm` | 70/210 | 相机高度警告 |
| `cuda_empty_cache_each_step` | false | 每步清空 CUDA 缓存 |

### 3.15 可视化

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `visualize_centered_crop_ratio` | 0.50 | 地图面板裁剪比例 | **保留** |
| `visualize_use_traversible_fallback` | false | 占用面板全空时用 traversible 代替 | **删除** |

### 3.16 诊断模式

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `diag_freeze_camera_extrinsics` | false | 冻结相机外参做 A/B 测试 | 可保留为高级调试 |
| `diag_disable_pose_bridge` | false | 禁用 GPS→地图位姿桥 | 可保留为高级调试 |
| `diag_sem_threshold_profile` | "current" | 语义阈值配置 (`current`/`legacy_strict`) | **删除 legacy_strict** |
| `diag_map_warp_mode` | "nearest" | 地图 warp 插值 (`nearest`/`bilinear`) | **删除 bilinear** |

---

## 四、方法行数分布（代码热点）

| 方法 | 行数 | 占比 |
|------|------|------|
| `__init__` | ~625 | 12.4% |
| `_sync_camera_extrinsics` | ~280 | 5.6% |
| `get_traversible` | ~400 | 8.0% |
| `act` | ~565 | 11.2% |
| `detect_objects` | ~200 | 4.0% |
| `visualize` | ~230 | 4.6% |
| `_plan` | ~285 | 5.7% |
| 日志/诊断方法合计 | ~500 | 10.0% |
| Debug JSONL dump 方法 | ~140 | 2.8% |
| 其他 | ~1800 | 35.7% |

**`__init__` 中 60% 以上是从 YAML 读取参数和转换为格子数。**

---

## 五、精简建议总结

### 必删（无争议）

1. `SG_Nav_0.py` — 旧版备份
2. `main()` + Habitat Challenge 入口
3. PSL 导入和 `add_predicates` / `add_rules`
4. `_auto_select_map_frame_transform` + 相关参数
5. `start_plan_flipud_mirror_*` 3 个参数
6. `map_frame_auto_select*` 2 个参数

### 建议删（需确认）

7. 脚本化相机控制 (scripted tilt) — 如果永远走 HTTP
8. OG occupancy 融合 (`og_occ_*`) — 如果不用 ScanSensor
9. Behavior bridge 参数 — 移到客户端配置
10. `legacy_strict` / `bilinear` 诊断配置
11. `visualize_use_traversible_fallback`
12. 大量日志开关 → 合并为单个 `log_level` / `verbose` 级别

### 建议硬编码（减少 YAML 参数）

13. 地图帧模式硬编码为 `flipud`
14. Forward guard 细节参数（4 个）
15. Random goal 参数（2 个）
16. Bootstrap 参数（2 个）
17. Map suspend 细节阈值（4 个）
18. Planner EMA / min turn / error ratio（3 个）
19. Possible goal escape 参数（2 个）

---

## 六、精简后预期参数清单

如果按照上述建议全部执行，YAML 参数将从 **120+** 减少到约 **40-50** 个核心参数。

请告诉我你具体希望精简哪些功能，我将据此开始代码修改。
