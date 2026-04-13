# SG-Nav 当前导航策略梳理

本文基于当前仓库里的实现整理，重点对应 [`SG_Nav.py`](../SG_Nav.py) 和 [`scenegraph.py`](../scenegraph.py)。

如果先给一句最短总结：

- `GLIP` 负责“看见什么目标/房间”。
- `SceneGraph` 负责“把观测组织成对象-关系-房间的结构化语义”。
- `VLM/LLM` 负责“补充关系推理和房间/群组语义判断”。
- `FMM` 负责“真正把目标点变成逐步动作”。

注意：代码里实际使用的是 **`FMM`**，不是 `FFM`。这里的 `FMM` 指 `Fast Marching Method` 路径规划器，对应 [`utils/utils_fmm/fmm_planner.py`](../utils/utils_fmm/fmm_planner.py) 和 [`SG_Nav.py`](../SG_Nav.py) 中的 `FMMPlanner`。

## 1. 先看整体架构

当前策略不是“纯 LLM 直接控制机器人”，而是一个明显的分层系统：

1. 底层建图层
   - 用深度图更新障碍图、自由空间图、房间图。
   - 关键模块在 [`SG_Nav.py`](../SG_Nav.py#L157) 到 [`SG_Nav.py`](../SG_Nav.py#L160)、[`SG_Nav.py`](../SG_Nav.py#L2142) 到 [`SG_Nav.py`](../SG_Nav.py#L2175)。

2. 感知层
   - `GLIP` 做开放词汇目标检测。
   - `GroundedSAM + SceneGraph` 做分割、3D 对象聚合、节点更新。
   - 关键位置：[`SG_Nav.py`](../SG_Nav.py#L128) 到 [`SG_Nav.py`](../SG_Nav.py#L141)、[`scenegraph.py`](../scenegraph.py#L963) 到 [`scenegraph.py`](../scenegraph.py#L980)。

3. 语义推理层
   - `SceneGraph` 内部调用 `VLM/LLM` 推关系、猜房间、算群组和目标相关性。
   - 关键位置：[`scenegraph.py`](../scenegraph.py#L744) 到 [`scenegraph.py`](../scenegraph.py#L960)。

4. 导航决策层
   - 先决定当前的 `goal_map` 是什么。
   - 再交给 `FMM` 算 short-term goal，再转成 `前进/左转/右转/停止`。
   - 关键位置：[`SG_Nav.py`](../SG_Nav.py#L1467) 到 [`SG_Nav.py`](../SG_Nav.py#L1747)、[`SG_Nav.py`](../SG_Nav.py#L2808) 到 [`SG_Nav.py`](../SG_Nav.py#L3127)。

也就是说，`LLM/VLM` 更像“高层语义顾问”，`FMM` 才是“真正执行局部导航”的控制器。

## 2. 当前导航主流程是什么样的

## 2.1 初始化阶段

初始化时，系统会建立以下几类核心模块：

- `GLIPDemo`：开放词汇检测器，用于目标和房间检测，见 [`SG_Nav.py`](../SG_Nav.py#L128) 到 [`SG_Nav.py`](../SG_Nav.py#L141)。
- `Semantic_Mapping / free_map_module / room_map_module`：分别维护障碍图、自由空间图、房间图，见 [`SG_Nav.py`](../SG_Nav.py#L157) 到 [`SG_Nav.py`](../SG_Nav.py#L167)。
- `SceneGraph`：在线场景图模块，见 [`SG_Nav.py`](../SG_Nav.py#L188)。
- 共现先验：
  - `obj.npy`：目标与其他物体共现先验。
  - `room.npy`：目标与房间共现先验。
  - 见 [`SG_Nav.py`](../SG_Nav.py#L178) 到 [`SG_Nav.py`](../SG_Nav.py#L186)。
- 运行时参数：
  - `detect_interval`
  - `scenegraph_update_interval`
  - `panorama_spin_until_step`
  - `planner_align_angle_deg`
  - `fmm_step_size`
  - `goal_stop_distance_threshold_m`
  - 见 [`SG_Nav.py`](../SG_Nav.py#L202) 到 [`SG_Nav.py`](../SG_Nav.py#L259)。

## 2.2 每一步 `act()` 的主流程

每个时间步，`act()` 大致按下面顺序执行：

1. 预处理观测
   - 清理非法深度。
   - 对齐 RGB 和深度尺寸。
   - 修正 `gps/compass`。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1477) 到 [`SG_Nav.py`](../SG_Nav.py#L1508)。

2. 更新场景图输入
   - 把 agent、目标、房间图、free map、当前观测、全局地图和位姿都塞给 `scenegraph`。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1513) 到 [`SG_Nav.py`](../SG_Nav.py#L1520)。

3. 按间隔更新 SceneGraph
   - `run_scenegraph_update = (total_steps % scenegraph_update_interval) == 0`
   - 真正调用的是 `scenegraph.update_scenegraph()`。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1521) 到 [`SG_Nav.py`](../SG_Nav.py#L1528)。

4. 更新地图
   - `update_map()` 更新障碍图。
   - `update_free_map()` 更新自由空间图。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1531) 到 [`SG_Nav.py`](../SG_Nav.py#L1535)。

5. 用 `GLIP` 做目标检测
   - `detect_objects()` 会决定当前是否：
     - `found_goal`
     - `found_possible_goal`
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1541) 到 [`SG_Nav.py`](../SG_Nav.py#L1548)、[`SG_Nav.py`](../SG_Nav.py#L1256) 到 [`SG_Nav.py`](../SG_Nav.py#L1465)。

6. 初始全景扫描
   - 前 16 步左右会执行一段固定的抬头/低头/转向扫描。
   - 如果在 `panorama_spin_until_step` 之前还没有发现目标线索，会继续转圈。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1550) 到 [`SG_Nav.py`](../SG_Nav.py#L1593)。

7. 构造可通行区域
   - `get_traversible()` 用障碍图、碰撞图、free map 生成局部可通行栅格。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1610) 到 [`SG_Nav.py`](../SG_Nav.py#L1618)、[`SG_Nav.py`](../SG_Nav.py#L2278) 到 [`SG_Nav.py`](../SG_Nav.py#L2681)。

8. 决定当前 long-term goal 来自哪里
   - 如果 `found_goal`：目标已经比较确定，直接用 `goal_gps`。
   - 否则如果 `found_possible_goal`：用远距离/弱确认的 `possible_goal_temp_gps`。
   - 否则优先做 frontier-based exploration（`fbe()`）。
   - frontier 都不行时，退化成 random goal。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1620) 到 [`SG_Nav.py`](../SG_Nav.py#L1651)。

9. 交给 `FMM` 做局部规划
   - `_plan()` 内部先 `_get_stg()`，再把 STG 转成动作。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1653) 到 [`SG_Nav.py`](../SG_Nav.py#L1659)、[`SG_Nav.py`](../SG_Nav.py#L2808) 到 [`SG_Nav.py`](../SG_Nav.py#L3127)。

10. 若卡住或局部停止，重新选 frontier 或随机点
   - 这是当前实现里一个很重要的保底机制。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1663) 到 [`SG_Nav.py`](../SG_Nav.py#L1747)。

## 2.3 当前策略的核心状态机

从代码逻辑看，当前导航至少有下面几种状态：

- `found_goal = True`
  - 说明目标已经被较强地确认。
  - 系统会直接把 `goal_map` 设到目标所在位置。

- `found_possible_goal = True`
  - 说明看到了疑似目标，但还不够近或者投票还不够稳。
  - 系统会朝这个“可能目标”先靠近。

- `frontier exploration`
  - 没有目标线索时，执行 FBE，去探索前沿区域。

- `random fallback`
  - frontier 失败或者卡住时，随机采样一个点脱困。

这也是当前策略最重要的特征：**语义目标驱动和 frontier exploration 是并行共存的，不是二选一。**

## 3. GLIP 在这套系统里的作用

虽然你重点问的是 `VLM/LLM`，但从代码上讲，真正承担“第一层目标发现”的其实是 `GLIP`，这点很重要。

`GLIP` 做两件事：

1. 检测目标物体
   - 在 `detect_objects()` 中，系统调用：
   - [`SG_Nav.py`](../SG_Nav.py#L1256) 到 [`SG_Nav.py`](../SG_Nav.py#L1262)
   - 然后根据 bbox 中心 + 深度估计目标的方向和距离。

2. 检测房间类别
   - 用 `rooms_captions` 做房间检测，再更新 `room_map`。
   - 见 [`SG_Nav.py`](../SG_Nav.py#L1544) 到 [`SG_Nav.py`](../SG_Nav.py#L1548)、[`SG_Nav.py`](../SG_Nav.py#L2165) 到 [`SG_Nav.py`](../SG_Nav.py#L2175)。

所以可以把它理解成：

- `GLIP` 是“目标/房间候选的入口”
- `SceneGraph + VLM/LLM` 是“高层语义组织与推理”
- `FMM` 是“执行控制”

## 4. VLM 的作用是什么

当前代码里，`VLM` 主要用于 **视觉关系判断**，而不是直接控制机器人。

## 4.1 VLM 的调用位置

`SceneGraph.update_edge()` 里先尝试用视觉模型为新边赋关系：

- 若两个节点在同一帧里都出现过，先取一张 joint image。
- 然后问：
  - `What is the spatial relationship between the A and the B in the image?`
- 见 [`scenegraph.py`](../scenegraph.py#L793) 到 [`scenegraph.py`](../scenegraph.py#L814)。

之后在 `discriminate_relation()` 里，VLM 还会做一次关系验证：

- 问的是：
  - `A 和 B 是否满足 relation？`
- 如果回答里有 `yes`，边才保留。
- 见 [`scenegraph.py`](../scenegraph.py#L1097) 到 [`scenegraph.py`](../scenegraph.py#L1121)。

## 4.2 VLM 负责什么，不负责什么

VLM 负责：

- 给对象对补“空间关系”
  - 例如 `next to`、`on`、`under`。
- 对候选关系做视觉确认。

VLM 不负责：

- 不直接做主目标检测。
  - 主目标检测还是 `GLIP`。
- 不直接选动作。
  - 动作输出还是 `FMM` + 局部规则。
- 不直接做 frontier 路径规划。

## 4.3 当前实现里的一个细节

代码里当前默认：

- `self.llm_name = 'llama3.2-vision'`
- `self.vlm_name = 'llama3.2-vision'`
- 见 [`scenegraph.py`](../scenegraph.py#L165) 到 [`scenegraph.py`](../scenegraph.py#L166)。

也就是说，**当前默认配置下，LLM 和 VLM 用的是同一个 Ollama 模型名**，但它们的“角色”仍然是分开的：

- `VLM` 调用时会带图像，见 [`scenegraph.py`](../scenegraph.py#L1019) 到 [`scenegraph.py`](../scenegraph.py#L1031)。
- `LLM` 调用时只给文本 prompt，见 [`scenegraph.py`](../scenegraph.py#L986) 到 [`scenegraph.py`](../scenegraph.py#L1017)。

所以“同一个模型实例名”和“系统里承担的功能角色”是两回事。

## 5. LLM 的作用是什么

当前代码里，`LLM` 的主要职责是 **文本层的语义补全与高层推断**。

## 5.1 LLM 做边关系 proposal

如果某些新边没有被 VLM 直接定下来，系统会批量把对象对发给 LLM，让它猜“最可能的单一空间关系”：

- 见 [`scenegraph.py`](../scenegraph.py#L826) 到 [`scenegraph.py`](../scenegraph.py#L869)。

这个阶段更像：

- 用 LLM 补足“可能关系”
- 再让判别逻辑去删掉不合理的关系

所以它不是最终裁决器，而是“候选生成器”。

## 5.2 LLM 做房间预测

在 `insert_goal()` 里，LLM 会回答：

- “目标最可能在哪个房间里？”
- 见 [`scenegraph.py`](../scenegraph.py#L913) 到 [`scenegraph.py`](../scenegraph.py#L960)。

这一步的作用不是直接把机器人送到终点，而是：

- 先从已有 scene graph 中挑一个最可能相关的房间
- 再在这个房间里选一个 group node 作为中期语义目标

## 5.3 LLM 做 graph correlation 评分

`graph_corr()` 会做多轮问答，估计：

- “某个对象群组和目标是否容易共现”
- 见 [`scenegraph.py`](../scenegraph.py#L1130) 到 [`scenegraph.py`](../scenegraph.py#L1144)。

这部分输出会影响：

- 哪个 room/group 更值得探索
- 最终 `scenegraph.score()` 给 frontier 的语义加分

## 5.4 LLM 不直接输出导航动作

这一点非常关键：

- LLM 不直接输出 `left/right/forward/stop`
- LLM 也不直接提供一条完整几何路径
- 它只是改变“哪里更值得去”

最后动作仍然由 `goal_map -> FMM STG -> 局部规则` 产生。

## 6. 场景图的作用是什么

如果只看名字，很容易把 `SceneGraph` 理解成“只是做个可视化图”。但在当前实现里，它其实承担了三层作用。

## 6.1 作用一：把连续观测变成结构化对象记忆

`update_scenegraph()` 会做：

1. `segment2d()`
   - GroundedSAM 分割出候选物体，见 [`scenegraph.py`](../scenegraph.py#L602) 到 [`scenegraph.py`](../scenegraph.py#L628)。
2. `mapping3d()`
   - 把 2D mask + depth 投到 3D，并跨帧合并成对象，见 [`scenegraph.py`](../scenegraph.py#L630) 到 [`scenegraph.py`](../scenegraph.py#L672)。
3. `get_caption()`
   - 给对象确定 caption，见 [`scenegraph.py`](../scenegraph.py#L673) 到 [`scenegraph.py`](../scenegraph.py#L680)。
4. `update_node()`
   - 更新对象节点、中心、所属房间，见 [`scenegraph.py`](../scenegraph.py#L681) 到 [`scenegraph.py`](../scenegraph.py#L742)。
5. `update_edge()`
   - 更新对象间关系，见 [`scenegraph.py`](../scenegraph.py#L744) 到 [`scenegraph.py`](../scenegraph.py#L894)。

这意味着 scene graph 不是“瞬时识别结果”，而是一个 **跨时间累积的结构化记忆**。

## 6.2 作用二：为小目标提供额外定位能力

对于 `small_objects`，当前实现不会只依赖 GLIP bbox，而是会结合 scene graph 的分割和节点信息：

- 见 [`SG_Nav.py`](../SG_Nav.py#L1303) 到 [`SG_Nav.py`](../SG_Nav.py#L1360)。

这里的意义是：

- 小目标 bbox 容易不稳定
- 分割 mask + 场景图节点能提供更细粒度的目标线索

所以 scene graph 不只是做探索语义，它还直接参与了目标确认。

## 6.3 作用三：指导 frontier 选择

`fbe()` 在选 frontier 时，不是只看距离，还会调用：

- `self.scenegraph.score(frontier_locations_16_raw_b, num_16_frontiers)`
- 见 [`SG_Nav.py`](../SG_Nav.py#L1916)。

而 `scenegraph.score()` 又综合了三部分：

1. 房间先验
   - 当前 frontier 周围像不像目标常见房间，见 [`scenegraph.py`](../scenegraph.py#L1062) 到 [`scenegraph.py`](../scenegraph.py#L1069)。

2. 物体共现先验
   - 当前 frontier 附近是否靠近与目标常共现的物体，见 [`scenegraph.py`](../scenegraph.py#L1070) 到 [`scenegraph.py`](../scenegraph.py#L1085)。

3. 基于 LLM 的中期语义目标
   - `insert_goal()` 会预测一个 `mid_term_goal`，离这个语义目标近的 frontier 会额外加分，见 [`scenegraph.py`](../scenegraph.py#L1087) 到 [`scenegraph.py`](../scenegraph.py#L1094)。

所以场景图的真正价值是：

- 它把“盲目探索”变成了“带语义偏好的探索”。

## 7. FMM 什么时候用

用户提到的 “FFM” 在当前代码里应理解为 `FMM`。它的使用频率非常高，几乎是每一步都在用。

## 7.1 FMM 的核心职责

`FMM` 的职责非常明确：

- 输入：`traversible + goal_map + start`
- 输出：`short-term goal (STG)` 和 stop/replan 信号

关键函数：

- `_get_stg()`：调用 `FMMPlanner.set_multi_goal()` 和 `get_short_term_goal()`，见 [`SG_Nav.py`](../SG_Nav.py#L3088) 到 [`SG_Nav.py`](../SG_Nav.py#L3127)。
- `_plan()`：把 STG 转成动作，见 [`SG_Nav.py`](../SG_Nav.py#L2808) 到 [`SG_Nav.py`](../SG_Nav.py#L3086)。

## 7.2 FMM 在哪些场景下会被调用

### 场景 A：已确认目标 `found_goal`

这时 `goal_map` 就是目标位置，FMM 负责：

- 走到目标附近
- 在靠近目标时做 stop/朝向修正

对应逻辑：

- 目标 goal_map 构造：[`SG_Nav.py`](../SG_Nav.py#L1621) 到 [`SG_Nav.py`](../SG_Nav.py#L1626)
- 近目标停止策略：[`SG_Nav.py`](../SG_Nav.py#L3049) 到 [`SG_Nav.py`](../SG_Nav.py#L3084)

### 场景 B：疑似目标 `found_possible_goal`

这时目标还没完全确认，但系统会先朝可能目标靠近，FMM 负责把这个“临时目标点”变成局部动作。

- 见 [`SG_Nav.py`](../SG_Nav.py#L1629) 到 [`SG_Nav.py`](../SG_Nav.py#L1634)。

### 场景 C：frontier exploration

没有目标线索时，`fbe()` 先选一个 frontier，然后 FMM 去执行。

- frontier 选择：[`SG_Nav.py`](../SG_Nav.py#L1863) 到 [`SG_Nav.py`](../SG_Nav.py#L1926)
- frontier goal_map 执行：[`SG_Nav.py`](../SG_Nav.py#L1635) 到 [`SG_Nav.py`](../SG_Nav.py#L1651)

### 场景 D：随机脱困

当 frontier 不可用或者机器人卡住时，会生成 random goal，然后仍然交给 FMM 执行。

- 见 [`SG_Nav.py`](../SG_Nav.py#L1641) 到 [`SG_Nav.py`](../SG_Nav.py#L1645)、[`SG_Nav.py`](../SG_Nav.py#L1709) 到 [`SG_Nav.py`](../SG_Nav.py#L1747)。

### 场景 E：frontier 候选距离评估

注意，FMM 不只是执行器，在 `fbe()` 里还被用来计算每个 frontier 的距离代价：

- 先从 agent 当前位置出发计算 `fmm_dist`
- 再把 frontier 远近折算成分数
- 见 [`SG_Nav.py`](../SG_Nav.py#L1884) 到 [`SG_Nav.py`](../SG_Nav.py#L1892)

也就是说，FMM 在当前实现里同时承担：

- 候选点评分时的距离评估器
- 实际执行时的局部路径规划器

## 7.3 FMM 之后怎么变成动作

`_plan()` 的动作规则是纯几何的，不是语言模型控制：

- 先算 agent 朝向和 STG 方向的夹角。
- 夹角大就转向。
- 对齐后才前进。
- 如果 forward guard 发现前方碰撞风险，就把前进改成转向。

对应位置：

- 对齐逻辑：[`SG_Nav.py`](../SG_Nav.py#L2886) 到 [`SG_Nav.py`](../SG_Nav.py#L2987)
- 前进安全保护：[`SG_Nav.py`](../SG_Nav.py#L2989) 到 [`SG_Nav.py`](../SG_Nav.py#L3020)
- 历史碰撞补偿：[`SG_Nav.py`](../SG_Nav.py#L2809) 到 [`SG_Nav.py`](../SG_Nav.py#L2864)

所以 FMM 不是“直接吐动作编号”，而是：

- 先给出 STG
- 再由一个 hand-crafted local policy 把 STG 转成离散动作

## 8. VLM、LLM、场景图、FMM 四者关系

可以把当前系统理解成下面这条链路：

`观测(rgb/depth/gps/compass)`
-> `GLIP / GroundedSAM / Mapping`
-> `SceneGraph(节点、边、房间、群组)`
-> `VLM/LLM 补关系和高层语义判断`
-> `生成更有语义的 frontier / mid-term goal / goal hint`
-> `goal_map`
-> `FMM`
-> `离散动作`

其中最容易混淆的点有三个：

1. `VLM/LLM` 不直接做低层控制
   - 它们影响的是“去哪儿更合理”，不是“这一帧具体走左还是走右”。

2. 场景图不是可有可无的展示层
   - 它直接参与小目标识别和 frontier 评分。

3. FMM 不是探索策略本身
   - 它是执行器。
   - 真正决定 long-term goal 的，是目标检测状态、frontier 策略和场景图语义评分。

## 9. 当前默认配置体现了什么策略倾向

从 [`configs/sgnav_minimal.rgbd.yaml`](../configs/sgnav_minimal.rgbd.yaml) 和 [`configs/scenegraph_runtime.yaml`](../configs/scenegraph_runtime.yaml) 看，当前策略偏向：

- 每 `5` 步做一次目标检测/场景图更新
  - `detect_interval: 5`
  - `scenegraph_update_interval: 5`
- 初始会保留较长的 panorama 扫描
  - `panorama_spin_until_step: 22`
- 场景图的边推理默认开启
  - `skip_edge_llm: false`
- FMM 对目标时允许比较保守地停靠
  - `goal_stop_distance_threshold_m: 1.0`

这说明当前实现不是“每步都强依赖 LLM/VLM”，而是：

- 感知和场景图更新是间歇触发的
- FMM 和地图才是每步都会运行的主干

## 10. 最后给一个简明结论

如果只看“当前导航策略到底是什么”，可以概括为：

- 这是一个 **语义增强的 frontier/object navigation 混合策略**。
- 目标一旦被看到，就进入目标驱动导航。
- 目标没看到时，就做 frontier exploration。
- frontier 不是盲选，而是被场景图、房间先验、物体共现先验、LLM 推理过的中期目标共同影响。
- 最终所有 long-term goal 都会被转换成 `goal_map`，再统一交给 `FMM` 做局部导航。

如果只看四个模块的职责边界，可以记成一句：

- `VLM`：看图判断对象关系。
- `LLM`：做文本语义补全和房间/群组推断。
- `场景图`：把跨帧对象、关系、房间组织成结构化记忆，并给探索提供语义偏置。
- `FMM`：把当前目标点稳定地执行成局部动作。

