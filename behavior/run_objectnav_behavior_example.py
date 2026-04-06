#!/usr/bin/env python3
"""
Skeleton evaluation loop: SG_Nav_Agent + OmniGibson-style observations.

Run from the SG-Nav repository root so `utils` and GLIP resolve:

  cd SG-Nav
  python behavior/run_objectnav_behavior_example.py --dry_run

Full stack (GPU, GLIP, SAM, Ollama, etc.) is heavy; use --dry_run to verify imports
and observation shaping only.

With OmniGibson available, replace fetch_rgb_depth_and_robot with your env/robot
(as in openpi-comet/scripts/eval_skill_flat.py patterns).
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dry_run():
    import numpy as np

    from behavior.omnigibson_adapter import build_habitat_shaped_observations

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640, 1), 2.0, dtype=np.float32)
    obs = build_habitat_shaped_observations(
        rgb, depth, gps_xy=np.zeros(2), compass_rad=0.0
    )
    assert obs["depth"].shape[-1] == 1
    assert obs["gps"].shape == (2,)
    print("dry_run: observation dict ok")


def _main_real(args):
    sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)

    import numpy as np
    from omegaconf import OmegaConf

    from behavior.omnigibson_adapter import (
        apply_sgnav_discrete_action,
        build_habitat_shaped_observations,
        base_xy_yaw_from_robot,
    )
    from utils.behavior_taxonomy import behavior_goal_to_mp3d
    from SG_Nav import SG_Nav_Agent

    cfg = OmegaConf.load(
        os.path.join(REPO_ROOT, "configs/sgnav_minimal.rgbd.yaml")
    )
    nav_args = argparse.Namespace(visualize=False, split_l=-1, split_r=-1)
    agent = SG_Nav_Agent(cfg, nav_args)
    agent.simulator = None

    behavior_name = args.behavior_goal or "chair"
    mp3d_goal, sg_hint = behavior_goal_to_mp3d(behavior_name)
    agent.reset(object_category=mp3d_goal, object_category_sg=sg_hint)

    # --- User replaces this with OmniGibson env + robot ---
    def fetch_rgb_depth_and_robot():
        raise RuntimeError(
            "Wire OmniGibson: return (rgb HxWx3, depth HxW or HxWx1, robot)."
        )

    for _step in range(args.max_steps):
        rgb, depth, robot = fetch_rgb_depth_and_robot()
        gps_xy, yaw = base_xy_yaw_from_robot(robot)
        obs = build_habitat_shaped_observations(rgb, depth, gps_xy, yaw)
        out = agent.act(obs)
        apply_sgnav_discrete_action(robot, int(out["action"]))
        # env.step(...) if needed for physics


def main():
    sys.path.insert(0, REPO_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--behavior_goal", type=str, default="chair")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    if args.dry_run:
        _dry_run()
        return
    _main_real(args)


if __name__ == "__main__":
    main()
