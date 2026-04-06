from __future__ import annotations

import cv2
import numpy as np
import torch
import skimage

def line_list(text, line_length=80):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list

def add_text(image: np.ndarray, text: str, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_text_list(image: np.ndarray, text_list: list, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_rectangle(image: np.ndarray, top_left: tuple, bottom_right: tuple, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def add_resized_image(base_image: np.ndarray, overlay_image: np.ndarray, position: tuple, size: tuple):
    resized_overlay = cv2.resize(overlay_image, size)

    h, w = resized_overlay.shape[:2]

    x, y = position

    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError("Overlay image goes out of the bounds of the base image.")

    base_image[y:y+h, x:x+w] = resized_overlay
    return base_image

def compute_crop_rect(img_height: int, img_width: int, point: tuple, size: tuple) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) using the same rules as ``crop_around_point``."""
    crop_width, crop_height = size
    px, py = point
    left = max(px - crop_width // 2, 0)
    top = max(py - crop_height // 2, 0)
    right = min(px + (crop_width - crop_width // 2), img_width)
    bottom = min(py + (crop_height - crop_height // 2), img_height)
    if right - left < crop_width:
        if left == 0:
            right = left + crop_width
        else:
            left = right - crop_width
    if bottom - top < crop_height:
        if top == 0:
            bottom = top + crop_height
        else:
            top = bottom - crop_height
    return left, top, right, bottom


def crop_around_point(image: np.ndarray, point: tuple, size: tuple):
    img_height, img_width = image.shape[:2]
    left, top, right, bottom = compute_crop_rect(img_height, img_width, point, size)
    return image[top:bottom, left:right]


def draw_agent(agent, map, pose, agent_size, color_index, alpha=1):
    h, w = int(map.shape[1]), int(map.shape[2])
    px = float(pose[0].item() if hasattr(pose[0], "item") else pose[0])
    py = float(pose[1].item() if hasattr(pose[1], "item") else pose[1])
    cx = int(px * 100.0 / agent.resolution)
    cy = int((agent.map_size_cm / 100.0 - py) * 100.0 / agent.resolution)
    x0 = max(0, cx - agent_size)
    x1 = min(w, cx + agent_size)
    y0 = max(0, cy - agent_size)
    y1 = min(h, cy + agent_size)
    if x1 <= x0 or y1 <= y0:
        return
    color_ori = map[:, y0:y1, x0:x1]
    color_new = torch.zeros_like(color_ori)
    color_new[color_index] = 1
    color_new = alpha * color_new + (1 - alpha) * color_ori
    map[:, y0:y1, x0:x1] = color_new

def draw_goal(agent, map, goal_size, color_index):
    skimage.morphology.disk(goal_size)
    h, w = map.shape[1], map.shape[2]

    def _paint(center_y, center_x):
        y0 = max(0, int(center_y) - goal_size)
        y1 = min(h, int(center_y) + goal_size)
        x0 = max(0, int(center_x) - goal_size)
        x1 = min(w, int(center_x) + goal_size)
        if y0 >= y1 or x0 >= x1:
            return
        map[:, y0:y1, x0:x1] = 0
        map[color_index, y0:y1, x0:x1] = 1

    # Prefer planner goal map when available so visualization matches actual planning target.
    goal_map = getattr(agent, "goal_map", None)
    if goal_map is not None:
        ys, xs = np.where(goal_map == 1)
        if len(ys) > 0:
            _paint(ys[0], xs[0])
            return

    if not agent.found_goal and agent.goal_loc is not None:
        _paint(int(agent.map_size_cm / 5) - int(agent.goal_loc[0]), int(agent.goal_loc[1]))
        return

    gy = int((agent.map_size_cm / 200 + agent.goal_gps[1]) * 100 / agent.resolution)
    gx = int((agent.map_size_cm / 200 + agent.goal_gps[0]) * 100 / agent.resolution)
    _paint(gy, gx)
