"""
Map BEHAVIOR / OmniGibson object names to MP3D ObjectNav goal categories used by SG-Nav
(co-occurrence rows in tools/obj.npy are indexed by MP3D task category names).

See utils_glip.CANONICAL_MP3D_GOAL_ORDER for MP3D goal ids aligned with tools/obj.npy.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple

from utils.utils_glip import CANONICAL_MP3D_GOAL_ORDER

MP3D_GOAL_CATEGORIES = list(CANONICAL_MP3D_GOAL_ORDER)

# Explicit BDDL / IG-style names -> MP3D ObjectNav-v1 task category
BEHAVIOR_NAME_TO_MP3D: dict[str, str] = {
    # Seating / furniture
    "chair": "chair",
    "office_chair": "chair",
    "rocking_chair": "chair",
    "sofa": "sofa",
    "couch": "sofa",
    "loveseat": "sofa",
    "ottoman": "cushion",
    "cushion": "cushion",
    "stool": "stool",
    "seating": "seating",
    "bench": "seating",
    "table": "table",
    "coffee_table": "table",
    "dining_table": "table",
    "desk": "table",
    "nightstand": "cabinet",
    "side_table": "table",
    "cabinet": "cabinet",
    "shelf": "cabinet",
    "bookshelf": "cabinet",
    "chest_of_drawers": "chest_of_drawers",
    "dresser": "chest_of_drawers",
    "picture": "picture",
    "painting": "picture",
    "poster": "picture",
    "mirror": "picture",
    "tv": "tv_monitor",
    "television": "tv_monitor",
    "tv_monitor": "tv_monitor",
    "sink": "sink",
    "plant": "plant",
    "potted_plant": "plant",
    "bed": "bed",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "shower": "shower",
    "towel": "towel",
    "counter": "counter",
    "countertop": "counter",
    "fridge": "cabinet",
    "refrigerator": "cabinet",
    "oven": "cabinet",
    "microwave": "cabinet",
    "dishwasher": "cabinet",
    "fireplace": "fireplace",
    "gym_equipment": "gym_equipment",
    "treadmill": "gym_equipment",
    "exercise_machine": "gym_equipment",
    "clothes": "clothes",
    "laundry_basket": "clothes",
    "washer": "clothes",
    "dryer": "clothes",
    # Common BDDL underscores -> tokens
    "floor_lamp": "cabinet",
    "table_lamp": "cabinet",
    "lamp": "cabinet",
    # Keep radio as explicit goal token so planner/detector target "radio" directly.
    "radio": "radio",
    "radio_receiver": "radio",
    "telephone": "cabinet",
    "computer": "table",
}


def _normalize_key(name: str) -> str:
    s = unicodedata.normalize("NFKC", name.strip().lower())
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def behavior_goal_to_mp3d(behavior_object_name: str) -> Tuple[str, str]:
    """
    Returns (mp3d_object_category, scenegraph_prompt_hint).

    mp3d_object_category is one of the SG-Nav / MP3D goal names when possible,
    otherwise the closest bucket or 'chair' as last resort.

    scenegraph_prompt_hint is passed to the scene graph as obj_goal_sg when useful.
    """
    raw = behavior_object_name.strip()
    key = _normalize_key(raw)
    if key in BEHAVIOR_NAME_TO_MP3D:
        mp3d = BEHAVIOR_NAME_TO_MP3D[key]
        return mp3d, _default_sg_prompt(mp3d, raw)

    # Heuristic: longest matching key substring
    best = None
    best_len = 0
    for k, v in BEHAVIOR_NAME_TO_MP3D.items():
        if k in key and len(k) > best_len:
            best = v
            best_len = len(k)
    if best is not None:
        return best, _default_sg_prompt(best, raw)

    # Token overlap with MP3D category strings
    tokens = set(key.split("_")) - {"", "a", "the", "of"}
    for cat in MP3D_GOAL_CATEGORIES:
        ck = _normalize_key(cat)
        if ck in key or key in ck:
            return cat, _default_sg_prompt(cat, raw)
        if ck in tokens:
            return cat, _default_sg_prompt(cat, raw)

    # Default bucket
    return "chair", raw.replace("_", " ")


def _default_sg_prompt(mp3d: str, original: str) -> str:
    if mp3d == "gym_equipment":
        return "treadmill. fitness equipment."
    if mp3d == "chest_of_drawers":
        return "drawers"
    if mp3d == "tv_monitor":
        return "tv"
    if original.lower() != mp3d:
        return f"{original.replace('_', ' ')}. {mp3d}."
    return mp3d


def is_known_mp3d_goal(name: str) -> bool:
    return name in MP3D_GOAL_CATEGORIES
