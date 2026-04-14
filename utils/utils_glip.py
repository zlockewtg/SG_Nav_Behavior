import csv
import copy
import gzip
import json
import os
import re
from pathlib import Path

# GLIP prompt
categories = [] # categories except doors 
categories_40 = []
categories_map = {}
categories_doors = []
with open('tools/matterport_category_mappings.tsv') as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for i, line in enumerate(tsv_file):
        line_ = [item for item in line[0].split('   ') if not item =='']
        if i == 0 or len(line_) < 4:
            continue
        if int(line_[3]) > 10:
            if 'door' in line_[-1] and line_[2] not in categories_doors:
                categories_doors.append(line_[2])
            else:
                categories.append(line_[2])
                categories_map[line_[2]] = line_[-1]
        if line_[-1] not in categories_40 and line_[-1] is not 'objects' and 'void' not in line_[-1]:
            categories_40.append(line_[-1])


categories_21 = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa',
'bed', 'chest_of_drawers', 'plant', 'sink', 'toilet', 'stool',
'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes']
# categories_21 = ['chair', 'table', 'picture',  'sofa',
# 'bed',  'plant', 'sink', 'toilet',   'clothes','camera']

categories_21_origin = copy.deepcopy(categories_21)


# categories_21.append('heater')
categories_21.append('window')
categories_21.append('radio')
categories_21.append('vidalia onion')
categories_21.append('parer')
categories_21.append('bowl')
categories_21.append('microwave')
# categories_21.append('treadmill')
# categories_21.append('exercise machine')
object_captions = '. '.join(categories_21) +'.'# version 1
rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']


def _mp3d_category_word_set() -> set[str]:
    """Lowercase tokens to avoid duplicating MP3D phrases when appending goal-specific GLIP prompts."""
    s: set[str] = set()
    for c in categories_21:
        s.add(c.lower())
        s.add(c.lower().replace("_", " "))
        for part in c.lower().split("_"):
            if len(part) >= 3:
                s.add(part)
    return s


_MP3D_CATEGORY_WORDS = _mp3d_category_word_set()


def extract_tokens_for_glip_from_goal_sg(goal_sg: str, max_tokens: int = 8) -> list[str]:
    """
    Pull open-vocab style words from obj_goal_sg (e.g. radio, receiver) for GLIP caption extension.
    Skips tokens that are already MP3D category names (so we do not repeat cabinet, chair, ...).
    """
    if not goal_sg or not str(goal_sg).strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", str(goal_sg)):
        w = m.group(0)
        low = w.lower().rstrip("_")
        if low in seen:
            continue
        if low in _MP3D_CATEGORY_WORDS:
            continue
        seen.add(low)
        out.append(w)
        if len(out) >= max_tokens:
            break
    return out


def compose_glip_object_caption(base_caption: str, extra_phrases: list[str]) -> str:
    """
    GLIP grounding string: ``phrase1. phrase2. ...``.
    Appends @extra_phrases if not already present (case-insensitive).
    """
    if not extra_phrases:
        return base_caption
    seen: set[str] = set()
    chunks: list[str] = []
    stem = base_caption.strip()
    if stem.endswith("."):
        stem = stem[:-1]
    for part in stem.split("."):
        p = part.strip()
        if not p:
            continue
        seen.add(p.lower())
        chunks.append(p)
    for phrase in extra_phrases:
        q = str(phrase).strip()
        if not q:
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        chunks.append(q)
    return ". ".join(chunks) + "."


rooms_captions = '. '.join(rooms)+'.'
door_captions = 'doorway. hallway.'# v2
# object_captions = '. '.join(categories_21)+'.' # + '. wall. door.'

# pre_defined_captions = rooms + pre_defined_captions


# LLM reasoning prompt
# room_prompt = "In which room will you most likely to find a "

# Must match row order of ``tools/obj.npy`` (21,) and ``tools/room.npy`` (21, 9).
CANONICAL_MP3D_GOAL_ORDER = (
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
)


def _load_projection_from_val_json(path: Path):
    with gzip.open(path, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
    projection_reverse = data["category_to_task_category_id"]
    projection: dict = {}
    for key, item in projection_reverse.items():
        projection[item] = key
    return projection, projection_reverse


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VAL_JSON_PATH = _REPO_ROOT / "tools" / "val.json.gz"
_SKIP_VAL_JSON = os.environ.get("SG_NAV_SKIP_VAL_JSON", "").strip().lower() in ("1", "true", "yes")

if _SKIP_VAL_JSON or not _VAL_JSON_PATH.is_file():
    projection = {i: CANONICAL_MP3D_GOAL_ORDER[i] for i in range(len(CANONICAL_MP3D_GOAL_ORDER))}
    projection_reverse = {}
else:
    projection, projection_reverse = _load_projection_from_val_json(_VAL_JSON_PATH)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1], 'y2': bb2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
