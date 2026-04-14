#!/usr/bin/env python3
"""
Skill-level evaluation based on skills_flat.jsonl segments.

Uses segments from extract_skill_segments_from_flat.py. Supports:
- Pre-sampled instances (load tro_state) - default
- BDDL online sampling (--online_sampling) - randomly create scene from BDDL each run

Success criterion for "move to": (1) min distance < --distance_threshold (default 1m; min of left/right EEF to target; base fallback),
(2) final distance < initial (moved closer). Distance-based early stop only runs after --min_steps_before_reach_early_stop
loop iterations (default 800): before that, the episode never ends just because distance is already within range; after that,
if min distance is still within threshold the rollout can stop early, otherwise it continues until --max_steps_per_segment.
Disable distance early stop entirely with --no_early_stop_on_reach_distance.
Episode may also end early after many consecutive low base-speed steps (--consecutive_low_speed_steps / --base_speed_threshold).
Other skills: run for max_steps (success metric TBD).

Usage:
  1. Extract segments: python scripts/extract_skill_segments_from_flat.py --output skill_segments_flat.json
  2. Start serve_b1k
  3. Run eval: python scripts/eval_skill_flat.py --segments skill_segments_flat.json --policy_url localhost:8000

  Move-to only (recommended): only test "move to" skill, scene from BDDL random sampling:
    python scripts/eval_skill_flat.py ... --move_to_only --total_segments 10

  BDDL segments (no skill_segments_flat.json needed): generate move-to prompts from BDDL:
    python scripts/eval_skill_flat.py --policy_url localhost:8000 --move_to_only --bddl_segments --total_segments 10

  BDDL segments filtered by skills_flat.jsonl (match objects from jsonl, new scene per segment; --bddl_segments is implied by --skills_flat):
    python scripts/eval_skill_flat.py --policy_url localhost:8000 --move_to_only --skills_flat scripts/skills_flat.jsonl --task turning_on_radio --total_segments 10

  move_to_segments.json (RFT schema: task + instance_id + tro_state + annotation object_id; same scene pipeline as generate_move_to_rft_data.py):
    python scripts/eval_skill_flat.py --policy_url localhost:8000 --move_to_segments_json data_generation/rft/move_to_segments.json --task chop_an_onion --total_segments 10
  Same, but each segment picks a random instance_id that has a matching *-tro_state.json on disk (per-segment, can differ; JSON instance_id ignored):
    ... --random_instance [--seed 42]

  Segment order: by default, segments are not shuffled globally; all move-to segments for one task run in file order,
  and tasks follow the order of first appearance in the segment list. Use --shuffle_segments for random order (old behavior).

  Offline (pre-sampled tro_state, keep prompt from skills_flat): add --no_online_sampling
  With BDDL random sampling: python scripts/eval_skill_flat.py ... --online_sampling

  Match BehaviorLeRobotDataset prompts (imperative vs Skill:/Objects:): add --use_skill_prompt_format
  if training used use_skill_prompt_format=True; default matches dataset default (imperative, no Goal line).

  Prompt format: "move to <物体名>" (e.g. "move to radio").

  Logs (under <log_dir>/<exp_name>/log/): eval.log (timestamped lines via logging) and eval_segments.jsonl
  (one JSON object per tested segment: prompt, distance_final_m / distance_min_m / distance_initial_m, success, ...).
"""

import json
import logging
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
import pathlib
from pathlib import Path

import cv2
import numpy as np
import torch as th

warnings.filterwarnings("ignore", message="Casting input x to numpy array", category=UserWarning, module="gymnasium")

project_root = Path(__file__).resolve().parents[1]
# Prefer openpi-comet's OmniGibson (has VisualGeomPrim) over any other installed copy
og_path = project_root / "BEHAVIOR-1K" / "OmniGibson"
if og_path.exists():
    sys.path.insert(0, str(og_path))
sys.path.insert(0, str(project_root / "BEHAVIOR-1K"))
sys.path.insert(0, str(project_root / "BEHAVIOR-1K" / "joylo"))
sys.path.insert(0, str(project_root / "BEHAVIOR-1K" / "bddl"))

from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
import omnigibson as og
from omnigibson.learning.utils.eval_utils import (
    ACTION_QPOS_INDICES,
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from hydra.utils import instantiate

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
gm.NO_OMNI_LOGS = True
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MOVE_TO_SUCCESS_DISTANCE_THRESHOLD = 1.0  # meters; reach / success gate (default early stop after min steps)
# Do not allow distance-based early stop until this loop index (after this many env.step calls from prior iterations).
MOVE_TO_MIN_STEPS_BEFORE_REACH_EARLY_STOP = 1200
MOVE_TO_SUCCESS_BASE_SPEED_THRESHOLD = 0.1  # For early exit only (see --base_speed_threshold)
MOVE_TO_SUCCESS_CONSECUTIVE_LOW_SPEED_STEPS = 1500  # For early exit only (see --consecutive_low_speed_steps)
DEFAULT_MAX_STEPS_PER_SEGMENT = 1500
DEFAULT_LOG_DIR = "/mnt/public/tgy/openpi-comet/logs"
# Aligned with data_generation/rft/generate_move_to_rft_data.py POST_RESET_PHYSICS_STEPS
POST_RESET_PHYSICS_STEPS = 50
TASK_SCENE_ROOMS_MAPPING_PATH = Path(__file__).resolve().parent / "task_scene_rooms_mapping.json"
TASK_MAPPING_PATH = Path(__file__).resolve().parent / "task_mapping.json"


def discover_instance_ids(task_name: str, scene_model: str, activity_definition_id: int = 0) -> list[int]:
    """List activity_instance_id values whose tro file exists (same basename as BehaviorTask.get_cached_activity_scene_filename)."""
    task_path = get_task_instance_path(scene_model)
    if task_path is None:
        return [0]
    instances_dir = os.path.join(
        task_path,
        "json",
        f"{scene_model}_task_{task_name}_instances",
    )
    if not os.path.isdir(instances_dir):
        return [0]
    prefix = f"{scene_model}_task_{task_name}_{activity_definition_id}_"
    suffix = "_template-tro_state.json"
    ids = []
    for fname in os.listdir(instances_dir):
        if fname.startswith(prefix) and fname.endswith(suffix):
            mid = fname[len(prefix) : -len(suffix)]
            try:
                ids.append(int(mid))
            except ValueError:
                pass
    return sorted(ids) if ids else [0]


def _load_move_to_segments_json(segments_path: Path) -> list[dict]:
    """Load move_to_segments.json / RFT-style list; dedupe by (task_name, instance_id, object_id).

    Same rules as data_generation/rft/generate_move_to_rft_data.extract_move_to_segments.
    """
    with open(segments_path) as f:
        all_segments = json.load(f)
    seen = set()
    out = []
    for seg in all_segments:
        skill_str = str(seg.get("skill_str", "")).strip().lower()
        prompt = str(seg.get("prompt", "")).strip().lower()
        is_move_to = (skill_str == "move to") or prompt.startswith(("move to", "move back to"))
        if not is_move_to:
            continue
        key = (seg["task_name"], seg["instance_id"], seg["object_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(seg))
    return out


def _order_segments_sequential_per_task(segments: list[dict]) -> list[dict]:
    """Group segments by task (normalized name), preserve order within each task; task order = first appearance in list."""
    if not segments:
        return []
    by_norm: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    order_norm: list[str] = []
    seen_norm: set[str] = set()
    for i, seg in enumerate(segments):
        tn = seg.get("task_name", "")
        norm = tn.lower().replace(" ", "_") if isinstance(tn, str) else str(tn)
        by_norm[norm].append((i, seg))
        if norm not in seen_norm:
            seen_norm.add(norm)
            order_norm.append(norm)
    out: list[dict] = []
    for norm in order_norm:
        for _, seg in sorted(by_norm[norm], key=lambda x: x[0]):
            out.append(seg)
    return out


def _normalize_move_to_prompt_from_seg(seg: dict) -> str:
    """Imperative prompt like 'move to <readable>' from object_id (RFT helper)."""
    p = str(seg.get("prompt", "")).strip()
    if p:
        p_low = p.lower()
        if p_low.startswith("move to ") or p_low.startswith("move back to "):
            return p
    object_id = str(seg.get("object_id", "object"))
    object_name = re.sub(r"_\d+$", "", object_id).replace("_", " ").strip()
    if not object_name:
        object_name = "object"
    return f"move to {object_name}"


def _finalize_move_to_segments_for_eval(segments: list[dict]) -> None:
    """Normalize instance_id, skill_str, prompt for eval loop."""
    for seg in segments:
        seg["instance_id"] = int(seg["instance_id"])
        seg["skill_str"] = seg.get("skill_str", "move to")
        if not str(seg.get("prompt", "")).strip():
            seg["prompt"] = _normalize_move_to_prompt_from_seg(seg)


def _find_target_object_annotation(env, object_id: str):
    """Resolve annotation-style object_id (e.g. wicker_basket_90); matches generate_move_to_rft_data.find_target_object."""
    task = getattr(env, "task", None)
    if task is not None and hasattr(task, "object_scope"):
        base_name = re.sub(r"_\d+$", "", object_id).replace("_", " ")
        base_simple = base_name.replace(" ", "_")
        for name, bddl_inst in task.object_scope.items():
            if bddl_inst.is_system or not bddl_inst.exists or bddl_inst.fixed_base:
                continue
            if "agent" in name.lower():
                continue
            obj = bddl_inst.wrapped_obj
            if obj is None:
                continue
            if name == object_id or base_simple in name or base_name in name.replace("_", " "):
                return obj
    try:
        obj = env.scene.object_registry("name", object_id)
        if obj is not None:
            return obj
    except Exception:
        pass
    base_name = re.sub(r"_\d+$", "", object_id).replace("_", " ")
    for obj in getattr(env.scene.object_registry, "objects", []):
        try:
            n = getattr(obj, "name", None) or str(obj)
            if base_name.replace(" ", "_") in n or base_name in n.replace("_", " "):
                return obj
        except Exception:
            continue
    return None


def _annotation_display_name(object_id: str) -> str:
    if not object_id:
        return ""
    base = re.sub(r"_\d+$", "", object_id)
    return base.replace("_", " ").strip()


# --- Prompt formatting aligned with BehaviorLeRobotDataset / build_orchestrators_from_annotations ---

# Same success logic and segment filtering; prompts always use "move to <物体名>" (e.g. "move to radio").
MOVE_TO_SKILL_ALIASES = frozenset({"move to", "move to object"})

_OBJECT_ID_LIKE = re.compile(r"^[a-zA-Z0-9_]+_\d+$")


def _is_move_to_like_skill(skill_desc: str) -> bool:
    return (skill_desc or "").strip().lower() in MOVE_TO_SKILL_ALIASES


def _annotation_object_id_to_readable(obj_id: str) -> str:
    """Convert object_id like 'radio_89' or 'trash_can_116' to readable form (dataset convention)."""
    base = re.sub(r"_\d+$", "", obj_id)
    return base.replace("_", " ")


def _flatten_strings(obj) -> list:
    if isinstance(obj, str):
        return [obj] if obj else []
    if isinstance(obj, list):
        result = []
        for x in obj:
            result.extend(_flatten_strings(x))
        return result
    return []


def _memory_tokens(memory_prefix: list | None) -> list[str]:
    if not memory_prefix:
        return []
    return [str(m).strip() for m in memory_prefix if m and str(m).strip()]


def _format_move_to_with_memory(
    obj_phrase: str,
    memory_prefix: list | None,
    *,
    ord_word: str | None = None,
) -> str:
    toks = _memory_tokens(memory_prefix)
    has_back = "back" in toks
    has_other = "the other" in toks
    stem = "move back to" if has_back else "move to"
    middle = f"{ord_word} {obj_phrase}".strip() if ord_word else obj_phrase
    if has_other:
        return f"{stem} the other {middle}"
    return f"{stem} {middle}"


def _format_skill_prompt(
    skill_desc: str,
    object_ids: list,
    memory_prefix: list,
    spatial_prefix: list,
    task_name: str,
) -> str:
    """Same structure as BehaviorLeRobotDataset._format_skill_prompt (no Goal line).

    For ``move to`` with memory_prefix ``back`` / ``the other``, folds memory into the skill
    phrase (e.g. ``Skill: move back to the other radio``) instead of a separate Context line.
    """
    objects_flat = _flatten_strings(object_ids)
    objects_str = ", ".join(objects_flat) if objects_flat else ""

    mem_toks = _memory_tokens(memory_prefix)
    extra_mem = [t for t in mem_toks if t not in {"back", "the other"}]
    inject_move_mem = _is_move_to_like_skill(skill_desc) and len(objects_flat) == 1 and (
        "back" in mem_toks or "the other" in mem_toks
    )

    if inject_move_mem:
        raw = objects_flat[0]
        obj_phrase = _object_phrase_for_prompt(raw) if raw else ""
        skill_phrase = _format_move_to_with_memory(obj_phrase, memory_prefix, ord_word=None)
        parts = [f"Skill: {skill_phrase}"]
        if extra_mem:
            parts.append(f"Context: {' '.join(extra_mem)}")
    else:
        parts = [f"Skill: {skill_desc}"]
        if objects_str:
            parts.append(f"Objects: {objects_str}")
        if memory_prefix and any(memory_prefix):
            mem_str = " ".join(m for m in memory_prefix if m).strip()
            if mem_str:
                parts.append(f"Context: {mem_str}")
    if spatial_prefix and any(spatial_prefix):
        sp_flat = []
        for s in spatial_prefix:
            if isinstance(s, list):
                sp_flat.extend(str(x) for x in s if x)
            elif s:
                sp_flat.append(str(s))
        sp_str = " ".join(sp_flat).strip()
        if sp_str:
            parts.append(f"Spatial: {sp_str}")

    return ". ".join(parts)


def _object_ids_nested_from_seg(seg: dict) -> list:
    """Normalize segment object_id to annotation-style nested lists."""
    oid = seg.get("object_id")
    if isinstance(oid, list) and oid and isinstance(oid[0], list):
        return oid
    if isinstance(oid, str):
        return [[oid]] if oid else [[]]
    if isinstance(oid, list) and oid and isinstance(oid[0], str):
        return [oid]
    if isinstance(oid, list):
        return oid
    return [[]]


def _skill_desc_from_seg(seg: dict) -> str:
    sd = seg.get("skill_str") or ""
    if sd:
        return sd
    desc = seg.get("skill_description")
    if isinstance(desc, list) and desc:
        return str(desc[0] or "")
    return str(desc or "")


def build_behavior_lerobot_task_prompt(
    skill_desc: str,
    object_ids_nested: list,
    memory_prefix: list | None,
    spatial_prefix: list | None,
    *,
    use_skill_prompt_format: bool,
    task_name: str = "",
    include_object_for_skills: set | None = None,
) -> str:
    """
    Match BehaviorLeRobotDataset.build_orchestrators_from_annotations task strings
    (use_skill_prompt_format True -> Skill:/Objects:/Context:/Spatial:; False -> imperative e.g. move to X).
    """
    if include_object_for_skills is None:
        include_object_for_skills = set(MOVE_TO_SKILL_ALIASES)
    memory_prefix = memory_prefix or []
    spatial_prefix = spatial_prefix or []

    if use_skill_prompt_format:
        return _format_skill_prompt(
            skill_desc,
            object_ids_nested,
            memory_prefix,
            spatial_prefix,
            task_name,
        )

    if skill_desc.lower() in include_object_for_skills and object_ids_nested:
        obj_ids = object_ids_nested[0] if object_ids_nested else []
        if obj_ids:
            obj_id = obj_ids[0]
            obj_readable = _object_phrase_for_prompt(obj_id)
            if _is_move_to_like_skill(skill_desc):
                return _format_move_to_with_memory(obj_readable, memory_prefix, ord_word=None)
            return f"{skill_desc} {obj_readable}"
        return skill_desc
    return skill_desc


def prompt_for_eval_segment(seg: dict, *, use_skill_prompt_format: bool) -> str:
    """Build policy prompt for one segment dict (jsonl row, BDDL segment, or extracted JSON)."""
    return build_behavior_lerobot_task_prompt(
        _skill_desc_from_seg(seg),
        _object_ids_nested_from_seg(seg),
        seg.get("memory_prefix"),
        seg.get("spatial_prefix"),
        use_skill_prompt_format=use_skill_prompt_format,
        task_name=str(seg.get("goal") or seg.get("task_name") or ""),
    )


def _target_display_name_from_segment(seg: dict, prompt: str) -> str:
    """Short label for video overlay; works with Skill:/Objects: and plain 'move to X' prompts."""
    obj_match = re.search(
        r"Objects:\s*([^\.]+?)(?:\.\s*(?:Goal|Context|Spatial)|$)",
        prompt,
    )
    if obj_match:
        return obj_match.group(1).strip().split(",")[0].strip()
    m = re.search(
        r"Skill:\s*(move(?:\s+back)?\s+to(?:\s+the\s+other)?)\s+(.+?)(?:\.\s*Context|\.\s*Spatial|$)",
        prompt,
        re.I | re.DOTALL,
    )
    if m:
        rest = m.group(2).strip()
        if rest.lower().startswith("skill:"):
            return rest
        return rest.split(".")[0].strip()
    plain = re.match(
        r"^move(?:\s+back)?\s+to(?:\s+the\s+other)?\s+(.+)$",
        prompt.strip(),
        re.I,
    )
    if plain:
        return plain.group(1).strip()
    oid = seg.get("object_id")
    if isinstance(oid, str) and oid:
        if _OBJECT_ID_LIKE.match(oid):
            return _annotation_object_id_to_readable(oid)
        return _bddl_object_to_readable(oid)
    return str(oid or "")


def _bddl_object_to_readable(obj_id: str) -> str:
    """Convert BDDL object ID (e.g. firewood.n.01_1) to readable form for prompt."""
    term = obj_id.lstrip("?")
    natural = term.split(".")[0].replace("_", " ")
    if "_" in term:
        suffix = term.split("_")[-1]
        if suffix.isdigit():
            natural += f" {suffix}"
        else:
            natural += suffix
    return natural


def _object_phrase_for_prompt(obj_id: str) -> str:
    """Readable phrase for imperative prompts; annotation-style ids vs BDDL."""
    s = str(obj_id)
    if _OBJECT_ID_LIKE.match(s):
        return _annotation_object_id_to_readable(s)
    return (_bddl_object_to_readable(s) or s).strip()


def _episode_index_to_instance(episode_index: int) -> int:
    """Extract instance_id from episode_index. episode_index = task_idx*1e4 + instance*10 + traj."""
    return int((episode_index // 10) % 1e3)


def _extract_object_type_from_skills_flat(obj_id: str) -> str:
    """Extract object type from skills_flat object_id (e.g. radio_89 -> radio, trash_can_116 -> trash_can)."""
    if not obj_id:
        return ""
    # Remove trailing _number (e.g. radio_89 -> radio, trash_can_116 -> trash_can)
    base = re.sub(r"_\d+$", "", obj_id)
    # Remove model suffix like _koagbh (coffee_table_koagbh_0 -> coffee_table)
    base = re.sub(r"_[a-z]+\d*$", "", base)
    return base.replace("_", " ").strip()


def _map_skills_flat_object_to_bddl(skills_flat_obj_id: str, task_name: str) -> str | None:
    """
    Map skills_flat object_id (e.g. radio_89) to BDDL object_id (e.g. radio_receiver.n.01_1)
    for scene lookup. Match by object type from jsonl.
    """
    from bddl.parsing import parse_problem

    obj_type = _extract_object_type_from_skills_flat(skills_flat_obj_id)
    if not obj_type:
        return None
    obj_type_lower = obj_type.lower()
    obj_words = set(obj_type_lower.split())

    try:
        __, objects, __, __ = parse_problem(task_name, 0, "omnigibson")
    except Exception:
        return None

    best_match = None
    best_score = 0
    for cat, insts in objects.items():
        for inst in (insts or []):
            if not isinstance(inst, str):
                continue
            readable = _bddl_object_to_readable(inst)
            readable_lower = readable.lower()
            readable_words = set(readable_lower.split())
            # Score: word overlap; prefer if obj_type is substring of readable (e.g. "radio" in "radio receiver")
            score = len(obj_words & readable_words)
            if obj_type_lower in readable_lower or readable_lower in obj_type_lower:
                score += 10
            # Partial: e.g. "can" in "ashcan" for trash_can
            if any(ow in readable_lower.split()[0] for ow in obj_words if len(ow) > 2):
                score += 5
            # Synonym: trash_can -> ashcan (WordNet)
            if "trash" in obj_words and "ashcan" in readable_lower:
                score += 10
            # Dataset naming vs BDDL: cutting_board_* -> chopping_board.n.01_*
            if (
                "board" in obj_words
                and "cutting" in obj_words
                and "chopping" in readable_lower
                and "board" in readable_lower
            ):
                score += 15
            if score > best_score:
                best_score = score
                best_match = inst
    return best_match


def _load_segments_from_skills_flat_for_bddl(
    skills_flat_path: Path,
    task_mapping_path: Path,
    tasks: list[str] | None = None,
    *,
    use_skill_prompt_format: bool = False,
    offline: bool = False,
) -> list[dict]:
    """
    Load move-to segments from skills_flat.jsonl, map object_id to BDDL format.
    Each segment will be tested in its own BDDL-sampled scene.
    """
    if not task_mapping_path.exists():
        logger.error(f"task_mapping.json not found: {task_mapping_path}")
        return []
    with open(task_mapping_path) as f:
        task_mapping = json.load(f)

    def _human_task_to_bddl(human_name: str) -> str | None:
        candidate = human_name.replace(" ", "_").lower()
        if candidate in task_mapping:
            return candidate
        for key in task_mapping:
            if key.lower().replace(" ", "_") == candidate:
                return key
        return None

    segments = []
    task_set = {t.lower().replace(" ", "_") for t in tasks} if tasks else None
    with open(skills_flat_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not _is_move_to_like_skill(row.get("skill_str") or ""):
                continue
            human_task = row.get("task_name", "")
            bddl_task = _human_task_to_bddl(human_task)
            if bddl_task is None:
                continue
            if task_set and bddl_task not in task_set:
                continue
            obj_ids = row.get("object_id", [[]])
            skills_flat_obj_id = obj_ids[0][0] if obj_ids and obj_ids[0] else ""
            if not skills_flat_obj_id:
                continue
            bddl_obj_id = _map_skills_flat_object_to_bddl(skills_flat_obj_id, bddl_task)
            if bddl_obj_id is None:
                logger.warning(f"No BDDL match for {skills_flat_obj_id} in task {bddl_task}, skipping")
                continue
            prompt_src = {
                "skill_str": row.get("skill_str", "move to"),
                "skill_description": row.get("skill_description"),
                "object_id": [[skills_flat_obj_id]],
                "memory_prefix": row.get("memory_prefix", []),
                "spatial_prefix": row.get("spatial_prefix", []),
                "goal": row.get("goal", human_task),
                "task_name": row.get("task_name", human_task),
            }
            prompt = prompt_for_eval_segment(
                prompt_src, use_skill_prompt_format=use_skill_prompt_format
            )
            instance_id = _episode_index_to_instance(row.get("episode_index", 0)) if offline else 0
            seg = {
                "task_name": bddl_task,
                "instance_id": instance_id,
                "object_id": bddl_obj_id,
                "prompt": prompt,
                "skill_str": row.get("skill_str") or "move to",
                "episode_index": row.get("episode_index", 0),
                "memory_prefix": row.get("memory_prefix", []),
            }
            segments.append(seg)
    if offline:
        logger.info(f"Loaded {len(segments)} move-to segments from skills_flat (offline/tro_state, instance_id from episode_index)")
    else:
        logger.info(f"Loaded {len(segments)} move-to segments from skills_flat (mapped to BDDL)")
    return segments


def extract_segments_from_bddl(
    available_tasks: dict,
    tasks: list[str] | None = None,
    max_segments_per_task: int | None = None,
    max_objects_per_activity: int | None = 5,
    *,
    use_skill_prompt_format: bool = False,
    move_to_skill_str: str = "move to",
) -> list[dict]:
    """
    Extract move-to segments from BDDL activity definitions.
    Returns list of segment dicts: task_name, instance_id, object_id, prompt, skill_str.
    move_to_skill_str: "move to" (prompts like "move to radio").
    """
    from bddl.activity import get_all_activities, get_instance_count
    from bddl.parsing import parse_problem

    domain_name = "omnigibson"
    segments = []
    task_counts = defaultdict(int)

    for activity_name in get_all_activities():
        if activity_name not in available_tasks:
            continue
        if tasks is not None and activity_name not in tasks:
            continue
        if max_segments_per_task is not None and task_counts[activity_name] >= max_segments_per_task:
            continue

        n_defs = get_instance_count(activity_name)
        for activity_definition_id in range(min(1, n_defs)):
            try:
                __, objects, __, __ = parse_problem(
                    activity_name, activity_definition_id, domain_name
                )
            except Exception as e:
                logger.warning(f"Failed to parse {activity_name} problem{activity_definition_id}: {e}")
                continue

            obj_instances = []
            for cat, insts in objects.items():
                if cat == "object" or "agent" in cat or "floor" in cat:
                    continue
                for inst in (insts or []):
                    if isinstance(inst, str):
                        obj_instances.append(inst)

            if max_objects_per_activity is not None:
                obj_instances = obj_instances[:max_objects_per_activity]

            goal_human = activity_name.replace("_", " ")
            for obj_id in obj_instances:
                if max_segments_per_task is not None and task_counts[activity_name] >= max_segments_per_task:
                    break
                prompt = build_behavior_lerobot_task_prompt(
                    move_to_skill_str,
                    [[obj_id]],
                    [],
                    [],
                    use_skill_prompt_format=use_skill_prompt_format,
                    task_name=goal_human,
                )
                seg = {
                    "task_name": activity_name,
                    "activity_definition_id": activity_definition_id,
                    "instance_id": 0,
                    "object_id": obj_id,
                    "prompt": prompt,
                    "skill_str": move_to_skill_str,
                }
                segments.append(seg)
                task_counts[activity_name] += 1

    logger.info(f"Extracted {len(segments)} move-to segments from BDDL across {len(task_counts)} tasks")
    return segments


def _get_tasks_by_scene_rooms(
    scene: str,
    rooms: list[str],
    mapping_path: pathlib.Path | None = None,
) -> list[str]:
    """Load task bddl_names from task_scene_rooms_mapping.json for the given scene and rooms.

    Returns tasks that occur in any of the specified rooms (room groups where in_rooms
    intersects with the given rooms).
    """
    if mapping_path is None:
        mapping_path = pathlib.Path(__file__).resolve().parent / "task_scene_rooms_mapping.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    if scene not in mapping:
        raise ValueError(
            f"Unknown scene: {scene}. Available: {list(mapping)}"
        )
    rooms_set = set(rooms)
    task_bddl_names: list[str] = []
    seen: set[str] = set()
    for room_group in mapping[scene]:
        in_rooms = set(room_group.get("in_rooms", []))
        if rooms_set == in_rooms:  # any overlap
            for t in room_group.get("tasks", []):
                bddl_name = t.get("bddl_name", "")
                if bddl_name and bddl_name not in seen:
                    seen.add(bddl_name)
                    task_bddl_names.append(bddl_name)
    return task_bddl_names


def _get_allowed_tasks_by_scene_rooms(
    scenes: list[str] | None,
    rooms: list[str] | None,
    mapping_path: Path,
) -> list[str] | None:
    """Get allowed task bddl_names for the given scene/room filters.

    Returns None if neither scene nor room is specified (no filtering).
    Otherwise returns list of task bddl_names that match.
    """
    if not scenes and not rooms:
        return None
    if not mapping_path.exists():
        return []
    with open(mapping_path) as f:
        mapping = json.load(f)
    scene_list = list(scenes) if scenes else list(mapping.keys())
    allowed: set[str] = set()
    for scene in scene_list:
        if scene not in mapping:
            continue
        if rooms:
            room_list = list(rooms)
        else:
            room_list = []
            for rg in mapping[scene]:
                room_list.extend(rg.get("in_rooms", []))
            room_list = list(set(room_list))
        tasks = _get_tasks_by_scene_rooms(scene, room_list, mapping_path)
        allowed.update(tasks)
    return list(allowed)


def _load_task_scene_room_mapping(mapping_path: Path) -> dict[str, tuple[str, str]]:
    """
    Load task_scene_rooms_mapping.json and build task_name (bddl) -> (scene, room_key).
    room_key: single room like "living_room", multi-room like "garage_kitchen" (sorted, joined).
    """
    if not mapping_path.exists():
        logger.warning(f"task_scene_rooms_mapping.json not found: {mapping_path}")
        return {}
    with open(mapping_path) as f:
        data = json.load(f)
    result = {}
    for scene, room_configs in data.items():
        for room_cfg in room_configs:
            in_rooms = room_cfg.get("in_rooms", [])
            room_key = "_".join(sorted(in_rooms)) if in_rooms else "unknown"
            for t in room_cfg.get("tasks", []):
                bddl_name = t.get("bddl_name", "").replace(" ", "_").lower()
                if bddl_name:
                    result[bddl_name] = (scene, room_key)
    return result


class SkillWebsocketPolicy:
    def __init__(self, host: str, port: int):
        from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
        self._client = WebsocketClientPolicy(host=host, port=port)
        self._prompt_override = None

    def set_prompt_override(self, prompt: str | None):
        self._prompt_override = prompt

    def forward(self, obs: dict) -> th.Tensor:
        if self._prompt_override is not None:
            obs = dict(obs)
            obs["prompt"] = self._prompt_override
        obs = _tensors_to_numpy(obs)
        return self._client.act(obs)

    def reset(self):
        self._client.reset()


def _tensors_to_numpy(obj):
    if isinstance(obj, th.Tensor):
        return obj.cpu().numpy()
    if isinstance(obj, dict):
        return {k: _tensors_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensors_to_numpy(x) for x in obj)
    return obj


def _find_target_object(scene, object_id: str, task=None):
    """Find target object by object_id. Uses object_scope when task is provided (online_sampling)."""
    all_objs = _find_all_matching_objects(scene, object_id, task)
    return all_objs[0][0] if all_objs else None


def _get_bddl_synset(object_id: str) -> str:
    """Extract BDDL synset from object_id (e.g. can__of__soda.n.01_1 -> can__of__soda.n.01)."""
    return re.sub(r"_\d+$", "", object_id)


def _find_all_matching_objects(scene, object_id: str, task=None):
    """
    Find all objects matching object_id (same type). Returns list of (obj, display_name).
    When scene has multiple identical objects (e.g. multiple radios), returns all of them.
    Uses BDDL synset for strict matching: can__of__soda.n.01 matches only cans, NOT ashcan.
    """
    if not object_id:
        return []

    target_synset = _get_bddl_synset(object_id)
    base_name = re.sub(r"_\d+$", "", object_id).replace("_", " ")
    result: list[tuple] = []
    seen: set = set()  # avoid duplicates by object id()

    def _add(obj, display_name: str):
        if obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            result.append((obj, display_name))

    # 1. Try object_scope first (BDDL format, works with online_sampling)
    # Match by BDDL synset only - e.g. can__of__soda.n.01 matches can__of__soda.n.01_1, _2, _3
    # but NOT ashcan.n.01 (avoids "can" in "ashcan" false match)
    if task is not None and hasattr(task, "object_scope"):
        scope = task.object_scope
        if object_id in scope:
            entity = scope[object_id]
            if getattr(entity, "exists", False) and not getattr(entity, "is_system", False):
                obj = getattr(entity, "wrapped_obj", None) or getattr(entity, "entity", None)
                if obj is not None:
                    _add(obj, _bddl_object_to_readable(object_id))
        for bddl_inst, entity in scope.items():
            if "agent.n." in bddl_inst or "floor.n." in bddl_inst:
                continue
            if not getattr(entity, "exists", True) or getattr(entity, "is_system", False):
                continue
            inst_synset = _get_bddl_synset(bddl_inst)
            if inst_synset == target_synset:
                obj = getattr(entity, "wrapped_obj", None) or getattr(entity, "entity", None)
                if obj is not None:
                    _add(obj, _bddl_object_to_readable(bddl_inst))

    if result:
        return result

    # 2. Try object_registry with multiple name variants (single-object lookup)
    base_simple = object_id.split(".")[0] + "_" + object_id.split("_")[-1] if "_" in object_id else object_id.split(".")[0]
    candidates = [
        object_id,
        base_simple,
        object_id.replace(".", "_"),
        re.sub(r"\.n\.\d+", "", object_id).replace("_", " "),
    ]
    for name in candidates:
        try:
            obj = scene.object_registry("name", name)
            if obj is not None:
                _add(obj, name)
                return result
        except Exception:
            pass

    # 3. Fallback: iterate object_registry by category (find all matching)
    try:
        base_cat_full = object_id.split(".")[0]
        for obj in getattr(scene.object_registry, "objects", []):
            try:
                n = getattr(obj, "name", None) or str(obj)
                if base_cat_full in (n or "") or base_name in (n or "").replace("_", " "):
                    _add(obj, n or str(obj))
            except Exception:
                continue
    except Exception:
        pass
    return result


def _get_robot_base_position(robot):
    try:
        pos, _ = robot.get_position_orientation()
        return np.array(pos)
    except Exception:
        return np.zeros(3)


def _get_arm_link_world_position(robot, side: str) -> np.ndarray | None:
    """World position of one arm end (EEF / wrist link). side: 'left' or 'right'."""
    # R1 / R1Pro canonical EEF names first (see omnigibson.robots.r1.R1.eef_link_names)
    if side == "left":
        names = (
            "left_eef_link",
            "left_realsense_link",
            "left_eef",
            "robot_r1_left_eef",
        )
    else:
        names = (
            "right_eef_link",
            "right_realsense_link",
            "right_eef",
            "robot_r1_right_eef",
        )
    links = getattr(robot, "links", None) or {}
    for name in names:
        link = links.get(name)
        if link is None:
            continue
        try:
            pos, _ = link.get_position_orientation()
            return np.asarray(pos, dtype=np.float64)
        except Exception:
            continue
    return None


def _min_distance_arms_to_point(robot, point: np.ndarray) -> float:
    """Shortest distance from left or right arm end to ``point``; falls back to base if no EEF links."""
    left = _get_arm_link_world_position(robot, "left")
    right = _get_arm_link_world_position(robot, "right")
    cand: list[float] = []
    if left is not None:
        cand.append(float(np.linalg.norm(left - point)))
    if right is not None:
        cand.append(float(np.linalg.norm(right - point)))
    if cand:
        return min(cand)
    base = _get_robot_base_position(robot)
    return float(np.linalg.norm(base - point))


def _get_object_position(obj):
    """World-frame point for distance-to-target.

    OmniGibson ``aabb_center`` is from collision geometry; some assets (e.g. wall nails) get a huge
    hull so the AABB center is far from the visible object. Prefer the object root pose from
    ``get_position_orientation()`` unless the AABB is tight and agrees with the root.
    """
    root: np.ndarray | None = None
    try:
        pos, _ = obj.get_position_orientation()
        root = np.asarray(pos, dtype=np.float64).reshape(3)
    except Exception:
        pass
    try:
        if hasattr(obj, "aabb_center"):
            ac = np.asarray(obj.aabb_center, dtype=np.float64).reshape(3)
            if root is None:
                return ac
            delta = float(np.linalg.norm(ac - root))
            max_ext = 0.0
            if hasattr(obj, "aabb_extent"):
                try:
                    max_ext = float(np.max(np.asarray(obj.aabb_extent, dtype=np.float64)))
                except Exception:
                    max_ext = 0.0
            if max_ext < 8.0 and delta < 4.0:
                return ac
    except Exception:
        pass
    if root is not None:
        return root
    return np.zeros(3)


class SkillEvaluator:
    def __init__(
        self,
        policy_url: str,
        env_wrapper_cfg=None,
        write_video: bool = True,
        video_path: Path | None = None,
        online_sampling: bool = False,
        partial_scene_load: bool = False,
        *,
        rft_style_tro: bool = False,
        use_annotation_object_lookup: bool = False,
    ):
        host, _, port_str = policy_url.replace("ws://", "").replace("wss://", "").rstrip("/").partition(":")
        port = int(port_str) if port_str else 8000
        self.policy = SkillWebsocketPolicy(host=host, port=port)
        self.env_wrapper_cfg = env_wrapper_cfg
        self.env = None
        self.robot = None
        self.write_video = write_video
        self.video_path = video_path
        self.online_sampling = online_sampling
        self.partial_scene_load = partial_scene_load
        self.rft_style_tro = rft_style_tro
        self.use_annotation_object_lookup = use_annotation_object_lookup
        self._cached_tro_state = None
        self.target = None

    def _robot_pose_registry_key(self) -> str:
        """Key used inside tro_state['robot_poses']; matches OmniGibson BaseRobot.model_name when present."""
        robot = self.robot
        if robot is None:
            return "R1Pro"
        name = getattr(robot, "model_name", None)
        if name is not None:
            return str(name)
        name = getattr(robot, "_model_name", None)
        if name is not None:
            return str(name)
        cls_name = robot.__class__.__name__
        return "R1Pro" if cls_name == "Robot" else cls_name

    def _reapply_tro_after_reset(self, tro_state: dict) -> None:
        """Match generate_move_to_rft_data.reapply_tro_from_cache (after env.reset())."""
        rk = self._robot_pose_registry_key()
        for tro_key, tro_data in tro_state.items():
            if tro_key == "robot_poses":
                if rk in tro_data:
                    rp = tro_data[rk][0]
                    self.robot.set_position_orientation(rp["position"], rp["orientation"])
            elif tro_key in self.env.task.object_scope:
                self.env.task.object_scope[tro_key].load_state(tro_data, serialized=False)
        if "robot_poses" in tro_state:
            self.env.scene.write_task_metadata(key="robot_poses", data=tro_state["robot_poses"])
    def load_env(self, task_name: str, max_steps: int):
        """Load environment. Aligned with omnigibson/learning/eval.py: scene always uses instance 0."""
        if self.env is not None:
            og.sim.stop()
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        available_tasks = load_available_tasks()
        assert task_name in available_tasks, f"Invalid task: {task_name}"
        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        # Scene load_room logic aligned with eval.py (omnigibson/learning/eval.py):
        # - partial_scene_load=True: load only task-relevant rooms (faster)
        # - partial_scene_load=False (default): load full scene (load_room_types=None)
        # For online_sampling (BDDL), use full scene to avoid "table.n.02_1: empty" etc.
        # For pre-sampled (tro_state), use partial when partial_scene_load for speed.
        if self.partial_scene_load and not self.online_sampling:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms
        cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        cfg["task"]["termination_config"]["max_steps"] = max_steps
        cfg["task"]["include_obs"] = False

        if self.online_sampling:
            cfg["task"]["online_object_sampling"] = True
            cfg["task"]["use_presampled_robot_pose"] = False
            cfg["task"]["activity_instance_id"] = 0
        else:
            # Pre-sampled (tro_state): use activity_instance_id=0 like omnigibson/learning/eval.py.
            # Scene template is always from instance 0; load_task_instance loads tro_state for any instance.
            cfg["task"]["activity_instance_id"] = 0
            # RFT / move_to_segments: reset() must not overwrite robot pose from generic task_metadata (generate_move_to_rft_data.py).
            if self.rft_style_tro:
                cfg["task"]["use_presampled_robot_pose"] = False

        env = og.Environment(configs=cfg)
        if self.env_wrapper_cfg is not None:
            env = instantiate(self.env_wrapper_cfg, env=env)
        self.env = env
        self.robot = env.scene.object_registry("name", "robot_r1")
        self.current_task_name = task_name
        return env

    def load_task_instance(self, instance_id: int):
        if self.online_sampling:
            self._cached_tro_state = None
            return True
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        if not os.path.exists(tro_file_path):
            self._cached_tro_state = None
            return False
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        self._cached_tro_state = tro_state

        rk = self._robot_pose_registry_key()
        for tro_key, tro_data in tro_state.items():
            if tro_key == "robot_poses":
                if rk not in tro_data:
                    self._cached_tro_state = None
                    return False
                robot_pos = tro_data[rk][0]["position"]
                robot_quat = tro_data[rk][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                self.env.scene.write_task_metadata(key=tro_key, data=tro_data)
            else:
                # Aligned with eval.py: direct access (assumes object_scope matches tro_state keys)
                self.env.task.object_scope[tro_key].load_state(tro_data, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        # Aligned with omnigibson/learning/eval.py - do NOT re-apply robot pose during stabilize.
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()

        if self.rft_style_tro:
            # data_generation/rft/generate_move_to_rft_data.load_env_for_task_instance: full env.reset() then re-apply TRO.
            self.env.reset()
            self._reapply_tro_after_reset(tro_state)
            for _ in range(POST_RESET_PHYSICS_STEPS):
                og.sim.step_physics()
                for entity in self.env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()
            setattr(self.env, "_eval_cached_tro_state", tro_state)
            return True

        self.env.scene.reset()
        return True

    def _preprocess_obs(self, obs: dict) -> dict:
        import omnigibson.utils.transform_utils as T
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose)))
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        if hasattr(self, "current_task_name") and self.current_task_name:
            obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.current_task_name]], dtype=th.int64)
        return obs

    def _write_video_frame(self, obs: dict, video_writer, distances: list[tuple[str, float]] | None = None):
        """
        Write frame to video. distances: list of (object_display_name, distance_in_meters).
        Distances use min(left EEF, right EEF) to object (base fallback).
        When scene has multiple identical objects, pass all (name, dist) to show them simultaneously.
        """
        try:
            left_key = ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"
            right_key = ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"
            head_key = ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"
            def to_np(x):
                return x.cpu().numpy() if isinstance(x, th.Tensor) else np.asarray(x)
            left_wrist_rgb = cv2.resize(to_np(obs[left_key]), (224, 224))
            right_wrist_rgb = cv2.resize(to_np(obs[right_key]), (224, 224))
            head_rgb = cv2.resize(to_np(obs[head_key]), (448, 448))
            frame = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])
            y_offset = 30
            if distances:
                for obj_name, dist in distances:
                    text = f"dist->{obj_name}: {dist:.3f}m"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 28
            else:
                target_name = getattr(self, "target_obj_name", None) or str(self.target)
                cv2.putText(frame, f"dist->{target_name}: N/A", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 28
            prompt_text = getattr(self, "current_prompt", "") or ""
            if prompt_text:
                prompt_short = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
                cv2.putText(frame, prompt_short, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            write_video(np.expand_dims(frame, 0), video_writer=video_writer, batch_size=1, mode="rgb")
        except (KeyError, TypeError):
            pass

    def run_segment(
        self,
        seg: dict,
        max_steps: int,
        distance_threshold: float,
        video_name: str | None = None,
        base_speed_threshold: float = MOVE_TO_SUCCESS_BASE_SPEED_THRESHOLD,
        consecutive_low_speed_steps: int = MOVE_TO_SUCCESS_CONSECUTIVE_LOW_SPEED_STEPS,
        *,
        early_stop_on_reach_distance: bool = True,
        min_steps_before_reach_early_stop: int = MOVE_TO_MIN_STEPS_BEFORE_REACH_EARLY_STOP,
    ) -> dict:
        self.policy.set_prompt_override(seg["prompt"])
        self.policy.reset()

        if not self.load_task_instance(seg["instance_id"]):
            return {
                "success": False,
                "min_distance": float("inf"),
                "initial_distance": float("inf"),
                "final_distance": float("inf"),
                "last_action_base_speed": float("inf"),
                "steps": 0,
                "error": "tro_not_found",
            }

        obs, _ = self.env.reset()
        if self.rft_style_tro and self._cached_tro_state is not None:
            self._reapply_tro_after_reset(self._cached_tro_state)
            for _ in range(POST_RESET_PHYSICS_STEPS):
                og.sim.step_physics()
                for entity in self.env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()

        oid = seg.get("object_id", "")
        if self.use_annotation_object_lookup and oid:
            tgt = _find_target_object_annotation(self.env, str(oid))
            dn = _annotation_display_name(str(oid)) or str(oid)
            all_matching = [(tgt, dn)] if tgt is not None else []
        else:
            all_matching = _find_all_matching_objects(
                self.env.scene, oid, task=getattr(self.env, "task", None)
            ) if oid else []
        target_obj = all_matching[0][0] if all_matching else None
        self.target = target_obj
        # Extract target object name from prompt for video display (e.g. "radio_89" or "radio receiver 1")
        prompt = seg.get("prompt", "")
        self.current_prompt = prompt
        self.target_obj_name = _target_display_name_from_segment(seg, prompt)
        if target_obj is None and seg.get("object_id"):
            logger.warning(f"Target object not found for object_id={seg['object_id']}, min_distance will be inf")
        min_distance = float("inf")
        initial_distance = float("inf")
        final_distance = float("inf")
        last_action_base_speed = float("inf")  # No action yet
        consecutive_low_speed_count = 0
        base_action_slice = ACTION_QPOS_INDICES["R1Pro"]["base"]

        video_writer = None
        if self.write_video and self.video_path and video_name:
            video_fpath = self.video_path / f"{video_name}.mp4"
            video_writer = create_video_writer(fpath=str(video_fpath), resolution=(448, 672))

        for step in range(max_steps):
            obs = self._preprocess_obs(obs)
            distances_for_video: list[tuple[str, float]] = []
            current_step_min = float("inf")
            for obj, display_name in all_matching:
                obj_pos = _get_object_position(obj)
                dist = _min_distance_arms_to_point(self.robot, obj_pos)
                distances_for_video.append((display_name, dist))
                min_distance = min(min_distance, dist)
                current_step_min = min(current_step_min, dist)
                if step == 0:
                    initial_distance = min(initial_distance, dist)
            final_distance = current_step_min

            if video_writer is not None:
                self._write_video_frame(obs, video_writer, distances_for_video)

            if (
                early_stop_on_reach_distance
                and min_distance < distance_threshold
                and step >= min_steps_before_reach_early_stop
            ):
                break

            action = self.policy.forward(obs)
            if isinstance(action, th.Tensor):
                action = action.cpu().numpy()
            action = np.asarray(np.squeeze(action), dtype=np.float32)
            # Base velocity: first 2 dims are x,y linear velocity
            base_vel = action[base_action_slice]
            last_action_base_speed = float(np.linalg.norm(base_vel[:2]))

            if last_action_base_speed < base_speed_threshold:
                consecutive_low_speed_count += 1
            else:
                consecutive_low_speed_count = 0

            # Early exit when base speed stays low (saves time; success still uses only distance criteria)
            if consecutive_low_speed_count >= consecutive_low_speed_steps:
                break

            obs, _, terminated, truncated, info = self.env.step(action, n_render_iterations=1)
            if terminated or truncated:
                break

        # Success: (1) distance < threshold, (2) final < initial (moved closer)
        success = min_distance < distance_threshold and final_distance < initial_distance

        if video_writer is not None:
            container, stream = video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()

        return {
            "success": success,
            "min_distance": float(min_distance),
            "steps": step + 1,
            "initial_distance": float(initial_distance),
            "final_distance": float(final_distance),
            "last_action_base_speed": float(last_action_base_speed),
        }


def _sanitize_for_json(obj):
    """Replace inf/nan with None for strict JSON (e.g. per-segment jsonl)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _normalize_eval_result(res: dict) -> None:
    """Ensure distance and speed fields exist for logging and metrics."""
    for k in ("min_distance", "final_distance", "initial_distance", "last_action_base_speed"):
        if k not in res:
            res[k] = float("inf")
        else:
            try:
                res[k] = float(res[k])
            except (TypeError, ValueError):
                res[k] = float("inf")
    if "steps" not in res:
        res["steps"] = 0


def _format_dist_m(x) -> str:
    if x is None:
        return "inf"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "?"
    if math.isinf(xf):
        return "inf"
    if math.isnan(xf):
        return "nan"
    return f"{xf:.4f}"


def _log_eval_segment_summary(run_idx: int, total: int, seg: dict, res: dict, *, jsonl_f) -> None:
    """Append one jsonl record and write a structured line to eval.log (via logger)."""
    _normalize_eval_result(res)
    prompt = seg.get("prompt", "") or ""
    record = {
        "run_index": run_idx,
        "total_in_run": total,
        "task_name": res.get("task_name"),
        "scene": res.get("scene"),
        "room_key": res.get("room_key"),
        "prompt": prompt,
        "success": bool(res.get("success")),
        "distance_final_m": res.get("final_distance"),
        "distance_min_m": res.get("min_distance"),
        "distance_initial_m": res.get("initial_distance"),
        "steps": int(res.get("steps", 0)),
        "skill_str": res.get("skill_str", ""),
        "object_id": seg.get("object_id"),
        "error": res.get("error"),
        "last_action_base_speed": res.get("last_action_base_speed"),
    }
    jsonl_f.write(json.dumps(_sanitize_for_json(record), ensure_ascii=False) + "\n")
    jsonl_f.flush()
    logger.info(
        "[segment %s/%s] success=%s | final_dist_m=%s min_dist_m=%s initial_dist_m=%s | steps=%s | task=%s | prompt=%s",
        run_idx,
        total,
        res.get("success"),
        _format_dist_m(res.get("final_distance")),
        _format_dist_m(res.get("min_distance")),
        _format_dist_m(res.get("initial_distance")),
        res.get("steps", 0),
        res.get("task_name", ""),
        prompt,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skill eval from skills_flat segments")
    parser.add_argument("--segments", type=str, default="skill_segments_flat.json")
    parser.add_argument("--policy_url", type=str, required=True)
    parser.add_argument("--max_steps_per_segment", type=int, default=DEFAULT_MAX_STEPS_PER_SEGMENT)
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=MOVE_TO_SUCCESS_DISTANCE_THRESHOLD,
        help="Min distance (m) for success and for reach-based early stop (only after --min_steps_before_reach_early_stop). Default 1.0.",
    )
    parser.add_argument(
        "--no_early_stop_on_reach_distance",
        action="store_true",
        help="Do not stop when min distance drops below --distance_threshold; run up to --max_steps_per_segment unless other early exits fire.",
    )
    parser.add_argument(
        "--min_steps_before_reach_early_stop",
        type=int,
        default=MOVE_TO_MIN_STEPS_BEFORE_REACH_EARLY_STOP,
        help=(
            "Run at least this many loop iterations before distance-based early stop is allowed; until then, continue even "
            f"if already within --distance_threshold. Default {MOVE_TO_MIN_STEPS_BEFORE_REACH_EARLY_STOP}."
        ),
    )
    parser.add_argument(
        "--base_speed_threshold",
        type=float,
        default=MOVE_TO_SUCCESS_BASE_SPEED_THRESHOLD,
        help="Base velocity (x,y norm) threshold for early episode exit only (not used in success criterion)",
    )
    parser.add_argument(
        "--consecutive_low_speed_steps",
        type=int,
        default=MOVE_TO_SUCCESS_CONSECUTIVE_LOW_SPEED_STEPS,
        help="Consecutive low base-speed steps before early exit (saves time; not used in success criterion)",
    )
    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--env_wrapper",
        type=str,
        default="omnigibson.learning.wrappers.RGBLowResWrapper",
        help="Env wrapper class path (must be class, not module). E.g. omnigibson.learning.wrappers.RGBLowResWrapper",
    )
    parser.add_argument("--no_env_wrapper", action="store_true", help="Disable env_wrapper (no observation wrapper)")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--no_write_video", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--online_sampling", action="store_true", help="Use BDDL online_object_sampling (random scene from BDDL)")
    parser.add_argument("--no_online_sampling", action="store_true", help="Use pre-sampled tro_state instead of online sampling; keeps prompt from skills_flat. Use with --move_to_only --skills_flat.")
    parser.add_argument(
        "--random_instance",
        action="store_true",
        help=(
            "Per segment, pick a random activity_instance_id that has a matching "
            "{scene}_task_{task}_{def}_{id}_template-tro_state.json (only IDs on disk). "
            "Each segment can get a different id; load_task_instance loads that id's tro. "
            "Works with --move_to_segments_json (overrides instance_id in JSON) and --no_online_sampling. "
            "Use --seed for reproducible picks."
        ),
    )
    parser.add_argument("--move_to_only", action="store_true", help="Only test 'move to' skill, scene from BDDL random sampling (or tro_state when --no_online_sampling)")
    parser.add_argument("--bddl_segments", action="store_true", help="Generate segments from BDDL (no skill_segments_flat.json needed); use with --move_to_only")
    parser.add_argument(
        "--skills_flat",
        type=str,
        default=None,
        help="With --move_to_only: load move-to rows from this jsonl, map object_id to BDDL (implies --bddl_segments). Add --no_online_sampling for pre-sampled tro_state.",
    )
    parser.add_argument("--max_objects_per_activity", type=int, default=5, help="Max objects per BDDL activity when using --bddl_segments")
    parser.add_argument("--total_segments", type=int, default=None, help="User-specified number of tests; randomly sample this many segments")
    parser.add_argument("--skills", type=str, nargs="*", default=None, help="Only these skills, e.g. 'move to'")
    parser.add_argument("--task", type=str, nargs="*", default=None, help="Only these tasks (BDDL name), e.g. turning_on_radio")
    parser.add_argument("--scene", type=str, nargs="*", default=None, help="Only these scenes, e.g. house_double_floor_lower")
    parser.add_argument("--room", type=str, nargs="*", default=None, help="Only these rooms (room_key or substring), e.g. living_room")
    parser.add_argument("--partial_scene_load", action="store_true", help="Load only task-relevant rooms (faster, less memory). Aligned with eval.py partial_scene_load. Not used when online_sampling.")
    parser.add_argument(
        "--move_to_segments_json",
        type=str,
        default=None,
        help=(
            "Path to move_to_segments.json (task_name, instance_id, object_id, prompt). "
            "Scene = task template + tro_state for that instance, same as "
            "data_generation/rft/generate_move_to_rft_data.load_env_for_task_instance. "
            "Implies --move_to_only --no_online_sampling, annotation-style target lookup, and (unless --full_scene_load) partial rooms. "
            "Add --random_instance to ignore JSON instance_id and sample a valid tro-backed id per segment."
        ),
    )
    parser.add_argument(
        "--full_scene_load",
        action="store_true",
        help="With --move_to_segments_json: load full scene instead of partial task rooms (RFT default uses partial).",
    )
    parser.add_argument(
        "--shuffle_segments",
        action="store_true",
        help="Randomly shuffle segment order before eval. Default: same-task move-to segments run in file order, "
        "tasks batched in order of first appearance (no global shuffle).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only load segments and print, do not run eval")
    parser.add_argument(
        "--use_skill_prompt_format",
        action="store_true",
        help="Use BehaviorLeRobotDataset skill-prompt lines (Skill:/Objects:/Context:/Spatial:, no Goal). "
        "Omit this flag for imperative prompts like 'move to radio' (dataset default use_skill_prompt_format=False).",
    )
    args = parser.parse_args()

    if args.skills_flat and not args.bddl_segments and not args.move_to_segments_json:
        args.bddl_segments = True
        logger.info(
            "--skills_flat was set without --bddl_segments; enabling --bddl_segments so object_ids are mapped to BDDL "
            "(annotation ids like cutting_board_76 do not exist under BDDL online sampling)."
        )

    if args.move_to_only:
        args.skills = ["move to", "move to object"]
        if not args.no_online_sampling:
            args.online_sampling = True
    if args.bddl_segments and not args.move_to_only:
        args.move_to_only = True
        args.skills = ["move to", "move to object"]
        if not args.no_online_sampling:
            args.online_sampling = True

    if args.move_to_segments_json:
        args.move_to_only = True
        args.skills = ["move to", "move to object"]
        args.no_online_sampling = True
        args.online_sampling = False
        if not args.full_scene_load:
            args.partial_scene_load = True

    if args.dry_run:
        if args.move_to_segments_json:
            mpath = Path(args.move_to_segments_json)
            if not mpath.exists():
                logger.error(f"move_to_segments_json not found: {mpath}")
                sys.exit(1)
            segments = _load_move_to_segments_json(mpath)
            _finalize_move_to_segments_for_eval(segments)
            if args.task:
                task_set = {t.lower().replace(" ", "_") for t in args.task}
                segments = [s for s in segments if s["task_name"].lower().replace(" ", "_") in task_set]
        elif args.bddl_segments:
            task_filter = [t.lower().replace(" ", "_") for t in args.task] if args.task else None
            if args.skills_flat:
                skills_flat_path = Path(args.skills_flat)
                if skills_flat_path.exists():
                    segments = _load_segments_from_skills_flat_for_bddl(
                        skills_flat_path,
                        TASK_MAPPING_PATH,
                        tasks=task_filter,
                        use_skill_prompt_format=args.use_skill_prompt_format,
                        offline=not args.online_sampling,
                    )
                else:
                    logger.error(f"skills_flat not found: {skills_flat_path}")
                    sys.exit(1)
            else:
                available_tasks = load_available_tasks()
                segments = extract_segments_from_bddl(
                    available_tasks,
                    tasks=task_filter,
                    max_objects_per_activity=args.max_objects_per_activity,
                    use_skill_prompt_format=args.use_skill_prompt_format,
                    move_to_skill_str="move to",
                )
        else:
            segments_path = Path(args.segments)
            if not segments_path.exists():
                logger.error(f"Segments file not found: {segments_path}")
                sys.exit(1)
            with open(segments_path) as f:
                segments = json.load(f)
            for seg in segments:
                seg["prompt"] = prompt_for_eval_segment(
                    seg, use_skill_prompt_format=args.use_skill_prompt_format
                )
            if args.move_to_only or args.skills:
                segments = [s for s in segments if s.get("skill_str") in (args.skills or list(MOVE_TO_SKILL_ALIASES))]
            if args.task:
                task_set = {t.lower().replace(" ", "_") for t in args.task}
                segments = [s for s in segments if s["task_name"].lower().replace(" ", "_") in task_set]
        task_to_scene_room = _load_task_scene_room_mapping(TASK_SCENE_ROOMS_MAPPING_PATH)
        if args.scene or args.room:
            allowed_tasks = _get_allowed_tasks_by_scene_rooms(
                scenes=args.scene,
                rooms=args.room,
                mapping_path=TASK_SCENE_ROOMS_MAPPING_PATH,
            )
            if allowed_tasks is not None:
                task_set = {t.lower().replace(" ", "_") for t in allowed_tasks}
                segments = [
                    s for s in segments
                    if s["task_name"].lower().replace(" ", "_") in task_set
                ]
        if not segments:
            logger.error("No segments after filter (check --task/--scene/--room)")
            sys.exit(1)
        if args.total_segments is not None or getattr(args, "random_instance", False):
            if args.seed is not None:
                random.seed(args.seed)
        if args.total_segments is not None:
            if args.shuffle_segments:
                random.shuffle(segments)
            else:
                segments = _order_segments_sequential_per_task(segments)
            segments = segments[: args.total_segments]
        for i, s in enumerate(segments[:15]):
            scene, room_key = task_to_scene_room.get(s["task_name"].lower().replace(" ", "_"), ("unknown", "unknown"))
            print(f"  [{i}] [{scene}][{room_key}] {s['task_name']} | {s['skill_str']} | {s.get('object_id','')} | {s['prompt'][:50]}...")
        by_sr = defaultdict(int)
        for s in segments:
            scene, room_key = task_to_scene_room.get(s["task_name"].lower().replace(" ", "_"), ("unknown", "unknown"))
            by_sr[f"[{scene}][{room_key}]"] += 1
        print(f"\nTotal: {len(segments)} segments (showing first 15)")
        print("By scene+room:", dict(by_sr))
        if args.move_to_only:
            mode = "no_online_sampling (tro_state)" if args.no_online_sampling else "online_sampling=True"
            print(f"Mode: move_to_only ({mode})")
        if args.move_to_segments_json:
            print("Mode: move_to_segments.json (task+instance_id+tro, annotation object_id; RFT pipeline)")
        if args.bddl_segments:
            print("Mode: bddl_segments (segments from BDDL, no file needed)")
        return

    write_video = not args.no_write_video
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"eval_skill_flat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = log_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    video_path = exp_dir / "video" if write_video else None
    metrics_dir = exp_dir / "metrics"
    log_dir_run = exp_dir / "log"
    if video_path:
        video_path.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_dir_run.mkdir(parents=True, exist_ok=True)

    log_file = log_dir_run / "eval.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    if args.move_to_only:
        base = "move_to_only"
        if args.no_online_sampling:
            base += "+no_online_sampling(tro_state)"
        if getattr(args, "random_instance", False):
            base += "+random_instance"
        if args.online_sampling:
            base += "+online_sampling"
        base += ("+bddl_segments" if args.bddl_segments else "") + ("+skills_flat" if (args.bddl_segments and args.skills_flat) else "")
        if args.move_to_segments_json:
            base += "+move_to_segments_rft"
        mode_info = base
    else:
        mode_info = f"online_sampling={args.online_sampling}"
    print(f"[eval_skill_flat] Experiment dir: {exp_dir} | {mode_info}", flush=True)

    segment_jsonl_path = log_dir_run / "eval_segments.jsonl"
    print(f"[eval_skill_flat] Per-segment log (prompt, distances, success): {segment_jsonl_path}", flush=True)

    if args.move_to_segments_json:
        mpath = Path(args.move_to_segments_json)
        if not mpath.exists():
            logger.error(f"move_to_segments_json not found: {mpath}")
            sys.exit(1)
        segments = _load_move_to_segments_json(mpath)
        _finalize_move_to_segments_for_eval(segments)
        if args.task:
            task_set = {t.lower().replace(" ", "_") for t in args.task}
            segments = [s for s in segments if s["task_name"].lower().replace(" ", "_") in task_set]
    elif args.bddl_segments:
        task_filter = [t.lower().replace(" ", "_") for t in args.task] if args.task else None
        if args.skills_flat:
            skills_flat_path = Path(args.skills_flat)
            if not skills_flat_path.exists():
                logger.error(f"skills_flat not found: {skills_flat_path}")
                sys.exit(1)
            segments = _load_segments_from_skills_flat_for_bddl(
                skills_flat_path,
                TASK_MAPPING_PATH,
                tasks=task_filter,
                use_skill_prompt_format=args.use_skill_prompt_format,
                offline=not args.online_sampling,
            )
        else:
            available_tasks = load_available_tasks()
            segments = extract_segments_from_bddl(
                available_tasks,
                tasks=task_filter,
                max_objects_per_activity=args.max_objects_per_activity,
                use_skill_prompt_format=args.use_skill_prompt_format,
                move_to_skill_str="move to",
            )
    else:
        segments_path = Path(args.segments)
        if not segments_path.exists():
            logger.error(f"Segments file not found: {segments_path}")
            sys.exit(1)
        with open(segments_path) as f:
            segments = json.load(f)
        for seg in segments:
            seg["prompt"] = prompt_for_eval_segment(
                seg, use_skill_prompt_format=args.use_skill_prompt_format
            )

    if args.skills and not args.bddl_segments:
        segments = [s for s in segments if s.get("skill_str") in args.skills]
    if args.task and not args.bddl_segments:
        task_set = {t.lower().replace(" ", "_") for t in args.task}
        segments = [s for s in segments if s["task_name"].lower().replace(" ", "_") in task_set]
    # Apply scene/room filter BEFORE taking total_segments, so we sample from matching segments
    task_to_scene_room = _load_task_scene_room_mapping(TASK_SCENE_ROOMS_MAPPING_PATH)
    if args.scene is not None or args.room is not None:
        allowed_tasks = _get_allowed_tasks_by_scene_rooms(
            scenes=args.scene,
            rooms=args.room,
            mapping_path=TASK_SCENE_ROOMS_MAPPING_PATH,
        )
        if allowed_tasks is not None:
            task_set = {t.lower().replace(" ", "_") for t in allowed_tasks}
            segments = [
                s for s in segments
                if s["task_name"].lower().replace(" ", "_") in task_set
            ]
    if args.seed is not None:
        random.seed(args.seed)
    if args.shuffle_segments:
        random.shuffle(segments)
    else:
        segments = _order_segments_sequential_per_task(segments)
    if args.total_segments is not None:
        segments = segments[: args.total_segments]
    elif args.max_segments:
        segments = segments[: args.max_segments]
    msg = f"[eval_skill_flat] Loaded {len(segments)} segments"
    if args.move_to_segments_json:
        msg += " from move_to_segments.json (tro per instance_id)"
    elif args.bddl_segments:
        msg += " from BDDL" + (" (filtered by skills_flat)" if args.skills_flat else "")
    if args.total_segments is not None:
        msg += f", user-specified total_segments={args.total_segments}"
    print(msg, flush=True)

    if args.move_to_only and not args.bddl_segments and segments and not args.move_to_segments_json:
        for s in segments[: min(32, len(segments))]:
            oid = s.get("object_id", "")
            if isinstance(oid, str) and _OBJECT_ID_LIKE.match(oid):
                logger.warning(
                    "Segments contain annotation-style object_id=%r but --move_to_only uses BDDL online_sampling; "
                    "those names are not in task.object_scope. Use --bddl_segments [--skills_flat scripts/skills_flat.jsonl] "
                    "or only use segments whose object_id is BDDL (e.g. chopping_board.n.01_1).",
                    oid,
                )
                break

    gm.HEADLESS = True

    # env_wrapper: same pattern as official eval.py (env_wrapper._target_=...)
    if args.no_env_wrapper:
        env_wrapper_cfg = None
    else:
        try:
            from omegaconf import OmegaConf
            env_wrapper_cfg = OmegaConf.create({"_target_": args.env_wrapper})
            logger.info(f"[eval_skill_flat] Using env_wrapper: {args.env_wrapper}")
        except Exception as e:
            logger.warning(f"[eval_skill_flat] Failed to create env_wrapper ({args.env_wrapper}): {e}, running without wrapper")
            env_wrapper_cfg = None

    evaluator = SkillEvaluator(
        policy_url=args.policy_url,
        env_wrapper_cfg=env_wrapper_cfg,
        write_video=write_video,
        video_path=video_path,
        online_sampling=args.online_sampling,
        partial_scene_load=args.partial_scene_load,
        rft_style_tro=bool(args.move_to_segments_json),
        use_annotation_object_lookup=bool(args.move_to_segments_json),
    )
    results_list = []
    n_success = 0

    segments_by_scene_room = defaultdict(list)
    for i, seg in enumerate(segments):
        task_name = seg["task_name"]
        scene, room_key = task_to_scene_room.get(task_name.lower().replace(" ", "_"), ("unknown", "unknown"))
        segments_by_scene_room[(scene, room_key)].append((i, seg))

    total_segments = sum(len(s) for s in segments_by_scene_room.values())
    if args.task or args.scene or args.room:
        filters = []
        if args.task:
            filters.append(f"task={args.task}")
        if args.scene:
            filters.append(f"scene={args.scene}")
        if args.room:
            filters.append(f"room={args.room}")
        print(f"[eval_skill_flat] Filtered by {', '.join(filters)}: {total_segments} segments", flush=True)
    if total_segments == 0:
        logger.error("No segments to evaluate (check --task/--scene/--room filters)")
        sys.exit(1)

    processed_count = 0
    new_scene_per_segment = args.bddl_segments and args.skills_flat and args.online_sampling
    segment_jsonl_f = open(segment_jsonl_path, "w", encoding="utf-8")
    try:
        for (scene, room_key), scene_room_segments in sorted(segments_by_scene_room.items()):
            scene_room_segments.sort(key=lambda x: x[0])
            last_task = None
            available_tasks = load_available_tasks()
            for seg_idx, seg in scene_room_segments:
                task_name = seg["task_name"]
                instance_id = seg.get("instance_id", 0)
                if task_name not in available_tasks:
                    logger.warning(f"Skipping task {task_name}")
                    continue
                if getattr(args, "random_instance", False) and not evaluator.online_sampling:
                    task_cfg = available_tasks[task_name][0]
                    scene_model = task_cfg.get("scene_model", "Scene")
                    adid = int(task_cfg.get("activity_definition_id", 0))
                    avail_ids = discover_instance_ids(task_name, scene_model, activity_definition_id=adid)
                    if avail_ids:
                        instance_id = random.choice(avail_ids)
                        seg["instance_id"] = instance_id
                if new_scene_per_segment:
                    # Each segment gets its own BDDL-sampled scene
                    if evaluator.env is not None:
                        evaluator.env.close()
                    print(f"[eval_skill_flat] [{scene}][{room_key}] Task {task_name}: loading env (segment {processed_count + 1}), new scene", flush=True)
                    evaluator.load_env(task_name, max_steps=args.max_steps_per_segment + 100)
                elif task_name != last_task:
                    if last_task is not None and evaluator.env is not None:
                        evaluator.env.close()
                    n_segs = sum(1 for _, s in scene_room_segments if s["task_name"] == task_name)
                    print(f"[eval_skill_flat] [{scene}][{room_key}] Task {task_name}: loading env, {n_segs} segments", flush=True)
                    evaluator.load_env(task_name, max_steps=args.max_steps_per_segment + 100)
                    last_task = task_name
                print(f"[eval_skill_flat] Running {processed_count + 1}/{total_segments}...", flush=True)
                vid_suffix = seg.get("episode_index", seg.get("object_id", 0))
                if isinstance(vid_suffix, str):
                    vid_suffix = vid_suffix.replace(".", "_")
                video_name = f"{task_name}_{seg_idx}_{vid_suffix}" if write_video else None
                try:
                    res = evaluator.run_segment(
                        seg,
                        max_steps=args.max_steps_per_segment,
                        distance_threshold=args.distance_threshold,
                        video_name=video_name,
                        base_speed_threshold=args.base_speed_threshold,
                        consecutive_low_speed_steps=args.consecutive_low_speed_steps,
                        early_stop_on_reach_distance=not args.no_early_stop_on_reach_distance,
                        min_steps_before_reach_early_stop=args.min_steps_before_reach_early_stop,
                    )
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    logger.warning(f"Segment failed: {err_msg}")
                    res = {
                        "success": False,
                        "min_distance": float("inf"),
                        "initial_distance": float("inf"),
                        "final_distance": float("inf"),
                        "last_action_base_speed": float("inf"),
                        "steps": 0,
                        "error": err_msg,
                    }
                res["segment_idx"] = seg_idx
                res["task_name"] = task_name
                res["scene"] = scene
                res["room_key"] = room_key
                res["skill_str"] = seg.get("skill_str", "")
                res["prompt"] = seg.get("prompt", "")
                res["instance_id"] = seg.get("instance_id", 0)
                _normalize_eval_result(res)
                results_list.append(res)
                if res["success"]:
                    n_success += 1
                processed_count += 1
                sr = n_success / processed_count if processed_count else 0.0
                _log_eval_segment_summary(
                    processed_count, total_segments, seg, res, jsonl_f=segment_jsonl_f
                )
                prompt_txt = seg.get("prompt", "") or ""
                print(
                    f"[eval_skill_flat] [{processed_count}/{total_segments}] success={res['success']} | "
                    f"final_dist_m={_format_dist_m(res.get('final_distance'))} "
                    f"min_dist_m={_format_dist_m(res.get('min_distance'))} "
                    f"initial_dist_m={_format_dist_m(res.get('initial_distance'))} | "
                    f"SR={sr:.1%}",
                    flush=True,
                )
                print(f"  prompt: {prompt_txt}", flush=True)
            if evaluator.env is not None:
                evaluator.env.close()
    finally:
        segment_jsonl_f.close()

    by_scene_room = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results_list:
        key = f"[{r.get('scene', 'unknown')}][{r.get('room_key', 'unknown')}]"
        by_scene_room[key]["total"] += 1
        if r.get("success"):
            by_scene_room[key]["success"] += 1
    for v in by_scene_room.values():
        v["success_rate"] = v["success"] / v["total"] if v["total"] else 0.0

    results = {
        "total": len(results_list),
        "success": n_success,
        "success_rate": n_success / len(results_list) if results_list else 0.0,
        "by_scene_room": dict(by_scene_room),
        "results": results_list,
    }
    print(f"[eval_skill_flat] Done: {results['success']}/{results['total']} ({results['success_rate']:.2%})", flush=True)
    for key, sr_data in sorted(results["by_scene_room"].items()):
        print(f"  {key}: {sr_data['success']}/{sr_data['total']} ({sr_data['success_rate']:.1%})", flush=True)

    output_path = Path(args.output) if args.output else metrics_dir / "skill_eval_results_flat.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval_skill_flat] Saved to {output_path}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
