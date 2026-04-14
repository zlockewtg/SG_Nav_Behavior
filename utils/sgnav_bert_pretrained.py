"""Resolve BERT hub id vs local directory for GLIP (fully offline when dir is complete)."""

from __future__ import annotations

import os

_BERT_HUB = "bert-base-uncased"


def resolve_bert_pretrained_id(hub_id: str) -> str:
    """If hub_id is bert-base-uncased and SGNAV_BERT_BASE_UNCASED_PATH points to a directory, use it."""
    if hub_id != _BERT_HUB:
        return hub_id
    p = os.environ.get("SGNAV_BERT_BASE_UNCASED_PATH", "").strip()
    if p and os.path.isdir(p):
        return p
    return hub_id
