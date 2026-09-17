"""Reusable utilities for the CLIP gender-bias geometry experiments."""

from .geometry import build_embedding_views, normalize_rows, remove_direction
from .metrics import absolute_gender_gap, signed_gender_gap
from .prompts import OCCUPATIONS, PROMPT_TEMPLATES, fill_prompt

__all__ = [
    "build_embedding_views",
    "normalize_rows",
    "remove_direction",
    "absolute_gender_gap",
    "signed_gender_gap",
    "OCCUPATIONS",
    "PROMPT_TEMPLATES",
    "fill_prompt",
]
