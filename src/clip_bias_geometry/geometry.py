"""Embedding-space transformations used in the CLIP bias study."""

from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a matrix row-wise."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def estimate_gender_direction(image_embeddings: np.ndarray, genders: np.ndarray) -> np.ndarray:
    """Estimate the linear male-minus-female direction from image embeddings."""
    x = np.asarray(image_embeddings, dtype=np.float64)
    g = np.asarray(genders)
    male = x[g == "Male"].mean(axis=0)
    female = x[g == "Female"].mean(axis=0)
    return male - female


def remove_direction(x: np.ndarray, direction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Remove each row's projection onto a single direction."""
    x = np.asarray(x, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    unit = direction / max(np.linalg.norm(direction), eps)
    return x - np.outer(x @ unit, unit)


def build_embedding_views(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    genders: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Construct the three embedding geometries used in the main experiment."""
    images = np.asarray(image_embeddings, dtype=np.float64)
    texts = np.asarray(text_embeddings, dtype=np.float64)
    image_mean = images.mean(axis=0, keepdims=True)
    gender_direction = estimate_gender_direction(images, genders)

    return {
        "standard_cosine": {
            "image": normalize_rows(images),
            "text": normalize_rows(texts),
        },
        "mean_centered": {
            "image": normalize_rows(images - image_mean),
            "text": normalize_rows(texts - image_mean),
        },
        "gender_direction_removed": {
            "image": normalize_rows(remove_direction(images, gender_direction)),
            "text": normalize_rows(remove_direction(texts, gender_direction)),
        },
    }
