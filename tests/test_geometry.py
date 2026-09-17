import numpy as np

from src.clip_bias_geometry.geometry import (
    build_embedding_views,
    normalize_rows,
    remove_direction,
)
from src.clip_bias_geometry.metrics import absolute_gender_gap, signed_gender_gap


def test_normalize_rows_has_unit_norm():
    x = np.array([[3.0, 4.0], [5.0, 12.0]])
    normalized = normalize_rows(x)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_remove_direction_eliminates_projection():
    x = np.array([[1.0, 2.0], [3.0, -1.0]])
    direction = np.array([1.0, 0.0])
    projected_out = remove_direction(x, direction)
    np.testing.assert_allclose(projected_out[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(projected_out[:, 1], x[:, 1])


def test_gender_gap_metrics():
    male = np.array([0.4, 0.6])
    female = np.array([0.1, 0.3])
    assert np.isclose(signed_gender_gap(male, female), 0.3)
    assert np.isclose(absolute_gender_gap(male, female), 0.3)


def test_build_embedding_views_returns_expected_geometries():
    images = np.array([
        [1.0, 0.2, 0.5],
        [0.9, 0.1, 0.4],
        [-0.8, 0.1, 0.4],
        [-1.0, 0.2, 0.5],
    ])
    texts = np.array([[0.5, 0.5, 0.2], [-0.3, 0.8, 0.1]])
    genders = np.array(["Male", "Male", "Female", "Female"])

    views = build_embedding_views(images, texts, genders)

    assert set(views) == {
        "standard_cosine",
        "mean_centered",
        "gender_direction_removed",
    }
    for view in views.values():
        assert view["image"].shape == images.shape
        assert view["text"].shape == texts.shape
        np.testing.assert_allclose(
            np.linalg.norm(view["image"], axis=1), 1.0, atol=1e-10
        )
