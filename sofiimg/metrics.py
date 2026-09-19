"""Quality measures applied to segment rankings.

Three properties are reported. Faithfulness and sparsity come from the
degradation score and from the sparsity rate, both computed by the objective.
Robustness is measured here, through the agreement between the ranking of an
instance and the ranking of its nearest neighbor.

Robustness rests on the assumption that visually similar images deserve similar
explanations. That assumption is common in the literature, yet it is not
correlated with the other two properties, and a method can be perfectly stable
while being unfaithful. Multimodal explanation spaces, where several distinct
rankings are equally valid, also escape the measure. The number is therefore
informative alongside the others and misleading on its own.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "rank_biased_overlap",
    "robustness_score",
    "nearest_neighbors",
    "fold_change",
]


def rank_biased_overlap(
    first: Sequence, second: Sequence, p: float = 0.8, extrapolate: bool = True
) -> float:
    """Similarity of two rankings, weighted toward the leading positions.

    Rank biased overlap compares the two lists prefix by prefix and discounts
    each depth geometrically, so agreement among the most relevant segments
    counts far more than agreement in the tail. A value of one denotes identical
    rankings and a value of zero denotes disjoint ones.

    Parameters
    ----------
    first, second : sequence
        The two rankings, from the most to the least relevant.
    p : float, default=0.8
        Persistence of the geometric weights. The expected evaluation depth is
        ``1 / (1 - p)``, so the default looks about five positions deep, which
        is the setting used in the paper.
    extrapolate : bool, default=True
        Whether the tail beyond the shared depth is extrapolated, which is the
        correction that makes the measure meaningful for finite lists.

    Returns
    -------
    float
        Similarity in the unit interval.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between zero and one.")
    first, second = list(first), list(second)
    if not first or not second:
        return 0.0

    short, long = (first, second) if len(first) <= len(second) else (second, first)
    depth_short, depth_long = len(short), len(long)

    # overlaps[d] counts the elements shared by the two prefixes of depth d
    seen_short, seen_long = set(), set()
    overlaps = np.zeros(depth_long + 1, dtype=float)
    for depth in range(1, depth_long + 1):
        if depth <= depth_short:
            seen_short.add(short[depth - 1])
        seen_long.add(long[depth - 1])
        overlaps[depth] = len(seen_short & seen_long)

    if not extrapolate:
        total = sum(
            (overlaps[depth] / depth) * (p ** (depth - 1))
            for depth in range(1, depth_long + 1)
        )
        return float(min(max((1.0 - p) * total, 0.0), 1.0))

    # extrapolated form of Webber, Moffat and Zobel, which assumes that the
    # agreement observed at the deepest common position continues below it
    visible = sum(
        (overlaps[depth] / depth) * (p ** depth)
        for depth in range(1, depth_long + 1)
    )
    unseen = sum(
        (overlaps[depth_short] * (depth - depth_short) / (depth_short * depth))
        * (p ** depth)
        for depth in range(depth_short + 1, depth_long + 1)
    )
    tail = (
        (overlaps[depth_long] - overlaps[depth_short]) / depth_long
        + overlaps[depth_short] / depth_short
    ) * (p ** depth_long)
    score = ((1.0 - p) / p) * (visible + unseen) + tail
    return float(min(max(score, 0.0), 1.0))


def nearest_neighbors(X: np.ndarray, standardize: bool = True) -> np.ndarray:
    """Position of the closest other instance, under a Euclidean distance.

    Parameters
    ----------
    X : ndarray
        Images shaped ``(n, height, width, n_channels)``.
    standardize : bool, default=True
        Whether every image is centered and scaled before the distance is
        computed, which removes brightness and contrast from the comparison.
    """
    data = np.asarray(X, dtype=float).reshape(len(X), -1)
    if standardize:
        center = data.mean(axis=1, keepdims=True)
        spread = data.std(axis=1, keepdims=True)
        spread[spread == 0] = 1.0
        data = (data - center) / spread
    squared = np.sum(data ** 2, axis=1)
    distances = squared[:, None] + squared[None, :] - 2.0 * (data @ data.T)
    np.fill_diagonal(distances, np.inf)
    return np.argmin(distances, axis=1)


def robustness_score(
    rankings: Sequence[Sequence],
    X: np.ndarray,
    p: float = 0.8,
    standardize: bool = True,
) -> float:
    """Average agreement between the ranking of an instance and that of its neighbor.

    Parameters
    ----------
    rankings : sequence of sequence
        One ranking per image, aligned with the rows of ``X``.
    X : ndarray
        The images the rankings belong to.
    p : float, default=0.8
        Persistence handed to :func:`rank_biased_overlap`.
    standardize : bool, default=True
        Whether the raw images are standardized before the neighbor search.

    Returns
    -------
    float
        Mean similarity over the images, in the unit interval.
    """
    if len(rankings) != len(X):
        raise ValueError("One ranking per image is required.")
    if len(X) < 2:
        raise ValueError("At least two images are needed for a neighbor search.")
    neighbors = nearest_neighbors(X, standardize=standardize)
    scores = [
        rank_biased_overlap(rankings[position], rankings[neighbors[position]], p=p)
        for position in range(len(X))
    ]
    return float(np.mean(scores))


def fold_change(reference: float, other: float) -> float:
    """How many times better a reference value is than a competing one.

    The paper reports superiority this way, for instance sixteen times better on
    the degradation score. Division by a value at or below zero is undefined, so
    the function returns infinity in that case rather than a misleading number.

    Parameters
    ----------
    reference : float
        Value of the method under study.
    other : float
        Value of the method it is compared against.

    Returns
    -------
    float
        Their ratio, infinity when the comparison value is not positive and the
        reference is, and not a number when both vanish.
    """
    if other <= 0.0:
        return float("inf") if reference > 0.0 else float("nan")
    return float(reference / other)
