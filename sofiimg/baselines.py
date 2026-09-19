"""Comparison of SOFI against attribution methods that score pixels.

Gradient methods, class activation maps and occlusion produce one relevance
value per pixel, whereas SOFI produces an order of segments. The two become
comparable once the pixel-wise scores are averaged inside each segment and the
resulting values are sorted, which is the alignment the paper applies before
every comparison. The helpers below perform that alignment and nothing else, so
any attribution library can feed them, whether it produces a saliency map at the
resolution of the image or an attention map at the resolution of the tokens of a
transformer, once that map has been resized.

Feature occlusion is implemented here, since it needs only the classifier and it
is the standard perturbation baseline. Gradient methods are not, since they
require access to the internals of the model, and their maps enter through
:func:`segment_scores` like any other pixel-wise attribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .segmentation import Segmentation

__all__ = [
    "segment_scores",
    "ranking_from_scores",
    "feature_occlusion",
    "random_ranking",
    "compare_rankings",
]


def segment_scores(pixelwise, segmentation: Segmentation, reduce: str = "mean") -> np.ndarray:
    """Collapse a pixel-wise saliency map onto one value per segment.

    Parameters
    ----------
    pixelwise : array
        Relevance of every pixel, shaped ``(height, width)`` or shaped like the
        image, in which case the channels are averaged first.
    segmentation : Segmentation
        Partition the values are pooled over.
    reduce : {"mean", "sum", "max", "absmean"}, default="mean"
        How the values inside a segment are combined. The mean is the choice of
        the paper, since it does not favor large segments, and it is what keeps
        a segment partition of uneven areas comparable.

    Returns
    -------
    ndarray
        One score per segment, in the order of the segmentation.
    """
    values = np.asarray(pixelwise, dtype=float)
    if values.ndim == 3:
        values = values.mean(axis=-1)
    if values.ndim != 2:
        raise ValueError(
            "A saliency map must be shaped (height, width) or like the image "
            f"itself, and the array received has shape {values.shape}."
        )
    if values.shape != segmentation.shape:
        raise ValueError(
            f"The saliency map covers {values.shape} pixels and the "
            f"segmentation covers {segmentation.shape}. Both must describe the "
            "same image for the pooling to be meaningful."
        )
    functions = {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "absmean": lambda window: np.mean(np.abs(window)),
    }
    if reduce not in functions:
        raise ValueError(f"reduce must be one of {sorted(functions)}.")
    collapse = functions[reduce]
    return np.array(
        [float(collapse(values[s.rows, s.cols])) for s in segmentation],
        dtype=float,
    )


def ranking_from_scores(scores, descending: bool = True) -> list:
    """Order the segments by an attribution score.

    Higher relevance means greater importance under the usual convention, so the
    default sorts in descending order. Methods whose scores fall as relevance
    grows, such as the probability left after an occlusion, need the ascending
    order instead.

    Parameters
    ----------
    scores : sequence of float
        One score per segment, in the order of the segmentation.
    descending : bool, default=True
        Whether a larger score means a more important segment.

    Returns
    -------
    list of int
        Positions of the segments, from the most to the least important.
    """
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values if descending else values, kind="stable")
    return [int(position) for position in order]


def feature_occlusion(
    explainer,
    instance,
    segmentation: Optional[Segmentation] = None,
    marginalization=None,
    label: Optional[int] = None,
) -> np.ndarray:
    """Relevance of every segment under feature occlusion.

    Each segment is marginalized on its own and the drop in the probability of
    the explained class becomes its score. The method is the standard
    perturbation baseline, and it is exactly the reference sweep SOFI starts
    from, which makes the contrast between a scored ranking and an optimized one
    direct.

    Parameters
    ----------
    explainer : SOFIExplainer
        The explainer whose segmentation, operator and model define the
        protocol, so that the scores are comparable with a SOFI ranking.
    instance : array
        The image to score.
    segmentation : Segmentation, optional
        Partition to score. The one of the explainer is used when omitted.
    marginalization : optional
        Operator for this call alone.
    label : int, optional
        Class whose probability is tracked. The predicted class is used by
        default.

    Returns
    -------
    ndarray
        One drop per segment, in the order of the segmentation.

    Examples
    --------
    >>> scores = feature_occlusion(explainer, instance)
    >>> ranking = ranking_from_scores(scores)
    """
    segmentation = (
        explainer.segment(instance) if segmentation is None else segmentation
    )
    objective, _ = explainer.build_objective(
        instance, segmentation, marginalization=marginalization, label=label
    )
    return objective.single_segment_drops()


def random_ranking(n_segments: int, random_state=None) -> list:
    """A random order of the segments, the natural lower reference of a comparison.

    Parameters
    ----------
    n_segments : int
        Number of segments to permute.
    random_state : int or Generator, optional
        Seed of the permutation, so that a comparison is reproducible.

    Returns
    -------
    list of int
        A permutation of the segment positions.
    """
    rng = np.random.default_rng(random_state)
    return [int(position) for position in rng.permutation(int(n_segments))]


def compare_rankings(explainer, instance, rankings: dict, **kwargs):
    """Score several rankings of one instance under the SOFI protocol.

    Parameters
    ----------
    explainer : SOFIExplainer
        The explainer whose segmentation and operator define the protocol.
    instance : array
        The image every ranking refers to.
    rankings : dict
        Maps a method name onto an order of segment positions.
    **kwargs : dict
        Further arguments of ``score_ranking``, such as ``segmentation``,
        ``marginalization`` or ``label``, applied to every ranking so that they
        all meet the same protocol.

    Returns
    -------
    DataFrame
        One row per method with its degradation score and its sparsity rate,
        sorted from the most to the least faithful.
    """
    import pandas as pd

    segmentation = kwargs.pop("segmentation", None) or explainer.segment(instance)
    rows = []
    for name, order in rankings.items():
        scored = explainer.score_ranking(
            instance, order, segmentation=segmentation, **kwargs
        )
        rows.append(
            {
                "method": name,
                "ds": scored.ds,
                "auc_morf": scored.auc_morf,
                "auc_lerf": scored.auc_lerf,
                "sparsity_rate": scored.sparsity_rate,
                "top_segment": scored.ranking[0],
            }
        )
    return pd.DataFrame(rows).sort_values("ds", ascending=False).reset_index(drop=True)
