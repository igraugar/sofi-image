"""Inspection of an image before it is explained.

An explanation is only meaningful when the instance is one the model handles
well and when the segmentation carries some structure. The report below puts
both conditions in front of the user before any search starts. It gathers three
groups of facts, namely the shape and statistics of the image, the confidence
and the class distribution the model assigns to it, and the geometry of the
partition together with a diagnosis of how well the segments follow the content.

The segment separation index deserves a word. It compares the variance that
remains inside the segments with the total variance of the image, in the manner
of a one-way analysis of variance. A value near one says the borders captured
almost every change of intensity, so the segments are homogeneous and the
partition is informative. A value near zero says the segments are no more
homogeneous than arbitrary patches would be, which usually points at a
compactness that is too large for the contrast of the image.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .segmentation import Segmentation

__all__ = ["describe_instance", "instance_report"]


def _image_statistics(instance: np.ndarray) -> Dict[str, float]:
    flat = instance.reshape(-1)
    rows = np.diff(instance, axis=0)
    cols = np.diff(instance, axis=1)
    gradient = float(np.mean(np.abs(rows))) if rows.size else 0.0
    if cols.size:
        gradient = 0.5 * (gradient + float(np.mean(np.abs(cols))))
    return {
        "minimum": float(np.min(flat)),
        "maximum": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "range": float(np.ptp(flat)),
        "mean_abs_gradient": gradient,
        "saturated_fraction": float(
            np.mean((flat <= np.min(flat) + 1e-9) | (flat >= np.max(flat) - 1e-9))
        ),
    }


def _region_separation(instance: np.ndarray, segmentation: Segmentation) -> float:
    """Share of the variance of the image explained by the segment means."""
    total, within = 0.0, 0.0
    for channel in range(instance.shape[-1]):
        plane = instance[..., channel]
        total += float(np.sum((plane - plane.mean()) ** 2))
        for segment in segmentation:
            values = plane[segment.rows, segment.cols]
            within += float(np.sum((values - values.mean()) ** 2))
    if total <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - within / total)))


def describe_instance(
    instance: np.ndarray,
    segmentation: Segmentation,
    probabilities: Optional[np.ndarray] = None,
    label: Optional[int] = None,
    class_names=None,
    true_label=None,
) -> Dict[str, object]:
    """Gather every fact the report prints, as a dictionary.

    Parameters
    ----------
    instance : array
        The image, shaped ``(height, width)`` or ``(height, width,
        n_channels)``.
    segmentation : Segmentation
        The partition to describe.
    probabilities : array, optional
        Class probabilities of the image. The section about the model is
        omitted when they are not given.
    label : int, optional
        Position of the explained class. The most probable one is used by
        default.
    class_names : sequence, optional
        Labels in the order of the probability columns, used for reporting.
    true_label : optional
        Ground truth label, which turns on the correctness verdict.

    Returns
    -------
    dict
        The same information :func:`instance_report` formats, so the values can
        be logged or tabulated instead of printed.
    """
    instance = np.asarray(instance, dtype=float)
    if instance.ndim == 2:
        instance = instance[..., None]
    areas = segmentation.areas

    facts: Dict[str, object] = {
        "height": int(instance.shape[0]),
        "width": int(instance.shape[1]),
        "n_channels": int(instance.shape[2]),
        "image": _image_statistics(instance),
        "segmentation": {
            "method": segmentation.method,
            "n_segments": segmentation.n_segments,
            "smallest": int(areas.min()),
            "largest": int(areas.max()),
            "mean_area": float(areas.mean()),
            "coverage": segmentation.coverage,
            "separation": _region_separation(instance, segmentation),
            "params": segmentation.params,
        },
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        position = int(np.argmax(probabilities)) if label is None else int(label)
        ordered = np.argsort(-probabilities)
        runner_up = int(ordered[1]) if probabilities.size > 1 else position
        names = (
            [str(name) for name in class_names]
            if class_names is not None
            else [str(k) for k in range(probabilities.size)]
        )
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)))
        )
        facts["model"] = {
            "predicted": names[position],
            "probability": float(probabilities[position]),
            "runner_up": names[runner_up],
            "runner_up_probability": float(probabilities[runner_up]),
            "margin": float(probabilities[position] - probabilities[runner_up]),
            "entropy": entropy,
            "normalized_entropy": entropy / float(np.log(max(probabilities.size, 2))),
            "true_label": None if true_label is None else str(true_label),
            "correct": None
            if true_label is None
            else bool(str(true_label) == names[position]),
        }
    return facts


def instance_report(facts: Dict[str, object], width: int = 78) -> str:
    """Format the facts collected by :func:`describe_instance` for a console.

    Parameters
    ----------
    facts : dict
        The dictionary returned by :func:`describe_instance`.
    width : int, default=78
        Width of the rules that separate the sections.

    Returns
    -------
    str
        The report, ready to be printed.
    """
    image = facts["image"]
    seg = facts["segmentation"]
    lines = ["=" * width]
    lines.append("Image under study")
    lines.append("=" * width)

    lines.append(
        f"  shape          {facts['height']} by {facts['width']} pixels, "
        f"{facts['n_channels']} channel(s)"
    )
    lines.append(
        f"  range          [{image['minimum']:.3f}, {image['maximum']:.3f}], "
        f"span {image['range']:.3f}"
    )
    lines.append(
        f"  center         mean {image['mean']:.3f}, median {image['median']:.3f}, "
        f"deviation {image['std']:.3f}"
    )
    lines.append(
        f"  detail         mean absolute gradient {image['mean_abs_gradient']:.4f}, "
        f"{image['saturated_fraction'] * 100:.1f}% of pixels at an extreme"
    )

    if "model" in facts:
        model = facts["model"]
        lines.append("-" * width)
        verdict = ""
        if model["correct"] is not None:
            verdict = ", correct" if model["correct"] else ", MISCLASSIFIED"
        lines.append(
            f"  prediction     class {model['predicted']} at probability "
            f"{model['probability']:.4f}{verdict}"
        )
        lines.append(
            f"  runner up      class {model['runner_up']} at "
            f"{model['runner_up_probability']:.4f}, margin {model['margin']:.4f}"
        )
        lines.append(
            f"  uncertainty    entropy {model['entropy']:.4f}, normalized "
            f"{model['normalized_entropy']:.3f}"
        )
        if model["margin"] < 0.1:
            lines.append(
                "  note           the decision is close to a boundary, so the "
                "explanation may be unstable"
            )
        if model["correct"] is False:
            lines.append(
                "  note           SOFI assumes the prediction is one the model "
                "handles well, and this image is misclassified"
            )

    lines.append("-" * width)
    lines.append(
        f"  segmentation   {seg['method']}, {seg['n_segments']} segments covering "
        f"{seg['coverage'] * 100:.0f}% of the image"
    )
    lines.append(
        f"  areas          smallest {seg['smallest']}, largest {seg['largest']}, "
        f"mean {seg['mean_area']:.1f} pixels"
    )
    lines.append(
        f"  separation     {seg['separation']:.3f} of the variance is explained by "
        "the segment means"
    )
    if seg["separation"] < 0.2:
        lines.append(
            "  note           the segments are barely more homogeneous than "
            "arbitrary patches, so consider a smaller compactness"
        )
    if seg["n_segments"] < 2:
        lines.append(
            "  note           a single segment leaves nothing to rank, so the "
            "explanation will be trivial"
        )
    if seg["n_segments"] > 80:
        lines.append(
            "  note           many segments make the search slower and the "
            "explanation harder to read"
        )
    lines.append("=" * width)
    return "\n".join(lines)
