"""Comparative studies over one image, gathered behind a single object.

Explaining one image is the work of the explainer, which returns an explanation
that draws itself. This module answers a different question, namely how the
settings should be chosen, and it answers it by varying one of them and watching
what moves, or by explaining a batch and summarizing it. Both are research
tasks, and neither stands on the path of a user who wants a single explanation.

Every study receives the image itself, returns a table and, on request, draws
one figure per setting. The tables carry the same columns throughout, namely the
degradation score, the fall of the response, its recovery at the end, the step
at which the predicted class changes and the cost in queries, so that two
studies can be read side by side.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .baselines import feature_occlusion, random_ranking, ranking_from_scores, segment_scores
from .perturbation import make_perturbation
from .plotting import plot_explanation, plot_segmentation_grid

__all__ = ["Experiment"]

_PALETTE = ("#3b6fb6", "#8e6fb6", "#b6763b", "#4f9e6a", "#b64f4f", "#1f9e9e")


def _settings(**values):
    """Keep the arguments that were given, so the rest fall back to the explainer."""
    return {name: value for name, value in values.items() if value is not None}


class Experiment:
    """Batch explanations and sensitivity studies over the settings.

    Parameters
    ----------
    explainer : SOFIExplainer
        The explainer under study. Its settings are the baseline every study
        departs from, and a study overrides one of them at a time.
    random_state : int or Generator, optional
        Seed of the random ranking used as the lower reference of a comparison.

    Attributes
    ----------
    explanations_ : dict
        Every explanation the last study produced, keyed by its setting.

    Examples
    --------
    >>> study = Experiment(explainer, random_state=42)
    >>> study.segment_counts(image, [10, 20, 40], mask=roi, display=original)
    >>> study.operators(image, ["mean", "blur", "lowneg"], mask=roi)
    >>> study.batch(X_test, y_test, masks=rois)
    """

    def __init__(self, explainer, random_state=None):
        self.explainer = explainer
        self.random_state = random_state
        self.explanations_: Dict[str, object] = {}

    # ------------------------------------------------------------ selection
    def _codes(self, values) -> np.ndarray:
        names = getattr(self.explainer.wrapper, "classes", None)
        if names is None:
            return np.asarray(values, dtype=int)
        lookup = {str(name): index for index, name in enumerate(names)}
        return np.array([lookup.get(str(v), -1) for v in np.asarray(values)], dtype=int)

    def reliable_instances(self, X, y, return_mask: bool = False):
        """Positions of the images the model classifies correctly.

        The premise of the method is that the response deteriorates as segments
        are marginalized, which holds only for an image the model handles well,
        so an explanation of a misclassified image describes an error rather
        than a decision.

        Parameters
        ----------
        X : array
            Images to test.
        y : array
            Their labels.
        return_mask : bool, default=False
            Whether the boolean mask over every image is returned as well.

        Returns
        -------
        ndarray
            Positions of the images that are classified correctly.
        """
        predicted = self.explainer.wrapper.predict_proba(np.asarray(X, dtype=float))
        mask = predicted.argmax(axis=1) == self._codes(y)
        positions = np.flatnonzero(mask)
        return (positions, mask) if return_mask else positions

    def confident_instances(self, X, y, label=None, k: Optional[int] = None):
        """Correctly classified images, ordered from the most confident.

        Parameters
        ----------
        X : array
            Images to test.
        y : array
            Their labels.
        label : optional
            Class the images must belong to. Every class is considered when the
            argument is left out.
        k : int, optional
            Number of positions returned.

        Returns
        -------
        list of int
            Positions into ``X``, from the most confident downwards.
        """
        data = np.asarray(X, dtype=float)
        positions = self.reliable_instances(data, y)
        if len(positions) == 0:
            return []
        proba = self.explainer.wrapper.predict_proba(data[positions])
        if label is not None:
            keep = np.array([str(v) == str(label) for v in np.asarray(y)[positions]])
            positions, proba = positions[keep], proba[keep]
        ranked = [int(positions[i]) for i in np.argsort(-proba.max(axis=1))]
        return ranked if k is None else ranked[: int(k)]

    def batch(self, X, y=None, masks=None, progress: bool = False, **kwargs):
        """Explain a collection of images and summarize the results.

        Parameters
        ----------
        X : array
            Images to explain.
        y : array, optional
            Their labels, reported beside the predicted class.
        masks : array, optional
            One region of interest per image.
        progress : bool, default=False
            Whether a counter is printed as the batch advances.
        **kwargs : dict
            Arguments of :meth:`SOFIExplainer.explain`.

        Returns
        -------
        DataFrame
            One row per image, with the columns every study reports.
        """
        import pandas as pd

        data = np.asarray(X, dtype=float)
        rows, self.explanations_ = [], {}
        for position in range(len(data)):
            explanation = self.explainer.explain(
                data[position],
                mask=None if masks is None else masks[position],
                **kwargs,
            )
            self.explanations_[str(position)] = explanation
            row = self._row(str(position), explanation)
            row["predicted"] = explanation.class_name
            if y is not None:
                row["true"] = str(np.asarray(y)[position])
                row["correct"] = row["true"] == row["predicted"]
            rows.append(row)
            if progress:
                print(f"  {position + 1} of {len(data)}", end="\r")
        return pd.DataFrame(rows)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _row(name: str, explanation) -> Dict[str, object]:
        return {
            "setting": name,
            "segments": explanation.n_segments,
            "ds": round(explanation.ds, 4),
            "drop": round(explanation.drop, 3),
            "surge": round(explanation.surge, 3),
            "sparsity_rate": round(explanation.sparsity_rate, 3),
            "flip_after": explanation.sparsity_point,
            "queries": explanation.n_model_calls,
            "seconds": round(explanation.runtime, 1),
        }

    def _report(self, results, plot, title_of, display=None, **plot_kwargs):
        """Collect the rows and draw one figure per setting."""
        import pandas as pd

        self.explanations_ = dict(results)
        if plot:
            import matplotlib.pyplot as plt

            for position, (name, explanation) in enumerate(results.items()):
                plot_explanation(
                    explanation,
                    display=display,
                    title=title_of(name),
                    color=_PALETTE[position % len(_PALETTE)],
                    **plot_kwargs,
                )
                plt.show()
        return pd.DataFrame([self._row(name, e) for name, e in results.items()])

    # --------------------------------------------------------------- studies
    def segment_counts(self, x, counts: Sequence[int], mask=None, display=None,
                       plot: bool = True, compactness: Optional[float] = None,
                       sigma: Optional[float] = None, **kwargs):
        """Vary how many segments the image is divided into.

        The count is the most consequential setting of the package and more is
        not simply better. A coarse partition names a large share of the image,
        because the smallest thing it can point at is already large, while a
        fine one costs more queries and leaves the search a smaller fraction of
        a larger space.

        Parameters
        ----------
        counts : sequence of int
            Segment counts to try.
        plot : bool, default=True
            Whether one figure is drawn per count.
        compactness, sigma : float, optional
            Arguments of the segmentation held fixed across the sweep. The
            settings of the explainer are used when they are left out.
        **kwargs : dict
            Arguments of :func:`~sofiimg.plotting.plot_explanation`, applied to
            every figure the study draws, such as ``figsize``, ``width_ratio``,
            ``font_scale``, ``alpha`` and ``markersize``.

        Returns
        -------
        DataFrame
            One row per count, with the share of the image the explanation
            names beside the usual columns.
        """
        instance = np.asarray(x, dtype=float)
        results, shares = {}, {}
        for count in counts:
            partition = self.explainer.segment(
                instance, "slic", mask=mask, n_segments=int(count),
                **_settings(compactness=compactness, sigma=sigma),
            )
            explanation = self.explainer.explain(instance, segmentation=partition)
            named = explanation.order[
                : (explanation.sparsity_point or explanation.n_segments)
            ]
            total = (
                float(np.asarray(mask).sum())
                if mask is not None
                else float(np.prod(partition.shape))
            )
            shares[f"{count} segments"] = sum(partition[i].area for i in named) / total
            results[f"{count} segments"] = explanation
        frame = self._report(
            results, plot, lambda name: f"SLIC with {name} requested",
            display=display, **kwargs
        )
        frame.insert(3, "share_named", [round(shares[n], 3) for n in frame["setting"]])
        return frame

    def compactness(self, x, values: Sequence[float], mask=None, display=None,
                    n_segments: int = 40, plot: bool = True,
                    sigma: Optional[float] = None, **kwargs):
        """Vary the weight SLIC gives to position against intensity.

        A large value produces squarer segments that follow the image less, and
        a small one produces segments that hug the contours and grow irregular.

        Parameters
        ----------
        values : sequence of float
            Compactness values to try.
        n_segments : int, default=40
            Segment count held fixed across the sweep.
        plot : bool, default=True
            Whether one figure is drawn per value.
        sigma : float, optional
            Argument of the segmentation held fixed across the sweep.
        **kwargs : dict
            Arguments of :func:`~sofiimg.plotting.plot_explanation`, applied to
            every figure the study draws.

        Returns
        -------
        DataFrame
            One row per value.
        """
        instance = np.asarray(x, dtype=float)
        results = {}
        for value in values:
            partition = self.explainer.segment(
                instance, "slic", mask=mask, n_segments=n_segments,
                compactness=float(value), **_settings(sigma=sigma),
            )
            results[f"compactness {value}"] = self.explainer.explain(
                instance, segmentation=partition
            )
        return self._report(
            results, plot, lambda name: f"SLIC with {name}", display=display, **kwargs
        )

    def operators(self, x, names: Optional[Sequence[str]] = None, mask=None,
                  display=None, segmentation=None, plot: bool = True,
                  marginalization_params: Optional[dict] = None, **kwargs):
        """Vary what neutralizing a segment means.

        Parameters
        ----------
        names : sequence of str, optional
            Operators to try. The registered ones are used when left out.
        segmentation : Segmentation, optional
            Partition held fixed across the sweep. One is computed from the
            settings of the explainer when it is left out.
        plot : bool, default=True
            Whether one figure is drawn per operator.
        marginalization_params : dict, optional
            Arguments handed to every operator, such as ``n_replicas`` and
            ``taper``.
        **kwargs : dict
            Arguments of :func:`~sofiimg.plotting.plot_explanation`, applied to
            every figure the study draws.

        Returns
        -------
        DataFrame
            One row per operator, ordered as they were given.
        """
        from .perturbation import CLASS_AGNOSTIC, CLASS_DIRECTED

        instance = np.asarray(x, dtype=float)
        if names is None:
            names = list(CLASS_AGNOSTIC) + list(CLASS_DIRECTED)
        partition = segmentation or self.explainer.segment(
            instance, mask=mask
        )
        results = {
            name: self.explainer.explain(
                instance,
                segmentation=partition,
                marginalization=(
                    make_perturbation(name, **marginalization_params)
                    if marginalization_params
                    else name
                ),
            )
            for name in names
        }
        frame = self._report(
            results, plot, lambda name: f"Marginalization with {name}",
            display=display, **kwargs
        )
        directed = set(CLASS_DIRECTED)
        frame.insert(
            1, "family",
            ["directed" if n in directed else "agnostic" for n in frame["setting"]],
        )
        return frame

    def rankings(self, x, saliency: Optional[Mapping[str, np.ndarray]] = None,
                 orders: Optional[Mapping[str, Sequence[int]]] = None, mask=None,
                 display=None, segmentation=None, plot: bool = True, **kwargs):
        """Compare the ranking of SOFI against rankings obtained elsewhere.

        Feature occlusion and a random order are included as the standard upper
        and lower references. A pixel-wise saliency map produced by any other
        method joins them once it has been averaged inside each segment, and an
        explicit order joins them directly.

        Parameters
        ----------
        saliency : mapping, optional
            Named saliency maps, each shaped like the image or like one of its
            channels.
        orders : mapping, optional
            Named orders of the segments, from the most to the least relevant.
        segmentation : Segmentation, optional
            Partition every ranking is scored on. One is computed from the
            settings of the explainer when it is left out.
        plot : bool, default=True
            Whether one figure is drawn per method.
        **kwargs : dict
            Arguments of :func:`~sofiimg.plotting.plot_explanation`, applied to
            every figure the study draws.

        Returns
        -------
        DataFrame
            One row per method, ordered by degradation score.
        """
        instance = np.asarray(x, dtype=float)
        partition = segmentation or self.explainer.segment(
            instance, mask=mask
        )
        instance = instance

        found = {"SOFI": self.explainer.explain(instance, segmentation=partition).order}
        found["Occlusion"] = ranking_from_scores(
            feature_occlusion(self.explainer, instance, partition)
        )
        for name, values in (saliency or {}).items():
            found[name] = ranking_from_scores(segment_scores(values, partition))
        for name, order in (orders or {}).items():
            found[name] = list(order)
        found["Random"] = random_ranking(
            partition.n_segments, random_state=self.random_state
        )

        results = {
            name: self.explainer.score_ranking(instance, order, segmentation=partition)
            for name, order in found.items()
        }
        frame = self._report(
            results, plot, lambda name: f"Ranking obtained with {name}",
            display=display, **kwargs
        )
        return frame.sort_values("ds", ascending=False).reset_index(drop=True)

    def segmentation_grid(self, x, counts: Sequence[int] = (20, 40, 80), mask=None,
                          display=None, ncols: int = 3, **kwargs):
        """Draw the image under several partitions, as one panel of borders.

        Parameters
        ----------
        counts : sequence of int, default=(20, 40, 80)
            Segment counts drawn.
        ncols : int, default=3
            Panels per row.
        **kwargs : dict
            Arguments of :func:`~sofiimg.plotting.plot_segmentation_grid`, and
            of the segmentation itself.

        Returns
        -------
        matplotlib Figure
            The panel.
        """
        instance = np.asarray(x, dtype=float)
        accepted = {"compactness", "sigma", "max_num_iter", "enforce_connectivity"}
        options = {k: v for k, v in kwargs.items() if k in accepted}
        drawing = {k: v for k, v in kwargs.items() if k not in accepted}
        partitions = {
            f"{count} segments": self.explainer.segment(
                instance, "slic", mask=mask, n_segments=int(count), **options
            )
            for count in counts
        }
        return plot_segmentation_grid(
            instance, partitions, ncols=ncols, display=display, **drawing
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Experiment({type(self.explainer).__name__})"
