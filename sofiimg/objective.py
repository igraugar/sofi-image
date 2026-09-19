"""Perturbation curves and the objective optimized by SOFI.

A ranking is assessed through two perturbation curves obtained after cumulative
marginalization of image segments. The MoRF curve removes the most
relevant segments first and should collapse immediately, while the LeRF curve
removes the least relevant segments first and should stay flat for as long as
possible. The degradation score is the integral between the two, and SOFI
maximizes it. The MoRF branch rewards sparsity, since a sparse explanation
concentrates the response of the model in a handful of segments, and the LeRF
branch rewards correctness, since the segments declared irrelevant must be the
ones whose removal leaves the response untouched.

The search works on the raw response of the model and on nothing else, namely
the probability assigned to the class predicted before any perturbation, as in
segment perturbation. No ground truth label takes part, which keeps explanations
free from the bias induced by the error of the model.

Normalization never enters the optimization. It is applied afterwards, for
reporting and for drawing, and it follows the convention of the literature. The
response of the unperturbed series is placed at one and the response of the
fully perturbed series at zero. Both anchors are properties of the model and the
instance rather than of a ranking, so every ranking evaluated on the same
instance shares them, the reported score is the optimized score divided by a
positive constant, and no ordering can change. The anchors are read off a
reference sweep that also covers the fully marginalized image, so in the ordinary
case they are exactly the two responses the convention prescribes. Should the
fully marginalized response fail to be the lowest one, which happens when a
segment is replaced with a constant the model reads with confidence, the
anchors widen to the extremes the sweep observed and a warning names the cause.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .perturbation import Perturbation
from .segmentation import Segmentation

__all__ = [
    "Objective",
    "RankingEvaluation",
    "curve_auc",
    "degradation_score",
    "modularity_gap",
]

_EPSILON = 1e-12


def curve_auc(scores: Sequence[float]) -> float:
    """Area under a perturbation curve, divided by the number of steps.

    The division turns the area into the mean height of the curve, so instances
    with different numbers of segments stay comparable.

    Parameters
    ----------
    scores : sequence of float
        A perturbation curve, starting with the unperturbed response.

    Returns
    -------
    float
        The mean height of the curve.
    """
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        return float(values.sum())
    area = np.trapezoid(values) if hasattr(np, "trapezoid") else np.trapz(values)
    return float(area / (values.size - 1))


def degradation_score(morf: Sequence[float], lerf: Sequence[float]) -> float:
    """Integral between the LeRF and the MoRF perturbation curves.

    Larger values denote explanations that are simultaneously sparser and more
    faithful. On the reported scale both areas belong to the unit interval, so
    the score runs from minus one to one and a sound ranking keeps it positive.

    Parameters
    ----------
    morf : sequence of float
        Curve of the ranking, most relevant segment first.
    lerf : sequence of float
        Curve of the reversed ranking.

    Returns
    -------
    float
        The area between the two curves.
    """
    return curve_auc(lerf) - curve_auc(morf)


@dataclass
class RankingEvaluation:
    """Both raw curves of a ranking together with its degradation score.

    Attributes
    ----------
    order : list of int
        Positions of the segments, from the most to the least relevant.
    morf, lerf : list of float
        Raw responses along the ranking and along its reverse, each starting
        with the unperturbed one.
    ds : float
        Area between the two curves on the raw scale.
    morf_proba : ndarray or None
        Full probability matrix of the MoRF steps, from which the change of
        class is read without a further query.
    """

    order: List[int]
    morf: List[float] = field(repr=False)
    lerf: List[float] = field(repr=False)
    ds: float = 0.0
    morf_proba: Optional[np.ndarray] = field(default=None, repr=False)


class Objective:
    """Evaluate segment rankings through cumulative marginalization.

    Parameters
    ----------
    wrapper : ClassifierWrapper
        Access to the classifier being explained.
    instance : ndarray
        The image under study, shaped ``(height, width, n_channels)``.
    segmentation : Segmentation
        Partition whose segments a ranking permutes.
    perturbation : Perturbation
        Operator already fitted on this instance.
    label : int, optional
        Position of the class whose probability is tracked. The predicted class
        of the unperturbed series is used by default.
    """

    def __init__(
        self,
        wrapper,
        instance: np.ndarray,
        segmentation: Segmentation,
        perturbation: Perturbation,
        label: Optional[int] = None,
    ):
        self.wrapper = wrapper
        self.segmentation = segmentation
        self.perturbation = perturbation
        self.n_calls_ = 0

        self.instance = np.asarray(instance, dtype=float)
        if self.instance.ndim == 2:
            self.instance = self.instance[..., None]

        proba = wrapper.predict_proba(self.instance[None, ...])[0]
        self.n_calls_ += 1
        self.baseline_proba_ = np.asarray(proba, dtype=float)
        self.label = int(np.argmax(proba)) if label is None else int(label)
        self.baseline_ = float(self.baseline_proba_[self.label])

        self.n_replicas = int(getattr(perturbation, "n_replicas", 1))
        self.template = np.repeat(self.instance[None, ...], self.n_replicas, axis=0)
        self._reference_sweep()

    # ------------------------------------------------------------- accessors
    @property
    def n_segments(self) -> int:
        return self.segmentation.n_segments

    @property
    def anchors_(self) -> Tuple[float, float]:
        """Responses placed at one and at zero when a curve is reported."""
        return self.anchor_high_, self.anchor_low_

    # -------------------------------------------------------- reference sweep
    def _reference_sweep(self) -> None:
        """Marginalize every segment alone and then follow the greedy ordering.

        The sweep serves three purposes. It orders the segments by the
        degradation each of them causes on its own, which is the greedy start of
        the search, it produces the response of the fully marginalized image,
        which anchors the floor of the reported scale, and it establishes
        whether the response reacts to marginalization at all.
        """
        alone, _ = self._sweep()
        alone = np.asarray(alone, dtype=float)
        self.single_segment_drops_ = self.baseline_ - alone
        self.greedy_order_ = [
            int(position)
            for position in np.argsort(-self.single_segment_drops_, kind="stable")
        ]
        self.reference_curve_, _ = self.curve(self.greedy_order_)
        self.full_score_ = float(self.reference_curve_[-1])

        observed = list(alone) + list(self.reference_curve_) + [self.baseline_]
        self.anchor_high_ = float(self.baseline_)
        self.anchor_low_ = float(min(observed))

        # the unperturbed response should be the highest one. A marginal excursion
        # above it is absorbed by the clipping, while a substantial one widens the
        # scale, so that a curve is never flattened against a false ceiling
        overshoot = float(max(observed)) - self.anchor_high_
        span = self.anchor_high_ - self.anchor_low_
        systematic = overshoot > max(0.1 * span, _EPSILON)

        if span <= _EPSILON:
            warnings.warn(
                "The response of the model never lowers when segments are "
                "marginalized, so the degradation score carries no information. "
                "Check that the instance is one the model handles well and that "
                "the marginalization operator neutralizes the segments.",
                stacklevel=5,
            )
        elif systematic:
            self.anchor_high_ = float(max(observed))
            warnings.warn(
                "Marginalization leaves the model substantially more confident than "
                "the unperturbed image, so the response it reaches is not the most "
                "degraded one and the reported curves cannot start at one. This "
                "points at the marginalization operator rather than at the ranking, "
                "since a flat patch is itself a pattern the model may read with "
                "confidence. Try marginalization='auto', which selects the operator "
                "per instance.",
                stacklevel=5,
            )

    def _sweep(self):
        """Raw response after marginalizing each segment on its own."""
        states = []
        for position in range(self.n_segments):
            state = self.template.copy()
            self._perturb(state, position)
            states.append(state)
        return self._predict_states(states)

    # --------------------------------------------------------- normalization
    def normalize(self, scores: Sequence[float]) -> List[float]:
        """Express a raw curve on the reported scale.

        The transformation is affine and shared by every ranking, so it serves
        reading and drawing alone. It never takes part in the search.

        Parameters
        ----------
        scores : sequence of float
            A raw curve.

        Returns
        -------
        list of float
            The same curve inside the unit interval.
        """
        span = self.anchor_high_ - self.anchor_low_
        if span <= _EPSILON:
            return [1.0 for _ in scores]
        return [
            float(min(max((float(score) - self.anchor_low_) / span, 0.0), 1.0))
            for score in scores
        ]

    # --------------------------------------------------------------- scoring
    def _perturb(self, state: np.ndarray, position: int) -> None:
        segment = self.segmentation[position]
        for replica in range(self.n_replicas):
            self.perturbation.apply(state[replica], segment, replica)

    def _predict_states(self, states: List[np.ndarray]):
        """Score a sequence of marginalized states with one batched query.

        Every state holds one replica per row, and the response is averaged over
        the replicas before it enters a curve.
        """
        if not states:
            return [], np.zeros((0, self.baseline_proba_.size))
        stacked = np.concatenate(states, axis=0)
        proba = self.wrapper.predict_proba(stacked)
        self.n_calls_ += 1
        proba = proba.reshape(len(states), self.n_replicas, -1).mean(axis=1)
        return [float(row[self.label]) for row in proba], proba

    # ----------------------------------------------------------------- curve
    def curve(
        self,
        order: Sequence[int],
        prefix_length: int = 0,
        prefix_scores: Optional[Sequence[float]] = None,
        prefix_proba: Optional[np.ndarray] = None,
    ):
        """Raw perturbation curve of a marginalization order.

        A swap between two positions leaves every step before the first of them
        untouched, so those scores are inherited through ``prefix_scores``
        instead of being recomputed. The result is identical to a full
        evaluation while the number of model queries drops.

        Parameters
        ----------
        order : sequence of int
            Positions of the segments, in the order they are marginalized.
        prefix_length : int, default=0
            Number of leading steps inherited instead of recomputed.
        prefix_scores : sequence of float, optional
            Curve of the incumbent, from which that prefix is taken.
        prefix_proba : ndarray, optional
            Probability matrix of the incumbent. When it is left out the matrix
            is not rebuilt, which suits the LeRF branch that never reads it.

        Returns
        -------
        tuple
            The curve and the probability matrix of its steps.
        """
        order = [int(position) for position in order]
        if prefix_scores is None or prefix_length <= 0:
            prefix_length = 0
            scores = [self.baseline_]
            proba_rows: Optional[List[np.ndarray]] = [self.baseline_proba_]
        else:
            prefix_length = min(prefix_length, len(order))
            scores = list(prefix_scores[: prefix_length + 1])
            # the LeRF branch never consults the probability matrix, so its
            # prefix is inherited without one and the matrix is left empty
            proba_rows = (
                None
                if prefix_proba is None
                else list(np.asarray(prefix_proba)[: prefix_length + 1])
            )

        state = self.template.copy()
        for position in order[:prefix_length]:
            self._perturb(state, position)

        states = []
        for position in order[prefix_length:]:
            self._perturb(state, position)
            states.append(state.copy())

        tail_scores, tail_proba = self._predict_states(states)
        scores.extend(tail_scores)
        if proba_rows is None:
            return scores, np.zeros((0, self.baseline_proba_.size))
        if len(tail_proba):
            proba_rows.extend(list(tail_proba))
        return scores, np.asarray(proba_rows, dtype=float)

    def marginalized_images(self, order: Sequence[int]) -> np.ndarray:
        """States of the image along a marginalization order.

        The first row is the untouched image and the row ``k`` holds it after
        the first ``k`` segments of the order have been marginalized. Only the
        first replica is returned, since the purpose is illustration.

        Parameters
        ----------
        order : sequence of int
            Positions of the segments, in the order they are marginalized.

        Returns
        -------
        ndarray
            States shaped ``(len(order) + 1, height, width, n_channels)``.
        """
        order = [int(position) for position in order]
        state = self.template.copy()
        states = [state[0].copy()]
        for position in order:
            self._perturb(state, position)
            states.append(state[0].copy())
        return np.asarray(states, dtype=float)

    # the name the time series package uses, kept so that code written against
    # one package reads the other without a change
    marginalized_series = marginalized_images

    def evaluate(
        self,
        order: Sequence[int],
        incumbent: Optional[RankingEvaluation] = None,
        swap: Optional[Tuple[int, int]] = None,
    ) -> RankingEvaluation:
        """Both raw curves of a ranking, reusing the prefixes left by a swap.

        The LeRF curve follows the reversed ranking, so a swap between the
        positions ``i`` and ``j`` of the ranking occupies the positions
        ``n - 1 - j`` and ``n - 1 - i`` of the reversed one. The two prefixes are
        therefore inherited from opposite ends of the incumbent.

        Parameters
        ----------
        order : sequence of int
            The ranking to evaluate.
        incumbent : RankingEvaluation, optional
            Evaluation the candidate was derived from, whose prefixes are reused.
        swap : tuple of int, optional
            The two positions that were exchanged, in increasing order.

        Returns
        -------
        RankingEvaluation
            Both curves and the degradation score of the ranking.
        """
        order = [int(position) for position in order]
        n_segments = len(order)
        reversed_order = order[::-1]

        if incumbent is None or swap is None:
            morf, morf_proba = self.curve(order)
            lerf, _ = self.curve(reversed_order)
        else:
            first, second = int(swap[0]), int(swap[1])
            morf, morf_proba = self.curve(
                order, first, incumbent.morf, incumbent.morf_proba
            )
            lerf, _ = self.curve(
                reversed_order, n_segments - 1 - second, incumbent.lerf
            )

        return RankingEvaluation(
            order=order,
            morf=morf,
            lerf=lerf,
            ds=degradation_score(morf, lerf),
            morf_proba=morf_proba,
        )

    # ---------------------------------------------------------------- priors
    def single_segment_drops(self) -> np.ndarray:
        """Degradation caused when each segment is marginalized on its own."""
        return self.single_segment_drops_

    def greedy_order(self) -> List[int]:
        """Segments sorted by the drop each of them causes on its own."""
        return list(self.greedy_order_)

    # -------------------------------------------------------------- sparsity
    def flip_point(self, order: Sequence[int]) -> Optional[int]:
        """Step at which a marginalization order first changes the predicted class.

        ``None`` says the class survives every perturbation. The measure is
        applied to the MoRF order to obtain the sparsity of an explanation, and
        to the LeRF order to see how much later the same change arrives when the
        ranking is walked backwards.

        Parameters
        ----------
        order : sequence of int
            Positions of the segments, in the order they are marginalized.

        Returns
        -------
        int or None
            The step at which the class changes.
        """
        _, proba = self.curve([int(position) for position in order])
        return self._flip_from_proba(proba)

    def _flip_from_proba(self, proba) -> Optional[int]:
        proba = np.asarray(proba)
        if proba.size == 0:
            return None
        predicted = np.argmax(proba[1:], axis=1)
        flips = np.flatnonzero(predicted != self.label)
        return None if flips.size == 0 else int(flips[0]) + 1

    def sparsity(self, evaluation: RankingEvaluation):
        """Segments needed before the predicted class changes.

        Returns the position at which the class changes, the sparsity rate and
        the probability of the explained class at that position. The rate is the
        position over the segment count, so lower values denote sparser
        explanations. A ranking whose cumulative marginalization never changes
        the class receives a rate of one and a position of ``None``.

        Parameters
        ----------
        evaluation : RankingEvaluation
            The evaluation whose MoRF branch is inspected.

        Returns
        -------
        tuple
            The step, the rate and the probability of the explained class there.
        """
        proba = evaluation.morf_proba
        if proba is None or len(proba) != len(evaluation.morf):
            _, proba = self.curve(evaluation.order)
        position = self._flip_from_proba(proba)
        proba = np.asarray(proba)
        if position is None:
            return None, 1.0, float(proba[-1][self.label])
        return (
            position,
            position / float(self.n_segments),
            float(proba[position][self.label]),
        )


def modularity_gap(objective: Objective, order: Optional[Sequence[int]] = None) -> float:
    """Largest departure from modularity along one marginalization order.

    A modular black box satisfies ``f(x, A) = f(x, empty) - sum of c(s)``, so the
    cumulative curve is fully determined by the drops each segment causes alone.
    The function compares the observed curve with that additive prediction and
    returns the largest absolute difference. A value near zero says the
    modularity assumption of Theorem 3 holds, in which case the greedy ranking is
    provably optimal and the search has nothing left to find. Larger values say
    the effect of marginalizing a segment depends on what was marginalized
    before, which is the ordinary situation.

    Parameters
    ----------
    objective : Objective
        The objective bound to the instance under study.
    order : sequence of int, optional
        Order along which the comparison is made. The greedy one is used when it
        is omitted.

    Returns
    -------
    float
        The largest absolute difference between the observed curve and the
        additive prediction.
    """
    drops = objective.single_segment_drops()
    order = objective.greedy_order() if order is None else [int(p) for p in order]
    observed, _ = objective.curve(order)
    predicted = objective.baseline_ - np.cumsum([drops[position] for position in order])
    predicted = np.concatenate([[objective.baseline_], predicted])
    return float(np.max(np.abs(np.asarray(observed) - predicted)))
