"""Sparseness Optimized Feature Importance for image classification."""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .explanation import SOFIExplanation
from .model import ClassifierWrapper
from .objective import Objective, degradation_score, modularity_gap
from .perturbation import DEFAULT_CANDIDATES, PerturbationContext, make_perturbation
from .report import describe_instance, instance_report
from .search import greedy_ranking, hill_climbing
from .segmentation import Segmentation, build_segmentation

__all__ = ["SOFIExplainer"]


def _finite(array: np.ndarray, what: str) -> np.ndarray:
    """Refuse arrays holding values that are not finite.

    A missing or infinite value propagates silently through the perturbation
    curves and leaves a degradation score that looks ordinary while meaning
    nothing, so it is refused where it enters rather than diagnosed later.
    """
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{what} holds values that are not finite. Impute or drop them "
            "before explaining, since a curve computed on them carries no "
            "information."
        )
    return array


def _as_instance(x) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 2:
        instance = array[..., None]
    elif array.ndim == 3:
        instance = array
    elif array.ndim == 4 and array.shape[0] == 1:
        instance = array[0]
    else:
        raise ValueError(
            "An image must be shaped (height, width), (height, width, "
            f"n_channels) or (1, height, width, n_channels), and the input has "
            f"shape {array.shape}."
        )
    if instance.size == 0:
        raise ValueError("An image cannot be empty.")
    return _finite(instance, "The image")


def _as_mask(mask, instance) -> np.ndarray:
    """Read a region of interest and refuse one that cannot describe the image.

    A mask is boolean, so any array is thresholded above zero, which accepts a
    binary image stored as bytes as readily as one stored as booleans. It also
    has to cover the image it restricts, since a region of interest computed at
    another resolution would rank pixels that are not the ones it names.
    """
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            "A region of interest must be shaped (height, width), and the mask "
            f"received has shape {np.asarray(mask).shape}."
        )
    if array.shape != instance.shape[:2]:
        raise ValueError(
            f"The region of interest covers {array.shape} pixels and the image "
            f"holds {instance.shape[:2]}. Resize the mask with nearest neighbor "
            "sampling before it is passed, so that no label is invented."
        )
    binary = array > 0 if array.dtype != bool else array
    if not binary.any():
        raise ValueError(
            "The region of interest is empty, so there is nothing to segment "
            "and nothing to rank."
        )
    return binary


def _as_dataset(X) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 3:
        data = array[..., None]
    elif array.ndim == 4:
        data = array
    else:
        raise ValueError(
            "A dataset must be shaped (n_images, height, width) or "
            f"(n_images, height, width, n_channels), and the input has shape "
            f"{array.shape}."
        )
    if len(data) == 0:
        raise ValueError("A dataset cannot be empty.")
    return data


def _budget(value, name: str, minimum: int = 0) -> int:
    """Read a non-negative search budget, refusing anything else."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, and {value!r} is not.") from None
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, and {number} is not.")
    return number


def _progress(iterable, enabled: bool, description: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional
        return iterable
    return tqdm(iterable, desc=description)


class SOFIExplainer:
    """Post hoc explainer that optimizes the sparseness of segment rankings.

    SOFI searches for an order of image segments whose cumulative marginalization
    degrades the response of a classifier as fast as possible. A ranking is
    judged through its degradation score, namely the integral between the curve
    obtained when the least relevant segments are removed first and the one
    obtained when the most relevant segments are removed first. Maximizing that
    integral rewards sparsity and correctness at once, and the outcome is the
    smallest set of segments that changes the decision of the model.

    The explainer is agnostic to the classifier, which it queries for class
    probabilities alone, and agnostic to the segmentation, which may come from
    domain experts or from a segment algorithm. Both choices belong to the
    user and both are reported alongside the explanation, since a ranking of
    segments means nothing without the segments it ranks.

    Segmentation applies to one image at a time, since segments differ from
    image to image, so a ranking describes one decision. Rankings can be pooled
    across images only when the segmentation is shared by all of them, which
    happens with a patch grid and with an expert label map.

    Parameters
    ----------
    model : object
        Fitted classifier. Estimators exposing ``predict_proba``, Keras models
        and PyTorch modules, including the architectures of ``timm`` and the
        image models of ``transformers``, are recognized without further
        declaration.
    X_train : array
        Training images shaped ``(n, height, width)`` or ``(n, height, width,
        n_channels)``, already preprocessed exactly as the model expects them.
        They are the only source of marginalization values, and the images
        explained afterwards never contribute statistics.
    y_train : array, optional
        Training labels, needed by the class directed operators alone.
    output_fn : str or callable, optional
        Method of the model that produces the outputs. A string names an
        attribute, such as ``"predict_proba"`` or ``"forward"``, and a callable
        receives the batch directly. The detection resolves it when the argument
        is left out.
    output : {"auto", "proba", "logits"}, default="auto"
        Whether the model already returns probabilities. Under ``"auto"`` a probe
        batch decides, and a softmax is applied only when the raw output leaves
        the probability simplex.
    channels_first : {"auto", True, False}, default="auto"
        Whether the model expects the channel axis before the two spatial axes.
    classes : sequence, optional
        Class labels in the order of the probability columns, for reporting.
    batch_size : int, optional
        Largest number of images sent to the model in one call, which keeps the
        memory bounded when a partition of many segments is multiplied by the
        replicas. The whole batch goes at once by default.
    segmentation : str, Segmentation or array, default="slic"
        How an image is divided. See ``segmentation_params`` for the arguments
        of each procedure.
    segmentation_params : dict, optional
        Arguments of the segmentation, such as ``n_segments``, ``compactness``
        and ``sigma`` for the segment procedures, ``patch_size`` for the
        grid, and ``mask`` for a label map.
    marginalization : str, Perturbation, number or sequence, default="mean"
        Operator that neutralizes a segment. ``"auto"`` runs the search once per
        candidate operator and keeps the best outcome, and an explicit sequence
        of names restricts those candidates.
    marginalization_params : dict, optional
        Arguments of the operator, such as ``n_replicas``, ``taper`` and the
        ``sigma`` of the blur.
    initialization : {"greedy", "sequential", "random"} or sequence, default="greedy"
        Starting ranking. ``"greedy"`` reuses the ordering produced by the
        reference sweep, at no extra cost. ``"sequential"`` runs the greedy
        selection of Algorithm 3, which is stronger and quadratic in the segment
        count. An explicit sequence lets prior knowledge enter the search.
    max_iterations : int, default=200
        Budget of proposed swaps across all restarts. Zero evaluates the initial
        ranking without any search.
    patience : int, optional
        Consecutive swaps without improvement tolerated before a restart or the
        end of the search. ``None`` sets it to the whole budget.
    n_restarts : int, default=0
        Restarts from a random ranking granted once the patience expires.
    surge_tolerance : float, default=0.5
        Largest recovery, as a fraction of the untouched response, that
        ``marginalization="auto"`` tolerates in a candidate operator. A curve
        that climbs back further than this once every segment is marginalized is
        discarded whatever area it encloses.
    min_drop : float, default=0.5
        Smallest fall of the response, as a fraction of the untouched one, that
        ``marginalization="auto"`` requires of a candidate. An operator that
        never moves the model is discarded before its area is even considered,
        since reporting would otherwise stretch a range of a few thousandths
        across the whole unit interval and hand it a large score.
    accept_equal : bool, default=False
        Whether swaps that leave the score unchanged are accepted.
    check_modularity : bool, default=True
        Whether the departure from the modularity assumption is measured, which
        says whether the greedy ranking is already optimal.
    random_state : int or Generator, optional
        Seed of the operator draws, of the swap operator and of the restarts.
    verbose : bool, default=True
        Whether the one line notice naming the resolved backend, output
        convention and channel layout is printed when the explainer is built.
    progress : bool, default=False
        Whether a progress bar follows the search. It is off by default, since a
        local explanation is quick and a bar per call would bury the output of a
        notebook, and it is worth turning on for long searches or large batches.

    Examples
    --------
    >>> explainer = SOFIExplainer(model, X_train, random_state=42)
    >>> explainer.inspect(X_test[0])
    >>> explanation = explainer.explain(X_test[0])
    >>> print(explanation.summary())
    """

    def __init__(
        self,
        model,
        X_train,
        y_train=None,
        output_fn=None,
        output: str = "auto",
        channels_first="auto",
        classes: Optional[Sequence] = None,
        batch_size: Optional[int] = None,
        segmentation: Union[str, Segmentation, Sequence] = "slic",
        segmentation_params: Optional[Dict] = None,
        mask=None,
        marginalization="mean",
        marginalization_params: Optional[Dict] = None,
        initialization: Union[str, Sequence] = "greedy",
        max_iterations: int = 200,
        patience: Optional[int] = None,
        n_restarts: int = 0,
        accept_equal: bool = False,
        surge_tolerance: float = 0.5,
        min_drop: float = 0.5,
        check_modularity: bool = True,
        random_state=None,
        verbose: bool = True,
        progress: bool = False,
    ):
        self.wrapper = ClassifierWrapper(
            model,
            output_fn=output_fn,
            output=output,
            channels_first=channels_first,
            classes=classes,
            batch_size=batch_size,
            verbose=verbose,
        )
        if X_train is None:
            raise ValueError(
                "Training data is required, since it is the only source of "
                "marginalization values."
            )
        self.X_train = _finite(_as_dataset(X_train), "The training data")
        self.batch_size = batch_size
        self.y_train = None if y_train is None else np.asarray(y_train)
        if self.y_train is not None and len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train hold a different number of instances.")

        self.segmentation = segmentation
        self.segmentation_params = dict(segmentation_params or {})
        self.mask = self.segmentation_params.pop("mask", None) if mask is None else mask
        self.marginalization = marginalization
        self.marginalization_params = dict(marginalization_params or {})
        self.initialization = initialization
        self.max_iterations = _budget(max_iterations, "max_iterations")
        self.patience = None if patience is None else _budget(patience, "patience", 1)
        self.n_restarts = _budget(n_restarts, "n_restarts")
        self.accept_equal = bool(accept_equal)
        self.surge_tolerance = float(surge_tolerance)
        self.min_drop = float(min_drop)
        self.check_modularity = bool(check_modularity)
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.progress = bool(progress)

        self.wrapper.calibrate(self.X_train[: min(4, len(self.X_train))])

    # ------------------------------------------------------------- inspection
    def _check_instance(self, instance: np.ndarray) -> None:
        """Refuse an image whose shape the training data cannot explain.

        Marginalization values come from the training data, so an image of a
        different size or a different number of channels has no admissible
        substitution and the model would receive an input it was never fitted
        for. Failing at this point turns a confusing error inside the estimator
        into a readable one.
        """
        expected = self.X_train.shape[1:]
        if instance.shape != expected:
            raise ValueError(
                f"The image is shaped {instance.shape} while the training data "
                f"holds images shaped {expected}. Both must agree, since the "
                "marginalization values are drawn from the training data, and "
                "the same preprocessing has to be applied to both."
            )

    def segment(self, x, segmentation=None, mask=None, **overrides) -> Segmentation:
        """Segmentation of one instance, without running any search.

        The procedure and its arguments default to the ones the explainer was
        built with, and either can be overridden for a single call, which is how
        several partitions of the same instance are obtained without building
        several explainers.

        Parameters
        ----------
        x : array
            The image to divide, shaped ``(height, width)`` or ``(height,
            width, n_channels)``.
        segmentation : str, Segmentation or array, optional
            Procedure for this call alone. ``method="slic"`` is accepted as a
            synonym, so the name of a procedure can be passed the way the
            procedures themselves take it.
        mask : array, optional
            Boolean region of interest. Segments are grown inside it alone
            and every pixel outside it stays out of the explanation, so nothing
            beyond the segment can ever be ranked or marginalized.
        **overrides : dict
            Arguments of the procedure, such as ``n_segments``, ``compactness``,
            ``sigma`` or ``patch_size``. They override the ones the explainer
            was built with, for this call alone.

        Returns
        -------
        Segmentation
            The partition, which can be passed to :meth:`explain`.
        """
        instance = _as_instance(x)
        params = dict(self.segmentation_params)
        params.update(overrides)
        declared = params.pop("method", None)
        segment = self.mask if mask is None else mask
        if segment is not None:
            params["mask"] = _as_mask(segment, instance)
        kind = segmentation if segmentation is not None else declared
        return build_segmentation(instance, self.segmentation if kind is None else kind, **params)

    def inspect(
        self,
        x,
        true_label=None,
        plot: bool = True,
        printout: bool = True,
        segmentation: Optional[Segmentation] = None,
        mask=None,
        ax=None,
        **plot_kwargs,
    ) -> Dict[str, object]:
        """Describe an instance and its segmentation before explaining it.

        The report gathers the shape and statistics of the image, the
        confidence of the model with its runner up class, and the geometry of
        the segmentation together with the share of variance its segment means
        explain. Warnings appear when the decision sits near a class boundary,
        when the image is misclassified, or when the segments barely separate the
        content, since none of those yields a ranking worth reading.

        Parameters
        ----------
        x : array
            The image to describe.
        true_label : optional
            Ground truth label of the instance. When it is given, the report
            says whether the prediction is correct and warns when it is not.
        plot : bool, default=True
            Whether the image and the borders of its segments are drawn.
        printout : bool, default=True
            Whether the report is printed. Setting it to ``False`` returns the
            same facts without writing anything, which suits logging.
        segmentation : Segmentation, optional
            Partition to describe. The one of the explainer is used when it is
            omitted.
        ax : matplotlib Axes, optional
            Target axes of the figure. A new one is created when it is omitted.
        **plot_kwargs : dict
            Further arguments of :func:`~sofiimg.plotting.plot_segmentation`,
            such as ``linewidth``, ``color``, ``cmap`` or ``display``, so that
            the figure follows the same settings as every other one.

        Returns
        -------
        dict
            The facts the report formats, so that they can be logged or
            tabulated instead of printed.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance, mask=mask)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        probabilities = self.wrapper.predict_proba(instance[None, ...])[0]
        facts = describe_instance(
            instance,
            seg,
            probabilities=probabilities,
            class_names=self.wrapper.classes,
            true_label=true_label,
        )
        if printout:
            print(instance_report(facts))
        if plot:
            from .plotting import plot_segmentation

            plot_segmentation(instance, seg, ax=ax, **plot_kwargs)
        facts["segmentation_object"] = seg
        return facts

    # ---------------------------------------------------------------- explain
    @staticmethod
    def _check_segmentation(instance: np.ndarray, seg: Segmentation) -> Segmentation:
        """Refuse a segmentation that was built for a different shape."""
        if seg.shape != instance.shape[:2]:
            raise ValueError(
                f"The segmentation covers {seg.height} by {seg.width} pixels and "
                f"the image holds {instance.shape[0]} by {instance.shape[1]}. A "
                "segmentation belongs to one image size, so build a new one or "
                "reuse the image it was made for."
            )
        return seg

    def explain(
        self,
        x,
        segmentation: Optional[Segmentation] = None,
        mask=None,
        display=None,
        marginalization=None,
        label: Optional[int] = None,
        initialization=None,
        max_iterations: Optional[int] = None,
        patience: Optional[int] = None,
        n_restarts: Optional[int] = None,
        random_state=None,
        verbose: Optional[bool] = None,
    ) -> SOFIExplanation:
        """Explain the prediction made for one instance.

        Parameters
        ----------
        x : array
            The image to explain, shaped ``(height, width)`` or ``(height,
            width, n_channels)``.
        segmentation : Segmentation, optional
            Partition to rank. It is computed from the settings of the explainer
            when omitted, which is the usual case.
        mask : array, optional
            Boolean region of interest for this call alone. Segments are
            grown inside it and everything outside stays out of the ranking.
        marginalization : optional
            Operator for this call alone, overriding the one of the explainer.
        label : int, optional
            Position of the class whose probability the search tracks. The
            predicted class is used by default, which makes the explanation
            independent of the ground truth.
        initialization : {"greedy", "sequential", "random"} or sequence, optional
            Starting ranking for this call alone.
        max_iterations : int, optional
            Budget of proposed swaps for this call alone. Zero evaluates the
            initial ranking without searching.
        patience : int, optional
            Swaps without improvement tolerated for this call alone.
        n_restarts : int, optional
            Restarts granted for this call alone.
        random_state : int or Generator, optional
            Seed for this call alone.
        verbose : bool, optional
            Whether a progress bar follows this search.

        Returns
        -------
        SOFIExplanation
            The ranking, both curves, the scores and the counters of the run.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance, mask=mask)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        spec = self.marginalization if marginalization is None else marginalization
        self._display = display
        show = self.progress if verbose is None else bool(verbose)
        seed = self.random_state if random_state is None else random_state

        settings = dict(
            label=label,
            initialization=initialization,
            max_iterations=max_iterations,
            patience=patience,
            n_restarts=n_restarts,
            seed=seed,
        )
        if isinstance(spec, str) and spec.lower() == "auto":
            spec = list(DEFAULT_CANDIDATES)
        if isinstance(spec, (list, tuple)):
            return self._explain_best_of(instance, seg, list(spec), verbose=show, **settings)
        return self._explain_once(instance, seg, spec, verbose=show, **settings)

    def explain_batch(self, X, labels=None, **kwargs) -> List[SOFIExplanation]:
        """Explain several instances, one search each.

        Every image receives its own segmentation, since segments are image
        specific. The explanations can be pooled afterwards only when the
        segmentation is shared, which ``aggregate_explanations`` checks.

        Parameters
        ----------
        X : array
            Images shaped ``(n, height, width)`` or ``(n, height, width,
            n_channels)``.
        labels : sequence of int, optional
            Class whose probability is tracked for each image. The predicted
            class of each of them is used by default.
        **kwargs : dict
            Further arguments of :meth:`explain`, applied to every instance.

        Returns
        -------
        list of SOFIExplanation
            One explanation per image, in the order they were given.
        """
        data = _as_dataset(X)
        show = kwargs.pop("verbose", self.progress)
        explanations = []
        for position in _progress(range(len(data)), show, "SOFI explanations"):
            explanations.append(
                self.explain(
                    data[position],
                    label=None if labels is None else int(labels[position]),
                    verbose=False,
                    **kwargs,
                )
            )
        return explanations

    def score_ranking(
        self,
        x,
        ranking: Sequence[int],
        segmentation: Optional[Segmentation] = None,
        marginalization=None,
        label: Optional[int] = None,
    ) -> SOFIExplanation:
        """Evaluate a ranking produced elsewhere under the same protocol.

        The degradation score and the sparsity rate become common ground for
        comparison, so any attribution method that ends in an order of segments
        can be measured against SOFI on identical terms.

        Parameters
        ----------
        x : array
            The image the ranking refers to.
        ranking : sequence of int
            Positions of the segments, from the most to the least relevant. It
            must be a permutation of the segments of the partition.
        segmentation : Segmentation, optional
            Partition the ranking refers to. The one of the explainer is used
            when it is omitted.
        marginalization : optional
            Operator for this call alone.
        label : int, optional
            Position of the class whose probability is tracked.

        Returns
        -------
        SOFIExplanation
            The same object :meth:`explain` returns, with no search performed.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance, mask=mask)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        return self._explain_once(
            instance,
            seg,
            self.marginalization if marginalization is None else marginalization,
            label=label,
            initialization=list(ranking),
            max_iterations=0,
            patience=None,
            n_restarts=0,
            seed=self.random_state,
            verbose=False,
        )

    # ------------------------------------------------------------- internals
    def build_objective(self, x, segmentation=None, marginalization=None, label=None, random_state=None, mask=None, display=None):
        """Bind an objective to one instance, without running any search.

        The objective is the piece that drives the model through cumulative
        marginalization, so it is what a custom search or an external baseline
        needs. Everything the explainer was configured with is honored.

        Parameters
        ----------
        x : array
            The image to bind the objective to.
        segmentation : Segmentation, optional
            Partition to rank. The one of the explainer is used when it is
            omitted.
        marginalization : optional
            Operator for this call alone.
        label : int, optional
            Position of the class whose probability is tracked.
        random_state : int or Generator, optional
            Seed of the draws the operator makes.

        Returns
        -------
        tuple
            The objective and the operator already fitted on the instance.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance, mask=mask)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        spec = self.marginalization if marginalization is None else marginalization
        self._display = display
        rng = np.random.default_rng(
            self.random_state if random_state is None else random_state
        )
        return self._build_objective(instance, seg, spec, label, rng)

    def _build_objective(self, instance, seg, spec, label, rng):
        """Fit the operator on the image and bind an objective to it."""
        operator = make_perturbation(spec, **self.marginalization_params)
        probe = self.wrapper.predict_proba(instance[None, ...])[0]
        if label is None:
            position = int(np.argmax(probe))
        else:
            position = int(label)
            if not 0 <= position < probe.size:
                raise ValueError(
                    f"label={label} names no class of this model, which returns "
                    f"{probe.size} of them. Pass the position of a column of the "
                    "probability matrix."
                )
        context = PerturbationContext(
            instance=instance,
            label=position,
            wrapper=self.wrapper,
            X_train=self.X_train,
            y_train=self.y_train,
            rng=rng,
            segmentation=seg,
        )
        operator.fit(context)
        operator.warn_if_taper_is_idle(seg)
        objective = Objective(self.wrapper, instance, seg, operator, label=position)
        return objective, operator

    def _initial_order(self, objective, initialization, rng) -> List[int]:
        spec = self.initialization if initialization is None else initialization
        n = objective.n_segments
        if isinstance(spec, str):
            key = spec.lower()
            if key == "random":
                return [int(p) for p in rng.permutation(n)]
            if key == "sequential":
                return greedy_ranking(objective)
            if key == "greedy":
                return objective.greedy_order()
            raise ValueError(
                "initialization must be 'greedy', 'sequential', 'random' or a ranking."
            )
        order = [int(position) for position in spec]
        if sorted(order) != list(range(n)):
            raise ValueError(
                f"An explicit ranking must be a permutation of the {n} segments."
            )
        return order

    def _explain_once(
        self,
        instance,
        seg,
        spec,
        label,
        initialization,
        max_iterations,
        patience,
        n_restarts,
        seed,
        verbose,
    ) -> SOFIExplanation:
        started = time.time()
        rng = np.random.default_rng(seed)
        objective, operator = self._build_objective(instance, seg, spec, label, rng)
        calls_before = objective.n_calls_

        start = self._initial_order(objective, initialization, rng)
        budget = self.max_iterations if max_iterations is None else int(max_iterations)
        result = hill_climbing(
            n_segments=objective.n_segments,
            evaluate=objective.evaluate,
            initial_order=start,
            max_iterations=budget,
            patience=self.patience if patience is None else patience,
            n_restarts=self.n_restarts if n_restarts is None else int(n_restarts),
            accept_equal=self.accept_equal,
            rng=rng,
            verbose=verbose,
        )
        evaluation = result["evaluation"]
        point, rate, _ = objective.sparsity(evaluation)
        lerf_point = objective.flip_point(list(evaluation.order)[::-1])
        gap = modularity_gap(objective) if self.check_modularity else None

        # the curves are reported exactly as the model produced them. Rescaling
        # them onto a common axis used to be the default, and it hid the failure
        # this package now reports openly, namely that a curve can plunge to the
        # floor of a rescaled axis while the probability it describes never moved
        morf = [float(v) for v in evaluation.morf]
        lerf = [float(v) for v in evaluation.lerf]
        # the states the image passes through are small next to the queries
        # already spent, and every figure needs them, so they are always carried
        # rather than hidden behind a flag
        states = objective.marginalized_images(evaluation.order)
        display = getattr(self, "_display", None)

        return SOFIExplanation(
            segmentation=seg,
            order=[int(p) for p in evaluation.order],
            morf_scores=morf,
            lerf_scores=lerf,
            morf_scores_raw=[float(v) for v in evaluation.morf],
            lerf_scores_raw=[float(v) for v in evaluation.lerf],
            ds=degradation_score(morf, lerf),
            ds_raw=float(evaluation.ds),
            anchor_high=float(objective.anchor_high_),
            anchor_low=float(objective.anchor_low_),
            raw_baseline=float(objective.baseline_),
            label=int(objective.label),
            class_name=self.wrapper.class_name(objective.label),
            sparsity_point=point,
            sparsity_rate=rate,
            sparsity_probability=float(evaluation.morf[point]) if point else float(evaluation.morf[-1]),
            lerf_sparsity_point=lerf_point,
            perturbation=operator.summary(),
            modularity_gap=gap,
            instance=instance.copy(),
            display=display,
            states=states,
            history=result["history"],
            n_iterations=result["n_iterations"],
            n_restarts_used=result["n_restarts_used"],
            n_evaluations=result["n_evaluations"],
            n_model_calls=objective.n_calls_ - calls_before,
            runtime=time.time() - started,
        )

    def _explain_best_of(self, instance, seg, candidates, verbose=False, **kwargs) -> SOFIExplanation:
        """Run one search per candidate operator and keep the best outcome.

        No single operator fits every problem, so the choice is treated as a
        hyperparameter resolved per instance. The criterion is not the
        degradation score alone.

        Two failures are screened out before any area is compared. The first is
        an operator that never moves the model, whose curve is a range of a few
        thousandths stretched across the unit interval by the reporting scale,
        so ``min_drop`` requires a real fall in the response. The second is a
        curve that climbs back at the end, which is the recurring failure of
        marginalization on images. Once every segment has been replaced the
        image is no longer a radiograph with pieces removed, it is a flat field
        or a uniformly blurred image, and the response of the model there owes
        nothing to the evidence that was taken away. Such a curve can carry a
        large area while describing nothing, so the selection first discards
        every candidate whose terminal response climbs by more than
        ``surge_tolerance`` of the untouched response above the lowest point it
        reached, and only then maximizes the score among what survives. When no
        candidate survives, the one with the smallest surge is kept and a
        warning says that every operator recovered.

        A small surge is left admissible on purpose. Marginalizing a segment
        that holds nothing but noise can genuinely help the classifier, so a
        curve is not required to be monotone, only to end near its floor rather
        than near its ceiling.
        """
        tolerance = float(kwargs.pop("surge_tolerance", self.surge_tolerance))
        floor = float(kwargs.pop("min_drop", self.min_drop))
        trials = []
        for spec in _progress(candidates, verbose, "SOFI operators"):
            try:
                explanation = self._explain_once(instance, seg, spec, verbose=False, **kwargs)
            except (ValueError, ImportError) as error:
                warnings.warn(
                    f"The '{spec}' operator was skipped, {type(error).__name__}, {error}",
                    stacklevel=3,
                )
                continue
            trials.append(
                {
                    "operator": explanation.perturbation.get("operator"),
                    "detail": explanation.perturbation.get("detail"),
                    "ds": explanation.ds,
                    "sparsity_rate": explanation.sparsity_rate,
                    "drop": explanation.drop,
                    "surge": explanation.surge,
                    "admissible": explanation.drop >= floor and explanation.surge <= tolerance,
                    "explanation": explanation,
                }
            )
        if not trials:
            raise ValueError(
                "Every candidate operator failed, so no explanation could be "
                "produced. Inspect the warnings above for the reasons."
            )

        admissible = [trial for trial in trials if trial["admissible"]]
        if admissible:
            winner = max(admissible, key=lambda trial: trial["ds"])
        else:
            moving = [trial for trial in trials if trial["drop"] >= floor]
            if moving:
                winner = min(moving, key=lambda trial: trial["surge"])
                warnings.warn(
                    "Every operator that moved the model left it recovering by "
                    f"more than {tolerance:.0%} of the untouched response once "
                    f"every segment was marginalized, the mildest being "
                    f"'{winner['operator']}' at {winner['surge']:.0%}. The "
                    "reported area is then hard to read as progressive evidence "
                    "removal.",
                    stacklevel=3,
                )
            else:
                winner = max(trials, key=lambda trial: trial["drop"])
                warnings.warn(
                    "No candidate operator lowered the response by as much as "
                    f"{floor:.0%} of its starting value, the largest fall being "
                    f"{winner['drop']:.1%} under '{winner['operator']}'. The "
                    "degradation scores in the trials describe a range of a few "
                    "thousandths stretched across the unit interval rather than "
                    "evidence that was removed, so none of them should be read "
                    "as a measurement.",
                    stacklevel=3,
                )
        best = winner["explanation"]
        for trial in trials:
            trial.pop("explanation")
        best.perturbation = dict(best.perturbation)
        best.perturbation["trials"] = trials
        return best

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SOFIExplainer(segmentation={self.segmentation!r}, "
            f"marginalization={self.marginalization!r}, backend={self.wrapper.backend})"
        )


def select_reliable_instances(
    wrapper_or_model,
    X,
    y,
    output_fn=None,
    output: str = "auto",
    channels_first="auto",
    classes: Optional[Sequence] = None,
    return_mask: bool = False,
):
    """Keep the instances a classifier gets right.

    The premise of the method is that the response deteriorates as segments are
    marginalized, which only holds for images the model handles well. The
    selection is deliberately left outside the explainer, so the criterion stays
    under the control of the user, and this helper covers the usual one.

    Parameters
    ----------
    wrapper_or_model : object
        A fitted classifier or an already built :class:`ClassifierWrapper`.
    output_fn : str or callable, optional
        Method of the model that produces the outputs, as in
        :class:`ClassifierWrapper`. Ignored when a wrapper is passed.
    output : {"auto", "proba", "logits"}, default="auto"
        Whether the model already returns probabilities. Ignored when a wrapper
        is passed.
    channels_first : {"auto", True, False}, default="auto"
        Whether the model expects the channel axis before the two spatial axes.
        Ignored when a wrapper is passed.
    classes : sequence, optional
        Class labels in the order of the probability columns, used to compare
        the predictions with the labels given.
    X : array
        Images shaped ``(n, height, width)`` or ``(n, height, width,
        n_channels)``.
    y : array
        Their labels, compared against the prediction after both have been
        mapped onto the positions of the probability columns.
    return_mask : bool, default=False
        Whether the boolean mask is returned as well, which is how the matching
        targets are selected.

    Returns
    -------
    ndarray
        The retained instances, and the mask when it is requested.
    """
    data = _as_dataset(X)
    labels = np.asarray(y)
    wrapper = (
        wrapper_or_model
        if isinstance(wrapper_or_model, ClassifierWrapper)
        else ClassifierWrapper(
            wrapper_or_model,
            output_fn=output_fn,
            output=output,
            channels_first=channels_first,
            classes=classes,
        )
    )
    if wrapper.n_classes is None:
        wrapper.calibrate(data[: min(4, len(data))])

    predicted = np.argmax(wrapper.predict_proba(data), axis=1)
    names = [str(name) for name in (wrapper.classes or range(wrapper.n_classes))]
    lookup = {name: position for position, name in enumerate(names)}
    truth = np.array([lookup.get(str(value), -1) for value in labels], dtype=int)
    if np.any(truth < 0):
        # labels that do not match the declared classes are compared as integers
        truth = labels.astype(int)

    mask = predicted == truth
    return (data[mask], mask) if return_mask else data[mask]
