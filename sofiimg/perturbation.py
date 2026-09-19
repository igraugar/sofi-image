"""Operators that neutralize a segment of an image.

Marginalizing a tabular feature is easy, since one statistic of the training
column carries no instance level information. A segment is harder. A flat
patch is itself a pattern, and a model may read the sharp rectangle it leaves
behind as evidence for some class, which keeps the marginalized image
informative and the degradation curve misleading. The operators below therefore
span three levels of sophistication, and the right one is problem dependent.

Two families exist and they carry different readings of the resulting ranking.

**Class-agnostic operators** erase the content of a segment without steering the
prediction anywhere. The substituted values come from the training
distribution, from the segment itself, from a blurred copy of the image or from
noise, and none of them consults the class being explained. A ranking obtained
this way answers the question the method was designed for, namely which segments
the model relies on. These operators are the default.

**Class-directed operators** choose the substitution so that the response of the
model falls as fast as possible, either through a search over constants or by
importing evidence from the other classes. They degrade the prediction sooner
and produce sparser explanations, and the reading changes accordingly. A
ranking obtained this way answers which segments move the model away from the
current class most quickly, which is a counterfactual question rather than a
reliance question. Every explanation states which family produced it, so the
two readings are never confused.

Every operator can request several replicas. The response is then averaged over
the draws, which approximates an expectation instead of the output at one
arbitrary point, at a cost that grows linearly with the draw count.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .segmentation import Segment

__all__ = [
    "PerturbationContext",
    "Perturbation",
    "ConstantPerturbation",
    "RandomConstantPerturbation",
    "NoisePerturbation",
    "SegmentMeanPerturbation",
    "BlurPerturbation",
    "InpaintPerturbation",
    "BackgroundPerturbation",
    "LineSearchPerturbation",
    "RoadPerturbation",
    "LowNegPerturbation",
    "OppositeClassMeanPerturbation",
    "NearestUnlikeNeighborPerturbation",
    "make_perturbation",
    "PERTURBATIONS",
    "CLASS_AGNOSTIC",
    "CLASS_DIRECTED",
    "DEFAULT_CANDIDATES",
]


@dataclass
class PerturbationContext:
    """Everything an operator may consult before it starts substituting.

    Attributes
    ----------
    instance : ndarray
        The image being explained, shaped ``(height, width, n_channels)``.
    label : int
        Position of the class whose probability the search tracks, normally the
        class the model predicts for the unperturbed image.
    wrapper : ClassifierWrapper
        Access to the black box, needed by the operators that search.
    X_train : ndarray, optional
        Training images shaped ``(n, height, width, n_channels)``, the only
        source of substitution values. The data being explained never
        contributes statistics.
    y_train : ndarray, optional
        Training labels, needed by the class-directed operators alone.
    rng : Generator
        Source of randomness, so that a run is reproducible.
    segmentation : Segmentation, optional
        The partition the ranking will permute. Only the operators that need to
        know what the fully marginalized image looks like consult it, and they
        need it because that image is the last point of every curve.
    """

    instance: np.ndarray
    label: int
    wrapper: object
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    segmentation: Optional[object] = None

    def require_training(self, operator: str) -> np.ndarray:
        """Training data, or a readable error naming the operator that needs it.

        Parameters
        ----------
        operator : str
            Name of the operator, used in the message.
        """
        if self.X_train is None:
            raise ValueError(
                f"The '{operator}' operator needs training data, since it is the "
                "only admissible source of substitution values. Pass X_train to "
                "the explainer."
            )
        return np.asarray(self.X_train, dtype=float)

    def require_labels(self, operator: str) -> np.ndarray:
        """Training labels, or a readable error naming the operator that needs them.

        Parameters
        ----------
        operator : str
            Name of the operator, used in the message.
        """
        if self.y_train is None:
            raise ValueError(
                f"The '{operator}' operator needs training labels, since it "
                "draws its substitution from the classes other than the one "
                "being explained. Pass y_train to the explainer."
            )
        return np.asarray(self.y_train)

    @property
    def statistics(self) -> Dict[str, float]:
        """Statistics of the training data pooled over every channel."""
        data = self.require_training("statistics")
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
        }

    @property
    def channel_statistics(self) -> Dict[str, np.ndarray]:
        """The same statistics computed for each channel separately.

        A preprocessing pipeline that standardizes the channels of an image
        with the constants of a pretrained backbone leaves them centered on
        different values, so one pooled constant would neutralize one channel
        while it introduced a visible cast in another. Every constant operator
        therefore reads the statistic of the channel it is writing into, which
        reduces to the pooled value when the image is grayscale.
        """
        data = self.require_training("channel statistics")
        axes = (0, 1, 2)
        return {
            "min": np.min(data, axis=axes),
            "max": np.max(data, axis=axes),
            "mean": np.mean(data, axis=axes),
            "median": np.median(data, axis=axes),
            "std": np.std(data, axis=axes),
        }


class Perturbation:
    """Base class of every substitution operator.

    Subclasses implement :meth:`_fit`, which prepares whatever the operator
    needs, and :meth:`replacement`, which returns the values that replace the
    pixels of one segment on one replica.

    Parameters
    ----------
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step. Operators
        that draw randomly benefit from several replicas, and deterministic
        ones ignore the argument.
    taper : int, default=0
        Width in pixels of the band over which the substituted values are
        blended into the surrounding image along the border of a segment. A flat
        patch introduces a sharp contour that no natural image holds, and
        convolutional models react to edges, so part of a measured degradation
        may be a boundary artifact rather than lost evidence. The default of
        zero reproduces the published behavior.
    """

    name = "perturbation"
    class_directed = False
    interpretation = (
        "Segments are ranked by how much the prediction relies on them."
    )

    def __init__(self, n_replicas: int = 1, taper: int = 0):
        if int(n_replicas) < 1:
            raise ValueError("n_replicas must be a positive integer.")
        if int(taper) < 0:
            raise ValueError("taper must be zero or a positive integer.")
        self.n_replicas = int(n_replicas)
        self.taper = int(taper)
        self.context: Optional[PerturbationContext] = None
        self._weights: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, context: PerturbationContext) -> "Perturbation":
        """Prepare the operator for one image.

        Parameters
        ----------
        context : PerturbationContext
            Everything the operator may consult, namely the image, the class
            being explained, the classifier, the training data and its labels,
            and the source of randomness.

        Returns
        -------
        Perturbation
            The operator itself, so that the call chains.
        """
        self.context = context
        self._weights = {}
        self._fit(context)
        return self

    def _fit(self, context: PerturbationContext) -> None:  # pragma: no cover - hook
        return None

    # -------------------------------------------------------------- applying
    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """Values that replace one segment on one replica.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the ``n_replicas`` independent substitutions is wanted.
            Deterministic operators ignore it.

        Returns
        -------
        ndarray
            One row per pixel of the segment and one column per channel. A
            single value or a one dimensional array is broadcast.
        """
        raise NotImplementedError

    def warn_if_taper_is_idle(self, segmentation) -> None:
        """Say so when the blending band is wider than the segments it applies to.

        The band is measured inwards from the border of a segment, so it needs a
        segment whose interior is deeper than the band itself. Announcing the
        shortfall prevents a reader from crediting the blending with a result it
        never touched.

        Parameters
        ----------
        segmentation : Segmentation
            The partition the operator will be applied to.
        """
        if self.taper <= 0:
            return
        smallest = int(min(segment.area for segment in segmentation))
        radius = 0.5 * np.sqrt(smallest)
        if radius <= self.taper:
            warnings.warn(
                f"taper={self.taper} spans a band wider than the smallest segment "
                f"of the partition, which holds {smallest} pixels. Those segments "
                "are blended over their whole extent, so they are never fully "
                "neutralized and their degradation is understated.",
                stacklevel=3,
            )

    def _taper_weights(self, segment: Segment) -> np.ndarray:
        """Blending weight of every pixel of a segment, one at its core.

        The weight follows the distance of a pixel to the border of its segment,
        so the substitution takes over gradually instead of stopping at a hard
        contour. The values are computed once per segment and reused.
        """
        if segment.index in self._weights:
            return self._weights[segment.index]
        from scipy.ndimage import distance_transform_edt

        row_min, row_max, col_min, col_max = segment.bbox
        # the transform runs inside the bounding box with a one pixel frame, so
        # the border of the segment is measured against the outside of it
        window = np.zeros((row_max - row_min + 2, col_max - col_min + 2), dtype=bool)
        window[segment.rows - row_min + 1, segment.cols - col_min + 1] = True
        distance = distance_transform_edt(window)
        values = distance[segment.rows - row_min + 1, segment.cols - col_min + 1]
        weights = np.clip(values / float(self.taper + 1), 0.0, 1.0)
        self._weights[segment.index] = weights
        return weights

    def apply(self, work: np.ndarray, segment: Segment, replica: int = 0) -> None:
        """Substitute one segment of ``work``, in place.

        Parameters
        ----------
        work : ndarray
            A single image shaped ``(height, width, n_channels)``, which the
            objective mutates cumulatively along a ranking.
        segment : Segment
            The segment to neutralize.
        replica : int, default=0
            Which of the independent substitutions is applied.
        """
        values = np.asarray(self.replacement(segment, replica), dtype=float)
        n_channels = work.shape[-1]
        if values.ndim == 0:
            values = np.full((segment.area, n_channels), float(values))
        elif values.ndim == 1:
            values = (
                np.repeat(values[None, :], segment.area, axis=0)
                if values.size == n_channels
                else np.repeat(values[:, None], n_channels, axis=1)
            )
        if self.taper > 0:
            weights = self._taper_weights(segment)[:, None]
            original = self.context.instance[segment.rows, segment.cols, :]
            values = weights * values + (1.0 - weights) * original
        work[segment.rows, segment.cols, :] = values

    # -------------------------------------------------------------- reporting
    def describe(self) -> str:
        return self.name

    def summary(self) -> Dict[str, object]:
        return {
            "operator": self.name,
            "class_directed": self.class_directed,
            "n_replicas": self.n_replicas,
            "taper": self.taper,
            "detail": self.describe(),
            "interpretation": self.interpretation,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.describe()})"


# ------------------------------------------------------- class agnostic
class ConstantPerturbation(Perturbation):
    """Replace a segment with one constant estimated on the training data.

    This is the family used in the paper. The statistic is global, so it is the
    same for every segment, which keeps the substituted level typical of the data
    the model was trained on. The training mean is the default of the package
    and it is the closest image counterpart of the gray patch that the occlusion
    literature places over a segment.

    Parameters
    ----------
    statistic : {"mean", "min", "max", "median", "zero"} or float
        Value that replaces the segment. A float is used verbatim, which covers
        a constant chosen from domain knowledge, such as the value the
        background of a radiograph takes after the preprocessing.
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step.
    taper : int, default=0
        Width in pixels of the band over which the substitution is blended into
        the surrounding image. Zero reproduces the published behavior.
    """

    name = "constant"

    def __init__(self, statistic="mean", n_replicas: int = 1, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.statistic = statistic
        self.value_: Optional[float] = None

    def _fit(self, context: PerturbationContext) -> None:
        n_channels = np.asarray(context.instance).shape[-1]
        if isinstance(self.statistic, (int, float, np.floating, np.integer)):
            self.values_ = np.full(n_channels, float(self.statistic))
        elif str(self.statistic).lower() == "zero":
            self.values_ = np.zeros(n_channels)
        else:
            key = str(self.statistic).lower()
            stats = context.channel_statistics
            if key not in stats:
                raise ValueError(
                    f"Unknown constant statistic '{self.statistic}'. Choose among "
                    "'mean', 'min', 'max', 'median', 'zero', or pass a number."
                )
            values = np.asarray(stats[key], dtype=float).reshape(-1)
            if values.size < n_channels:
                values = np.resize(values, n_channels)
            self.values_ = values[:n_channels]
        self.value_ = float(np.mean(self.values_))

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The constant, one value per channel.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per channel, broadcast over the pixels of the segment.
        """
        return self.values_

    def describe(self) -> str:
        if self.values_.size > 1 and float(np.ptp(self.values_)) > 1e-9:
            return (
                f"constant {self.statistic} per channel, from "
                f"{self.values_.min():.4f} to {self.values_.max():.4f}"
            )
        return f"constant {self.statistic} at {self.value_:.4f}"


class RandomConstantPerturbation(Perturbation):
    """Replace a segment with a constant drawn uniformly from the training range.

    A fresh value is drawn for every segment and every replica, so the operator
    carries no systematic level of its own. Several replicas are recommended,
    since a single draw is one arbitrary point of the range.

    Parameters
    ----------
    n_replicas : int, default=5
        Number of independent substitutions averaged at every step.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    """

    name = "random"

    def __init__(self, n_replicas: int = 5, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.draws_: Dict[tuple, np.ndarray] = {}

    def _fit(self, context: PerturbationContext) -> None:
        stats = context.channel_statistics
        self.low_ = np.asarray(stats["min"], dtype=float).reshape(-1)
        self.high_ = np.asarray(stats["max"], dtype=float).reshape(-1)
        self.draws_ = {}
        self._rng = context.rng

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """One uniform draw per channel, fixed per segment and replica.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per channel.
        """
        key = (segment.index, replica)
        if key not in self.draws_:
            self.draws_[key] = self._rng.uniform(self.low_, self.high_)
        return self.draws_[key]

    def describe(self) -> str:
        return f"uniform draw on [{self.low_.min():.3f}, {self.high_.max():.3f}]"


class NoisePerturbation(Perturbation):
    """Replace a segment with Gaussian noise matched to the training data.

    Level and dispersion follow the training set, so the substituted patch stays
    plausible in scale while carrying no spatial structure. The draws are fixed
    once per segment and replica, so competing rankings meet identical
    conditions.

    Parameters
    ----------
    n_replicas : int, default=5
        Number of independent draws averaged at every step.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    scale : float, default=1.0
        Multiplier applied to the deviation of the training data, so that the
        noise can be made milder or harsher than the content it replaces.
    """

    name = "noise"

    def __init__(self, n_replicas: int = 5, taper: int = 0, scale: float = 1.0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.scale = float(scale)
        self.draws_: Dict[tuple, np.ndarray] = {}

    def _fit(self, context: PerturbationContext) -> None:
        stats = context.channel_statistics
        self.center_ = np.asarray(stats["mean"], dtype=float).reshape(-1)
        self.spread_ = np.asarray(stats["std"], dtype=float).reshape(-1) * self.scale
        self.draws_ = {}
        self._rng = context.rng

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """Gaussian noise, fixed per segment and replica.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        key = (segment.index, replica)
        if key not in self.draws_:
            self.draws_[key] = self._rng.normal(
                self.center_, self.spread_, size=(segment.area, self.center_.size)
            )
        return self.draws_[key]

    def describe(self) -> str:
        return (
            f"Gaussian noise per channel, spread from {self.spread_.min():.3f} "
            f"to {self.spread_.max():.3f}"
        )


class SegmentMeanPerturbation(Perturbation):
    """Flatten a segment onto its own average color.

    Level and position are preserved and the texture is destroyed, which
    isolates the contribution of the local pattern from the contribution of the
    brightness. The substitution is local to the image, so no training statistic
    enters, and the marginalized image stays close to the data manifold. This is
    the operator the segment attribution literature applies most often.
    """

    name = "segment_mean"

    def _fit(self, context: PerturbationContext) -> None:
        self.instance_ = np.asarray(context.instance, dtype=float)

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The average color of the segment itself.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per channel.
        """
        return self.instance_[segment.rows, segment.cols, :].mean(axis=0)

    def describe(self) -> str:
        return "the average color of the segment itself"


class BlurPerturbation(Perturbation):
    """Replace a segment with its blurred counterpart.

    The segment keeps its brightness and its coarse layout and loses the detail,
    so what is removed is the texture and the edges the model may be reading
    rather than the presence of tissue. No contour is introduced, since the
    blurred image agrees with the original away from the segment, which makes
    this the closest image counterpart of the linear interpolation the time
    series version applies between two breakpoints. It is the operator to prefer
    when a flat patch would itself look like a lesion.

    Parameters
    ----------
    sigma : float, default=8.0
        Width of the Gaussian kernel, in pixels. A width comparable to the size
        of a segment removes its content, and a much smaller one leaves it
        legible.
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    """

    name = "blur"

    def __init__(self, sigma: float = 8.0, n_replicas: int = 1, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        if float(sigma) <= 0:
            raise ValueError("sigma must be a positive number of pixels.")
        self.sigma = float(sigma)

    def _fit(self, context: PerturbationContext) -> None:
        from scipy.ndimage import gaussian_filter

        instance = np.asarray(context.instance, dtype=float)
        # the whole image is blurred once and every segment reads its own pixels
        # from the result, so the cost does not grow with the segment count
        self.blurred_ = gaussian_filter(
            instance, sigma=(self.sigma, self.sigma, 0.0), mode="nearest"
        )

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The blurred image restricted to the segment.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        return self.blurred_[segment.rows, segment.cols, :]

    def describe(self) -> str:
        return f"a Gaussian blur of width {self.sigma:.1f} pixels"


class InpaintPerturbation(Perturbation):
    """Replace a segment with a biharmonic reconstruction from its border.

    The substituted patch is the smoothest continuation of the surrounding
    tissue, so the marginalized image holds no segment that looks removed and the
    model is asked what the image would say had the evidence never been there.
    This is the most faithful marginalization of the package and the most
    expensive one, since a linear system is solved for every segment. The results
    are cached, so a search that revisits a ranking pays the cost once.

    Parameters
    ----------
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step. The
        operator is deterministic, so more than one draw only adds cost.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    """

    name = "inpaint"

    def __init__(self, n_replicas: int = 1, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.cache_: Dict[int, np.ndarray] = {}

    def _fit(self, context: PerturbationContext) -> None:
        self.instance_ = np.asarray(context.instance, dtype=float)
        self.cache_ = {}

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The inpainted image restricted to the segment.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        if segment.index in self.cache_:
            return self.cache_[segment.index]
        from skimage.restoration import inpaint_biharmonic

        # only a frame around the segment takes part in the reconstruction, since
        # a biharmonic solve over the whole image would cost far more and change
        # nothing away from the border
        pad = 8
        row_min = max(0, segment.bbox[0] - pad)
        row_max = min(self.instance_.shape[0], segment.bbox[1] + pad)
        col_min = max(0, segment.bbox[2] - pad)
        col_max = min(self.instance_.shape[1], segment.bbox[3] + pad)
        window = self.instance_[row_min:row_max, col_min:col_max, :]
        hole = np.zeros(window.shape[:2], dtype=bool)
        hole[segment.rows - row_min, segment.cols - col_min] = True
        filled = inpaint_biharmonic(window, hole, channel_axis=-1)
        values = filled[segment.rows - row_min, segment.cols - col_min, :]
        self.cache_[segment.index] = values
        return values

    def describe(self) -> str:
        return "a biharmonic reconstruction from the border of the segment"


class BackgroundPerturbation(Perturbation):
    """Replace a segment with the same pixels of training images.

    The substituted patch is a real piece of a real image, so the marginalized
    instance stays on the data manifold and the model is never asked about an
    input unlike anything it saw. Averaging over several draws approximates the
    expectation of the response when the segment carries no instance specific
    information, which is the definition marginalization is built on. The
    operator assumes the images are registered, which radiographs of one
    protocol are and photographs of arbitrary scenes are not.

    Parameters
    ----------
    n_replicas : int, default=10
        Number of training images drawn. Larger values smooth the expectation
        and raise the cost linearly.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    exclude_label : bool, default=False
        Whether images of the explained class are excluded from the draws.
        Leaving it false keeps the operator class-agnostic, which is the
        recommended setting.
    """

    name = "background"

    def __init__(self, n_replicas: int = 10, taper: int = 0, exclude_label: bool = False):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.exclude_label = bool(exclude_label)
        self.class_directed = bool(exclude_label)

    def _fit(self, context: PerturbationContext) -> None:
        data = context.require_training(self.name)
        if self.exclude_label:
            labels = np.asarray(context.require_labels(self.name))
            keep = _as_positions(labels, context) != int(context.label)
            if not keep.any():
                raise ValueError(
                    "Every training image belongs to the class being explained, "
                    "so no background outside it exists."
                )
            data = data[keep]
        size = min(self.n_replicas, len(data))
        positions = context.rng.choice(len(data), size=size, replace=False)
        self.pool_ = data[np.sort(positions)]
        self.n_replicas = size

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching pixels of one training image.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        source = self.pool_[replica % len(self.pool_)]
        return source[segment.rows, segment.cols, :]

    def describe(self) -> str:
        scope = (
            "images outside the explained class"
            if self.exclude_label
            else "training images"
        )
        return f"the matching pixels of {len(self.pool_)} {scope}"


# -------------------------------------------------------- class directed
_DIRECTED_READING = (
    "Segments are ranked by how quickly they move the model away from the "
    "explained class, which is a counterfactual reading rather than a reliance "
    "reading."
)


class LineSearchPerturbation(Perturbation):
    """Constant found by the golden section search of the paper.

    The interval spanned by the training data is narrowed according to the
    golden ratio, and the constant retained is the one whose flat image receives
    the lowest probability for the explained class. The extremes, the training
    mean, a uniform draw and the midpoint of the final interval are evaluated
    alongside the outcome of the search, exactly as in Algorithm 2.

    Under ``admissible``, the search is restricted to constants whose flat image
    is not assigned the explained class at all. The substitution is then known
    in advance to carry no evidence for that class, which is the strongest
    guarantee a constant can offer. When no such constant exists, the operator
    says so and falls back to the plain minimizer, and the fact that the model
    reads a uniform field as the explained class is itself worth knowing.

    Parameters
    ----------
    tolerance : float, default=1e-3
        Width of the interval below which the search stops.
    max_iterations : int, default=100
        Upper bound on the narrowing steps.
    admissible : bool, default=False
        Whether constants assigned to the explained class are rejected.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    """

    name = "line_search"
    class_directed = True
    interpretation = _DIRECTED_READING

    def __init__(
        self,
        tolerance: float = 1e-3,
        max_iterations: int = 100,
        admissible: bool = False,
        taper: int = 0,
    ):
        super().__init__(n_replicas=1, taper=taper)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.admissible = bool(admissible)

    def _fit(self, context: PerturbationContext) -> None:
        stats = context.statistics
        instance = np.asarray(context.instance, dtype=float)
        wrapper, label = context.wrapper, int(context.label)
        self.fallback_ = False

        def probe(value: float):
            candidate = np.full_like(instance, float(value))
            probabilities = wrapper.predict_proba(candidate[None, ...])[0]
            return float(probabilities[label]), int(np.argmax(probabilities))

        phi = (1.0 + 5.0 ** 0.5) / 2.0
        low, high = stats["min"], stats["max"]
        c = high - (high - low) / phi
        d = low + (high - low) / phi
        f_c, _ = probe(c)
        f_d, _ = probe(d)

        for _ in range(self.max_iterations):
            if abs(high - low) < self.tolerance:
                break
            if f_c < f_d:
                high, d, f_d = d, c, f_c
                c = high - (high - low) / phi
                f_c, _ = probe(c)
            else:
                low, c, f_c = c, d, f_d
                d = low + (high - low) / phi
                f_d, _ = probe(d)

        candidates = [
            stats["min"],
            stats["max"],
            stats["mean"],
            float(context.rng.uniform(stats["min"], stats["max"])),
            0.5 * (low + high),
        ]
        scored = [(value,) + probe(value) for value in candidates]

        if self.admissible:
            outside = [item for item in scored if item[2] != label]
            if outside:
                best = min(outside, key=lambda item: item[1])
                self.value_ = float(best[0])
                self.escaped_ = True
                self.probability_ = float(best[1])
                return
            self.fallback_ = True

        best = min(scored, key=lambda item: item[1])
        self.value_ = float(best[0])
        self.probability_ = float(best[1])
        self.escaped_ = bool(best[2] != label)

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The constant found by the golden section search.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            The constant, broadcast over the pixels of the segment.
        """
        return np.asarray(self.value_, dtype=float)

    def describe(self) -> str:
        verdict = (
            "leaves the explained class"
            if self.escaped_
            else "stays inside the explained class"
        )
        note = ", no admissible constant existed" if self.fallback_ else ""
        return (
            f"constant {self.value_:.4f} from the golden section search, which "
            f"{verdict} at probability {self.probability_:.3f}{note}"
        )



class RoadPerturbation(Perturbation):
    """Noisy linear imputation, the removal operator of the ROAD evaluation.

    Rong and colleagues showed that the ordinary removal operators leak the shape
    of the removed area back to the model. A constant patch, a blurred patch or a
    zeroed one all leave a silhouette, and a classifier can read the silhouette
    itself, so a curve measured that way partly reports the mask rather than the
    evidence. Their remedy, published as ROAD, replaces every removed pixel with
    a weighted average of its immediate neighbors, weighting the four orthogonal
    ones fully and the four diagonal ones by the inverse of the square root of
    two, and adds a little noise so that the reconstruction carries no exact
    information either. The values of the removed area are the solution of the
    linear system those equations define, which is why the operator is called a
    noisy linear imputation rather than a blur.

    The reconstruction is solved on a small window around the segment rather than
    over the whole image, since the equations away from the border contribute
    nothing, and the result is cached, so a search that revisits a ranking pays
    the cost once. Being class agnostic, it answers the question the method was
    designed for.

    Parameters
    ----------
    noise : float, default=0.01
        Deviation of the noise added to the reconstruction, as a multiple of the
        deviation of the training data. Zero recovers a plain linear imputation.
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.

    References
    ----------
    Rong, Leemann, Borisov, Kasneci and Kasneci, A Consistent and Efficient
    Evaluation Strategy for Attribution Methods, ICML 2022.
    """

    name = "road"

    def __init__(self, noise: float = 0.01, n_replicas: int = 1, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.noise = float(noise)
        self.cache_: Dict[int, np.ndarray] = {}

    def _fit(self, context: PerturbationContext) -> None:
        self.instance_ = np.asarray(context.instance, dtype=float)
        self.spread_ = float(np.std(context.require_training(self.name)))
        self.cache_ = {}
        self._rng = context.rng

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The noisy linear reconstruction restricted to the segment.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        if segment.index not in self.cache_:
            self.cache_[segment.index] = self._impute(segment)
        values = self.cache_[segment.index]
        if self.noise > 0:
            values = values + self._rng.normal(
                0.0, self.noise * self.spread_, size=values.shape
            )
        return values

    def _impute(self, segment: Segment) -> np.ndarray:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        pad = 2
        height, width, n_channels = self.instance_.shape
        row_min = max(0, segment.bbox[0] - pad)
        row_max = min(height, segment.bbox[1] + pad)
        col_min = max(0, segment.bbox[2] - pad)
        col_max = min(width, segment.bbox[3] + pad)
        window = self.instance_[row_min:row_max, col_min:col_max, :]
        rows, cols = window.shape[:2]

        hole = np.zeros((rows, cols), dtype=bool)
        hole[segment.rows - row_min, segment.cols - col_min] = True
        index = np.full((rows, cols), -1, dtype=int)
        unknown = np.argwhere(hole)
        index[hole] = np.arange(len(unknown))

        # the eight neighbors, orthogonal ones weighted fully and diagonal ones
        # by the inverse of the square root of two, exactly as ROAD prescribes
        offsets = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 2.0 ** -0.5), (-1, 1, 2.0 ** -0.5),
            (1, -1, 2.0 ** -0.5), (1, 1, 2.0 ** -0.5),
        ]
        data, ridx, cidx = [], [], []
        rhs = np.zeros((len(unknown), n_channels), dtype=float)
        for position, (r, c) in enumerate(unknown):
            total = 0.0
            for dr, dc, weight in offsets:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                total += weight
                if hole[rr, cc]:
                    data.append(-weight)
                    ridx.append(position)
                    cidx.append(index[rr, cc])
                else:
                    rhs[position] += weight * window[rr, cc, :]
            if total <= 0:  # pragma: no cover - a segment with no neighbor
                total = 1.0
                rhs[position] += window[r, c, :]
            data.append(total)
            ridx.append(position)
            cidx.append(position)

        matrix = sparse.csr_matrix(
            (data, (ridx, cidx)), shape=(len(unknown), len(unknown))
        )
        filled = np.column_stack(
            [spsolve(matrix, rhs[:, channel]) for channel in range(n_channels)]
        )
        filled = np.asarray(filled, dtype=float).reshape(len(unknown), n_channels)

        order = index[segment.rows - row_min, segment.cols - col_min]
        return filled[order]

    def describe(self) -> str:
        return (
            "a noisy linear imputation from the neighbors of the segment, with "
            f"noise at {self.noise:.3f} of the training deviation"
        )


class LowNegPerturbation(Perturbation):
    """Replace a segment with the matching patch of a low scoring reference.

    The reference is a training image the model assigns a near zero probability
    of the explained class, drawn at random from the pool of the lowest scoring
    ones and then held fixed for the whole run. Every substituted patch is
    therefore a real piece of a real image that the model already reads as
    belonging elsewhere, so the marginalized instance stays inside the training
    distribution at every step and the fully marginalized image is a plausible
    instance of another class rather than a flat field.

    That is what removes the surge. A constant leaves an image the model has
    never seen and may score arbitrarily, whereas the terminal state of this
    operator is the reference itself, whose score is known in advance to be low.
    The operator assumes the images are registered, which radiographs of one
    protocol are and photographs of arbitrary scenes are not.

    Parameters
    ----------
    pool_size : int, default=10
        Number of lowest scoring images the reference is drawn from. One is
        enough when the pool is reliable, and a larger pool trades a little
        determinism for robustness to a single unusual image.
    max_candidates : int, default=200
        Largest number of training images scored while the pool is assembled,
        which is what bounds the cost of the operator.
    reference : array, optional
        An explicit reference image, which skips the search entirely.
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step. The operator
        is deterministic once the reference is fixed.
    taper : int, default=0
        Width in pixels of the blending band along the border of a segment.
    """

    name = "lowneg"
    class_directed = True
    interpretation = _DIRECTED_READING

    def __init__(
        self,
        pool_size: int = 10,
        max_candidates: int = 200,
        reference=None,
        n_replicas: int = 1,
        taper: int = 0,
    ):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.pool_size = int(pool_size)
        self.max_candidates = int(max_candidates)
        self.reference = reference

    def _fit(self, context: PerturbationContext) -> None:
        instance = np.asarray(context.instance, dtype=float)
        if self.reference is not None:
            template = np.asarray(self.reference, dtype=float)
            if template.ndim == 2:
                template = template[..., None]
            if template.shape != instance.shape:
                raise ValueError(
                    f"The reference image is shaped {template.shape} and the "
                    f"image being explained holds {instance.shape}. Both must "
                    "agree, since the substitution copies matching pixels."
                )
            self.template_ = template
            self.score_ = float(
                context.wrapper.predict_proba(template[None, ...])[0, int(context.label)]
            )
            self.pool_used_ = 1
            return

        data = context.require_training(self.name)
        if len(data) > self.max_candidates:
            positions = context.rng.choice(len(data), size=self.max_candidates, replace=False)
            data = data[np.sort(positions)]

        # the labels are consulted when they exist, so that the reference comes
        # from another class rather than merely from a low scoring image of the
        # explained one, and the operator still works without them
        if context.y_train is not None:
            codes = _as_positions(np.asarray(context.y_train), context)
            if len(codes) == len(context.X_train):
                keep = codes[: len(data)] != int(context.label) if len(data) == len(codes) else None
                if keep is not None and keep.any():
                    data = data[keep]

        scores = context.wrapper.predict_proba(data)[:, int(context.label)]
        pool = np.argsort(scores)[: max(1, min(self.pool_size, len(scores)))]
        chosen = int(context.rng.choice(pool))
        self.template_ = data[chosen]
        self.score_ = float(scores[chosen])
        self.pool_used_ = int(len(pool))

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching pixels of the reference image.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        return self.template_[segment.rows, segment.cols, :]

    def describe(self) -> str:
        return (
            f"the matching pixels of a reference drawn from the {self.pool_used_} "
            f"lowest scoring images, which the model scores {self.score_:.4f}"
        )


class OppositeClassMeanPerturbation(Perturbation):
    """Replace a segment with the average image of the other classes.

    The paper describes this as the starting point of the constant search. Used
    on its own, it imports the appearance typical of everything the explained
    class is not, so the marginalized instance drifts toward the alternatives
    rather than toward nothing.
    """

    name = "opposite_mean"
    class_directed = True
    interpretation = _DIRECTED_READING

    def _fit(self, context: PerturbationContext) -> None:
        data = context.require_training(self.name)
        labels = np.asarray(context.require_labels(self.name))
        codes = _as_positions(labels, context)
        keep = codes != int(context.label)
        if not keep.any():
            raise ValueError(
                "Every training image belongs to the explained class, so the "
                "average of the other classes does not exist."
            )
        self.template_ = data[keep].mean(axis=0)
        self.n_used_ = int(keep.sum())

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching pixels of the mean image of the other classes.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        return self.template_[segment.rows, segment.cols, :]

    def describe(self) -> str:
        return (
            f"the mean image of {self.n_used_} instances outside the explained class"
        )


class NearestUnlikeNeighborPerturbation(Perturbation):
    """Replace a segment with the same pixels of the nearest unlike neighbor.

    The donor is the training image of another class closest to the one being
    explained under a Euclidean distance. Every substituted patch is therefore
    real, plausible and drawn from an image the model assigns elsewhere, which
    makes the degradation both fast and realistic. The operator follows the
    logic of counterfactual explanation methods, and its ranking should be read
    accordingly.
    """

    name = "nearest_unlike"
    class_directed = True
    interpretation = _DIRECTED_READING

    def _fit(self, context: PerturbationContext) -> None:
        data = context.require_training(self.name)
        labels = np.asarray(context.require_labels(self.name))
        codes = _as_positions(labels, context)
        keep = codes != int(context.label)
        if not keep.any():
            raise ValueError(
                "No training image outside the explained class was found, so "
                "the nearest unlike neighbor does not exist."
            )
        pool = data[keep]
        instance = np.asarray(context.instance, dtype=float)
        distances = np.linalg.norm(
            pool.reshape(len(pool), -1) - instance.reshape(1, -1), axis=1
        )
        self.position_ = int(np.argmin(distances))
        self.template_ = pool[self.position_]
        self.distance_ = float(distances[self.position_])

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching pixels of the nearest unlike neighbor.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per pixel of the segment and per channel.
        """
        return self.template_[segment.rows, segment.cols, :]

    def describe(self) -> str:
        return (
            "the matching pixels of the nearest unlike neighbor at distance "
            f"{self.distance_:.3f}"
        )


def _as_positions(labels: np.ndarray, context: PerturbationContext) -> np.ndarray:
    """Map training labels onto probability column positions."""
    classes = getattr(context.wrapper, "classes", None)
    if classes is not None:
        lookup = {str(value): position for position, value in enumerate(classes)}
        mapped = [lookup.get(str(value)) for value in labels]
        if all(item is not None for item in mapped):
            return np.array(mapped, dtype=int)
    try:
        return labels.astype(int)
    except (TypeError, ValueError):
        _, codes = np.unique(labels, return_inverse=True)
        return codes


# ------------------------------------------------------------------ registry
PERTURBATIONS = {
    "zero": lambda **kw: ConstantPerturbation("zero", **kw),
    "mean": lambda **kw: ConstantPerturbation("mean", **kw),
    "median": lambda **kw: ConstantPerturbation("median", **kw),
    "min": lambda **kw: ConstantPerturbation("min", **kw),
    "max": lambda **kw: ConstantPerturbation("max", **kw),
    "random": RandomConstantPerturbation,
    "noise": NoisePerturbation,
    "segment_mean": SegmentMeanPerturbation,
    "blur": BlurPerturbation,
    "inpaint": InpaintPerturbation,
    "background": BackgroundPerturbation,
    "road": RoadPerturbation,
    "lowneg": LowNegPerturbation,
    "line_search": LineSearchPerturbation,
    "admissible_line_search": lambda **kw: LineSearchPerturbation(admissible=True, **kw),
    "opposite_mean": OppositeClassMeanPerturbation,
    "nearest_unlike": NearestUnlikeNeighborPerturbation,
}

CLASS_AGNOSTIC = (
    "zero",
    "mean",
    "median",
    "min",
    "max",
    "random",
    "noise",
    "segment_mean",
    "blur",
    "inpaint",
    "background",
    "road",
)

CLASS_DIRECTED = (
    "lowneg",
    "line_search",
    "admissible_line_search",
    "opposite_mean",
    "nearest_unlike",
)

# candidates explored under marginalization="auto". Six are class agnostic and
# one is class directed, and the selection reports which family won, since the
# reading of the ranking depends on it
DEFAULT_CANDIDATES = ("mean", "min", "max", "blur", "segment_mean", "road", "lowneg")

# aliases kept so that the settings of the paper can be named as published, and
# so that the operators of the time series package answer to their own names
_ALIASES = {
    "constant_min": "min",
    "constant_max": "max",
    "constant_mean": "mean",
    "local_search": "line_search",
    "nun": "nearest_unlike",
    "gray": "mean",
    "noisy_linear": "road",
    "road_imputation": "road",
    "linear": "blur",
    "region_mean": "segment_mean",
}


def make_perturbation(spec, **kwargs) -> Perturbation:
    """Resolve the ``marginalization`` argument of the explainer.

    Parameters
    ----------
    spec : Perturbation, str or float
        A ready operator, which is returned untouched, the name of a registered
        one, listed in ``PERTURBATIONS``, or a number, which becomes a constant
        substitution at that value.
    **kwargs : dict
        Arguments of the operator, such as ``n_replicas``, ``taper`` and
        ``sigma``.

    Returns
    -------
    Perturbation
        The operator, not yet fitted on any image.
    """
    if isinstance(spec, Perturbation):
        return spec
    if isinstance(spec, (int, float, np.floating, np.integer)):
        return ConstantPerturbation(float(spec), **kwargs)
    key = _ALIASES.get(str(spec).lower(), str(spec).lower())
    if key not in PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation '{spec}'. Registered operators are "
            f"{sorted(PERTURBATIONS)}, and a number is read as a constant."
        )
    factory = PERTURBATIONS[key]
    try:
        return factory(**kwargs)
    except TypeError as error:
        raise TypeError(
            f"The '{key}' operator does not accept every argument given in "
            f"marginalization_params, {error}. Arguments such as 'sigma' belong "
            "to one operator alone."
        ) from error
