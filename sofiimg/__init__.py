"""SOFI for image classification.

Sparseness Optimized Feature Importance is a model agnostic post hoc explainer.
An explanation is an order of image segments whose cumulative marginalization
degrades the response of a classifier as fast as possible, and its quality is
the area between the two perturbation curves that order induces. Maximizing that
area rewards sparsity and faithfulness at once, so the explanation isolates the
smallest set of segments that changes the decision.

The segments are the unit of interpretation and they belong to the user. SLIC
segments are the default, since they follow the content of the image and
stay compact enough to point at, a rectangular patch grid is what makes several
explanations comparable and what aligns an explanation with the tokens of a
vision transformer, and and a label map produced by any
other library enters through the same argument.

This package is the image counterpart of ``sofits``. Parameter names, the
explainer interface, the explanation object and the figures of the degradation
curves are shared, so code written against one reads the other.
"""

from .animation import animate_marginalization
from .baselines import (
    compare_rankings,
    feature_occlusion,
    random_ranking,
    ranking_from_scores,
    segment_scores,
)
from .data import ImageDataset, load_dataset, load_image_folder
from .experiment import Experiment
from .explainer import SOFIExplainer, select_reliable_instances
from .explanation import SOFIExplanation, aggregate_explanations, noise_onset
from .metrics import (
    fold_change,
    nearest_neighbors,
    rank_biased_overlap,
    robustness_score,
)
from .model import ClassifierWrapper
from .objective import (
    Objective,
    RankingEvaluation,
    curve_auc,
    degradation_score,
    modularity_gap,
)
from .perturbation import (
    CLASS_AGNOSTIC,
    CLASS_DIRECTED,
    BackgroundPerturbation,
    BlurPerturbation,
    ConstantPerturbation,
    InpaintPerturbation,
    LineSearchPerturbation,
    LowNegPerturbation,
    NearestUnlikeNeighborPerturbation,
    NoisePerturbation,
    OppositeClassMeanPerturbation,
    Perturbation,
    PerturbationContext,
    RandomConstantPerturbation,
    RoadPerturbation,
    SegmentMeanPerturbation,
    make_perturbation,
)
from .plotting import (
    plot_degradation_curve,
    plot_explanation_grid,
    plot_explanation_overlay,
    plot_explanation,
    plot_image,
    plot_importance_map,
    plot_marginalization,
    plot_optimization_trace,
    plot_overlay_grid,
    plot_segments,
    plot_segmentation,
    plot_segmentation_grid,
)
from .report import describe_instance, instance_report
from .search import greedy_ranking, hill_climbing
from .segmentation import (
    Segment,
    Segmentation,
    boundaries_of,
    build_segmentation,
    resize_labels,
    segmentation_from_mask,
    slic_segmentation,
)

__version__ = "1.0.0"

__all__ = [
    "SOFIExplainer",
    "Experiment",
    "SOFIExplanation",
    "ClassifierWrapper",
    "Segment",
    "Segmentation",
    "slic_segmentation",
    "segmentation_from_mask",
    "build_segmentation",
    "boundaries_of",
    "resize_labels",
    "Perturbation",
    "PerturbationContext",
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
    "CLASS_AGNOSTIC",
    "CLASS_DIRECTED",
    "Objective",
    "RankingEvaluation",
    "curve_auc",
    "degradation_score",
    "modularity_gap",
    "hill_climbing",
    "greedy_ranking",
    "aggregate_explanations",
    "noise_onset",
    "select_reliable_instances",
    "rank_biased_overlap",
    "robustness_score",
    "nearest_neighbors",
    "fold_change",
    "describe_instance",
    "instance_report",
    "segment_scores",
    "ranking_from_scores",
    "feature_occlusion",
    "random_ranking",
    "compare_rankings",
    "ImageDataset",
    "load_dataset",
    "load_image_folder",
    "plot_image",
    "plot_segmentation",
    "plot_segmentation_grid",
    "plot_degradation_curve",
    "plot_marginalization",
    "plot_segments",
    "plot_explanation_overlay",
    "plot_overlay_grid",
    "plot_explanation",
    "animate_marginalization",
    "plot_importance_map",
    "plot_optimization_trace",
    "plot_explanation_grid",
    "__version__",
]
