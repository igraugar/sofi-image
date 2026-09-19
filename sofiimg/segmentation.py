"""Segments of an image, the units SOFI ranks.

A segment is a connected group of pixels that spans every color channel, since
color is not an independent signal the way a sensor channel is. The
:class:`Segmentation` object holds the label map that assigns each pixel to a
segment, the segments themselves with their area, bounding box and centroid,
and the metadata used by the report and the figures.

SLIC is the only procedure the package offers. It clusters pixels in a space
that joins their color to their position, so a segment is homogeneous in
appearance and compact in the plane, and both properties are what make a segment
a unit a reader can point at rather than an arbitrary window. A ready
:class:`Segmentation` or a label map produced elsewhere is also accepted, which
is how segments computed by another library enter the explainer.

A region of interest may restrict every one of them. Passing a boolean mask
grows the segments inside it alone and leaves every pixel outside it out of
the partition, so nothing beyond the segment can be ranked or marginalized. That
matters far more for images than it would for a time series. A radiographic
classifier trained on unrestricted images can settle on positioning cues, on soft
tissue outside the thorax or on a border artifact, and an explanation of such a
model faithfully reports evidence that carries no clinical meaning. Restricting
the partition to the anatomy of interest keeps the explanation inside the segment
a reader is willing to reason about.

Segmentation always applies to one instance, since the segments of one image say
nothing about the segments of another. The patch grid is the exception, and it is
also what makes an explanation comparable with the token grid of a vision
transformer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Segment",
    "Segmentation",
    "boundaries_of",
    "resize_labels",
    "slic_segmentation",
    "segmentation_from_mask",
    "build_segmentation",
]

_ALIASES = {
    "segment": "slic",
    "segments": "slic",
}


def boundaries_of(labels, thickness: int = 1) -> np.ndarray:
    """Borders of a label map, one pixel wide before any dilation.

    A pixel belongs to a border when the label to its right or below differs
    from its own, which draws a single line between two segments instead of the
    two adjacent lines a symmetric convention produces.

    Parameters
    ----------
    labels : array
        Integer label map.
    thickness : int, default=1
        Width of the line in pixels.

    Returns
    -------
    ndarray
        A boolean array shaped like the label map.
    """
    array = np.asarray(labels)
    edges = np.zeros(array.shape, dtype=bool)
    edges[:, :-1] |= array[:, :-1] != array[:, 1:]
    edges[:-1, :] |= array[:-1, :] != array[1:, :]
    if int(thickness) > 1:
        from scipy.ndimage import binary_dilation

        edges = binary_dilation(edges, iterations=int(thickness) - 1)
    return edges


def resize_labels(labels, shape) -> np.ndarray:
    """Carry a label map onto another resolution, without inventing labels.

    Nearest neighbor sampling is the only admissible rule, since a label is a
    name rather than a quantity and any interpolation between two of them would
    produce a segment that does not exist. The purpose is display, namely drawing
    a partition computed at the resolution the model receives over the original
    image at its own resolution.

    Parameters
    ----------
    labels : array
        Integer label map.
    shape : tuple of int
        Height and width wanted.

    Returns
    -------
    ndarray
        The label map at that resolution.
    """
    array = np.asarray(labels)
    height, width = int(shape[0]), int(shape[1])
    if array.shape[:2] == (height, width):
        return array
    rows = np.minimum(
        (np.arange(height) * array.shape[0] / height).astype(int), array.shape[0] - 1
    )
    cols = np.minimum(
        (np.arange(width) * array.shape[1] / width).astype(int), array.shape[1] - 1
    )
    return array[np.ix_(rows, cols)]


@dataclass(frozen=True)
class Segment:
    """One connected group of pixels, spanning every channel of the image.

    Attributes
    ----------
    index : int
        Position of the segment inside its segmentation, which is also the
        position a ranking permutes and the number in its label.
    rows, cols : ndarray
        Pixel coordinates of the segment. They are held explicitly rather than
        as a boolean image, since a segmentation of two hundred segments
        would otherwise carry two hundred full sized masks.
    bbox : tuple of int
        Smallest rectangle holding the segment, as ``(row_min, row_max,
        col_min, col_max)`` with the two maxima excluded, following the
        convention of a Python slice.
    centroid : tuple of float
        Center of mass of the segment, in row and column order.
    """

    index: int
    rows: np.ndarray = field(repr=False)
    cols: np.ndarray = field(repr=False)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: Tuple[float, float] = (0.0, 0.0)

    @property
    def area(self) -> int:
        """Number of pixels the segment holds."""
        return int(self.rows.size)

    @property
    def length(self) -> int:
        """Alias of :attr:`area`, kept so that shared code reads the same."""
        return self.area

    @property
    def label(self) -> str:
        return f"s{self.index}"

    def mask(self, shape) -> np.ndarray:
        """Boolean image that is true on the pixels of the segment.

        Parameters
        ----------
        shape : tuple of int
            Height and width of the image the mask belongs to.

        Returns
        -------
        ndarray
            A boolean array of that shape.
        """
        canvas = np.zeros(tuple(shape)[:2], dtype=bool)
        canvas[self.rows, self.cols] = True
        return canvas

    def slice(self) -> Tuple[slice, slice]:
        """Row and column slices of the bounding box of the segment."""
        row_min, row_max, col_min, col_max = self.bbox
        return slice(row_min, row_max), slice(col_min, col_max)


class Segmentation:
    """Partition of one image into the units SOFI ranks.

    Parameters
    ----------
    labels : array
        Integer label map shaped ``(height, width)``. Every distinct
        non-negative value becomes one segment, and negative entries stay
        outside every segment and are never marginalized.
    n_channels : int, default=1
        Number of channels of the images the partition applies to. It is
        carried for validation alone, since a segment spans every channel.
    method : str, default="manual"
        Name of the procedure that produced the partition, kept for the report
        and the figures.
    params : dict, optional
        Arguments handed to that procedure.
    """

    def __init__(
        self,
        labels,
        n_channels: int = 1,
        method: str = "manual",
        params: Optional[Dict] = None,
    ):
        label_map = np.asarray(labels)
        if label_map.ndim != 2:
            raise ValueError(
                "A segmentation is described by a label map shaped (height, "
                f"width), and the array received has shape {label_map.shape}."
            )
        label_map = label_map.astype(int, copy=True)
        values = [int(v) for v in np.unique(label_map) if v >= 0]
        if not values:
            raise ValueError(
                "The label map holds no non-negative value, so there is nothing "
                "to rank. Every pixel was left outside every segment."
            )

        segments: List[Segment] = []
        renumbered = np.full(label_map.shape, -1, dtype=int)
        for position, value in enumerate(values):
            rows, cols = np.nonzero(label_map == value)
            renumbered[rows, cols] = position
            segments.append(
                Segment(
                    index=position,
                    rows=rows,
                    cols=cols,
                    bbox=(
                        int(rows.min()),
                        int(rows.max()) + 1,
                        int(cols.min()),
                        int(cols.max()) + 1,
                    ),
                    centroid=(float(rows.mean()), float(cols.mean())),
                )
            )

        self.segments = segments
        self.mask = renumbered
        self.height = int(label_map.shape[0])
        self.width = int(label_map.shape[1])
        self.n_channels = int(n_channels)
        self.method = str(method)
        self.params = dict(params or {})

    # ------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, position: int) -> Segment:
        return self.segments[int(position)]

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def shape(self) -> Tuple[int, int]:
        """Height and width of the images the partition applies to."""
        return (self.height, self.width)

    @property
    def labels(self) -> List[str]:
        return [segment.label for segment in self.segments]

    @property
    def areas(self) -> np.ndarray:
        """Pixel count of every segment, in the order of the segmentation."""
        return np.array([segment.area for segment in self.segments], dtype=int)

    @property
    def lengths(self) -> np.ndarray:
        """Alias of :attr:`areas`, kept so that shared code reads the same."""
        return self.areas

    @property
    def coverage(self) -> float:
        """Fraction of the pixels that belong to some segment."""
        return float(np.mean(self.mask >= 0))

    def boundaries(self, thickness: int = 1, labels=None) -> np.ndarray:
        """Boolean image that is true on the borders between segments.

        The border is one pixel wide by default, which is the thinnest line an
        image can carry and the one that hides the least of the content beneath
        it. A wider line is obtained by dilation rather than by a different
        convention, so the geometry of the partition never changes.

        Parameters
        ----------
        thickness : int, default=1
            Width of the line in pixels.
        labels : array, optional
            Label map the borders are computed from, which is how the borders
            of a partition are drawn at the resolution of the original image
            rather than at the resolution the model receives.

        Returns
        -------
        ndarray
            A boolean array shaped like the label map.
        """
        return boundaries_of(self.mask if labels is None else labels, thickness)

    def describe(self):
        """One row per segment, with its area, bounding box and centroid."""
        import pandas as pd

        return pd.DataFrame(
            {
                "segment": self.labels,
                "area": [s.area for s in self.segments],
                "row_min": [s.bbox[0] for s in self.segments],
                "row_max": [s.bbox[1] for s in self.segments],
                "col_min": [s.bbox[2] for s in self.segments],
                "col_max": [s.bbox[3] for s in self.segments],
                "centroid_row": [round(s.centroid[0], 1) for s in self.segments],
                "centroid_col": [round(s.centroid[1], 1) for s in self.segments],
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Segmentation(method={self.method}, n_segments={self.n_segments}, "
            f"shape={self.height}x{self.width})"
        )


# ---------------------------------------------------------------- builders
def _as_instance(x) -> np.ndarray:
    """Return one image shaped ``(height, width, n_channels)``."""
    array = np.asarray(x, dtype=float)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[0] == 1:
        return array[0]
    raise ValueError(
        "An image must be shaped (height, width) or (height, width, "
        f"n_channels), and the array received has shape {array.shape}."
    )


def _for_skimage(instance: np.ndarray, rescale: bool = True):
    """The image in the form the ``skimage`` segmenters expect.

    The preprocessing that suits a classifier rarely suits a segment
    algorithm, since a standardized image is centered on zero and a scaled one
    may span any interval. The values are therefore mapped onto the unit
    interval before the partition is computed, which changes no geometry and
    keeps the compactness argument comparable across problems.
    """
    image = np.asarray(instance, dtype=float)
    if rescale:
        low, high = float(np.min(image)), float(np.max(image))
        image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    if image.shape[-1] == 1:
        return image[..., 0], None
    return image, -1


def slic_segmentation(
    x,
    n_segments: int = 50,
    compactness: float = 0.2,
    sigma: float = 1.0,
    max_num_iter: int = 10,
    enforce_connectivity: bool = True,
    convert2lab: bool = False,
    rescale: bool = True,
    mask=None,
) -> Segmentation:
    """Segments obtained with simple linear iterative clustering.

    SLIC clusters the pixels in a space that joins their color to their
    position, so a segment is homogeneous in appearance and compact in the
    plane. Both properties matter to an explanation. Homogeneity makes a
    segment a plausible unit of meaning rather than an arbitrary window, and
    compactness keeps the segment small enough to point at, which is what lets a
    reader say where the evidence lies.

    Parameters
    ----------
    x : array
        One image, shaped ``(height, width)`` or ``(height, width, n_channels)``.
    n_segments : int, default=50
        Approximate number of segments requested. SLIC returns a number
        close to it rather than exactly it, since the clusters are seeded on a
        grid and small ones are absorbed afterwards.
    compactness : float, default=0.2
        Weight of the spatial term against the color term. Larger values give
        squarer segments that follow the image less, and smaller values give
        segments that hug the contours and grow irregular. The default suits the
        unit interval ``rescale`` maps the values onto, and a value near ten
        belongs to the CIELAB space that ``convert2lab`` produces, where the
        color term spans a hundred units rather than one.
    sigma : float, default=1.0
        Width of the Gaussian smoothing applied before the clustering, which
        keeps noise from fragmenting the segments.
    max_num_iter : int, default=10
        Iterations of the clustering.
    enforce_connectivity : bool, default=True
        Whether every segment is forced to be connected. Turning it off
        leaves segments that a reader cannot point at, so it is on by default.
    convert2lab : bool, default=False
        Whether a three channel image is converted to the CIELAB space first.
        It suits photographs in genuine RGB and it is off by default, since a
        preprocessed medical image is not one.
    rescale : bool, default=True
        Whether the values are mapped onto the unit interval before the
        clustering. It leaves the geometry unchanged and makes ``compactness``
        mean the same thing regardless of the preprocessing.
    mask : array, optional
        Boolean region of interest. Segments are grown inside it alone and
        every pixel outside it stays out of the partition, so a background, or
        any anatomy the explanation should not reason about, is excluded
        altogether rather than merely deprioritized.

    Returns
    -------
    Segmentation
        The partition.

    Examples
    --------
    >>> segmentation = slic_segmentation(image, n_segments=60, compactness=8.0)
    >>> segmentation.n_segments
    58
    """
    from skimage.segmentation import slic

    instance = _as_instance(x)
    image, channel_axis = _for_skimage(instance, rescale=rescale)
    if int(n_segments) < 1:
        raise ValueError("n_segments must be a positive integer.")

    labels = slic(
        image,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        max_num_iter=int(max_num_iter),
        enforce_connectivity=bool(enforce_connectivity),
        convert2lab=bool(convert2lab) if channel_axis is not None else False,
        channel_axis=channel_axis,
        start_label=0,
        mask=None if mask is None else np.asarray(mask, dtype=bool),
    )
    labels = np.asarray(labels, dtype=int)
    if mask is not None:
        # SLIC numbers a masked partition from one and leaves zero outside
        labels = labels - 1
    params = {
        "n_segments": int(n_segments),
        "compactness": float(compactness),
        "sigma": float(sigma),
        "max_num_iter": int(max_num_iter),
        "enforce_connectivity": bool(enforce_connectivity),
        "convert2lab": bool(convert2lab),
        "rescale": bool(rescale),
    }
    segmentation = Segmentation(
        labels, n_channels=instance.shape[-1], method="slic", params=params
    )
    _warn_on_shortfall(segmentation, n_segments, "slic")
    return segmentation


def segmentation_from_mask(mask, n_channels: int = 1, method: str = "mask") -> Segmentation:
    """Segmentation described by a label map supplied by the user.

    Every distinct non-negative value becomes one segment, and negative entries
    stay outside every segment and are never marginalized, which is how a
    background is excluded. The output of any segmentation library the package
    does not wrap enters the explainer this way.

    Parameters
    ----------
    mask : array
        Integer label map shaped ``(height, width)``, or a boolean image, in
        which case the true pixels form one segment.
    n_channels : int, default=1
        Number of channels of the images the partition applies to.
    method : str, default="mask"
        Name recorded for the report and the figures.

    Returns
    -------
    Segmentation
        The partition the label map describes.
    """
    array = np.asarray(mask)
    if array.dtype == bool:
        array = np.where(array, 0, -1)
    return Segmentation(array, n_channels=n_channels, method=method)


def _warn_on_shortfall(segmentation: Segmentation, requested, method: str) -> None:
    """Say so when a procedure returns a count far from the one requested.

    SLIC seeds its clusters on a grid and merges the small ones afterwards, so
    the count obtained is close to the count asked for rather than equal to it.
    A small difference is ordinary and passes in silence, while a large one
    usually means the image is too small or too flat for the request, and
    passing it over would misrepresent what the explainer is about to rank.
    """
    if requested is None:
        return
    obtained = segmentation.n_segments
    requested = int(requested)
    if abs(obtained - requested) > max(2, 0.25 * requested):
        warnings.warn(
            f"The '{method}' procedure produced {obtained} segments where "
            f"{requested} were requested. A small image, a flat one or a large "
            "compactness leaves the clustering little to separate.",
            stacklevel=3,
        )


def build_segmentation(x, segmentation="slic", **kwargs) -> Segmentation:
    """Resolve the ``segmentation`` argument of the explainer.

    Accepted values are a ready :class:`Segmentation`, ``"slic"`` together with
    its arguments, ``"mask"`` together with a label map, or an integer array read
    directly as a label map.

    Parameters
    ----------
    x : array
        The image to divide.
    segmentation : str, Segmentation or array, default="slic"
        The partition or the procedure that produces it.
    **kwargs : dict
        Arguments of that procedure, including ``mask``, which restricts SLIC
        to a region of interest.

    Returns
    -------
    Segmentation
        The partition.
    """
    instance = _as_instance(x)
    height, width, n_channels = instance.shape

    if isinstance(segmentation, Segmentation):
        if segmentation.shape != (height, width):
            raise ValueError(
                "The segmentation was built for an image of "
                f"{segmentation.height} by {segmentation.width} pixels and the "
                f"instance holds {height} by {width}."
            )
        return segmentation

    if not isinstance(segmentation, str):
        array = np.asarray(segmentation)
        if array.shape[:2] != (height, width):
            raise ValueError(
                f"A label map must be shaped {(height, width)} to describe this "
                f"instance, and the array received has shape {array.shape}."
            )
        return segmentation_from_mask(array, n_channels=n_channels)

    name = _ALIASES.get(segmentation.lower(), segmentation.lower())
    if name == "mask":
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError(
                "segmentation='mask' needs the mask argument, which holds one "
                "integer label per pixel."
            )
        return segmentation_from_mask(mask, n_channels=n_channels)

    if name != "slic":
        raise ValueError(
            f"Unknown segmentation procedure '{segmentation}'. SLIC is the only "
            "procedure the package offers, and segments computed elsewhere enter "
            "through 'mask' together with a label map, or as a label map passed "
            "directly."
        )
    accepted = {
        "n_segments",
        "compactness",
        "sigma",
        "max_num_iter",
        "enforce_connectivity",
        "convert2lab",
        "rescale",
        "mask",
    }
    options = {k: v for k, v in kwargs.items() if k in accepted}
    return slic_segmentation(instance, **options)
