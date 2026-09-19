"""Visualization of the segmentations and the degradation curves of SOFI.

The figures never touch the font settings of the session. Sizes are taken from
the active matplotlib or seaborn theme, so one global call such as
``set_plot_style(font_scale=1.2)`` governs the title, the axes, the ticks, the
legend and the annotation alike. Colors behave the same way through
``set_curve_colors``, and a single figure can depart from the session through its
own arguments.

Two conventions govern every figure that draws an image.

The first concerns resolution. A classifier is fed a small square, often cropped
and always downscaled, because that is what it was trained on. A reader is not,
and a partition drawn over a downscaled radiograph hides the very detail the
explanation is about. Every figure therefore accepts the original image
through its ``display`` argument and carries the label map onto that resolution
by nearest neighbor sampling, which invents no segment and moves no border. The
explanation itself is unaffected, since it was computed on the array the model
receives.

The second concerns what an overlay means. Marginalizing a segment replaces its
content, and drawing the replacement would put a gray patch or a blurred blob in
front of the reader, which is the mechanism rather than the finding. What the
reader needs to know is which segments were removed, so the segments are painted
in a translucent color over the untouched image and outlined, and the color
carries no magnitude of its own.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .segmentation import boundaries_of, resize_labels

__all__ = [
    "MORF_COLOR",
    "LERF_COLOR",
    "BOUNDARY_COLOR",
    "SEGMENT_COLOR",
    "plot_image",
    "plot_segmentation",
    "plot_segmentation_grid",
    "plot_degradation_curve",
    "plot_explanation_grid",
    "plot_segments",
    "plot_explanation_overlay",
    "plot_overlay_grid",
    "plot_marginalization",
    "plot_explanation",
    "plot_importance_map",
    "plot_optimization_trace",
]


# defaults of the color arguments, so that a figure drawn without any argument
# looks like every other figure of the package
MORF_COLOR = "#03719c"
LERF_COLOR = "#1A1A1A"
BOUNDARY_COLOR = "#0343df"
SEGMENT_COLOR = "#3b6fb6"

# a muted palette holding no grey and no black, since either is read as tissue
# once it is laid over a radiograph
_PALETTE = (
    "#3b6fb6",
    "#8e6fb6",
    "#b6763b",
    "#b64f4f",
    "#4f9e6a",
    "#1f9e9e",
)


_YLABEL = "Predicted probability"


def _theme(font_scale: float = 1.1, style: str = "whitegrid"):
    """Typography and background of one figure, applied for that figure alone.

    Nothing is stored between calls. Every figure therefore looks the same
    whether it is the first of a session or the hundredth, and a figure that
    needs different typography says so in its own arguments.
    """
    import matplotlib as mpl

    settings = {}
    try:
        import seaborn as sns

        settings.update(sns.axes_style(style))
        settings.update(sns.plotting_context("notebook", font_scale=font_scale))
    except ImportError:  # pragma: no cover - seaborn is optional
        settings["font.size"] = 10.0 * font_scale
    return mpl.rc_context(settings)


def _themed(function):
    """Give a figure the ``font_scale`` and ``style`` arguments of :func:`_theme`."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, font_scale: float = 1.1, style: str = "whitegrid", **kwargs):
        with _theme(font_scale, style):
            return function(*args, **kwargs)

    return wrapper


def _fit_grid(figure, panels, ncols, gap: float = 0.16, pad: float = 0.14):
    """Give a grid of image panels a height that matches its width.

    An axes holding an image keeps the aspect of that image, so it shrinks
    inside whatever cell the layout gives it. When the figure is not shaped like
    the grid it carries, every cell is larger than the panel drawn in it and the
    surplus appears as bands, between the panels when the grid has several rows
    or columns and around them when it does not. No layout engine reclaims that
    space, since as far as the engine is concerned every cell is full.

    The remedy is to derive the height rather than to accept it. The width the
    caller asked for fixes the width of a panel, the aspect of the image fixes
    its height, and the height of the figure then follows from the number of
    rows together with the room the headings need, which is measured on a first
    pass rather than assumed. The panels end up exactly filling their cells at
    any width, any number of columns and any typography, so the figure carries
    no blank band and the headings keep their spacing.
    """
    visible = [ax for ax in panels if ax.get_visible()]
    if not visible:
        return figure

    figure.canvas.draw()
    width, _ = figure.get_size_inches()
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(visible) / ncols))

    drawn = [ax.get_images() for ax in visible]
    shapes = [image[0].get_array().shape for image in drawn if image]
    if not shapes:
        return figure
    aspect = float(shapes[0][0]) / float(shapes[0][1])

    head = max(ax.title.get_window_extent().height / figure.dpi for ax in visible)
    crown = (
        figure._suptitle.get_window_extent().height / figure.dpi
        if figure._suptitle is not None
        else 0.0
    )

    panel = (width - 2.0 * pad - (ncols - 1) * gap) / ncols
    if panel <= 0:  # pragma: no cover - a width smaller than its own padding
        return figure
    height = (
        nrows * (panel * aspect + head)
        + (nrows - 1) * gap
        + crown
        + 2.0 * pad
    )
    figure.set_size_inches(width, height)
    # the engine is rebuilt rather than reused, so that the layout follows from
    # the new size alone and not from the one the figure was created with
    figure.set_layout_engine(
        "constrained", w_pad=pad / 2.0, h_pad=pad / 2.0, wspace=0.0, hspace=0.0
    )
    figure.canvas.draw()
    return figure


def _heading(figure, title, pad: float = 0.12):
    """Place the heading of a figure, with air between it and the panels.

    A constrained layout packs a suptitle tight against the panel titles beneath
    it, which reads as one crowded block. Widening the vertical padding of the
    layout separates the two, and every figure of the package uses the same
    value so that a heading sits at the same height throughout.
    """
    engine = figure.get_layout_engine()
    if engine is not None:
        current = getattr(engine, "get", lambda: {})() or {}
        engine.set(h_pad=max(float(pad), float(current.get("h_pad", 0.0) or 0.0)))
    if title is not None:
        figure.suptitle(title)
    return figure


def _tighten_pair(figure, image_ax, curve_ax, gap: float = 0.04, margin: float = 0.015):
    """Remove the blank band a square image leaves inside a wider cell.

    An axes holding an image keeps the aspect of that image, so it shrinks
    inside whatever cell the layout gives it and leaves the rest of the cell
    empty. No layout engine reclaims that space, because as far as the engine is
    concerned the cell is full, and the band grows or shrinks with every change
    of size or width ratio.

    The panels are therefore placed once by the engine, measured as drawn, and
    then adjusted. The image is pushed against the left of its cell and the
    curve is moved until it clears the image by a fixed gap, keeping its right
    edge, so it absorbs the band. Only measured quantities take part, so the
    result holds for any width ratio and any figure size.
    """
    image_ax.set_anchor("W")
    figure.canvas.draw()

    image_box = image_ax.get_position()
    curve_box = curve_ax.get_position()
    # the room the curve keeps to its left for the tick labels and the axis
    # title, measured rather than assumed
    tight = curve_ax.get_tightbbox().transformed(figure.transFigure.inverted())
    label = max(0.0, curve_box.x0 - tight.x0)

    figure.set_layout_engine("none")
    image_ax.set_position(
        [float(margin), image_box.y0, image_box.width, image_box.height]
    )
    left = float(margin) + image_box.width + float(gap) + label
    width = curve_box.x1 - left
    if width > 0.05:
        curve_ax.set_position([left, curve_box.y0, width, curve_box.height])
    return figure


def _new_axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _as_display(image, reference=None) -> np.ndarray:
    """Map an image onto the unit interval so that matplotlib can draw it.

    Parameters
    ----------
    image : array
        The image to draw, shaped ``(height, width)`` or ``(height, width,
        n_channels)``.
    reference : array, optional
        Array whose extremes fix the mapping, which is how a sequence of states
        keeps one common scale instead of stretching each panel to its own.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    source = array if reference is None else np.asarray(reference, dtype=float)
    low, high = float(np.min(source)), float(np.max(source))
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _draw(ax, image, cmap):
    ax.imshow(image, cmap=None if image.ndim == 3 else cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return ax


def _darken(color, factor: float = 0.55):
    """A darker shade of a color, used for the border of a painted segment.

    A border in the color of its own fill disappears against it, and a border in
    black belongs to no segment in particular. Darkening the fill keeps one
    segment in one color while still separating two segments that touch.
    """
    import matplotlib.colors as mcolors

    red, green, blue = mcolors.to_rgb(color)
    return (red * factor, green * factor, blue * factor)


def _overlay(ax, mask, color, alpha=1.0):
    """Paint a boolean mask over the axes in one color, leaving the rest clear."""
    import matplotlib.colors as mcolors

    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[mask] = mcolors.to_rgba(color, alpha=float(alpha))
    ax.imshow(rgba, interpolation="nearest")


@_themed
def plot_image(image, ax=None, title=None, figsize=(4.4, 4.4), cmap="gray", display=None):
    """Draw one image, with its values mapped onto the unit interval.

    Parameters
    ----------
    image : array
        The image, shaped ``(height, width)`` or ``(height, width,
        n_channels)``.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure.
    figsize : tuple, default=(4.4, 4.4)
        Size of the figure created when no axes is supplied.
    cmap : str, default="gray"
        Colormap of a single channel image. It is ignored for a color one.
    display : array, optional
        Image drawn instead of ``image``, normally the original image at its
        own resolution.

    Returns
    -------
    matplotlib Axes
        The axes the image was drawn on.
    """
    ax = _new_axes(ax, figsize)
    _draw(ax, _as_display(image if display is None else display), cmap)
    if title is not None:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------- segmentation
@_themed
def plot_segmentation(
    instance,
    segmentation,
    ax=None,
    title: Optional[str] = None,
    figsize=(4.6, 4.6),
    linewidth: float = 1.0,
    color: Optional[str] = None,
    cmap: str = "gray",
    alpha: float = 1.0,
    annotate: bool = False,
    fontsize: Optional[float] = None,
    display=None,
):
    """Draw an image with the borders of its segments laid over it.

    The borders are drawn in blue, which separates them from the gray levels of
    a radiograph and from the content of a photograph alike. They are one pixel
    wide by default, which is the thinnest line an image can carry, and
    ``linewidth`` widens them by dilation when a larger figure needs it.

    Passing the original image through ``display`` draws the partition at the
    resolution a reader can actually see. The label map is carried onto that
    resolution by nearest neighbor sampling, so no border moves and no segment
    appears or disappears. A partition of forty segments computed on a ninety-six
    pixel square becomes illegible when it is drawn on that square and stays
    perfectly readable when it is drawn on the original image.

    Parameters
    ----------
    instance : array
        The image the partition was computed on, shaped ``(height, width)`` or
        ``(height, width, n_channels)``.
    segmentation : Segmentation
        Partition whose borders are drawn.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure. A default one names the method and the count.
    figsize : tuple, default=(4.6, 4.6)
        Size of the figure created when no axes is supplied.
    linewidth : float, default=1.0
        Thickness of the borders in pixels, rounded to a whole number, since a
        border drawn on an image is itself made of pixels.
    color : str, optional
        Color of the borders for this figure alone. The color in force for the
        session is used when it is left out.
    cmap : str, default="gray"
        Colormap of a single channel image.
    alpha : float, default=1.0
        Opacity of the borders, which lets the content show through them.
    annotate : bool, default=False
        Whether the number of every segment is written at its centroid, which is
        how a ranking is read back onto the image.
    fontsize : float, optional
        Size of those numbers. The size of the session is used by default.
    display : array, optional
        Original image drawn instead of ``instance``, at its own resolution.

    Returns
    -------
    matplotlib Axes
        The axes the image was drawn on.
    """
    if float(linewidth) <= 0:
        raise ValueError("linewidth must be a positive number of pixels.")

    image = _as_display(instance if display is None else display)
    ax = _new_axes(ax, figsize)
    _draw(ax, image, cmap)

    labels = resize_labels(segmentation.mask, image.shape[:2])
    edges = boundaries_of(labels, thickness=int(round(float(linewidth))))
    _overlay(ax, edges, color or BOUNDARY_COLOR, alpha)

    if annotate:
        scale_row = image.shape[0] / segmentation.height
        scale_col = image.shape[1] / segmentation.width
        for segment in segmentation:
            ax.text(
                segment.centroid[1] * scale_col,
                segment.centroid[0] * scale_row,
                str(segment.index),
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )

    ax.set_title(
        title
        if title is not None
        else f"{segmentation.method}, {segmentation.n_segments} segments"
    )
    return ax


@_themed
def plot_segmentation_grid(
    instance,
    segmentations,
    ncols: int = 3,
    figsize=None,
    panel_size=(3.8, 3.8),
    title=None,
    **kwargs,
):
    """Draw one image under several segmentation procedures.

    Every panel holds the same image with the borders of one procedure, so the
    vocabularies the explainer would rank can be compared side by side.

    Parameters
    ----------
    instance : array
        The image every panel shows.
    segmentations : dict or sequence
        Mapping from a heading onto a ``Segmentation``, or a sequence of them,
        in which case the method name becomes the heading.
    ncols : int, default=3
        Number of panels per row.
    figsize : tuple, optional
        Size of the whole figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(3.8, 3.8)
        Size of one panel, from which the size of the figure is derived.
    title : str, optional
        Heading placed above the grid.
    **kwargs : dict
        Further arguments of :func:`plot_segmentation`, such as ``linewidth``,
        ``color``, ``cmap`` or ``display``.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    import matplotlib.pyplot as plt

    items = (
        list(segmentations.items())
        if isinstance(segmentations, dict)
        else [(seg.method, seg) for seg in segmentations]
    )
    if not items:
        raise ValueError("At least one segmentation is required.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(items) / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
    flat = np.atleast_1d(axes).ravel()
    for (heading, segmentation), ax in zip(items, flat):
        plot_segmentation(
            instance,
            segmentation,
            ax=ax,
            title=f"{heading} with {segmentation.n_segments} segments",
            **kwargs,
        )
    for ax in flat[len(items) :]:
        ax.axis("off")
    _heading(figure, title)
    return _fit_grid(figure, flat[: len(items)], ncols)


# ----------------------------------------------------------- degradation curve
@_themed
def plot_degradation_curve(
    explanation,
    ax=None,
    title: Optional[str] = None,
    annotate_ds: bool = True,
    legend: bool = True,
    mark_sparsity: bool = True,
    figsize=(6.6, 4.2),
    linewidth: float = 1.8,
    linestyle: str = "-",
    marker: str = "s",
    markersize: float = 8.0,
    marker_edge_color: str = "white",
    marker_edge_width: float = 0.7,
    morf_color: Optional[str] = None,
    lerf_color: Optional[str] = None,
    fill_color: str = "gray",
    fill_alpha: float = 0.22,
    star_marker=(8, 1, 0),
    star_scale: float = 2.0,
    xlabel: str = "Marginalized segments",
    ylabel: str = "Predicted probability",
    ylim=(-0.05, 1.22),
    yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
    n_xticks: int = 11,
    legend_loc: str = "lower left",
    legend_ncol: int = 2,
    annotation: Optional[str] = None,
):
    """Draw the MoRF and the LeRF curves together with their enclosed area.

    The curves hold the probability of the explained class exactly as the model
    produced it, so the vertical axis is the unit interval and two figures from
    different images are read on the same scale without any rescaling. The shaded segment between them is the degradation
    score, which places the quantity maximized by the search in the figure
    itself. A star on each curve marks the step at which the predicted class
    first changes, and the distance between the two is what the ranking buys.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure. A default one names the explained class.
    annotate_ds : bool, default=True
        Whether the degradation score is written inside the axes.
    legend : {True, False, "inside"}, default=True
        Whether the two curves are labeled. ``True`` lays the legend out as a
        single row underneath the axes, so it never covers the curves.
        ``"inside"`` places it in the lower left corner of the axes instead,
        which is what keeps two panels of a paired figure the same height.
    mark_sparsity : bool, default=True
        Whether the class change of each curve receives a star.
    figsize : tuple, default=(6.6, 4.2)
        Size of the figure created when no axes is supplied.
    linewidth : float, default=1.8
        Thickness of both curves.
    markersize : float, default=8.0
        Size of the square marker placed on every step.
    morf_color, lerf_color : str, optional
        Color of each curve for this figure alone. The colors in force for the
        session are used when they are left out, and ``set_curve_colors``
        changes those.
    """
    import matplotlib.pyplot as plt

    morf = np.asarray(explanation.morf_scores, dtype=float)
    lerf = np.asarray(explanation.lerf_scores, dtype=float)
    steps = np.arange(len(morf))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")

    # the curves hold probabilities, so the axis is fixed to the unit interval
    # with a margin that leaves room for the annotation
    bottom, top = float(ylim[0]), float(ylim[1])

    ax.fill_between(steps, morf, lerf, color=fill_color, alpha=fill_alpha, zorder=2)
    for values, color, label, order in (
        (lerf, lerf_color or LERF_COLOR, "LeRF", 3),
        (morf, morf_color or MORF_COLOR, "MoRF", 4),
    ):
        ax.plot(
            steps,
            values,
            marker=marker,
            markersize=markersize,
            markeredgecolor=marker_edge_color,
            markeredgewidth=marker_edge_width,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=order,
        )

    if mark_sparsity:
        marks = (
            (explanation.sparsity_point, morf, morf_color or MORF_COLOR),
            (explanation.lerf_sparsity_point, lerf, lerf_color or LERF_COLOR),
        )
        for point, curve, color in marks:
            if point is None:
                continue
            ax.plot(
                [int(point)],
                [curve[int(point)]],
                marker=star_marker,
                markersize=star_scale * markersize,
                color=color,
                markeredgecolor=marker_edge_color,
                markeredgewidth=marker_edge_width + 0.1,
                zorder=5,
            )

    ax.set_ylim(bottom, top)
    ax.set_yticks(list(yticks))
    ax.set_xlim(-0.4, len(morf) - 0.6)
    step = max(1, int(np.ceil(len(morf) / max(1, int(n_xticks)))))
    ax.set_xticks(steps[::step])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        title if title is not None else f"SOFI, class {explanation.class_name}"
    )

    if legend == "inside":
        ax.legend(
            loc=legend_loc, ncol=legend_ncol, frameon=True, framealpha=0.9,
            handlelength=1.6, columnspacing=1.2, borderpad=0.4,
        )
    elif legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.26),
            ncol=legend_ncol,
            frameon=False,
            borderaxespad=0.0,
            handlelength=2.0,
            columnspacing=2.0,
        )

    if annotate_ds:
        ax.text(
            0.98,
            0.96,
            annotation if annotation is not None else f"DS: {explanation.ds:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="#001f3f",
                boxstyle="round,pad=0.35",
            ),
        )
    return ax


@_themed
def plot_explanation_grid(
    picks,
    title=None,
    ncols: int = 2,
    figsize=None,
    panel_size=(6.3, 3.9),
    **kwargs,
):
    """Draw several explanations on one grid of panels.

    Panels are packed as closely as the labels allow, so the heading sits just
    above the first row and consecutive rows are separated by the height of a
    legend and nothing more.

    Parameters
    ----------
    picks : sequence
        Explanations, pairs holding a heading and an explanation, or objects
        exposing an ``explanation`` and a ``title``.
    title : str, optional
        Heading of the grid.
    ncols : int, default=2
        Number of panels per row.
    figsize : tuple, optional
        Size of the figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(6.3, 3.9)
        Size of one panel, used to derive the size of the figure.
    **kwargs : dict
        Further arguments handed to every panel, such as ``linewidth``,
        ``markersize``, ``morf_color`` and ``lerf_color``.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    import matplotlib.pyplot as plt

    items = list(picks)
    if not items:
        raise ValueError("At least one explanation is required.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(items) / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
    figure.get_layout_engine().set(h_pad=0.12, w_pad=0.02, hspace=0.06, wspace=0.03)
    flat = np.atleast_1d(axes).ravel()

    for item, ax in zip(items, flat):
        if isinstance(item, tuple):
            heading, explanation = item
        else:
            explanation = getattr(item, "explanation", item)
            heading = getattr(item, "title", None)
        plot_degradation_curve(explanation, ax=ax, title=heading, **kwargs)
    for ax in flat[len(items) :]:
        ax.axis("off")
    return _heading(figure, title)


# ------------------------------------------------------------ segment overlays
@_themed
def plot_segments(
    image,
    segmentation,
    segments,
    ax=None,
    title: Optional[str] = None,
    figsize=(4.6, 4.6),
    color: Optional[str] = None,
    alpha: float = 0.45,
    linewidth: float = 1.0,
    edge_color: Optional[str] = None,
    cmap: str = "gray",
    display=None,
):
    """Paint a set of segments over the image, translucently and outlined.

    This is the figure a reader looks at. The image stays untouched and the
    segments the explanation names are painted over it, so the eye is drawn to
    the location rather than to the substitution that produced the measurement.

    Parameters
    ----------
    image : array
        The image the partition was computed on.
    segmentation : Segmentation
        Partition the segments belong to.
    segments : sequence of int
        Positions of the segments painted. An empty sequence leaves the image
        bare, which is what the first panel of a marginalization sequence shows.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure.
    figsize : tuple, default=(4.6, 4.6)
        Size of the figure created when no axes is supplied.
    color : str, optional
        Color of the painted segments. The color in force for the session is
        used when it is left out, and ``set_curve_colors(segment=...)`` changes
        that one.
    alpha : float, default=0.45
        Opacity of the paint, low enough that the content stays readable.
    linewidth : float, default=1.0
        Thickness of the outline in pixels. Zero leaves the segments unoutlined.
    edge_color : str, optional
        Color of the outline. A darker shade of the paint is used by default,
        which keeps one segment in one color while still separating two segments
        that touch.
    cmap : str, default="gray"
        Colormap of a single channel image.
    display : array, optional
        Original image drawn instead of ``image``, at its own resolution.

    Returns
    -------
    matplotlib Axes
        The axes the segments were drawn on.
    """
    image = _as_display(image if display is None else display)
    ax = _new_axes(ax, figsize)
    _draw(ax, image, cmap)

    chosen = [int(position) for position in segments]
    if chosen:
        labels = resize_labels(segmentation.mask, image.shape[:2])
        painted = np.isin(labels, chosen)
        fill = color or SEGMENT_COLOR
        _overlay(ax, painted, fill, alpha)
        if float(linewidth) > 0:
            # the borders follow the label map rather than the painted area, so
            # two segments that touch are still told apart, which is what makes a
            # block of contiguous segments readable as several segments
            inside = np.where(painted, labels, -1)
            edges = (
                boundaries_of(inside, thickness=int(round(float(linewidth))))
                & painted
            )
            _overlay(ax, edges, edge_color or _darken(fill), 1.0)

    if title is not None:
        ax.set_title(title)
    return ax


def _flip_point(explanation, k=None) -> int:
    """How many leading segments an overlay paints when no count is given.

    The default is the step at which the predicted class first changes, since
    that is the claim a sparse explanation makes, namely that these segments and
    no others carry the decision. A ranking that never changes the class has no
    such step, and the start of the noise region takes its place, because the
    ranking says nothing beyond it.
    """
    if k is not None:
        return max(0, min(int(k), explanation.n_segments))
    if explanation.sparsity_point is not None:
        return int(explanation.sparsity_point)
    if explanation.noise_onset is not None:
        return int(explanation.noise_onset)
    return explanation.n_segments


@_themed
def plot_explanation_overlay(
    explanation,
    k: Optional[int] = None,
    ax=None,
    title: Optional[str] = None,
    figsize=(4.6, 4.6),
    color: Optional[str] = None,
    alpha: float = 0.45,
    linewidth: float = 1.0,
    edge_color: Optional[str] = None,
    cmap: str = "gray",
    display=None,
    annotate: bool = False,
):
    """Paint the leading segments of an explanation over the image.

    The segments painted by default are the ones whose cumulative marginalization
    changes the predicted class, which is exactly the claim the explanation
    makes. Everything after that step is left bare, since it is not part of the
    claim, and ``k`` overrides the count when a fixed budget is wanted instead.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    k : int, optional
        Number of leading segments painted. The step at which the class changes
        is used when it is left out, and the start of the noise region takes its
        place for a ranking that never changes the class.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure.
    figsize : tuple, default=(4.6, 4.6)
        Size of the figure created when no axes is supplied.
    color : str, optional
        Color of the painted segments, which is how one method is told from
        another in a comparison.
    alpha : float, default=0.45
        Opacity of the paint.
    linewidth : float, default=1.0
        Thickness of the outline in pixels.
    edge_color : str, optional
        Color of the outline.
    cmap : str, default="gray"
        Colormap of a single channel image.
    display : array, optional
        Original image drawn instead of the array the model receives.
    annotate : bool, default=False
        Whether the number of segments painted is added to the heading.

    Returns
    -------
    matplotlib Axes
        The axes the segments were drawn on.
    """
    count = _flip_point(explanation, k)
    heading = title if title is not None else f"SOFI, class {explanation.class_name}"
    if annotate:
        heading = f"{heading} ({count} segments)"
    return plot_segments(
        explanation.instance,
        explanation.segmentation,
        explanation.order[:count],
        ax=ax,
        title=heading,
        figsize=figsize,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        edge_color=edge_color,
        cmap=cmap,
        display=display,
    )


@_themed
def plot_overlay_grid(
    items,
    ncols: int = 3,
    figsize=None,
    panel_size=(3.8, 4.0),
    title=None,
    colors: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    annotate: bool = True,
    **kwargs,
):
    """Compare several explanations of one image, each painted in its own color.

    Every panel shows the same image with the segments one method selected, so
    the comparison is read at a glance rather than from a table. The colors
    carry no magnitude and serve only to tell the panels apart.

    Parameters
    ----------
    items : dict or sequence
        Mapping from a heading onto an explanation, or a sequence of pairs. A
        value may also be a pair holding an explanation and a color, which
        overrides the palette for that panel alone.
    ncols : int, default=3
        Number of panels per row.
    figsize : tuple, optional
        Size of the whole figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(3.8, 4.0)
        Size of one panel.
    title : str, optional
        Heading placed above the grid.
    colors : sequence of str, optional
        Palette cycled across the panels. A muted default is used when it is
        left out.
    k : int, optional
        Number of leading segments painted in every panel. Each explanation uses
        its own class change step when it is left out, which is what makes the
        panels comparable as claims rather than as fixed budgets.
    annotate : bool, default=True
        Whether the number of segments painted is written in each heading.
    **kwargs : dict
        Further arguments of :func:`plot_explanation_overlay`, such as
        ``alpha``, ``linewidth``, ``cmap`` and ``display``.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    import matplotlib.pyplot as plt

    entries = list(items.items()) if isinstance(items, dict) else list(items)
    if not entries:
        raise ValueError("At least one explanation is required.")

    palette = list(colors) if colors else list(_PALETTE)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(entries) / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
    flat = np.atleast_1d(axes).ravel()
    for position, (heading, value) in enumerate(entries):
        explanation, color = value if isinstance(value, tuple) else (value, None)
        plot_explanation_overlay(
            explanation,
            k=k,
            ax=flat[position],
            title=str(heading),
            color=color or palette[position % len(palette)],
            annotate=annotate,
            **kwargs,
        )
    for ax in flat[len(entries) :]:
        ax.axis("off")
    _heading(figure, title)
    return _fit_grid(figure, flat[: len(entries)], ncols)


@_themed
def plot_marginalization(
    explanation,
    steps: Optional[Sequence[int]] = None,
    ncols: int = 3,
    figsize=None,
    panel_size=(3.6, 3.9),
    title=None,
    cmap: str = "gray",
    color: Optional[str] = None,
    alpha: float = 0.45,
    linewidth: float = 1.0,
    edge_color: Optional[str] = None,
    display=None,
):
    """Show the image along the cumulative marginalization of the ranking.

    Segments are never marginalized in isolation, so an explanation cannot be
    drawn by shading each of them with a score of its own. What the ranking
    prescribes is a sequence, the first segment on its own, then the first two,
    then the first three, and this figure paints that sequence over the
    untouched image. Each panel reports the fidelity that survives the removal
    it shows, so the growth of the painted area and the collapse of the response
    are read together.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    steps : sequence of int, optional
        Steps drawn, counted from zero for the untouched image. A spread from
        the start of the ranking to just past the class change is chosen when
        omitted.
    ncols : int, default=3
        Number of panels per row.
    figsize : tuple, optional
        Size of the whole figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(3.6, 3.9)
        Size of one panel.
    title : str, optional
        Heading placed above the grid.
    cmap : str, default="gray"
        Colormap of a single channel image.
    color : str, optional
        Color of the painted segments.
    alpha : float, default=0.45
        Opacity of the paint.
    linewidth : float, default=1.0
        Thickness of the outline in pixels.
    edge_color : str, optional
        Color of the outline.
    display : array, optional
        Original image drawn instead of the array the model receives.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    import matplotlib.pyplot as plt

    n_states = explanation.n_segments + 1
    if steps is None:
        boundary = explanation.sparsity_point or explanation.noise_onset
        last = n_states - 1 if boundary is None else min(boundary + 1, n_states - 1)
        count = min(6, last + 1)
        steps = sorted(set(np.linspace(0, last, count).round().astype(int).tolist()))
    steps = [int(s) for s in steps]
    if any(s < 0 or s >= n_states for s in steps):
        raise ValueError(f"Steps must lie between 0 and {n_states - 1}.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(steps) / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
    flat = np.atleast_1d(axes).ravel()

    for step, ax in zip(steps, flat):
        fidelity = explanation.morf_scores[step]
        if step == 0:
            heading = f"untouched, fidelity {fidelity:.2f}"
        else:
            heading = (
                f"after {step} ({explanation.ranking[step - 1]}), "
                f"fidelity {fidelity:.2f}"
            )
        plot_segments(
            explanation.instance,
            explanation.segmentation,
            explanation.order[:step],
            ax=ax,
            title=heading,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            edge_color=edge_color,
            cmap=cmap,
            display=display,
        )
    for ax in flat[len(steps) :]:
        ax.axis("off")

    _heading(
        figure,
        title
        if title is not None
        else f"Cumulative marginalization of class {explanation.class_name}",
    )
    return _fit_grid(figure, flat[: len(steps)], ncols)


@_themed
def plot_explanation(
    explanation,
    k: Optional[int] = None,
    figsize=(8.0, 4.0),
    title=None,
    display=None,
    overlay_title: str = "Relevant segments",
    curve_title: str = "MoRF and LeRF curves",
    color: Optional[str] = None,
    alpha: float = 0.45,
    linewidth: float = 1.0,
    edge_color: Optional[str] = None,
    cmap: str = "gray",
    width_ratio: float = 1.45,
    **curve_kwargs,
):
    """Draw an explanation as a pair, the segments beside the two curves.

    This is the figure to reach for whenever one explanation has to be shown.
    Several explanations are shown by calling it once for each, so that every
    result carries its own heading and stands on its own, rather than by packing
    them into one canvas where the headings compete.

    An overlay says where the evidence lies and a curve says how much removing
    it costs, and neither answers for the other. A segment painted on a image is
    persuasive whether or not the response actually fell, and a degradation
    score is unreadable without knowing which segments produced it, so the two
    panels belong together.

    The panels are given the same height and the width of the left one is set to
    match it, so the square image fills its axes instead of floating in a band
    of white. The legend of the curve sits inside its own axes for the same
    reason, since a legend placed underneath would shorten one panel and not the
    other.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    k : int, optional
        Number of leading segments painted. The step at which the predicted
        class changes is used when it is left out.
    figsize : tuple, default=(10.2, 4.0)
        Size of the whole figure.
    title : str, optional
        Heading placed above both panels.
    display : array, optional
        Original image drawn instead of the array the model receives.
    overlay_title, curve_title : str
        Headings of the two panels.
    color, alpha, linewidth, edge_color, cmap : optional
        Arguments of :func:`plot_explanation_overlay`.
    width_ratio : float, default=1.45
        Width of the curve panel relative to the image. The default keeps the
        image square without leaving a gap between the two.
    **curve_kwargs : dict
        Arguments of :func:`plot_degradation_curve`, such as ``morf_color`` and
        ``markersize``.

    Returns
    -------
    matplotlib Figure
        The figure holding the two panels.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, 2, figsize=figsize, layout="constrained",
        gridspec_kw={"width_ratios": [1.0, float(width_ratio)]},
    )
    figure.get_layout_engine().set(w_pad=0.02, h_pad=0.12, wspace=0.02, hspace=0.0)

    plot_explanation_overlay(
        explanation, k=k, ax=axes[0], title=overlay_title, color=color, alpha=alpha,
        linewidth=linewidth, edge_color=edge_color, cmap=cmap, display=display,
        annotate=False,
    )
    plot_degradation_curve(
        explanation, ax=axes[1], title=curve_title, legend="inside", **curve_kwargs
    )
    _heading(figure, title)
    return _tighten_pair(figure, axes[0], axes[1])


@_themed
def plot_importance_map(
    explanation,
    ax=None,
    title: Optional[str] = None,
    figsize=(5.2, 4.6),
    cmap: str = "inferno",
    image_cmap: str = "gray",
    alpha: float = 0.55,
    top_k: Optional[int] = None,
    colorbar: bool = True,
    outline: bool = True,
    linewidth: float = 1.0,
    color: Optional[str] = None,
    display=None,
):
    """Draw the ranking as a graded heat map laid over the image.

    Every segment is painted with the rank based importance the explanation
    assigns to it. The weight of a segment follows its position in the ranking
    and nothing else, and the segments inside the noise region of the curve
    receive no weight and stay transparent. It is the graded alternative to
    :func:`plot_explanation_overlay`, which paints one color and no gradient.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure.
    figsize : tuple, default=(5.2, 4.6)
        Size of the figure created when no axes is supplied.
    cmap : str, default="inferno"
        Colormap of the importance values.
    image_cmap : str, default="gray"
        Colormap of a single channel image underneath.
    alpha : float, default=0.55
        Opacity of the heat map.
    top_k : int, optional
        Number of leading segments painted. Every segment carrying a weight is
        painted when it is omitted.
    colorbar : bool, default=True
        Whether the scale of the importance is drawn beside the axes.
    outline : bool, default=True
        Whether the borders of the painted segments are drawn.
    linewidth : float, default=1.0
        Thickness of those borders in pixels.
    color : str, optional
        Color of those borders for this figure alone.
    display : array, optional
        Original image drawn underneath, at its own resolution.

    Returns
    -------
    matplotlib Axes
        The axes the map was drawn on.
    """
    if explanation.instance is None and display is None:
        raise ValueError(
            "The explanation carries no image, so the map cannot be drawn over "
            "one. Pass the image through the display argument."
        )

    image = _as_display(explanation.instance if display is None else display)
    ax = _new_axes(ax, figsize)
    _draw(ax, image, image_cmap)

    weights = explanation.importances
    painted = set(range(explanation.n_segments))
    if top_k is not None:
        painted = set(explanation.order[: int(top_k)])

    labels = resize_labels(explanation.segmentation.mask, image.shape[:2])
    surface = np.full(labels.shape, np.nan, dtype=float)
    for position in painted:
        if weights[position] > 0:
            surface[labels == position] = weights[position]

    if np.all(np.isnan(surface)):
        raise ValueError(
            "No segment carries a weight, so nothing can be painted. The whole "
            "ranking fell inside the noise region of the curve."
        )

    image = ax.imshow(
        np.ma.masked_invalid(surface),
        cmap=cmap,
        alpha=float(alpha),
        vmin=0.0,
        vmax=float(np.nanmax(surface)),
        interpolation="nearest",
    )

    if outline:
        shown = np.where(np.isnan(surface), 0, 1)
        edges = boundaries_of(shown, thickness=int(round(float(linewidth)))) & (shown > 0)
        _overlay(ax, edges, color or BOUNDARY_COLOR)

    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Importance")

    ax.set_title(
        title
        if title is not None
        else f"SOFI importance, class {explanation.class_name}"
    )
    return ax


@_themed
def plot_optimization_trace(
    explanation, ax=None, figsize=(6.6, 4.0), title: Optional[str] = None
):
    """Draw how the degradation score evolved across the proposals of the search.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result of a run, which carries the history of the search.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when it is omitted.
    figsize : tuple, default=(6.6, 4.0)
        Size of the figure created when no axes is supplied.
    title : str, optional
        Heading of the figure.

    Returns
    -------
    matplotlib Axes
        The axes the trace was drawn on.
    """
    if not explanation.history:
        raise ValueError("The explanation holds no history, so no trace can be drawn.")
    ax = _new_axes(ax, figsize)
    iterations = [record["iteration"] for record in explanation.history]
    candidates = [record["candidate_ds"] for record in explanation.history]
    best = [record["best_ds"] for record in explanation.history]

    ax.plot(iterations, candidates, color="gray", linewidth=1.0, alpha=0.8, label="Proposal")
    ax.plot(iterations, best, color=MORF_COLOR, linewidth=1.8, label="Incumbent")
    for position in [r["iteration"] for r in explanation.history if r.get("restart")]:
        ax.axvline(position, color="gray", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Proposal")
    ax.set_ylabel("Degradation score")
    ax.set_title(title if title is not None else "SOFI search trajectory")
    ax.legend(loc="lower right", frameon=False)
    return ax
