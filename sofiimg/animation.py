"""Animate a SOFI explanation as a two panel film, written as a GIF and an MP4.

The animation replays the process a degradation curve records. The left panel
adds segments in the order a ranking prescribes, and the right panel grows the
corresponding curve one point at a time, so a viewer sees which segment produced
which fall.

Four details make it readable, and each of them is a correction of the obvious
way to do it.

The two curves are built one after the other rather than together. They follow
different orders, since the LeRF curve marginalizes the reverse of the ranking
the MoRF curve follows, so no single sequence of segments can drive both.
Showing them at once would suggest that one segment is responsible for a point
on each curve, which is false. The LeRF pass runs first, the left panel is then
cleared, and the MoRF pass starts fresh.

Color carries the ranking rather than the order of play. A segment is green when
it belongs to the set whose cumulative marginalization changes the predicted
class, and red-orange otherwise, in both passes and on the markers of both
curves. The LeRF pass therefore opens in red-orange and turns green only at the
very end, while the MoRF pass opens in green, which is the claim of the method
made visible.

The segments are painted over the original image rather than over the array the
model receives. Drawing the substituted pixels would put a patch of the operator
in front of the viewer, which is the mechanism rather than the finding.

The closing frames drop the unimportant segments, paint what remains in blue,
shade the area between the two curves and write the degradation score, so the
film ends on the explanation itself.

Run the file to rebuild the animation of the notebook:

    python examples/animate_marginalization.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from . import plotting
from .plotting import _as_display, _darken, _draw, _overlay, _theme, _tighten_pair
from .segmentation import boundaries_of, resize_labels

__all__ = ["animate_marginalization"]

RELEVANT_COLOR = "#4f9e6a"
IRRELEVANT_COLOR = "#b64f4f"


def _relevant_count(explanation, k: Optional[int] = None) -> int:
    """How many leading segments of the ranking carry the decision.

    The step at which the predicted class first changes is the natural answer,
    since that is the claim a sparse explanation makes. A ranking that never
    changes the class has no such step, and the start of the noise region takes
    its place, because the ranking says nothing beyond it.
    """
    if k is not None:
        return max(0, min(int(k), explanation.n_segments))
    if explanation.sparsity_point is not None:
        return int(explanation.sparsity_point)
    if explanation.noise_onset is not None:
        return int(explanation.noise_onset)
    return explanation.n_segments


def _draw_image_panel(
    ax,
    picture,
    labels,
    order,
    upto,
    relevant,
    cmap,
    alpha,
    linewidth,
    title,
    relevant_color,
    irrelevant_color,
    single_color=None,
    annotation=None,
):
    """One frame of the left panel, holding the first ``upto`` segments of an order."""
    ax.cla()
    _draw(ax, picture, cmap)

    chosen = list(order[:upto])
    if chosen:
        # the segments are grouped by color and painted in two passes, so a
        # boolean mask is built twice rather than once per segment
        groups = {}
        for position in chosen:
            key = single_color or (
                relevant_color if position in relevant else irrelevant_color
            )
            groups.setdefault(key, []).append(position)
        for key, group in groups.items():
            painted = np.isin(labels, group)
            _overlay(ax, painted, key, alpha)
            if float(linewidth) > 0:
                # the borders follow the label map rather than the painted area,
                # so two segments that touch are still told apart
                inside = np.where(painted, labels, -1)
                edges = (
                    boundaries_of(inside, thickness=int(round(float(linewidth))))
                    & painted
                )
                _overlay(ax, edges, _darken(key), 1.0)

    ax.set_title(title)

    # the key sits under the image, so a viewer never has to guess what a color
    # means. It names one entry alone once the closing frames drop the rest
    from matplotlib.patches import Patch

    if single_color is None:
        handles = [
            Patch(facecolor=relevant_color, edgecolor=_darken(relevant_color),
                  alpha=alpha, label="Important"),
            Patch(facecolor=irrelevant_color, edgecolor=_darken(irrelevant_color),
                  alpha=alpha, label="Unimportant"),
        ]
    else:
        handles = [
            Patch(facecolor=single_color, edgecolor=_darken(single_color),
                  alpha=alpha, label="Relevant segments")
        ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    if annotation is not None:
        ax.text(
            0.98,
            0.04,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="#001f3f",
                boxstyle="round,pad=0.35",
            ),
        )


def _draw_curve_panel(
    ax,
    explanation,
    morf,
    lerf,
    stage,
    revealed,
    linewidth,
    markersize,
    morf_color,
    lerf_color,
    show_ds,
):
    """One frame of the right panel.

    ``stage`` is zero while the LeRF curve grows, one while the MoRF curve grows
    over a completed LeRF curve, and two for the closing frames that shade the
    area between them.
    """
    ax.cla()
    n = len(morf) - 1
    steps = np.arange(len(morf))

    def draw(values, upto, base, label):
        # every setting below matches plot_degradation_curve, so a frame of the
        # animation and a figure of the notebook are indistinguishable
        visible = steps[: upto + 1]
        ax.plot(
            visible,
            values[: upto + 1],
            marker="s",
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=base,
            linewidth=linewidth,
            label=label,
            zorder=3 if base == lerf_color else 4,
        )

    if stage == 2:
        ax.fill_between(steps, morf, lerf, color="gray", alpha=0.22, zorder=2)

    if stage == 0:
        draw(lerf, revealed, lerf_color, "LeRF")
        # the MoRF entry is present from the first frame, so the legend never
        # changes size and the layout never shifts
        ax.plot([], [], color=morf_color, linewidth=linewidth, marker="s",
                markersize=markersize, markeredgecolor="white",
                markeredgewidth=0.7, label="MoRF")
    else:
        draw(lerf, n, lerf_color, "LeRF")
        draw(morf, revealed if stage == 1 else n, morf_color, "MoRF")

    marks = []
    if explanation.lerf_sparsity_point is not None:
        if stage >= 1 or explanation.lerf_sparsity_point <= revealed:
            marks.append((explanation.lerf_sparsity_point, lerf, lerf_color))
    if explanation.sparsity_point is not None:
        if stage == 2 or (stage == 1 and explanation.sparsity_point <= revealed):
            marks.append((explanation.sparsity_point, morf, morf_color))
    for point, curve, color in marks:
        ax.plot(
            [int(point)],
            [curve[int(point)]],
            marker=(8, 1, 0),
            markersize=2 * markersize,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )

    ax.set_ylim(-0.05, 1.22)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.4, n - 0.6 + 1)
    ax.set_xticks(steps[:: max(1, int(np.ceil(len(morf) / 11)))])
    ax.set_xlabel("Marginalized segments")
    ax.set_ylabel(plotting._YLABEL)
    ax.set_title("MoRF and LeRF curves")
    # the legend sits inside the axes and behind the curves, so that a marker
    # landing in that corner stays visible over it
    legend = ax.legend(
        loc="lower left", ncol=2, frameon=True, framealpha=0.9,
        handlelength=1.6, columnspacing=1.2, borderpad=0.4,
    )
    legend.set_zorder(1)

    if show_ds:
        ax.text(
            0.98,
            0.96,
            f"DS: {explanation.ds:.2f}",
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


@plotting._themed
def animate_marginalization(
    explanation,
    display=None,
    filename: Optional[str] = None,
    formats: Sequence[str] = ("gif",),
    title: str = "Sparseness Optimized Feature Importance",
    k: Optional[int] = None,
    figsize=None,
    panel_size=(4.0, 4.3),
    width_ratio: float = 1.55,
    dpi: int = 110,
    cmap: str = "gray",
    alpha: float = 0.5,
    segment_linewidth: float = 1.0,
    linewidth: float = 1.8,
    curve_markersize: float = 8.0,
    relevant_color: str = RELEVANT_COLOR,
    irrelevant_color: str = IRRELEVANT_COLOR,
    final_color: Optional[str] = None,
    morf_color: Optional[str] = None,
    lerf_color: Optional[str] = None,
    annotate_probability: bool = True,
    step_ms: int = 260,
    start_hold_ms: int = 900,
    pass_hold_ms: int = 1200,
    end_hold_ms: int = 3000,
):
    """Render the marginalization of one explanation as a GIF and an MP4.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    display : array, optional
        Original image drawn instead of the array the model receives, at its own
        resolution. The label map is carried onto that resolution by nearest
        neighbor sampling, so no border moves.
    filename : str, optional
        Path without an extension. A temporary file is used when it is left
        out, which is enough for a notebook that only displays the animation.
    formats : sequence of str, default=("gif",)
        Which files to write. Adding ``"mp4"`` writes a video as well, which
        needs ffmpeg.
    title : str
        Heading placed above both panels.
    k : int, optional
        Number of leading segments treated as relevant, and therefore painted
        green and kept in the closing frames. The step at which the predicted
        class changes is used when it is left out.
    figsize : tuple, optional
        Size of the figure holding the two panels. It is derived from
        ``panel_size`` and ``width_ratio`` when it is left out.
    panel_size : tuple, default=(4.0, 4.3)
        Width and height of the image panel, from which the figure is sized.
    width_ratio : float, default=1.55
        Width of the curve panel relative to the image panel, chosen so that the
        square image fills its axes rather than floating in a band of white.
    dpi : int, default=110
        Resolution of every frame.
    cmap : str, default="gray"
        Colormap of a single channel image.
    alpha : float, default=0.5
        Opacity of the painted segments.
    segment_linewidth : float, default=1.0
        Thickness of the segment outlines in pixels.
    linewidth : float, default=1.8
        Thickness of both curves, matching the figures of the notebook.
    curve_markersize : float, default=8.0
        Size of the square markers on the curves. The stars that mark a change
        of class are scaled from it.
    relevant_color, irrelevant_color : str
        Colors of the segments that do and do not carry the decision.
    final_color : str, optional
        Color of the closing frames. The session color is used by default.
    morf_color, lerf_color : str, optional
        Colors of the two curves. The settings in force for the session are used
        when they are left out, so one call to ``set_curve_colors`` governs the
        animation and the figures alike.
    annotate_probability : bool, default=True
        Whether the surviving probability is written in the left panel. It is
        the height of the growing curve at the same step, so the panels agree.
    step_ms : int, default=260
        Time each step is held on screen.
    start_hold_ms : int, default=900
        Extra time the untouched image is held at the start of each pass.
    pass_hold_ms : int, default=1200
        Time the completed LeRF curve is held before the second pass begins.
    end_hold_ms : int, default=3000
        Time the closing explanation is held, with the score shown.

    Returns
    -------
    list of str
        The paths the animation was written to.
    """
    picture = _as_display(explanation.instance if display is None else display)
    labels = resize_labels(explanation.segmentation.mask, picture.shape[:2])

    morf = np.asarray(explanation.morf_scores, dtype=float)
    lerf = np.asarray(explanation.lerf_scores, dtype=float)
    morf_order = list(explanation.order)
    lerf_order = list(reversed(explanation.order))
    n = explanation.n_segments
    count = _relevant_count(explanation, k)
    relevant = set(morf_order[:count])

    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (panel_size[0] * (1.0 + float(width_ratio)), panel_size[1])
    figure, (left, right) = plt.subplots(
        1, 2, figsize=figsize, layout="constrained",
        gridspec_kw={"width_ratios": [1.0, float(width_ratio)]},
    )
    figure.get_layout_engine().set(w_pad=0.03, h_pad=0.12, wspace=0.02, hspace=0.0)
    figure.suptitle(title)

    # one entry per frame, holding the pass, the step inside it and the time the
    # frame stays on screen
    schedule = [(0, 0, start_hold_ms)]
    schedule += [(0, step, step_ms) for step in range(1, n + 1)]
    schedule += [(0, n, pass_hold_ms)]
    schedule += [(1, 0, start_hold_ms)]
    schedule += [(1, step, step_ms) for step in range(1, n + 1)]
    schedule += [(2, n, end_hold_ms)]

    frames, durations, tightened = [], [], False
    for stage, step, duration in schedule:
        if stage == 0:
            _draw_image_panel(
                left, picture, labels, lerf_order, step, relevant, cmap, alpha,
                segment_linewidth, "Least relevant segments first",
                relevant_color, irrelevant_color,
                annotation=(
                    f"Probability: {lerf[step]:.3f}" if annotate_probability else None
                ),
            )
        elif stage == 1:
            _draw_image_panel(
                left, picture, labels, morf_order, step, relevant, cmap, alpha,
                segment_linewidth, "Most relevant segments first",
                relevant_color, irrelevant_color,
                annotation=(
                    f"Probability: {morf[step]:.3f}" if annotate_probability else None
                ),
            )
        else:
            _draw_image_panel(
                left, picture, labels, morf_order, count, relevant, cmap, alpha,
                segment_linewidth, "Relevant segments",
                relevant_color, irrelevant_color,
                single_color=final_color or plotting.SEGMENT_COLOR,
                annotation=None,
            )
        _draw_curve_panel(
            right, explanation, morf, lerf, stage, step, linewidth,
            curve_markersize,
            morf_color or plotting.MORF_COLOR,
            lerf_color or plotting.LERF_COLOR,
            show_ds=stage == 2,
        )
        if not tightened:
            # the panels are placed once, on the first frame, and every frame
            # afterwards reuses those positions
            _tighten_pair(figure, left, right)
            tightened = True

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
        durations.append(duration)

    plt.close(figure)

    if filename is None:
        filename = str(Path(tempfile.mkdtemp()) / "sofi_marginalization")
    filename = str(filename)
    for suffix in (".gif", ".mp4"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]

    written = []
    for kind in formats:
        name = f"{filename}.{kind}"
        if kind == "gif":
            frames[0].save(
                name, save_all=True, append_images=frames[1:],
                duration=durations, loop=0, optimize=True,
            )
        elif kind == "mp4":
            _write_mp4(frames, durations, name)
        else:
            raise ValueError(f"Unknown format '{kind}', use 'gif' or 'mp4'.")
        written.append(name)
    return written


def _write_mp4(frames, durations, name: str, fps: int = 25) -> None:
    """Write the same frames as a video, repeating each for its own duration.

    A video runs at one constant rate, so a frame that should linger is written
    several times rather than given a longer exposure. That is what keeps the
    timing of the video identical to the timing of the GIF.
    """
    try:
        import imageio_ffmpeg
    except ImportError as error:  # pragma: no cover - depends on the system
        raise RuntimeError(
            "Writing an MP4 needs ffmpeg. Install the imageio-ffmpeg package, "
            "which carries a copy, or drop 'mp4' from the formats argument."
        ) from error

    width, height = frames[0].size
    # an even size is required by the H.264 encoder
    width -= width % 2
    height -= height % 2
    writer = imageio_ffmpeg.write_frames(
        name, (width, height), fps=fps, quality=8, macro_block_size=None
    )
    writer.send(None)
    for frame, duration in zip(frames, durations):
        pixels = np.asarray(frame)[:height, :width].tobytes()
        for _ in range(max(1, int(round(fps * duration / 1000.0)))):
            writer.send(pixels)
    writer.close()
