"""Shared design system for the full-page plates (Figures 1 and 4).

Both figures are read the same way, so they are drawn the same way. Each
panel is a fixed vertical stack inside its own frame:

    header strip   badge, title, one-line subtitle
    plot           never occluded by anything
    legend         identity, directly under the axis
    note           the caveat, in grey, at the foot of the frame

Nothing is allowed to sit on top of the data. An earlier version floated the
note box inside the axes and it covered curves in half the panels; the band
below the plot costs a little height and gives all of it back in legibility.

Columns are wide on purpose. The gutter is only as large as two y-axis
labels need, because plate area spent on white space is plate area not spent
on the plot.

Palette: categorical slots blue / orange / violet / green, validated for
colour-vision deficiency (worst adjacent CVD dE 18.4, normal-vision dE 24.0)
and for contrast against the panel surface. Identity is always carried by
position and label as well as by hue, so the plates survive greyscale
printing.
"""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import FancyBboxPatch      # noqa: E402

BLUE, ORANGE, VIOLET, GREEN = "#2a78d6", "#eb6834", "#8a5cd1", "#12836a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d8d7d2"
SURF = "#f7f7f5"
HEADS = ["#e8f0fb", "#fdeee7", "#f1ebfa"] * 4
BADGE = [BLUE, ORANGE, VIOLET] * 4

RC = {
    "font.size": 6.3, "axes.titlesize": 6.6, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED,
}

# --- the vertical budget inside one frame, in figure fractions ------------
HEAD_H = 0.048      # header strip
GAP_TOP = 0.005     # between header and the top of the axes
XLAB_H = 0.029      # tick labels plus the x label, under the axes
LEG_H = 0.032       # legend band (keep legends to at most two rows)
NOTE_H = 0.038      # grey note band (at most three lines)
PAD_BOT = 0.008     # frame padding below the note
AX_PAD_L = 0.058    # room inside the frame for the y label and ticks
AX_PAD_R = 0.014

FRAME_OVERHEAD = HEAD_H + GAP_TOP + XLAB_H + LEG_H + NOTE_H + PAD_BOT


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, lw=0.4, color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.2, colors=INK2)


def panels(fig, nrow=3, ncol=2, left=0.020, right=0.980, gutter=0.020,
           top=0.938, bottom=0.088, vgap=0.007):
    """Lay out ``nrow x ncol`` framed panels and return ``(axes, frames)``.

    ``frames`` are the rectangles the headers, legends and notes are placed
    against; ``axes`` sit inside them with room reserved for the labels.
    """
    fw = (right - left - gutter * (ncol - 1)) / ncol
    fh = (top - bottom - vgap * (nrow - 1)) / nrow
    axes, frames = [], []
    for r in range(nrow):
        fy = top - (r + 1) * fh - r * vgap
        for c in range(ncol):
            fx = left + c * (fw + gutter)
            ax = fig.add_axes([
                fx + AX_PAD_L,
                fy + PAD_BOT + NOTE_H + LEG_H + XLAB_H,
                fw - AX_PAD_L - AX_PAD_R,
                fh - FRAME_OVERHEAD,
            ])
            ax.set_zorder(2)
            ax.patch.set_visible(False)
            tidy(ax)
            axes.append(ax)
            frames.append((fx, fy, fw, fh))
    return axes, frames


def head(bg, frame, k, title, sub):
    """Rounded frame plus a pastel header strip with a numbered badge."""
    fx, fy, fw, fh = frame
    bg.add_patch(FancyBboxPatch(
        (fx, fy), fw, fh,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        facecolor="white", edgecolor=GRID, lw=0.7, zorder=-10))
    bg.add_patch(FancyBboxPatch(
        (fx + 0.004, fy + fh - HEAD_H), fw - 0.008, HEAD_H - 0.004,
        boxstyle="round,pad=0.002,rounding_size=0.007",
        facecolor=HEADS[k - 1], edgecolor="none", zorder=-9))
    cy = fy + fh - HEAD_H / 2 - 0.002
    bg.plot([fx + 0.021], [cy + 0.008], marker="o", ms=8.0,
            color=BADGE[k - 1], zorder=-8, clip_on=False)
    bg.text(fx + 0.021, cy + 0.008, str(k), ha="center", va="center",
            color="white", fontsize=5.8, fontweight="bold", zorder=-7)
    bg.text(fx + 0.035, cy + 0.009, title, ha="left", va="center",
            color=INK, fontsize=7.3, fontweight="bold", zorder=-7)
    bg.text(fx + 0.035, cy - 0.011, sub, ha="left", va="center",
            color=BADGE[k - 1], fontsize=5.8, zorder=-7)


_CMD = re.compile(r"\\[a-zA-Z]+")


def _visual_len(token):
    """Rendered width of a token, in character-widths.

    Counting raw characters over-measures anything with mathtext in it:
    ``$\\alpha^\\star$`` is fourteen characters and renders as about two. Each
    backslash command collapses to roughly one glyph, and the delimiters and
    grouping braces render as nothing, so they are removed before counting.
    """
    t = _CMD.sub("x", token)
    return len(re.sub(r"[${}^_\\]", "", t))


def _wrap(text, max_visual):
    """Greedy wrap that keeps ``$...$`` groups intact.

    Mathtext often contains spaces -- ``$n = 8,000$`` -- so the text cannot
    simply be split on whitespace; the spaces inside a math group are
    protected first and restored afterwards.
    """
    protected, out, in_math = [], [], False
    for ch in text:
        if ch == "$":
            in_math = not in_math
        out.append("\x00" if (ch == " " and in_math) else ch)
    words = "".join(out).split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        wl = _visual_len(w)
        if cur and cur_len + 1 + wl > max_visual:
            lines.append(" ".join(cur))
            cur, cur_len = [w], wl
        else:
            cur.append(w)
            cur_len += (1 if cur_len else 0) + wl
    if cur:
        lines.append(" ".join(cur))
    return [ln.replace("\x00", " ") for ln in lines]


def note(bg, frame, text, fontsize=5.3, maxlines=3):
    """The grey caveat band at the foot of the frame, below the legend.

    ``text`` is given as one string and wrapped to the width of the box, so
    the line breaks follow the layout instead of being typed against a guess
    at it. The wrap is then *measured* rather than trusted: a character-count
    estimate is unreliable for mathtext, where ``$\\alpha^\\star$`` is fourteen
    characters and two glyphs, so the rendered extent is checked and the
    target narrowed until the text is inside the box.
    """
    fx, fy, fw, fh = frame
    fig = bg.figure
    box_w = fw - 0.020
    inner = box_w - 0.016
    fig_w_in = fig.get_size_inches()[0]
    target = int(inner * fig_w_in / (0.50 * fontsize / 72.0))

    txt, size = None, fontsize
    for attempt in range(6):
        lines = _wrap(text, max(12, target))
        if len(lines) > maxlines:
            size = max(fontsize - 0.9, fontsize * (maxlines / len(lines)) ** 0.5)
            target = int(target * fontsize / size)
            continue
        if txt is not None:
            txt.remove()
        txt = bg.text(fx + 0.017, fy + PAD_BOT + (NOTE_H - 0.004) / 2,
                      "\n".join(lines), ha="left", va="center", color=INK2,
                      fontsize=size, zorder=-7, linespacing=1.32)
        fig.canvas.draw()
        w_frac = (txt.get_window_extent(fig.canvas.get_renderer())
                  .transformed(fig.transFigure.inverted()).width)
        if w_frac <= inner:
            break
        target = max(12, int(target * inner / w_frac) - 1)

    bg.add_patch(FancyBboxPatch(
        (fx + 0.010, fy + PAD_BOT), box_w, NOTE_H - 0.004,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor="#f4f4f1", edgecolor=GRID, lw=0.5, zorder=-8))


def legend(fig, ax, frame, ncol=2, fontsize=5.5, **kw):
    """Legend in the band between the x label and the note."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    fx, fy, fw, fh = frame
    ax.legend(h, l, loc="upper center",
              bbox_to_anchor=(fx + fw / 2, fy + PAD_BOT + NOTE_H + LEG_H),
              bbox_transform=fig.transFigure, ncol=ncol, frameon=False,
              fontsize=fontsize, handlelength=1.3, borderpad=0.0,
              labelspacing=0.25, columnspacing=1.0, handletextpad=0.45,
              labelcolor=INK2, **kw)


def banner(bg, text, y=0.948, h=0.038, fontsize=9.0):
    bg.add_patch(FancyBboxPatch(
        (0.020, y), 0.960, h,
        boxstyle="round,pad=0.003,rounding_size=0.007",
        facecolor="#1c2c46", edgecolor="none", zorder=-9))
    bg.text(0.5, y + h / 2, text, ha="center", va="center", color="white",
            fontsize=fontsize, fontweight="bold", zorder=-8)


def ledger(bg, title, items, y=0.005, h=0.075, fontsize=5.3):
    bg.add_patch(FancyBboxPatch(
        (0.020, y), 0.960, h,
        boxstyle="round,pad=0.003,rounding_size=0.007",
        facecolor=SURF, edgecolor=GRID, lw=0.7, zorder=-9))
    bg.text(0.034, y + h - 0.010, title, ha="left", va="center", color=INK,
            fontsize=6.4, fontweight="bold", zorder=-8)
    for i, (c, txt) in enumerate(items):
        yy = y + h - 0.024 - i * 0.0132
        bg.plot([0.038], [yy], marker="o", ms=3.2, color=c, zorder=-8)
        bg.text(0.048, yy, txt, ha="left", va="center", color=INK2,
                fontsize=fontsize, zorder=-8)


def provenance(bg, text, y=0.0012):
    bg.text(0.978, y, text, ha="right", va="bottom", color=GRID,
            fontsize=4.4, zorder=-8)
