"""Shared matplotlib look for every esnupi report figure.

Figures are saved as SVG with live text (svg.fonttype = none) so Chrome sets
the labels in the report's own fonts and the PDF stays vector. Column width
is 6.6in; draw at 1:1 so 7.5pt tick labels print at 7.5pt.

Palette (validated with the dataviz validator on #ffffff, adjacent + all pairs):
  BLUE #153bfa (site brand) → ORANGE #eb6834 → AQUA #1baf7a (aqua < 3:1, always direct-label it).
GREY is for baselines/benchmarks only, never a data series of its own.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE, AQUA = "#153bfa", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, MUTED, GRID, GREY, WASH = "#111318", "#5a5f6b", "#e4e6eb", "#9aa0ab", "#eef1ff"
COL_W = 6.6

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "Helvetica Neue",
    "font.size": 7.5,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 7.5,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.transparent": False,
    "figure.facecolor": "white",
})


def pct(decimals=0):
    return FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%".replace("-", "−"))


def num(decimals=0, prefix="", suffix=""):
    return FuncFormatter(lambda v, _: f"{prefix}{v:,.{decimals}f}{suffix}".replace("-", "−"))


def label_end(ax, x, y, text, dx=4, dy=0, **kw):
    """Direct label at a line's end, in ink (identity comes from the swatch-coloured line)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                fontsize=7, color=INK, **kw)


def save(fig, path):
    fig.savefig(path, format="svg")
    plt.close(fig)
