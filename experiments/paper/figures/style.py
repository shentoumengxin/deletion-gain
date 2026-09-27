"""Shared publication style: blue/orange for classes, orange/gray for methods.

Use TrueType Liberation Serif with STIX math so PDF text remains selectable and
embedded font metadata is valid. All active plots are designed for 5.5-inch width.
"""
from experiments.paper.paths import paper_data_root, figure_output_root, RESULTS_ROOT
import matplotlib as mpl

OURS = "#d55e00"
BASELINE = "#4a4a4a"
CONTROL = "#8a8a8a"
CONTROL2 = "#c6c6c6"
INK = "#1a1a1a"
#: Figures that contrast two data CLASSES rather than two methods use these: the accent
#: goes to the planted entries, because they are what the test has to find, and genuine
#: entries take a mid grey. Aliasing ATTACK to OURS (an earlier shortcut for the
#: superseded generators) silently collapsed both classes of fig3 onto one colour.
PLANTED = OURS
#: Light blue. Blue against vermillion is the safest pair there is for colourblind
#: readers -- safer than the green it replaces, which sat close to vermillion for
#: deuteranopes -- and it keeps a lightness gap so the panels survive greyscale.
GENUINE = "#0b5394"
ATTACK = PLANTED

# The canvas, in inches. Every figure is placed with \includegraphics[width=...], so the
# PDF is rescaled to the column and what a reader actually sees is
#     printed point size = natural point size x (target width / FULL_W),
#     printed height     = target width x (natural height / FULL_W).
# The second depends only on the aspect ratio. So shrinking the canvas while keeping each
# figure's aspect ratio makes every label print LARGER at exactly the same footprint on the
# page -- which is how the type got legible without costing the layout a line. Raising the
# point sizes instead would have grown the figures too, and it cost a page when tried.
# Change this and you change the printed size of every label; keep each figsize height in
# the same proportion to it.
FULL_W = 5.5
HALF_W = 2.65


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
