"""F7 variant -- the cut-granularity trade-off as a ranked chart instead of a plane.

The scatter version put 32 cells on a cost/detection plane and needed four marker
codes, a step frontier, range bars and a second panel to be read. Almost every cell
in it was a cell nobody chooses. This variant drops the plane and ranks the ten cuts
a reader is actually deciding between, by the one number that decides: the block rate
on the attack class each cut does WORST on. Cost and the benign fence become two
printed columns, so the three quantities line up per row instead of being spread
across two panels.

What it says, in the order the rows say it:

1. A cost/detection frontier exists and the deployed union sits on it. `count:4` +
   `width:2` blocks 0.895 on its worst class -- the top of the eligible ranking --
   at 1.00x cost by construction. Bold cost = on the frontier (no cheaper cut in
   the 32 blocks more).
2. The previously deployed `count:6` + `width:2` is NOT on it: four rows down at
   0.874, and its cost column reads 1.32 -- it costs more and blocks less.
3. A plain coarse `count:4` reaches 0.864 at 0.16x. Fifth in rank, cheapest but
   one on the chart, and on the frontier.
4. Unioning is not free. `excess` over a union is the pointwise max of its
   components, so the union's benign quantile is at least each component's, and
   the fence column shows it directly -- 4+2's 6.8 is above both count:4's 3.9
   and width:2's 4.7, and 6+2's 10.3 is above count:6's 8.1 and the same 4.7. The deployed union clears 0.0068 on the ComQA arm where
   the previous one clears 0.0103 -- the fence is what turns a score into a block,
   so a wider cut is not automatically better.
5. The three fine-only cuts are set apart below the rule: they cannot cut some
   benign entries at all (4.6% / 38% / 75% of the ComQA arm), those entries route
   to a miss, and the 5% fence budget never charges for it -- so their nominal
   block rates are not comparable and they are off the frontier. `width:2` alone
   would otherwise top the whole ranking, which is exactly why the deployed cut
   unions a coarse component onto it: to keep short entries scoreable.

Conventions kept from the scatter version, both load-bearing:
  cost   = per-set dots_per_hit.pooled.mean / the deployed cell's, then the GEOMETRIC
           mean over CAP/SCP/KCA, because absolute span counts scale with entry length.
  detect = block_at_budget summarized by the WORST of the three attack classes; one
           policy runs over all traffic and the cut ranking is anti-correlated between
           classes (Spearman -0.26), so a mean would hide a real disagreement.
  a cell whose benign-unjudgeable share exceeds 1% in any set is not frontier-eligible.

Ten of the 32 cells are drawn. Each omitted cell is a shown cut's span/runs twin,
is dominated by a shown cut (costs at least as much and blocks no more on its worst
class), or blocks under 0.6 -- that last is count:2 alone (0.05x, 0.551), which is on
the frontier but far off the bottom of this axis. Singles are shown in their span
form (the form the paper quotes: count:4 at 0.16x); unions exist only in runs form.

Sources: experiments/paper/results/v3/granularity32/g32_{cap,scp,kca}.json
  -> grid[<policy>].dots_per_hit.pooled.mean   (cost)
  -> grid[<policy>].block_at_budget            (detection, per set)
  -> grid[<policy>].threshold_scoring          (benign fence; cap == scp, both the
                                                ComQA benign arm)
  -> grid[<policy>].n_benign_unjudgeable       (with top-level n_benign_total)
Produced by experiments/paper/rq3_mechanism/rq3_granularity.py --mode sweep.
"""
from experiments.paper.paths import paper_data_root, figure_output_root, RESULTS_ROOT
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.paper.figures.style import apply, OURS, GENUINE, BASELINE, CONTROL, CONTROL2, INK, FULL_W

V3 = os.environ.get(
    "V3_RESULTS",
    str(paper_data_root() / "figures/v3"))
GRAN = os.path.join(V3, "granularity32")
OUT = os.path.join(figure_output_root(), "fig7_tradeoff.pdf")

SETS = ("cap", "scp", "kca")
#: the benign arm the fence column reports; cap and scp share the ComQA benign entries
#: (identical threshold_scoring), kca uses the Natural Questions arm.
FENCE_SET = "cap"
DEPLOYED = "multi[count:4+width:2:cap16]/runs"
PREVIOUS = "multi[count:6+width:2:cap16]/runs"
UNJUDGEABLE_TOL = 0.01

#: (spec, printed label, tag) -- order is fixed by worst-class block rate within each
#: group and asserted against the data at run time, so a data change cannot silently
#: leave the rows mis-sorted.
ELIGIBLE = [
    (DEPLOYED,                           "4 equal + 2-word", "deployed"),
    ("multi[count:8+width:3:cap16]/runs", "8 equal + 3-word", None),
    ("multi[count:3+width:2:cap16]/runs", "3 equal + 2-word", None),
    (PREVIOUS,                           "6 equal + 2-word", None),
    ("count:4",                          "4 equal",          None),
    ("count:6",                          "6 equal",          None),
    ("count:3",                          "3 equal",          None),
]
INELIGIBLE = [
    ("width:2:cap16", "2-word"),
    ("width:3:cap16", "3-word"),
    ("width:4:cap16", "4-word"),
]

# ---- geometry, in row units (y grows downward) --------------------------------
Y_HEAD = -1.30
Y_RULE = 6.80
Y_GROUP = 7.42
Y_INELIG0 = 8.15
YLIM = (10.75, -2.15)
XLIM = (0.753, 0.9045)
XTICKS = (0.78, 0.82, 0.86, 0.90)
#: axes-fraction x of the right edge of each printed column.
COL_COST = 1.070
COL_FENCE = 1.258


def load():
    grids = {}
    for s in SETS:
        with open(os.path.join(GRAN, f"g32_{s}.json"), encoding="utf-8") as fh:
            grids[s] = json.load(fh)
    dep = {s: grids[s]["grid"][DEPLOYED]["dots_per_hit"]["pooled"]["mean"] for s in SETS}
    cells = {}
    for spec in grids[SETS[0]]["grid"]:
        per = {s: grids[s]["grid"][spec] for s in SETS}
        blocks = [per[s]["block_at_budget"] for s in SETS]
        cells[spec] = {
            "rel": float(np.exp(np.mean([
                np.log(per[s]["dots_per_hit"]["pooled"]["mean"] / dep[s]) for s in SETS]))),
            "worst": float(min(blocks)),
            "fence": float(per[FENCE_SET]["threshold_scoring"]),
            "unj": max(per[s]["n_benign_unjudgeable"] / grids[s]["n_benign_total"]
                       for s in SETS),
        }
    return cells


def frontier(cells):
    """Specs not dominated on (low cost, high worst-class block), among eligible cells."""
    pts = sorted(((v["rel"], v["worst"], k) for k, v in cells.items()
                  if v["unj"] <= UNJUDGEABLE_TOL), key=lambda p: (p[0], -p[1]))
    out, best = set(), -np.inf
    for x, y, spec in pts:
        if y > best + 1e-12:
            out.add(spec)
            best = y
    return out


def main():
    apply()
    cells = load()
    front = frontier(cells)

    for group in (ELIGIBLE, INELIGIBLE):
        ws = [cells[r[0]]["worst"] for r in group]
        assert ws == sorted(ws, reverse=True), f"rows out of rank order: {ws}"

    fig = plt.figure(figsize=(FULL_W, 1.5971))
    ax = fig.add_axes([0.1420, 0.170, 0.6640, 0.820])

    for x in XTICKS:
        ax.plot([x, x], [-0.50, YLIM[0] - 0.15], color=CONTROL2, lw=0.4,
                zorder=0, solid_capstyle="butt")

    labels, lab_col, lab_wt = [], [], []

    def draw(y, spec, label, tag, eligible):
        v = cells[spec]
        if spec == DEPLOYED:
            col, mfc, ms, mew = OURS, OURS, 5.4, 0.0
        elif eligible:
            col, mfc, ms, mew = BASELINE, BASELINE, 4.2, 0.0
        else:
            col, mfc, ms, mew = CONTROL, "white", 4.2, 1.0
        # leader: stops short of a tag so no text ever sits on a line or a marker
        stop = v["worst"] - (0.026 if tag else 0.006)
        ax.plot([XLIM[0] + 0.002, stop], [y, y], ls=(0, (1, 2)), lw=0.55,
                color=CONTROL2, zorder=1, solid_capstyle="butt")
        ax.plot([v["worst"]], [y], "o", ms=ms, mfc=mfc, mec=col, mew=mew, zorder=3)
        if tag:
            ax.text(v["worst"] - 0.005, y, tag, ha="right", va="center",
                    fontsize=7, color=col, zorder=4)
        labels.append(label)
        lab_col.append(col if spec is DEPLOYED else (INK if eligible else CONTROL))
        lab_wt.append("bold" if spec is DEPLOYED else "normal")
        # Only the DEPLOYED row's numbers are coloured. Colouring PREVIOUS's cost made
        # the superseded cut louder than the deployed one, which inverts the message.
        num = OURS if spec is DEPLOYED else (INK if eligible else CONTROL)
        on_front = spec in front
        if on_front:
            ax.plot([COL_COST - 0.088], [y], marker="D", ms=2.4, mfc=INK, mec=INK,
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=4)
        ax.text(COL_COST, y, f"{v['rel']:.2f}", ha="right", va="center", fontsize=7,
                color=num, fontweight="bold" if spec is DEPLOYED else "normal",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=4)
        ax.text(COL_FENCE, y, f"{v['fence'] * 1e3:.1f}", ha="right", va="center",
                fontsize=7, color=num, transform=ax.get_yaxis_transform(),
                clip_on=False, zorder=4)

    ys = []
    for i, (spec, label, tag) in enumerate(ELIGIBLE):
        draw(i, spec, label, tag, True)
        ys.append(i)
    for j, (spec, label) in enumerate(INELIGIBLE):
        y = Y_INELIG0 + j
        share = cells[spec]["unj"]
        draw(y, spec, f"{label}  ({share * 100:.0f}%)" if share >= 0.1
             else f"{label}  ({share * 100:.1f}%)", None, False)
        ys.append(y)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for t, c, w in zip(ax.get_yticklabels(), lab_col, lab_wt):
        t.set_color(c)
        t.set_fontsize(7)
        t.set_fontweight(w)
    ax.tick_params(axis="y", length=0, pad=2.5)

    # group rule + heading for the cells the frontier cannot include
    ax.plot(XLIM, [Y_RULE] * 2, color=CONTROL2, lw=0.6, clip_on=False, zorder=1)
    ax.text(XLIM[0] + 0.001, Y_GROUP,
            "not Pareto-eligible: % of benign keys too short to segment",
            ha="left", va="center", fontsize=6.4, color=CONTROL, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    # column heads, and -- in the same empty band -- what the ten rows are ten of
    ax.text(COL_COST, Y_HEAD, "cost", ha="right", va="center",
            fontsize=7, color=INK, clip_on=False,
            transform=ax.get_yaxis_transform())
    ax.text(COL_FENCE, Y_HEAD, r"$\eta\times10^{3}$", ha="right",
            va="center", fontsize=7, color=INK, clip_on=False,
            transform=ax.get_yaxis_transform())

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(list(XTICKS))
    ax.set_xticklabels([f"{t:.2f}" for t in XTICKS], fontsize=7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(XLIM[0], XLIM[1])
    ax.set_xlabel("worst-class block rate at 5% FPR (CAP / SCP / KCA)", fontsize=7.5,
                  labelpad=1.5)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}")

    print(f"\n{'policy':<36} {'cost':>5} {'worst':>6} {'fence':>7} {'unj':>6} {'F':>2}")
    for spec, _, _ in ELIGIBLE:
        v = cells[spec]
        print(f"{spec:<36} {v['rel']:>5.2f} {v['worst']:>6.3f} {v['fence']:>7.5f} "
              f"{v['unj']:>6.3f} {'*' if spec in front else '':>2}")
    for spec, _ in INELIGIBLE:
        v = cells[spec]
        print(f"{spec:<36} {v['rel']:>5.2f} {v['worst']:>6.3f} {v['fence']:>7.5f} "
              f"{v['unj']:>6.3f} {'':>2}")
    print(f"frontier ({len(front)}): {sorted(front)}")


if __name__ == "__main__":
    main()
