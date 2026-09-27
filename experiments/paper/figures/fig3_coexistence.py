"""F3 -- coexistence: what cosine merges, deletion gain pulls apart.

Two panels from data/perrow/deletion_v3_cap_e5.jsonl: the v3 CAP-800 set (267
compress-append / 267 blend / 266 fuse) against the 499 human_comqa genuine entries,
scored under the deployed cut multi[count:4+width:2:cap16]/runs on e5-small-v2 in
float16, i.e. the exact rows Table 1's CAP column is read from. Regenerated 2026-09-01 by
experiments/paper/rq1_detection/dump_perrow_gain.py through the authoritative serving path
(deletion.build_profile + deletion.excess, per row). The same script re-dumped the
6+2 cut as a control: its excess distribution matches the earlier
deletion_v3_cap_e5.jsonl to 4.2e-05 (genuine) and 2.6e-05 (attack), i.e. within the
encoder's batch-composition band, which is what puts the 4+2 rows on the same footing:
(a) density of base_cos (cached entry vs its benign anchor) for genuine vs
    attack -- the two classes overlap heavily above the retrieval threshold
    tau = 0.90, so cosine alone cannot separate them.
(b) scatter of (base_cos, excess_span): at every matched cosine, genuine rows
    sit at/below gain = 0 while attacks sit well above it.
"""
from experiments.paper.paths import paper_data_root, figure_output_root, RESULTS_ROOT
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from experiments.paper.figures.style import apply, PLANTED, GENUINE, CONTROL, CONTROL2, FULL_W, HALF_W

DATA = os.path.join(paper_data_root(), "perrow",
                    "deletion_v3_cap_e5_42.jsonl")
OUT = os.path.join(figure_output_root(), "fig3_coexistence.pdf")

TAU = 0.90


def main():
    apply()
    rows = [json.loads(l) for l in open(DATA)]
    gc = np.array([r["base_cos"] for r in rows if r["arm"] == "genuine"])
    ac = np.array([r["base_cos"] for r in rows if r["arm"] == "attack"])
    ge = np.array([r["excess_span"] for r in rows if r["arm"] == "genuine"])
    ae = np.array([r["excess_span"] for r in rows if r["arm"] == "attack"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FULL_W, 1.75), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- (a) cosine coexistence -------------------------------------------
    bins = np.linspace(0.85, 1.0, 46)
    hg, _ = np.histogram(gc, bins=bins, density=True)
    ha, _ = np.histogram(ac, bins=bins, density=True)
    ctr = 0.5 * (bins[:-1] + bins[1:])

    ax1.fill_between(ctr, hg, step="mid", color=GENUINE, alpha=0.28, lw=0)
    ax1.fill_between(ctr, ha, step="mid", color=PLANTED, alpha=0.28, lw=0)
    ax1.step(ctr, hg, where="mid", color=GENUINE, lw=0.9)
    ax1.step(ctr, ha, where="mid", color=PLANTED, lw=0.9)
    # overlap region: min of the two densities
    ax1.fill_between(ctr, np.minimum(hg, ha), step="mid",
                     color="#6b5a48", alpha=0.45, lw=0)

    ymax = max(hg.max(), ha.max())
    ax1.set_ylim(0, ymax * 1.42)
    # the tau line stops below the note block so the two never cross
    ax1.axvline(TAU, ymin=0, ymax=0.60, color=CONTROL, lw=0.7, ls=":")
    ax1.text(TAU + 0.003, ymax * 0.62, r"retrieval $\tau=0.90$",
             va="bottom", ha="left", fontsize=8.5, color=CONTROL)
    # One note block in the empty upper-left band, above both histograms, instead of
    # gray text laid over the bars: medians, the retrieval shares, and the overlap key.
    ax1.text(0.852, ymax * 1.38,
             f"Poisoned, median {np.median(ac):.3f}", fontsize=8.5, color=PLANTED,
             ha="left", va="top")
    ax1.text(0.852, ymax * 1.22,
             f"Benign, median {np.median(gc):.3f}", fontsize=8.5, color=GENUINE,
             ha="left", va="top")
    ax1.text(0.852, ymax * 1.02, "Dark shading: overlap", fontsize=8.5,
             color="#4a4a4a", ha="left", va="top")
    ax1.set_xlim(0.85, 1.0)
    ax1.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax1.set_xlabel("Cosine similarity to query")
    ax1.set_ylabel("Density")
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)

    # ---- (b) deletion gain separates --------------------------------------
    ax2.axhline(0.0, color=CONTROL2, lw=0.7, zorder=1)
    ax2.scatter(ac, ae, s=2.5, color=PLANTED, alpha=0.30, lw=0, zorder=2,
                rasterized=False)
    ax2.scatter(gc, ge, s=2.5, color=GENUINE, alpha=0.35, lw=0, zorder=3,
                rasterized=False)
    ax2.axvline(TAU, color=CONTROL, lw=0.7, ls=":", zorder=1)

    med_ge, med_ae = np.median(ge), np.median(ae)
    ax2.annotate(f"Poisoned, median {med_ae:+.3f}", xy=(0.928, med_ae),
                 xytext=(0.862, 0.058), fontsize=8.5, color=PLANTED,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=PLANTED,
                                 shrinkA=1, shrinkB=1))
    ax2.annotate(f"Benign, median {med_ge:+.3f}", xy=(0.988, med_ge),
                 xytext=(0.900, -0.038), fontsize=8.5, color=GENUINE,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=GENUINE,
                                 shrinkA=1, shrinkB=1))
    ax2.text(0.8515, 0.003, "gain = 0", fontsize=8.5, color="#6a6a6a",
             ha="left", va="bottom")

    ax2.set_xlim(0.85, 1.0)
    ax2.set_ylim(-0.048, 0.070)
    # Pinned, not left to the autolocator: the canvas is sized so the labels print
    # large, which makes the axis physically small, and the autolocator answers that
    # by dropping to two ticks. The reader loses the 0.025 gridline that says how far
    # above zero the poisoned cloud sits.
    ax2.set_yticks([-0.025, 0.000, 0.025, 0.050])
    ax2.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax2.set_xlabel("Cosine similarity to query")
    ax2.set_ylabel("Deletion Gain")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("wrote", OUT)
    print(f"genuine n={len(gc)} cos_med={np.median(gc):.4f} exc_med={med_ge:+.4f}")
    print(f"attack  n={len(ac)} cos_med={np.median(ac):.4f} exc_med={med_ae:+.4f}")
    print(f"share above tau: genuine {(gc >= TAU).mean():.3f}, attack {(ac >= TAU).mean():.3f}")


if __name__ == "__main__":
    main()
