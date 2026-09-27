"""F6 -- the three RQ3/RQ4 evidence panels in one figure.

Replaces the old fig4_mechanism.pdf (its panel (b) survives as panel (b) here),
the inline tikz granularity plot that lived in tab/granularity.tex,
and fig5_baselines.pdf (now panels (a) and (b)). Every literal traces to a v3 JSON under
experiments/paper/results/v3/ (override the location with $V3_RESULTS).

The granularity panel that used to sit here is superseded by fig7_tradeoff.pdf,
which sweeps 32 cuts on all three attack classes instead of three cuts on one.

(a) Evasion vs effectiveness of the segmentation-aware search: the retrieved candidates,
    split into those the deployed 4+2 rule (Ours) blocks and those it accepts. Bar length is
    the share of retrieved candidates; hatching marks candidates whose response is a strict
    poisoning. Overall and by attacker round (round 1: first candidates; round 2: rewrites
    after the attacker sees the DG scores of its round-1 candidates), pooled over the three
    rule-informed attacker runs.
    Source: results/supp_appendix_20260923/search_figure3_rounds.json
      -> pooled.{overall,round1,round2}.{retrieved,accepted,blocked}.{n_candidates,strict}
      (rq2_robustness/supp_search_figure3.py); the accepted totals are checked
      against search_4p2.json -> rule_informed_draws.figure3_left_pooled.deployed_joint_rule.
(b) Detection at deployable cost: block@5% vs per-hit latency (log x), one
    point per attack class (CAP, SCP, KCA).
    Sources:
      DG + Answer check BR: results/answer_check_20260909/main_table.json
        -> cells["e5-small-v2|{set}"].answer_rules.either.excess_block_rate_joint_eta
      joint latency: results/answer_check_20260916/serving_microbenchmark.json
        -> sets.{LMP,SCP,KCA}.answer_checked.ms_median
        (single-thread cpu-server CPU microbenchmark; baselines retain their source hardware)
      cosine block (reference lines, no fabricated latency point -- it is the
        score the cache already computes for retrieval):
        detection/e5-small-v2/{lmp,scp,kca}.json -> cosine_block_rate
      cond. perplexity: detection/baselines/{lmp,scp,kca}_pplnli.json
        -> baselines_ppl_nli.perplexity.{block_at_budget,ms_per_row}
      multi-encoder agreement: data/perrow/multiview_*_e5.json + ci_table1.json
      LLM judge: data/perrow/judge_vote_{cap,scp,kca}.json (10 votes of the
        reported prompt at temperature 1) + ci_table1.json 'LLM judge (vote)'
        block@5%; latency = votes x median uncached call (~4.4 s)
      erase-and-check: ci_table1.json 'erase-and-check' block@5%; latency
        = one batched classifier pass over the copies (eac_cap_timing.json)
      LaCache k=20: detection/lacache_k20.json
        -> sets.{LMP,SCP,KCA}.encoders.e5-small-v2.all_planted.POOLED
           .block_at_5pct; latency 108.5 ms is the figure ITS OWN paper
           reports (arXiv 2608.01718 Table 4 / appendix E, k=20 forced
           decoding) -- not same-hardware with our CPU costs.
"""
from experiments.paper.paths import paper_data_root, figure_output_root, RESULTS_ROOT
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from experiments.paper.figures.style import apply, OURS, BASELINE, CONTROL, CONTROL2, INK, FULL_W

V3 = os.environ.get(
    "V3_RESULTS",
    os.path.join(paper_data_root(), "figures", "v3"))
OUT = os.path.join(figure_output_root(), "fig6_panels.pdf")

# LaCache's own reported per-query latency (arXiv 2608.01718 Table 4 /
# appendix E, k=20 forced decoding). Their hardware, not ours.
LACACHE_REPORTED_MS = 108.5

def load(rel):
    with open(os.path.join(V3, rel)) as f:
        return json.load(f)


def main():
    apply()

    # Preserve the approved right panel's physical geometry. Allocate the added
    # width to the left panel so automatic layout cannot stretch the cost plot.
    extra_width = 0.65
    width, height = FULL_W + extra_width, 2.0
    fig = plt.figure(figsize=(width, height))
    def axes_inches(left, bottom, span, rise):
        return fig.add_axes([left / width, bottom / height, span / width, rise / height])
    ax2 = axes_inches(0.734726, 0.404892, 2.130202 + extra_width, 1.393993)
    ax3 = axes_inches(3.328128 + extra_width, 0.404892, 2.130202, 1.393993)

    # ---- (a) evasion vs effectiveness: blocked / accepted, hatched = poisoned ----
    from matplotlib.patches import Patch
    with open(RESULTS_ROOT / "supp_appendix_20260923/search_figure3_rounds.json") as f:
        rounds = json.load(f)["pooled"]
    with open(RESULTS_ROOT / "supp_appendix_20260923/search_4p2.json") as f:
        check = json.load(f)["rule_informed_draws"]["figure3_left_pooled"][
            "deployed_joint_rule"]["overall"]
    assert (rounds["overall"]["accepted"]["n_candidates"], rounds["overall"]["accepted"]["strict"]) \
        == (check["n_candidates"], check["strict"])
    groups = ["overall", "round1", "round2"]
    labels = ["Overall", "Round 1", "Round 2"]
    for g in groups:
        cell = rounds[g]
        for k in ("n_candidates", "strict"):
            assert cell["retrieved"][k] == cell["accepted"][k] + cell["blocked"][k]
    positions = [2.3, 1, 0]
    offset, bar_h = 0.2, 0.36
    light_ours = "#f2b98f"
    series = [("blocked", "Blocked", CONTROL2, +offset),
              ("accepted", "Accepted", light_ours, -offset)]
    for key, name, color, dy in series:
        for index, (g, y) in enumerate(zip(groups, positions)):
            cell, total = rounds[g][key], rounds[g]["retrieved"]["n_candidates"]
            share = 100 * cell["n_candidates"] / total
            poisoned = 100 * cell["strict"] / total
            ax2.barh(y + dy, share, height=bar_h, color=color, zorder=2,
                     edgecolor="none")
            ax2.barh(y + dy, poisoned, height=bar_h, color="none", hatch="//////",
                     edgecolor=INK, linewidth=0, zorder=3)
            # label: the poisoned share of this bar, placed after the hatched part
            # (blocked) or after the bar (accepted); one decimal from 10% up, as in the text
            rate = 100 * cell["strict"] / cell["n_candidates"]
            rate_text = f"{rate:.1f}%" if rate >= 10 else f"{rate:.2f}%"
            text = rate_text
            x = (poisoned if key == "blocked" else share) + 1.5
            ax2.text(x, y + dy, text, ha="left", va="center", fontsize=7.5,
                     fontweight="bold" if index == 0 else "normal", color=INK, zorder=4)
    ax2.axhline(1.65, color=CONTROL2, lw=0.6, zorder=1)
    ax2.set_yticks(positions, labels)
    ax2.get_yticklabels()[0].set_fontweight("bold")
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_color(CONTROL2)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, 3.35)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("Share of retrieved candidates (%)", labelpad=2)
    handles = [Patch(facecolor=CONTROL2, edgecolor="none", label="Blocked by Ours"),
               Patch(facecolor=light_ours, edgecolor="none", label="Accepted by Ours"),
               Patch(facecolor="white", edgecolor=INK, hatch="//////", linewidth=0.4,
                     label="Poisoned")]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=3,
               fontsize=7.5, frameon=False, handlelength=1, handletextpad=0.4,
               columnspacing=0.9, borderaxespad=0)

    # ---- (c) block@5% vs per-hit latency, one point per attack class ------
    sets = ["lmp", "scp", "kca"]
    det = {s: load(f"detection/e5-small-v2/{s}.json") for s in sets}
    ppl = {s: load(f"detection/baselines/{s}_pplnli.json")["baselines_ppl_nli"]
           for s in sets}
    lac = load("detection/lacache_k20.json")["sets"]
    # SCP and KCA LaCache rescored on the 798 rows the other rows use (2026-09-05).
    lac_798 = {"SCP": json.load(open(os.path.join(paper_data_root(), "perrow", "scp_lacache_k20_798.json"))),
               "KCA": json.load(open(os.path.join(paper_data_root(), "perrow", "kca_lacache_k20_full.json")))}
    with open(RESULTS_ROOT / "answer_check_20260909/main_table.json") as f:
        joint_detection = json.load(f)["cells"]
    with open(RESULTS_ROOT / "answer_check_20260916/serving_microbenchmark.json") as f:
        joint_cost = json.load(f)["sets"]

    # LLM judge (2026-09-05): placed at the 5% budget by voting (data/perrow/judge_vote.py,
    # 10 samples of the reported prompt at temperature 1; block from ci_table1.json
    # 'LLM judge (vote)'); ms/hit = 10 calls x the median uncached call latency of that run.
    perrow = os.path.join(paper_data_root(), "perrow")
    jv = {s: json.load(open(os.path.join(perrow, f"judge_vote_{s if s != 'lmp' else 'cap'}.json")))
          for s in sets}
    judge_ms = {s: jv[s]["votes"] * jv[s]["ms_per_call_median"] for s in sets}

    # (label, [(latency ms, block@5%) per set], colour, marker)
    # erase-and-check as released (2026-09-05, data/perrow/eac_original.py): suffix mode,
    # d=20, the authors' DistilBERT classifier on the entry; block@5% of the continuous read
    # max_i P(harmful) from ci_table1.json 'erase-and-check'; ms/hit = one batched classifier
    # pass over the <=21 copies (eac_{set}.json ms_per_hit_median, 4 torch threads).
    # ms/hit from the idle-box re-timing on the CAP arm (eac_cap_timing.json), the Table 1
    # convention (CAP-arm measured cost); the per-set eac_*.json timings were taken under load.
    eac_ms = json.load(open(os.path.join(perrow, "eac_cap_timing.json")))["ms_per_hit_median"]
    eac = {s: {"ms_per_hit_median": eac_ms} for s in sets}
    ci = json.load(open(os.path.join(paper_data_root(), "perrow", "ci_table1.json")))
    ec_block = {s: ci[S]["methods"]["erase-and-check"]["att"][0]
                for s, S in zip(sets, ["CAP", "SCP", "KCA"])}
    # multi-encoder agreement (replaces NLI, 2026-09-04): -min cosine over e5/MiniLM/bge,
    # block from ci_table1.json 'multi-encoder min-cos'; ms/hit = two auxiliary query
    # encodes, unbatched (data/perrow/multiview_{set}_e5.json ms_per_hit_median).
    mv = {s: json.load(open(os.path.join(paper_data_root(), "perrow", f"multiview_{s if s != 'lmp' else 'cap'}_e5.json")))
          for s in sets}
    mv_block = {s: ci[S]["methods"]["multi-encoder min-cos"]["att"][0]
                for s, S in zip(sets, ["CAP", "SCP", "KCA"])}

    methods = [
        ("Conditional perplexity", "Perplexity",
         [(ppl[s]["perplexity"]["ms_per_row"],
           ppl[s]["perplexity"]["block_at_budget"]) for s in sets],
         BASELINE, "s"),
        ("Multi-embedding agreement", "Multi-embedding",
         [(mv[s]["ms_per_hit_median"], mv_block[s]) for s in sets],
         BASELINE, "^"),
        ("LaCache (reported latency)", "LaCache",
         [(LACACHE_REPORTED_MS,
           lac_798[S]["all_planted"]["lacache_main"]["POOLED"]["block"] if S in lac_798 else
           lac[S]["encoders"]["e5-small-v2"]["all_planted"]["POOLED"]["block_at_5pct"])
          for S in ["LMP", "SCP", "KCA"]],
         BASELINE, "D"),
        ("LLM judge", "LLM judge",
         [(judge_ms[s], ci[S]["methods"]["LLM judge (vote)"]["att"][0])
          for s, S in zip(sets, ["CAP", "SCP", "KCA"])],
         BASELINE, "v"),
        ("Erase-and-check", "Erase-and-check",
         [(eac[s]["ms_per_hit_median"], ec_block[s]) for s in sets],
         BASELINE, "o"),
        ("Ours", "Ours",
         [(joint_cost[S]["answer_checked"]["ms_median"],
           joint_detection[f"e5-small-v2|{s}"]["answer_rules"]["either"]
               ["excess_block_rate_joint_eta"])
          for s, S in zip(sets, ["LMP", "SCP", "KCA"])],
         OURS, "*"),
    ]

    # cosine: no separately metered added cost -- it is the score the cache already
    # computes for retrieval. One band across the three classes, never a point.
    cos_blocks = [det[s]["cosine_block_rate"] for s in sets]
    cos_mean = sum(cos_blocks) / len(cos_blocks)
    ax3.axhline(cos_mean, color=CONTROL2, lw=1.0, ls=":", zorder=1)

    # One marker per method, averaged over the three attack classes. Plotting a point
    # per class put fifteen dots here and stacked the judge's and LaCache's classes at
    # an identical x, which read as scatter noise; range stems were no clearer. The
    # panel's claim is about cost, which spans four orders of magnitude, so a per-class
    # spread of a few points adds nothing it can show. Per-class values are Table 2.
    # Five points do not earn a legend: inside the axes it covered the markers, below
    # them it left a band of white under the other two panels. Each point carries its
    # own short name instead; the caption expands them.
    # NLI sits left of its marker and the judge sits above its own: both above and the
    # judge's label ran back into NLI's, since the two markers are close in y.
    offsets = {"Perplexity": (8, -5, "left"),
               "Multi-embedding": (-5, -26, "right"),
               "LaCache": (6, 9, "left"), "LLM judge": (-2, 12, "right"),
               "Erase-and-check": (-6, 17, "right"),
               "Ours": (6, 7, "left")}
    for label, short, pts, color, marker in methods:
        ours = color == OURS
        x = sum(q[0] for q in pts) / len(pts)
        y = sum(q[1] for q in pts) / len(pts)
        ax3.scatter([x], [y], s=95 if ours else 34, color=color, zorder=3,
                    marker=marker, edgecolor="white", linewidth=0.5, label=label)
        dx, dy, ha = offsets[short]
        ax3.annotate(short, (x, y), textcoords="offset points", xytext=(dx, dy),
                     ha=ha, va="center", fontsize=8.5, color=color,
                     arrowprops=dict(arrowstyle="-",color=color,lw=.5),
                     fontweight="bold" if ours else "normal", zorder=4)

    ax3.set_xscale("log")
    ax3.set_xlim(0.006, 2.2e4)
    ax3.set_ylim(-0.10, 1.18)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_xticks([1e-1, 1e1, 1e3])
    ax3.set_xlabel("Added latency (ms, log scale)", labelpad=1)
    ax3.set_ylabel("Mean BR at 5% FPR", labelpad=2)
    ax3.annotate("Cosine (no added cost)", (ax3.get_xlim()[0], cos_mean),
                 textcoords="offset points", xytext=(4, 3), ha="left",
                 va="bottom", fontsize=8.5, color=CONTROL)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)
    for g in groups:
        print(g, {k: f"{rounds[g][k]['strict']}/{rounds[g][k]['n_candidates']}"
                  for k in ("retrieved", "blocked", "accepted")})
    print("judge_ms", judge_ms)
    for label, _short, pts, *_ in methods:
        print(label, [(round(x, 3), round(y, 3)) for x, y in pts])
    print("cosine blocks", [round(c, 3) for c in cos_blocks])


if __name__ == "__main__":
    main()
