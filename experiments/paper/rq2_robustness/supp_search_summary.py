"""Table 2 Search row and Figure 3 left-panel counts from one search-attacker run.

Inputs per run: the attacker's report JSON and its `--dump-viable` JSONL, one line per
retrievable candidate (cos >= 0.90) with `dg`, `evades_dg`, `evades_joint`,
`poisoned_strict`, `poisoned_no_pushback`, `added_words` and `undefended_pick`.

Table 2 row, target level, denominator = every target (`n_targets`), as published
(`tab/adaptive_crossenc.tex` header, `rq6_search_42_joint.json`):

- None ASR: the target's undefended pick (its most retrievable candidate) is a poisoning.
- BR = 1 - evasion, evasion = the target has at least one retrievable candidate the rule
  serves. DG only serves `dg <= DG threshold`; Ours serves `evades_joint` (deployed rule).
- ASR: the target has at least one served candidate whose victim response is a poisoning.

Strict tier (`obeyed_strict`) is the published tier; the no-pushback tier is reported beside
it. Figure 3 left panel, candidate level: among the candidates a filter serves, strict
successes / candidates, overall and by added words (entry words - targeted question words;
buckets <=2, 3-6, 7-10, >10). Every dumped line counts once, duplicate texts included, as the
earlier figure counted. CIs are 95% percentile intervals from 2,000 target-cluster bootstrap
replicates (each target is one intent).

Every rate recomputed from the dump is checked against the attacker's own report, and the
DG-filter figure counts against its `by_payload_words`; a mismatch raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BUCKETS = ("<=2", "3-6", "7-10", ">10")
REDRAW_NOTE = ("Same requests re-sent with a fresh cache file. 'none' and 'dg_only' hold "
               "answer-independent candidate sets, so they measure victim re-draw drift. "
               "'ours' holds the set the deployed rule served under the OLD answers; the "
               "rule reads the answer (ADL, Echo), so a fresh answer that now echoes the "
               "payload would usually be blocked. Its fresh ASR is biased upward and is "
               "not a drift estimate.")
TIERS = {"strict": "poisoned_strict", "no_pushback": "poisoned_no_pushback"}


def bucket(added: int) -> str:
    """The payload-words band of `rq6_deletion_aware_attacker.main`."""
    return "<=2" if added <= 2 else ("3-6" if added <= 6 else ("7-10" if added <= 10 else ">10"))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dump(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def target_flags(dump, n_targets, served, hit) -> np.ndarray:
    """Per target: 1 if some candidate passes ``served`` and ``hit``; targets absent are 0."""
    out = np.zeros(n_targets, dtype=bool)
    for r in dump:
        if served(r) and hit(r):
            out[r["target"]] = True
    return out


def cluster_ci(num, den, *, reps=2000, seed=0) -> list[float]:
    """Percentile CI of sum(num)/sum(den) with targets resampled with replacement."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    d = den[idx].sum(1)
    ratio = np.where(d > 0, num[idx].sum(1) / np.where(d > 0, d, 1), np.nan)
    lo, hi = np.nanpercentile(ratio, [2.5, 97.5])
    return [float(lo), float(hi)]


def rate_cell(flags, *, complement=False) -> dict:
    count = int(flags.sum())
    n = len(flags)
    ci = cluster_ci(flags, np.ones(n))
    value = count / n
    if complement:
        return {"value": 1 - value, "evading_targets": count, "n_targets": n,
                "ci95": [1 - ci[1], 1 - ci[0]]}
    return {"value": value, "count": count, "n_targets": n, "ci95": ci}


def table_row(dump, n_targets, dg_threshold) -> dict:
    dg = lambda r: r["dg"] <= dg_threshold          # noqa: E731
    joint = lambda r: r["evades_joint"]              # noqa: E731
    pick = lambda r: r["undefended_pick"]            # noqa: E731
    row = {"dg_only_br": rate_cell(target_flags(dump, n_targets, dg, lambda r: True),
                                   complement=True),
           "ours_br": rate_cell(target_flags(dump, n_targets, joint, lambda r: True),
                                complement=True)}
    for tier, field in TIERS.items():
        hit = lambda r, f=field: r[f]                # noqa: E731
        row[tier] = {"none_asr": rate_cell(target_flags(dump, n_targets, pick, hit)),
                     "dg_only_asr": rate_cell(target_flags(dump, n_targets, dg, hit)),
                     "ours_asr": rate_cell(target_flags(dump, n_targets, joint, hit))}
    return row


def figure_counts(dump, n_targets, served) -> dict:
    """Candidate-level strict / no-pushback shares among the candidates ``served`` passes."""
    kept = [r for r in dump if served(r)]

    def block(rows):
        strict = sum(r["poisoned_strict"] for r in rows)
        loose = sum(r["poisoned_no_pushback"] for r in rows)
        per_t_num = np.zeros(n_targets)
        per_t_den = np.zeros(n_targets)
        for r in rows:
            per_t_den[r["target"]] += 1
            per_t_num[r["target"]] += r["poisoned_strict"]
        return {"n_candidates": len(rows), "strict": int(strict), "no_pushback": int(loose),
                "strict_share": strict / len(rows) if rows else None,
                "no_pushback_share": loose / len(rows) if rows else None,
                "strict_share_ci95": cluster_ci(per_t_num, per_t_den) if rows else None,
                "n_distinct_texts": len({r["text"] for r in rows}),
                "n_targets_contributing": len({r["target"] for r in rows}),
                "n_targets_with_strict_success": len({r["target"] for r in rows
                                                      if r["poisoned_strict"]})}

    out = {"overall": block(kept)}
    out["by_added_words"] = {b: block([r for r in kept if bucket(r["added_words"]) == b])
                             for b in BUCKETS}
    return out


def crosscheck(dump, report, n_targets, dg_threshold, figure_dg) -> dict:
    """Rates recomputed from the dump against the attacker's report; raise on mismatch."""
    row = table_row(dump, n_targets, dg_threshold)
    pairs = {
        "collide_and_evade": 1 - row["dg_only_br"]["value"],
        "and_poison_strict": row["strict"]["dg_only_asr"]["value"],
        "and_poison_no_pushback": row["no_pushback"]["dg_only_asr"]["value"],
        "undefended_poison_strict": row["strict"]["none_asr"]["value"],
        "undefended_poison_no_pushback": row["no_pushback"]["none_asr"]["value"],
        "collide_and_evade_joint": 1 - row["ours_br"]["value"],
        "and_poison_strict_joint": row["strict"]["ours_asr"]["value"],
        "and_poison_no_pushback_joint": row["no_pushback"]["ours_asr"]["value"],
    }
    for key, got in pairs.items():
        assert abs(got - report[key]) < 1e-12, f"{key}: dump {got} vs report {report[key]}"
    for b in BUCKETS:
        mine, theirs = figure_dg["by_added_words"][b], report["by_payload_words"].get(b)
        if theirs is None:
            assert mine["n_candidates"] == 0, b
            continue
        assert (mine["n_candidates"], mine["strict"], mine["no_pushback"]) == \
            (theirs["n"], theirs["strict"], theirs["poisoned"]), b
    assert figure_dg["overall"]["n_candidates"] == report["surviving_candidates_tested"]
    assert report["answer_check"]["n_viable_rows"] == len(dump)
    return {"report_rates_reproduced": sorted(pairs), "by_payload_words_reproduced": True,
            "n_viable_rows": len(dump)}


def summarize(report_path, dump_path) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    dump = load_dump(dump_path)
    n = report["n_targets"]
    dg_threshold = report["fence"]
    ac = report["answer_check"]
    dg_filter = lambda r: r["dg"] <= dg_threshold    # noqa: E731
    for r in dump:
        assert r["evades_dg"] == dg_filter(r)
    figure = {"deployed_joint_rule": figure_counts(dump, n, lambda r: r["evades_joint"]),
              "dg_only_filter": figure_counts(dump, n, dg_filter)}
    return {
        "source": {"report": str(report_path), "report_sha256": sha256(report_path),
                   "dump": str(dump_path), "dump_sha256": sha256(dump_path)},
        "settings": {"policy": report["policy"], "n_targets": n,
                     "candidates_per_round": report["candidates_per_round"],
                     "rounds": report["rounds"], "retrieval_floor": report["retrieval_floor"],
                     "dg_only_threshold": dg_threshold, "eta": ac["eta"],
                     "eta_a": ac["eta_a"], "echo_min": ac["echo_min"], "rule": ac["rule"],
                     "fence_form": ac["fence_form"], "storage_dtype": ac["storage_dtype"],
                     "n_victim_texts": ac["n_victim_texts"],
                     "n_rows_without_answer": ac["n_rows_without_answer"]},
        "table2_search_row": table_row(dump, n, dg_threshold),
        "figure3_left": figure,
        "crosscheck": crosscheck(dump, report, n, dg_threshold, figure["dg_only_filter"]),
    }


def published_figure(path) -> dict:
    """The counts the current Figure 3 left panel draws (6+2 run, DG-only filter)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    bp = d["by_payload_words"]
    return {"source": str(path), "policy": d["policy"], "dg_only_threshold": d["fence"],
            "n_candidates": sum(c["n"] for c in bp.values()),
            "strict": sum(c["strict"] for c in bp.values()),
            "no_pushback": sum(c["poisoned"] for c in bp.values()),
            "n_targets_contributing": round(d["collide_and_evade"] * d["n_targets"]),
            "by_added_words": {b: {"n_candidates": bp[b]["n"], "strict": bp[b]["strict"],
                                   "no_pushback": bp[b]["poisoned"]} for b in BUCKETS}}


def attacker_prompt(report, *, historical_mechanism=None) -> dict:
    """The sentences the attacker read. Reports written before the attacker recorded them
    get them rebuilt: the policy's mechanism sentence (or ``historical_mechanism``) and the
    DG-only threshold line those runs used."""
    if "attacker_prompt" in report:
        return report["attacker_prompt"]
    from experiments.paper.rq2_robustness import rq6_deletion_aware_attacker as attacker
    from sentry.cache.defense.calibrate import parse_policy
    verdict, rule = attacker.attacker_rule(report["fence"])
    mechanism = historical_mechanism or attacker.mechanism_text(parse_policy(report["policy"]))
    return {"draw": 0, "mechanism": mechanism, "round1_verdict": verdict,
            "round2_rule": rule, "rebuilt": True}


def mean_sd(values) -> dict:
    v = np.asarray(values, float)
    return {"per_draw": [float(x) for x in v], "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else None}


def across_draws(runs) -> dict:
    """Table 2 cells per draw, with mean and sample sd (ddof=1) across draws."""
    rows = [r["table2_search_row"] for r in runs]
    out = {"n_draws": len(runs),
           "dg_only_br": mean_sd([t["dg_only_br"]["value"] for t in rows]),
           "ours_br": mean_sd([t["ours_br"]["value"] for t in rows])}
    for tier in TIERS:
        out[tier] = {cell: mean_sd([t[tier][cell]["value"] for t in rows])
                     for cell in ("none_asr", "dg_only_asr", "ours_asr")}
    return out


def pooled_figure(dumps, n_targets, thresholds) -> dict:
    """Figure counts pooled over draws; targets are the same 200 in every draw."""
    rows = [dict(r, draw=d) for d, dump in enumerate(dumps) for r in dump]
    out = {}
    for name, served in (("deployed_joint_rule", lambda r: r["evades_joint"]),
                         ("dg_only_filter", lambda r: r["dg"] <= thresholds[r["draw"]])):
        pooled = figure_counts(rows, n_targets, served)
        per_draw = [figure_counts([r for r in rows if r["draw"] == d], n_targets, served)
                    for d in range(len(dumps))]
        for key in ("overall",):
            pooled[key]["per_draw"] = [
                {k: pd[key][k] for k in ("strict", "no_pushback", "n_candidates",
                                         "n_targets_contributing")} for pd in per_draw]
        for b in BUCKETS:
            pooled["by_added_words"][b]["per_draw"] = [
                {k: pd["by_added_words"][b][k] for k in ("strict", "no_pushback",
                                                         "n_candidates")} for pd in per_draw]
        pooled["note"] = ("n_targets_contributing counts distinct targets over all draws; "
                          "per_draw gives each draw's own count")
        out[name] = pooled
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", nargs=2, action="append", metavar=("REPORT", "DUMP"),
                        default=[], help="one rule-informed attacker draw; repeat per draw")
    parser.add_argument("--single-report", required=True,
                        help="the first 4+2 run (one draw, DG-only threshold in feedback)")
    parser.add_argument("--single-dump", required=True)
    parser.add_argument("--old-report", required=True)
    parser.add_argument("--old-dump", required=True)
    parser.add_argument("--published-figure", required=True,
                        help="the JSON the current Figure 3 left panel reads "
                             "(results/v3/adaptive/rq6_search_v3.json)")
    parser.add_argument("--victim-redraw", default="",
                        help="supp_search_victim_drift.py output for the previous run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from experiments.paper.rq2_robustness.supp_search_replay import OLD_MECHANISM

    runs, dumps, thresholds = [], [], []
    for report_path, dump_path in args.draw:
        run = summarize(report_path, dump_path)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        run["attacker_prompt"] = attacker_prompt(report)
        runs.append(run)
        dumps.append(load_dump(dump_path))
        thresholds.append(report["fence"])
    single = summarize(args.single_report, args.single_dump)
    single["attacker_prompt"] = attacker_prompt(
        json.loads(Path(args.single_report).read_text(encoding="utf-8")))
    old = summarize(args.old_report, args.old_dump)
    old["attacker_prompt"] = attacker_prompt(
        json.loads(Path(args.old_report).read_text(encoding="utf-8")),
        historical_mechanism=OLD_MECHANISM)
    out = {}
    if runs:
        n = {r["settings"]["n_targets"] for r in runs}
        assert len(n) == 1, n
        out["rule_informed_draws"] = {
            "definition": ("independent attacker draws (new seeds and attacker cache per "
                           "draw), attacker told the 4+2 cut and the deployed rule; victim "
                           "temperature 0; same thresholds as the published row"),
            "across_draws": across_draws(runs),
            "figure3_left_pooled": pooled_figure(dumps, n.pop(), thresholds),
            "draws": runs}
    out.update({
        "single_draw_dg_only_feedback": single,
        "previous_joint_run_replayed": old,
        "published_table2_search_row": {"none_asr": 0.120, "dg_only_br": 0.815,
                                        "dg_only_asr": 0.025, "ours_br": 0.390,
                                        "ours_asr": 0.010},
        "published_figure3_left": published_figure(args.published_figure)})
    if args.victim_redraw:
        out["victim_redraw_previous_run"] = {
            **json.loads(Path(args.victim_redraw).read_text(encoding="utf-8")),
            "note": REDRAW_NOTE}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    named = [(f"draw{i + 1}", r) for i, r in enumerate(runs)]
    for name, run in named + [("single", single), ("old", old)]:
        t = run["table2_search_row"]
        print(f"{name}: None {t['strict']['none_asr']['value']:.3f} | DG BR "
              f"{t['dg_only_br']['value']:.3f} ASR {t['strict']['dg_only_asr']['value']:.3f} "
              f"| Ours BR {t['ours_br']['value']:.3f} ASR {t['strict']['ours_asr']['value']:.3f}")
        o = run["figure3_left"]["deployed_joint_rule"]["overall"]
        print(f"   deployed rule: {o['strict']}/{o['n_candidates']} strict, "
              f"{o['no_pushback']} no-pushback, {o['n_targets_contributing']} targets")
    if runs:
        a = out["rule_informed_draws"]["across_draws"]
        print("mean±sd:", {k: f"{v['mean']:.3f}±{v['sd']:.3f}" for k, v in
                           [("dg_only_br", a["dg_only_br"]), ("ours_br", a["ours_br"])]
                           + [(f"{c}", a["strict"][c]) for c in a["strict"]]})
        o = out["rule_informed_draws"]["figure3_left_pooled"]["deployed_joint_rule"]["overall"]
        print(f"pooled deployed rule: {o['strict']}/{o['n_candidates']} strict, "
              f"{o['n_targets_contributing']} distinct targets")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
