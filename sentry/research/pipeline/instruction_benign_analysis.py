"""Held-out analysis for the instruction stress test (no model calls).

Two layers sit here. The first is the Deletion-Gain-only analysis the stress test
always had: a threshold fitted on bare benign calibration hits, then false-veto rates
per pairing condition with intent-cluster intervals, matched AUROCs, and a cosine-only
baseline at the same budget. The second, added by the answer-checked study, reports the
same populations under every answer rule -- ``dg_only``, ``adl``, ``echo``, ``either``
and ``cosine_only`` -- so that what the answer check buys (false vetoes it removes) and
what it costs (attacks it lets through) are read off one table.

Nothing here is fitted on a test intent or on an attack. ``eta`` and ``eta_a`` both come
from the **bare benign calibration hits** of the same corpus and policy, at the same
budget. Every rate is retrieval-conditioned and states its denominator: a retrieval miss
is not a defense success. An unjudgeable entry, and an entry whose answer fields cannot
be read, is vetoed by every rule -- fail-closed, never dropped.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from .instruction_benign import (TEMPLATES, _jsonl_files, cluster_rate, fit_thresholds,
                                 operating_rates, read_jsonl, write_jsonl)


def auc(positive, negative):
    if len(positive) < 2 or len(negative) < 2:
        return None
    n = np.sort(negative)
    return float(np.mean((np.searchsorted(n, positive, side="left") +
                          np.searchsorted(n, positive, side="right")) / (2*len(n))))


def matched_discrimination(attack, benign, field, width, min_intents):
    abins, bbins = defaultdict(list), defaultdict(list)
    for rows, bins in ((attack, abins), (benign, bbins)):
        for r in rows:
            bins[int(np.floor(r[field]/width))].append(r)
    cells, aa, bb = [], [], []
    for key in sorted(abins.keys() & bbins.keys()):
        a, b = abins[key], bbins[key]
        if min(len({r["intent_id"] for r in a}), len({r["intent_id"] for r in b})) < min_intents:
            continue
        cells.append({"lower": key*width, "n_attack": len(a), "n_benign": len(b),
                      "dg_auroc": auc([r["dg"] for r in a], [r["dg"] for r in b]),
                      "cosine_auroc": auc([-r["base_cos"] for r in a], [-r["base_cos"] for r in b])})
        aa.extend(a)
        bb.extend(b)
    def pooled(stat):
        return float(np.average([c[stat] for c in cells], weights=[c["n_attack"] for c in cells])) if cells else None
    return {"dg_auroc": pooled("dg_auroc"), "cosine_auroc": pooled("cosine_auroc"),
            "attack_rows_supported": len(aa), "attack_rows_total": len(attack),
            "attack_intents_supported": len({r["intent_id"] for r in aa}),
            "benign_rows_supported": len(bb), "benign_rows_total": len(benign),
            "benign_intents_supported": len({r["intent_id"] for r in bb}),
            "min_distinct_intents_per_arm_per_bin": min_intents, "bins": cells}


def discrimination(attack, benign, cfg):
    a, b = [r for r in attack if r["judgeable"]], [r for r in benign if r["judgeable"]]
    return {"n_attack": len(a), "n_benign": len(b),
            "dg_auroc": auc([r["dg"] for r in a], [r["dg"] for r in b]),
            "cosine_auroc": auc([-r["base_cos"] for r in a], [-r["base_cos"] for r in b]),
            "cosine_matched": matched_discrimination(a, b, "base_cos", cfg.cosine_bin_width, cfg.min_bin_intents),
            "word_matched": matched_discrimination(a, b, "words", cfg.word_bin_width, cfg.min_bin_intents)}


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    index = np.searchsorted(np.cumsum(weights)/np.sum(weights), q, side="left")
    return float(values[min(index, len(values)-1)])


def recalibration_thresholds(rows, cfg, constraint_condition):
    """A prespecified, equal-weight three-stratum traffic mixture; no attack scores.

    Calibrate on two template families only; other template families remain unseen.
    The broad mixture is a harmless-traffic policy, not a claim of safe answer reuse.
    """
    cal = [r for r in rows if r["split"] == "calibration" and r["judgeable"]
           and r["retrieval_cos"] >= cfg.cache_threshold]
    strata = [[r for r in cal if r["condition"] == "bare"],
              [r for r in cal if r["template"] == "polite_tell" and r["condition"] == "entry_only"],
              [r for r in cal if r["template"] == "one_sentence" and r["condition"] == constraint_condition]]
    if not all(strata):
        return None
    weights = [1/(3*len(s)) for s in strata for _ in s]
    flat = sum(strata, [])
    return {"dg": weighted_quantile([r["dg"] for r in flat], weights, 1-cfg.fpr_budget),
            "neg_cos": weighted_quantile([-r["base_cos"] for r in flat], weights, 1-cfg.fpr_budget),
            "n": len(flat), "stratum_counts": [len(s) for s in strata],
            "stratum_weights": [1/3]*3, "seen_templates": ["polite_tell", "one_sentence"],
            "constraint_condition": constraint_condition}


def paired_effect(rows, bare, thresholds, cfg):
    pairs = [(r, bare[r["intent_id"]]) for r in rows if r["intent_id"] in bare
             and r["retrieval_cos"] >= cfg.cache_threshold
             and bare[r["intent_id"]]["retrieval_cos"] >= cfg.cache_threshold]
    if not pairs:
        return None
    groups = [r["intent_id"] for r, _ in pairs]
    def blocked(r):
        return int(not r["judgeable"] or r["dg"] > thresholds["dg"])
    diff = cluster_rate([blocked(r)-blocked(b) for r,b in pairs], groups,
                        cfg.bootstrap_iterations, cfg.seed)
    judgeable = [(r,b) for r,b in pairs if r["judgeable"] and b["judgeable"]]
    return {"common_hit_rows": len(pairs), "dg_veto_difference": diff,
            "mean_dg_difference": float(np.mean([r["dg"]-b["dg"] for r,b in judgeable])) if judgeable else None,
            "mean_cosine_difference": float(np.mean([r["base_cos"]-b["base_cos"] for r,b in pairs]))}


# --- the answer rules ----------------------------------------------------------

#: Every rule reported side by side. ``dg_only`` is what shipped before the answer
#: check; ``cosine_only`` is the project's mandatory baseline -- the cache already
#: thresholds on similarity, so a statistic that does not beat similarity at the same
#: budget has added nothing. The middle three are the answer rules the fence can run
#: (``sentry.cache.defense.deletion.ANSWER_RULES`` minus ``"none"``, which is
#: ``dg_only`` under another name).
RULES = ("dg_only", "adl", "echo", "either", "cosine_only")

#: The five pairing conditions of the stress test. Crossed with the two template kinds
#: (polite, constraint) plus the unwrapped ``bare`` row, they are the ten-condition table.
PAIRINGS = ("entry_only", "query_only", "both_same", "both_paraphrase", "exact_core")

#: The answer columns an ablation may read in place of ``adl_best``.
ADL_FIELDS = ("adl_best", "adl_best_first20", "adl_best_first_sentence", "adl_delta_best")

#: Rough grouping of a row's ``set`` for the seen / unseen / new-question reading.
#: Unknown set names fall through to ``"other"`` rather than being forced into a bucket.
SET_GROUPS = {"instruction_benign": "seen_wrappers", "wrapped_benign": "seen_wrappers",
              "unseen_wrappers": "unseen_wrappers", "unseen_wrapper": "unseen_wrappers",
              "fresh_benign": "new_questions", "composition_native": "new_questions",
              "composition_wrapped": "new_questions",
              "non_echo_attack": "attacks", "surface_form_attack": "attacks"}

#: Jaccard cuts at which an "unseen" instruction is called strictly unseen (ruling 10:
#: the frozen twelve overlap the seen templates, so the subset is reported at two cuts
#: rather than at one hand-picked one). Fixed here, never chosen from a score.
UNSEEN_COLLISION_CUTS = (0.5, 0.25)

#: Every instruction the built templates ever used, both wordings.
SEEN_INSTRUCTIONS = tuple(text for _, _, primary, paraphrase in TEMPLATES
                          for text in (primary, paraphrase))


#: What every ``veto_given_hit`` in this file is a rate over. Stored once per cell
#: rather than on each of the thousands of entries that share it.
DENOMINATOR_NOTE = ("veto_given_hit is over retrieval hits (retrieval_cos >= "
                    "cache_threshold); veto_all_pairs is over every constructed row. A "
                    "retrieval miss is never counted as a defense success.")


#: Metadata columns the built samples always carry and an extra set may not. The scored
#: file now mixes both -- ``unseen_wrappers``, the projection cohorts, ``surface_form_attack``,
#: the rq5 sets and ``non_echo_attack`` all arrive with their own subset of columns -- so
#: the analysis fills the gaps once, up front, instead of subscripting into a KeyError two
#: hundred thousand rows in. A filled column groups the row under a visible "" bucket.
ROW_DEFAULTS = {"corpus": "", "policy": "", "split": "", "condition": "", "template": "",
                "position": "", "kind": "", "set": "", "record_id": "", "malicious": False}


def normalise_rows(rows, path=None):
    """Fill the optional metadata columns; refuse a row that has no intent id.

    ``intent_id`` is the one column with no safe default: every split, every group and
    every bootstrap resamples over it, so defaulting it to ``""`` would merge unrelated
    rows into a single cluster and narrow every interval in the file. Everything else
    defaults to empty and stays countable. The counts come back so the run can say which
    columns it had to fill and how often.
    """
    filled = Counter()
    for index, row in enumerate(rows):
        if not row.get("intent_id"):
            raise ValueError(f"{path or 'scores'} row {index}: no intent_id, which is the "
                             "unit every split and every bootstrap groups by")
        for key, default in ROW_DEFAULTS.items():
            if row.get(key) is None:
                row[key] = default
                filled[key] += 1
    return rows, dict(filled)


def dg_fires(row, eta):
    """Does Deletion Gain veto this row? An unjudgeable entry always does (fail-closed).

    Written with the unjudgeable test first because ``dg`` is ``None`` on exactly those
    rows and ``None > eta`` raises. The outcome is the brief's rule unchanged.
    """
    return bool(not row["judgeable"] or row["dg"] > eta)


def veto(row, eta, eta_a, rule, echo_min=1, adl_field="adl_best"):
    """The rule under test, applied to one scored row.

    ``dg_only`` is Deletion Gain alone. The three answer rules only ever *withdraw* a
    veto Deletion Gain already raised -- they never raise one of their own -- and they
    withdraw nothing when the answer cannot be read: no answer, an unreadable answer, or
    a missing ablation column all veto. ``adl_field`` names the answer-loss column, so an
    ablation reads a truncated or delta loss through the same code the headline uses.
    """
    fires = dg_fires(row, eta)
    if not fires or rule == "dg_only":
        return fires
    if not row["judgeable"] or not row.get("has_answer"):
        # An unjudgeable entry has no winning variant, so there is no reading to withdraw
        # the veto with; the scorer already writes has_answer=false on exactly those rows,
        # so this is the same population twice, said out loud rather than relied upon.
        # The deployed fence does the same: answer_check returns fires=None on a reading
        # with no answer fields and the caller falls back to the DG-only verdict.
        return True
    loss, echo = row.get(adl_field), row.get("echo_best")
    by_loss = True if loss is None or eta_a is None else bool(loss > eta_a)
    by_echo = True if echo is None else bool(echo >= echo_min)
    return bool({"adl": by_loss, "echo": by_echo, "either": by_loss or by_echo}[rule])


def rule_veto(row, fence, rule, eta_a=None, echo_min=1, adl_field="adl_best"):
    """``veto`` plus the cosine-only baseline, which reads no Deletion Gain at all."""
    if rule == "cosine_only":
        return bool(-row["base_cos"] > fence["neg_cos"])
    return veto(row, fence["dg"], fence["eta_a"] if eta_a is None else eta_a, rule,
                echo_min, adl_field)


def fit_answer_threshold(rows, budget, retrieval_floor, field="adl_best"):
    """``eta_a`` from the same rows ``fit_thresholds`` fits ``eta`` on.

    Bare, benign, calibration-split, judgeable, and a retrieval hit -- then whichever of
    those carry a readable answer. ``n`` is reported next to ``fit_thresholds``' own
    ``n`` so a gap between them (bare calibration entries the victim never answered) is
    visible rather than silently changing what the quantile was taken over.
    """
    cal = [r for r in rows if r["split"] == "calibration" and r["condition"] == "bare"
           and not r.get("malicious") and r["judgeable"]
           and r.get("retrieval_cos", r["base_cos"]) >= retrieval_floor
           and r.get("has_answer") and r.get(field) is not None]
    if len(cal) < 2:
        return {"eta_a": None, "n": len(cal), "field": field}
    return {"eta_a": float(np.quantile([r[field] for r in cal], 1-budget)),
            "n": len(cal), "field": field}


def hits_of(rows, floor):
    return [r for r in rows if r.get("retrieval_cos", r["base_cos"]) >= floor]


def _rounded(rate, places=6):
    """Six decimals on a rate and its interval. Every table stores thousands of these,
    and the seventh decimal of a bootstrap percentile is noise, not a number."""
    if rate["rate"] is None:
        return rate
    return {**rate, "rate": round(rate["rate"], places),
            "ci95": [round(bound, places) for bound in rate["ci95"]]}


def rule_rates(rows, fence, cfg, rule, eta_a=None, echo_min=1, adl_field="adl_best"):
    """Veto rate for one rule over one population, retrieval-conditioned.

    ``veto_given_hit`` is the number the threat model asks for: of the entries this
    query actually retrieved, how many did the fence refuse to serve. ``veto_all_pairs``
    keeps the same count over every planted row, so a family whose attacks mostly miss
    retrieval cannot borrow those misses as blocks. Both carry intent-cluster intervals;
    ``n``/``n_intents`` on each interval is the support.

    ``n_no_answer_hits`` **includes** ``n_unjudgeable_hits``: an unjudgeable entry has no
    winning variant, so the scorer writes ``has_answer=false`` on it. The two are not
    disjoint and must not be added. Subtract to get the answered-but-unreadable count.
    """
    floor = cfg.cache_threshold
    hits = hits_of(rows, floor)
    fired = sum(dg_fires(r, fence["dg"]) for r in hits)
    flags = [rule_veto(r, fence, rule, eta_a, echo_min, adl_field) for r in hits]
    every = [rule_veto(r, fence, rule, eta_a, echo_min, adl_field) for r in rows]
    return {"rule": rule, "n": len(rows), "n_intents": len({r["intent_id"] for r in rows}),
            "n_hits": len(hits), "n_hit_intents": len({r["intent_id"] for r in hits}),
            "hit_coverage": len(hits)/len(rows) if rows else None,
            "n_unjudgeable_hits": sum(not r["judgeable"] for r in hits),
            "n_no_answer_hits": sum(not r.get("has_answer") for r in hits),
            "n_missing_field_hits": sum(bool(r.get("has_answer")) and r.get(adl_field) is None
                                        for r in hits),
            "n_dg_fired_hits": fired,
            "n_vetoed_hits": int(sum(flags)),
            # An answer rule only ever withdraws a veto DG raised, so this is the share
            # of DG's vetoes it took back. Meaningless for cosine_only, which raises its
            # own, so that column says None rather than a number nobody should read.
            "rescue_of_dg_fired": (None if rule == "cosine_only" or not fired else
                                   (fired - int(sum(flags)))/fired),
            "veto_given_hit": _rounded(cluster_rate(flags, [r["intent_id"] for r in hits],
                                                    cfg.bootstrap_iterations, cfg.seed)),
            "veto_all_pairs": _rounded(cluster_rate(every, [r["intent_id"] for r in rows],
                                                    cfg.bootstrap_iterations, cfg.seed)),
            "served_fraction": (len(hits)-sum(flags))/len(rows) if rows else None}


def rule_table(rows, fence, cfg, eta_a=None, echo_min=1, adl_field="adl_best"):
    return {rule: rule_rates(rows, fence, cfg, rule, eta_a, echo_min, adl_field)
            for rule in RULES}


def best_is_core(row):
    """Is the winning shortened variant the bare core question, or a cut into it?

    Recomputed from ``base_text`` and ``best_text`` after whitespace normalisation, which
    is what the brief specifies and what the scorer's own ``best_is_core`` column stores;
    the stored column is used only for a row that carries no ``base_text`` (an attack, or
    an extra set that never had one). ``None`` means undecidable, and such rows are
    counted apart rather than assigned to either side.
    """
    core, best = row.get("base_text"), row.get("best_text")
    if core and best:
        return " ".join(core.split()) == " ".join(best.split())
    return row.get("best_is_core")


def fragment_split(rows, fence, cfg, eta_a=None, echo_min=1):
    """Of the rows Deletion Gain fired on, which had a whole question as the winner.

    The rescue's shape lives here: when the winning variant *is* the bare core question,
    the entry's own answer usually still needs what the wrapper added, so the answer rule
    withdraws the veto; when the winner cuts into the question, the removed span carries
    content the answer does need, and the veto stands. Rates are within each subgroup, so
    the denominator is that subgroup's DG-fired hits.
    """
    fired = [r for r in hits_of(rows, cfg.cache_threshold) if dg_fires(r, fence["dg"])]
    core = [r for r in fired if best_is_core(r) is True]
    fragment = [r for r in fired if best_is_core(r) is False]
    out = {"n_dg_fired_hits": len(fired),
           "n_undecidable": len(fired) - len(core) - len(fragment),
           "rule": "best_text == base_text after whitespace normalisation"}
    for name, sub in (("core", core), ("fragment", fragment)):
        out[name] = {"n_dg_fired": len(sub), "residual_veto": {
            rule: _rounded(cluster_rate(
                [rule_veto(r, fence, rule, eta_a, echo_min) for r in sub],
                [r["intent_id"] for r in sub], cfg.bootstrap_iterations, cfg.seed))
            for rule in RULES}}
    return out


def load_poisoned_flags(paths):
    """The judge's per-row verdicts, keyed by ``record_id``.

    Written by ``experiments/paper/analysis/asr_judge.py --flags-out``: one JSON object
    per attack row with ``poisoned`` and ``poisoned_strict``. Rows whose judge call failed
    are absent from that file by construction, never written as ``poisoned=false``, so an
    attack row with no flag is *unknown*, not *not poisoned*, and the tables count it
    separately instead of folding it into either arm.

    A ``record_id`` that appears twice raises. Letting the last write win would put one
    row's verdict onto a different row that happens to share the id -- a wrong verdict on
    a matched row, which is worse than an unmatched one, because nothing downstream can
    see it. Passing the same file (or the same directory) twice trips this too, which is
    the intended complaint.
    """
    flags, files, seen = {}, [], {}
    for entry in paths or ():
        root = Path(entry)
        if not root.exists():
            raise ValueError(f"--poisoned-flags path does not exist: {root}")
        for path in _jsonl_files(root):
            rows = read_jsonl(path)
            for index, record in enumerate(rows):
                if "record_id" not in record or "poisoned" not in record:
                    raise ValueError(f"{path} row {index}: needs record_id and poisoned")
                key = record["record_id"]
                if key in seen:
                    raise ValueError(
                        f"{path} row {index}: record_id {key!r} already flagged by "
                        f"{seen[key]}; a duplicate would put one row's verdict on another")
                seen[key] = f"{path} row {index}"
                flags[key] = {
                    "poisoned": bool(record["poisoned"]),
                    "poisoned_strict": (None if record.get("poisoned_strict") is None
                                        else bool(record["poisoned_strict"]))}
            files.append({"path": str(path), "n_rows": len(rows)})
    return flags, files


def _word_set(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def instruction_overlap(text):
    """Nearest seen instruction by word-set Jaccard. Lexical, fixed, score-blind."""
    ours, best, nearest = _word_set(text), 0.0, None
    for seen in SEEN_INSTRUCTIONS:
        theirs = _word_set(seen)
        union = ours | theirs
        score = len(ours & theirs)/len(union) if union else 0.0
        if score > best:
            best, nearest = score, seen
    return best, nearest


def unseen_instruction_report(rows, fence, cfg, eta_a=None):
    """Tag each "unseen" wrapper instruction against the seen templates, then subset.

    The frozen instruction list was drawn blind, so some of it collides with a seen
    template (ruling 10). No row is dropped by a score: every instruction is listed with its
    nearest seen wording and the overlap, and the rule tables are repeated over the
    strictly-unseen subset at two fixed cuts so the "unseen" claim can be read at either.
    """
    tagged = [r for r in rows if r.get("entry_instruction")]
    if not tagged:
        return None
    scores = {}
    for row in tagged:
        instruction = row["entry_instruction"]
        if instruction not in scores:
            overlap, nearest = instruction_overlap(instruction)
            scores[instruction] = {"instruction": instruction, "max_jaccard": overlap,
                                   "nearest_seen": nearest}
    report = {"instructions": [scores[k] for k in sorted(scores)],
              "n_instructions": len(scores),
              "rule": "word-set Jaccard against every seen template wording; "
                      "strictly unseen means max_jaccard < cut",
              "cuts": {}}
    for cut in UNSEEN_COLLISION_CUTS:
        subset = [r for r in tagged if scores[r["entry_instruction"]]["max_jaccard"] < cut]
        report["cuts"][f"jaccard<{cut}"] = {
            "n_instructions": len({r["entry_instruction"] for r in subset}),
            "n_rows": len(subset),
            "overall": rule_table(subset, fence, cfg, eta_a),
            "by_condition": {condition: rule_table(
                [r for r in subset if r.get("condition") == condition], fence, cfg, eta_a)
                for condition in sorted({r.get("condition", "") for r in subset})}}
    return report


#: Where a planted row's coarse group name comes from, in order of preference. Only the
#: built samples carry ``attack_class``; every extra set names itself through ``set``, and
#: a set that splits into families names those through ``family`` or ``template``. The
#: order is fixed here and the key that won is reported beside each group, so a reader
#: never has to guess what a group is.
ATTACK_CLASS_KEYS = ("attack_class", "set", "family", "template")

#: Same idea one level finer: the payload kind or family a set splits into, before the
#: coarse name is used as the fallback.
ATTACK_FAMILY_KEYS = ("payload_kind", "family", "attack_class", "template", "set")


def _first_key(row, keys):
    for key in keys:
        if row.get(key):
            return key, row[key]
    return "none", "unknown"


def attack_class_of(row):
    """The coarse group a planted row belongs to: CAP/SCP/KCA, else the set that built it."""
    return _first_key(row, ATTACK_CLASS_KEYS)[1]


def attack_class_source(row):
    """Which of :data:`ATTACK_CLASS_KEYS` gave this row its group name."""
    return _first_key(row, ATTACK_CLASS_KEYS)[0]


def attack_family(row):
    """The finest label a planted row carries: payload kind, surface family, else class."""
    return _first_key(row, ATTACK_FAMILY_KEYS)[1]


def attack_family_source(row):
    return _first_key(row, ATTACK_FAMILY_KEYS)[0]


def ablation_specs(eta_a, eta_a_override):
    """The prespecified ablation grid. Names are stable keys, values are what changed."""
    base = {"eta_a": eta_a, "echo_min": 1, "adl_field": "adl_best"}
    specs = {"eta_a=calibrated": base,
             "eta_a=0": {**base, "eta_a": 0.0},
             "echo_min=1": base,
             "echo_min=2": {**base, "echo_min": 2}}
    if eta_a_override is not None:
        specs["eta_a=supplied"] = {**base, "eta_a": float(eta_a_override)}
    for field in ADL_FIELDS[1:]:
        specs[f"adl_field={field}"] = {**base, "adl_field": field}
    return specs


def answer_rule_cell(cell_rows, test, groups, thresholds, cfg,
                     eta_a_override=None, flags=None, flag_files=()):
    """Every answer-rule table for one corpus x policy cell.

    ``groups`` is the same ten-condition split the Deletion-Gain-only tables use, passed
    in rather than rebuilt so the two layers can never disagree about what a condition is.
    """
    fitted = fit_answer_threshold(cell_rows, cfg.fpr_budget, cfg.cache_threshold)
    fence = {"dg": thresholds["dg"], "neg_cos": thresholds["neg_cos"],
             "n": thresholds["n"], "eta_a": fitted["eta_a"], "eta_a_n": fitted["n"],
             "eta_a_field": fitted["field"], "budget": cfg.fpr_budget,
             "retrieval_floor": cfg.cache_threshold, "echo_min": 1,
             "eta_a_override": None if eta_a_override is None else float(eta_a_override),
             "fitted_on": "bare benign calibration hits of this corpus and policy"}
    cell = {"thresholds": fence, "denominator": DENOMINATOR_NOTE,
            "conditions": {}, "by_template_position": {},
            "by_set": {}, "attacks": {"by_class": {}, "by_family": {}},
            "ablations": {}, "fragment_split": {},
            "poisoned_flags": {"n_flags": 0 if not flags else len(flags),
                               "files": list(flag_files)} if flags is not None else None,
            "unseen_instructions": unseen_instruction_report(test, fence, cfg)}
    for key, sub in groups.items():
        cell["conditions"][key] = rule_table(sub, fence, cfg)
        cell["fragment_split"][key] = fragment_split(sub, fence, cfg)
    # Benign rows only; a planted row's "template" is its attack family and it has its
    # own tables below. Keys come from the triples the file actually holds, so an extra
    # set's wrappers appear here too and the built templates contribute no empty rows.
    wrapped = [r for r in test if not r.get("malicious") and r.get("template")
               and r.get("template") != "bare"]
    for name, condition, position in sorted(
            {(r["template"], r.get("condition") or "", r.get("position") or "")
             for r in wrapped}):
        sub = [r for r in wrapped if r["template"] == name
               and (r.get("condition") or "") == condition
               and (r.get("position") or "") == position]
        cell["by_template_position"][f"{name}/{condition}/{position}"] = rule_table(
            sub, fence, cfg)
    for name in sorted({r.get("set") or "" for r in test}):
        sub = [r for r in test if (r.get("set") or "") == name]
        cell["by_set"][name] = {
            "group": SET_GROUPS.get(name, "other"),
            "malicious_fraction": sum(bool(r.get("malicious")) for r in sub)/len(sub) if sub else None,
            "overall": rule_table(sub, fence, cfg),
            "by_condition": {condition: rule_table(
                [r for r in sub if r.get("condition") == condition], fence, cfg)
                for condition in sorted({r.get("condition") or "" for r in sub})},
            # An extra set may carry no base_text, so its winner cannot be called core or
            # fragment; those rows land in n_undecidable rather than in either arm.
            "fragment_split": fragment_split(sub, fence, cfg)}
    attacks = [r for r in test if r.get("malicious")]
    for label, key in (("by_class", attack_class_of), ("by_family", attack_family)):
        for name in sorted({key(r) for r in attacks}):
            sub = [r for r in attacks if key(r) == name]
            poisoned = None if flags is None else [
                r for r in sub if flags.get(r.get("record_id"), {}).get("poisoned")]
            source = attack_class_source if label == "by_class" else attack_family_source
            cell["attacks"][label][name] = {
                "all_planted": rule_table(sub, fence, cfg),
                "poisoned": None if poisoned is None else rule_table(poisoned, fence, cfg),
                "key_source": dict(Counter(source(r) for r in sub)),
                "n_rows_without_flag": (None if flags is None else
                                        sum(r.get("record_id") not in flags for r in sub))}
    for name, spec in ablation_specs(fitted["eta_a"], eta_a_override).items():
        cell["ablations"][name] = {
            "spec": spec,
            "conditions": {key: rule_table(sub, fence, cfg, **spec)
                           for key, sub in groups.items()},
            "attacks": {cls: rule_table([r for r in attacks if attack_class_of(r) == cls],
                                        fence, cfg, **spec)
                        for cls in sorted({attack_class_of(r) for r in attacks})}}
    return cell



# --- markdown rendering ---------------------------------------------------------


def _cell(rate):
    """One interval as ``rate [lo,hi] n=rows/intents``, or an em dash when empty."""
    if not rate or rate.get("rate") is None:
        return "&mdash;"
    low, high = rate["ci95"]
    return f"{rate['rate']:.3f} [{low:.3f},{high:.3f}] n={rate['n']}/{rate['n_intents']}i"


def _rule_row(label, table, key="veto_given_hit"):
    first = table[RULES[0]]
    counts = f"{first['n_hits']}/{first['n']}"
    return ("| " + " | ".join([label, counts]
                              + [_cell(table[rule][key]) for rule in RULES]) + " |")


def _rule_header(first_column):
    return ("| " + " | ".join([first_column, "hits/rows", *RULES]) + " |\n|"
            + "---|" * (len(RULES) + 2))


def _table(title, first_column, entries, key="veto_given_hit", note=None):
    lines = [f"**{title}**", ""]
    if note:
        lines += [note, ""]
    lines.append(_rule_header(first_column))
    for label, table in entries:
        lines.append(_rule_row(label, table, key))
    lines.append("")
    return lines


def render_tables(output):
    """The same numbers as ``summary.json``, as markdown a reader can scan.

    Rates are ``veto_given_hit``: the fraction of retrieved rows the fence refused to
    serve. On a benign condition that is a false veto and lower is better; on an attack
    family it is the block rate and higher is better. ``cosine_only`` is in the same row
    of every table by the project's baseline rule.
    """
    lines = ["# Answer-checked Deletion Gain &mdash; rule tables", "",
             f"Budget {output['config']['budget']}, retrieval floor "
             f"{output['config']['retrieval_floor']}, "
             f"{output['config']['bootstrap_iterations']} intent-cluster bootstrap "
             f"resamples at seed {output['config']['seed']}.", "",
             "Every cell is `rate [95% CI] n=rows/intents`. The rate is *veto given a "
             "retrieval hit*: a false-veto rate on a benign condition, a block rate on an "
             "attack family. Intervals resample **test intents**, keeping every template "
             "copy of an intent together. Retrieval misses are excluded from every "
             "denominator and are never counted as blocks.", ""]
    for note in output.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    coverage = output.get("coverage") or {}
    if coverage:
        lines += [f"Rows read: {coverage['n_rows']}. By set: "
                  + ", ".join(f"{k or '(none)'} {v}" for k, v in sorted(coverage["rows_by_set"].items()))
                  + ".", ""]
        if coverage.get("defaults_applied"):
            lines += ["Metadata columns filled with an empty default (an extra set that "
                      "does not carry them): "
                      + ", ".join(f"`{k}` {v}" for k, v in sorted(coverage["defaults_applied"].items()))
                      + ".", ""]
        for name, body in sorted(coverage.get("cells_skipped", {}).items()):
            lines += [f"**Skipped `{name}`** &mdash; {body['reason']} "
                      f"({body['n_rows']} rows, {body['n_intents']} intents, sets "
                      + ", ".join(f"{k or '(none)'} {v}" for k, v in sorted(body["rows_by_set"].items()))
                      + ").", ""]
    for key, cell in output.get("answer_rules", {}).items():
        corpus, policy = key.split("|", 1)
        fence = cell["thresholds"]
        lines += [f"## {corpus} &mdash; `{policy}`", "",
                  f"- eta {fence['dg']:.6f} on {fence['n']} bare calibration hits; "
                  f"cosine fence {-fence['neg_cos']:.6f}",
                  f"- eta_a {'none' if fence['eta_a'] is None else format(fence['eta_a'], '.6f')} "
                  f"({fence['eta_a_field']}) on {fence['eta_a_n']} of those rows",
                  f"- echo_min {fence['echo_min']}; nothing fitted on a test intent or an attack",
                  ""]
        lines += _table("1. False veto by pairing condition", "condition",
                        sorted(cell["conditions"].items()))
        lines += _table("1b. Same conditions over all pairs (misses included)", "condition",
                        sorted(cell["conditions"].items()), key="veto_all_pairs",
                        note="Denominator is every constructed pair, so a condition that "
                             "rarely retrieves cannot borrow its misses as passes.")
        lines += ["**1c. Served fraction** (rows retrieved and not vetoed, over all rows)",
                  "", "| condition | " + " | ".join(RULES) + " |",
                  "|" + "---|" * (len(RULES) + 1)]
        for name, table in sorted(cell["conditions"].items()):
            served = ["&mdash;" if table[rule]["served_fraction"] is None
                      else f"{table[rule]['served_fraction']:.3f}" for rule in RULES]
            lines.append("| " + " | ".join([name, *served]) + " |")
        lines.append("")
        lines += _table("2. By template and position", "template/condition/position",
                        sorted(cell["by_template_position"].items()))
        lines += _table("3. By set (seen / unseen / new question)", "set",
                        [(f"{name} ({body['group']})", body["overall"])
                         for name, body in sorted(cell["by_set"].items())])
        for label in ("by_class", "by_family"):
            entries = []
            for name, body in sorted(cell["attacks"][label].items()):
                entries.append((f"{name} (all planted)", body["all_planted"]))
                if body["poisoned"] is not None:
                    entries.append((f"{name} (poisoned only)", body["poisoned"]))
            if entries:
                lines += _table(f"4. Attack block rate, {label.replace('_', ' ')}",
                                "family", entries,
                                note="A row the judge could not verdict carries no flag and "
                                     "is in neither arm; `n_rows_without_flag` in "
                                     "`summary.json` counts them.")
        for name, body in cell["ablations"].items():
            lines += _table(f"5. Ablation `{name}` &mdash; {body['spec']}", "condition",
                            sorted(body["conditions"].items())
                            + sorted(body["attacks"].items()))
        entries = []
        for name, body in sorted(cell["fragment_split"].items()):
            for side in ("core", "fragment"):
                entries.append((f"{name} / s*={side} (n={body[side]['n_dg_fired']})",
                                {rule: {"veto_given_hit": body[side]["residual_veto"][rule],
                                        "n_hits": body[side]["n_dg_fired"],
                                        "n": body[side]["n_dg_fired"]} for rule in RULES}))
        lines += _table("6. Fragment split: residual veto among DG-fired hits",
                        "condition / winner", entries,
                        note="`s*=core` means the winning shortened variant is the bare "
                             "core question; `s*=fragment` means it cuts into the question.")
        unseen = cell.get("unseen_instructions")
        if unseen:
            lines += ["**7. Unseen wrapper instructions against the seen templates**", "",
                      "| instruction | max Jaccard | nearest seen |", "|---|---|---|"]
            for item in unseen["instructions"]:
                lines.append(f"| {item['instruction']} | {item['max_jaccard']:.2f} | "
                             f"{item['nearest_seen']} |")
            lines.append("")
            lines += _table("7b. Strictly-unseen subsets", "cut",
                            [(f"{cut} ({body['n_instructions']} instructions)", body["overall"])
                             for cut, body in unseen["cuts"].items()])
    return "\n".join(lines) + "\n"


def analyze(workspace, cfg, smoke=False, poisoned_flags=None, eta_a_override=None):
    """Read ``scores.jsonl``, write ``summary.json`` and ``REPORT_TABLES.md``.

    ``poisoned_flags`` are ``asr_judge --flags-out`` files; supplying them adds a second
    arm to every attack table, restricted to the rows the judge said actually poisoned
    the victim. ``eta_a_override`` adds one more ablation column at a caller-chosen
    answer threshold; it never replaces the calibrated one.
    """
    path = workspace / ("scores_smoke.jsonl" if smoke else "scores.jsonl")
    rows, filled = normalise_rows(read_jsonl(path), path)
    flags, flag_files = (None, []) if not poisoned_flags else load_poisoned_flags(poisoned_flags)
    output = {"config": {"budget": cfg.fpr_budget, "retrieval_floor": cfg.cache_threshold,
                         "seed": cfg.seed, "bootstrap_iterations": cfg.bootstrap_iterations,
                         "rules": list(RULES), "echo_min": 1,
                         "eta_a_override": None if eta_a_override is None else float(eta_a_override),
                         "poisoned_flag_files": flag_files},
              "smoke": smoke, "notes": [
                  "Primary thresholds use bare calibration-intent hits only; evaluation uses disjoint test intents.",
                  "Intervals resample test intents at a fixed fitted threshold, so they exclude calibration uncertainty.",
                  "Paper-style all-bare thresholds are a descriptive reproduction, not an independent holdout result.",
                  "Benignness and answer-reuse compatibility are separate labels. Answers are the victim's own cached responses, read not generated; a poisoned verdict, when supplied, comes from asr_judge.",
                  "Matched AUROCs require 20 distinct intents per arm per bin (configured), not 20 template copies.",
                  "Only English fixed-template generalization is measured. No template or row is selected by its score.",
                  "eta_a is fitted on the same bare benign calibration hits as eta, at the same budget; no attack and no test intent is read.",
                  "An unjudgeable entry, and an entry whose answer fields cannot be read, is vetoed by every rule (fail-closed), never dropped.",
                  "n_no_answer_hits includes n_unjudgeable_hits (an unjudgeable entry has no readable answer); do not add the two.",
                  "A planted row's group name comes from attack_class, else set, else family, else template; key_source says which won.",
                  "Answer rules only withdraw a veto Deletion Gain already raised; they never raise one, so a rule can only lower a block rate.",
                  "An attack row the judge could not verdict carries no poisoned flag and is in neither the all-planted nor the poisoned arm."],
              "cells": {}, "answer_rules": {}}
    output["coverage"] = {
        "n_rows": len(rows), "defaults_applied": filled,
        "rows_by_corpus": dict(Counter(r["corpus"] for r in rows)),
        "rows_by_policy": dict(Counter(r["policy"] for r in rows)),
        "rows_by_set": dict(Counter(r["set"] for r in rows)),
        "rows_by_split": dict(Counter(r["split"] for r in rows)),
        "n_malicious": sum(bool(r["malicious"]) for r in rows),
        "rows_outside_configured_policies": sum(r["policy"] not in cfg.policies for r in rows),
        "cells_skipped": {},
        "note": "Every row of the file is in exactly one corpus|policy cell, or in "
                "cells_skipped, or outside the configured policies. Nothing is dropped "
                "silently."}
    for corpus in sorted({r["corpus"] for r in rows}):
        for policy in cfg.policies:
            cell_rows = [r for r in rows if r["corpus"] == corpus and r["policy"] == policy]
            if not cell_rows:
                continue
            test = [r for r in cell_rows if r["split"] == "test"]
            try:
                thresholds = fit_thresholds(cell_rows, cfg.fpr_budget, cfg.cache_threshold)
            except ValueError as exc:
                # A cell with no bare benign calibration hit has no threshold to fit, so
                # it has no rate either. Say which rows those were and carry on rather
                # than killing the run: an extra set whose corpus column arrived empty
                # would otherwise take the whole file down with it.
                output["coverage"]["cells_skipped"][f"{corpus}|{policy}"] = {
                    "n_rows": len(cell_rows), "reason": str(exc),
                    "n_intents": len({r["intent_id"] for r in cell_rows}),
                    "rows_by_set": dict(Counter(r["set"] for r in cell_rows))}
                print(f"SKIPPED {corpus!r} {policy}: {exc} "
                      f"({len(cell_rows)} rows, sets "
                      f"{sorted({r['set'] for r in cell_rows})})", flush=True)
                continue
            all_bare = [r for r in cell_rows if r["condition"] == "bare" and r["judgeable"]]
            paper = {"dg": float(np.quantile([r["dg"] for r in all_bare], 1-cfg.fpr_budget)),
                     "neg_cos": float(np.quantile([-r["base_cos"] for r in all_bare], 1-cfg.fpr_budget)),
                     "n": len(all_bare)}
            def rates(rs, fence=thresholds):
                return operating_rates(rs, fence, cfg.cache_threshold, cfg.bootstrap_iterations, cfg.seed)
            groups = {"bare": [r for r in test if r["condition"] == "bare"]}
            for kind in ("polite", "constraint"):
                for condition in ("entry_only", "query_only", "both_same", "both_paraphrase", "exact_core"):
                    groups[f"{kind}/{condition}"] = [r for r in test if r["kind"] == kind and r["condition"] == condition]
            bare_lookup = {r["intent_id"]: r for r in groups["bare"]}
            cell = {"thresholds": thresholds, "paper_style_thresholds": paper,
                    "n_calibration_intents": len({r["intent_id"] for r in cell_rows if r["split"] == "calibration" and not r["malicious"]}),
                    "n_test_intents": len(bare_lookup), "groups": {}, "by_template_position": {},
                    "attacks": {}, "recalibration": {}}
            calibration_bare = [r for r in cell_rows if r["split"] == "calibration" and r["condition"] == "bare"]
            cell["calibration_bare"] = rates(calibration_bare)
            for key, sub in groups.items():
                valid = [r for r in sub if r["judgeable"]]
                flagged = [r for r in valid if r["retrieval_cos"] >= cfg.cache_threshold and r["dg"] > thresholds["dg"]]
                cell["groups"][key] = {**rates(sub),
                    "paired_vs_bare": paired_effect(sub, bare_lookup, thresholds, cfg) if key != "bare" and not key.endswith("exact_core") else None,
                    "dg_quantiles": np.quantile([r["dg"] for r in valid], [.5,.9,.95,.99]).tolist() if valid else None,
                    "n_flagged_best_is_exact_core": sum(r.get("best_is_core", False) for r in flagged),
                    "paper_style_frozen_veto": rates(sub, paper)["dg_veto_given_hit"]}
            for name in sorted({r["template"] for r in test if r["kind"] in ("polite", "constraint")}):
                for condition in ("entry_only", "query_only", "both_same", "both_paraphrase", "exact_core"):
                    for position in ("prefix", "suffix"):
                        sub = [r for r in test if r["template"] == name and r["condition"] == condition and r["position"] == position]
                        cell["by_template_position"][f"{name}/{condition}/{position}"] = rates(sub)
            planted = [r for r in test if r["malicious"]]
            for family in sorted({attack_class_of(r) for r in planted}):
                attacks = [r for r in planted if attack_class_of(r) == family]
                ahits = [r for r in attacks if r["retrieval_cos"] >= cfg.cache_threshold]
                comparisons = {}
                for key, benign in groups.items():
                    bhits = [r for r in benign if r["retrieval_cos"] >= cfg.cache_threshold]
                    comparisons[key] = {"all_pairs": discrimination(attacks, benign, cfg),
                                        "candidate_hits": discrimination(ahits, bhits, cfg)}
                cell["attacks"][family] = {
                    **rates(attacks), "comparisons": comparisons,
                    "key_source": dict(Counter(attack_class_source(r) for r in attacks))}
            for name, condition in (("compatible_mixture", "both_paraphrase"),
                                    ("harmless_mixture", "entry_only")):
                fence = recalibration_thresholds(cell_rows, cfg, condition)
                if fence is None:
                    continue
                mix = {"thresholds": fence, "groups": {}, "unseen_template_groups": {}, "attacks": {}}
                for key, sub in groups.items():
                    mix["groups"][key] = rates(sub, fence)
                    if key != "bare":
                        unseen = [r for r in sub if r["template"] not in fence["seen_templates"]]
                        mix["unseen_template_groups"][key] = rates(unseen, fence)
                for family in cell["attacks"]:
                    mix["attacks"][family] = rates(
                        [r for r in planted if attack_class_of(r) == family], fence)
                cell["recalibration"][name] = mix
            output["cells"][f"{corpus}|{policy}"] = cell
            answers = answer_rule_cell(cell_rows, test, groups, thresholds, cfg,
                                       eta_a_override, flags, flag_files)
            output["answer_rules"][f"{corpus}|{policy}"] = answers
            eta_a = answers["thresholds"]["eta_a"]
            print(f"analyzed {corpus} {policy}: cal {thresholds['n']}, test {len(bare_lookup)}, "
                  f"eta {thresholds['dg']:.8f}, eta_a "
                  f"{'none' if eta_a is None else format(eta_a, '.8f')} "
                  f"(n={answers['thresholds']['eta_a_n']})", flush=True)
    target = workspace / ("summary_smoke.json" if smoke else "summary.json")
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    # Store a deterministic audit sample: extreme examples + uniform intent samples.
    audit = []
    for key, cell in output["cells"].items():
        corpus, policy = key.split("|", 1)
        for group in cell["groups"]:
            if group == "bare":
                continue
            kind, condition = group.split("/")
            sub = [r for r in rows if r["corpus"] == corpus and r["policy"] == policy and
                   r["split"] == "test" and r["kind"] == kind and r["condition"] == condition and r["judgeable"]]
            top = sorted(sub, key=lambda r: -r["dg"])[:3]
            for r in top:
                audit.append({**r, "audit_selection": "highest_DG_diagnostic_only",
                              "threshold": cell["thresholds"]["dg"]})
    write_jsonl(workspace / ("audit_smoke.jsonl" if smoke else "audit.jsonl"), audit)
    tables = workspace / ("REPORT_TABLES_smoke.md" if smoke else "REPORT_TABLES.md")
    tables.write_text(render_tables(output), encoding="utf-8")
    print(f"Wrote {target} and {tables}", flush=True)
