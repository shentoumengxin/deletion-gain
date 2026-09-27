"""v3 detection matrix: excess + cosine per (set, encoder), with the poisoned column.

Replicates ``sentry.cache.defense.calibrate.evaluate_policy`` line for line so the
pooled/per-family numbers are bit-identical to the shipped tool, and adds the two things
the tool does not carry:

  * ``record_id`` on every scored attack row, joined to ``poisoned_flags.jsonl`` (with the
    KCA f2 ``gcg-f2-NNNN`` -> ``kca-f2-NNNN`` remap) to report TPR restricted to the rows
    the judge marked poisoned, alongside the all-planted TPR.
  * both flat-fence heights explicitly: the ``scoring`` height (fit on the WHOLE benign
    arm, the boundary the block rates are read against) and the ``holdout`` height (fit on
    the fit-half, the one calibrate writes to disk).

Optional ``--with-ppl-nli`` adds the conditional-perplexity and bidirectional-NLI
baselines on the same rows, timed in ms/row (for the e5 baseline column).

``--dump-rows`` writes one JSONL row per scored entry -- both arms, the winning
variant's ``answer_loss`` and its echo count already net of the anchor's content words --
so a bootstrap can refit ``eta_a``, refit the joint height and re-read the rule on each
replicate without re-embedding anything. :func:`rule_rates_from_rows` rebuilds a cell's
rates from that dump by calling the same ``fit_rule_fence`` / ``joint_fence`` /
``blocks_under``; a test pins the rebuild against ``run_cell`` exactly.

``--answers`` joins each entry to the victim answer generated for it (by the sha256 of
the **entry text**, the key ``experiments/paper/rq1_detection/gen_answers.py`` writes) and
adds the ``answer_rules`` table: what each of ``dg_only`` / ``adl`` / ``echo`` /
``either`` blocks, with the cosine-only baseline beside them. The join is by text rather
than by ``record_id``/``intent_id`` because the answer files are text-keyed, and a text
join cannot hand one entry another entry's answer. The DG-only column is the published
one: it is the same arithmetic as the top-level fields and this file refuses to write a
report where the two disagree.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import (
    ANSWER_RULE_COLUMNS, AnswerColumns, achieved_under, answer_rule_pairing, auroc,
    blocks_under, fit_rule_fence, joint_fence, matched_auroc, rule_needs_eta_a,
    _intent_holdout, parse_policy,
)
from sentry.cache.defense.deletion import build_profile, excess
from sentry.cache.defense.fence import CalibrationRow, ExcessFence, achieved_block_rate
# The answer join lives in one place. `load_answer_files` implements the rule
# `gen_answers.py:load_answers` owns -- append-only files, last non-empty response per
# hash wins -- and a test pins the two against each other
# (tests/test_instruction_benign.py::test_answer_loading_agrees_with_gen_answers).
# Restating it here would be a second chance to join differently.
from sentry.research.pipeline.instruction_benign import (  # noqa: E402
    digest as entry_digest, load_answer_files as load_answers,
)

_ENTRY_ROLES = {"canonical": "genuine", "ndss": "attack"}
_ANCHOR_ROLES = ("legal", "benign_query")


def load_rows(path, attack_roles, benign_generators):
    """Entry-side rows with record_id retained; mirrors load_pair_rows + filter_benign."""
    records = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    entry_roles = dict(_ENTRY_ROLES)
    for role in attack_roles:
        entry_roles[role] = "attack"
    anchors = {}
    for role in _ANCHOR_ROLES:
        for r in records:
            if r.get("query_role") == role:
                anchors.setdefault(r["intent_id"], r["text"])
    rows = []
    kept = {"genuine": Counter(), "attack": Counter()}
    dropped = {"genuine": 0, "attack": 0}
    allowed_benign = set(benign_generators) if benign_generators else None
    for r in records:
        arm = entry_roles.get(r.get("query_role"))
        if arm is None:
            continue
        gen = r.get("generator", "unknown")
        anchor = anchors.get(r["intent_id"])
        if anchor is None or anchor == r["text"]:
            dropped[arm] += 1
            continue
        if arm == "genuine" and allowed_benign is not None and gen not in allowed_benign:
            continue
        rows.append({"text": r["text"], "anchor": anchor, "arm": arm, "family": gen,
                     "intent_id": r["intent_id"], "record_id": r.get("record_id")})
        kept[arm][gen] += 1
    comp = {"kept": {a: dict(c) for a, c in kept.items()}, "dropped_no_anchor": dropped}
    return rows, comp


def load_flags(path):
    flags = {}
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        f = json.loads(l)
        flags[f["record_id"]] = f
    return flags


def poisoned_of(record_id, flags):
    """Direct join, then the KCA f2 gcg-f2-NNNN -> kca-f2-NNNN remap. None if absent."""
    if record_id in flags:
        return flags[record_id]
    if record_id and record_id.startswith("gcg-f2-"):
        alt = "kca-f2-" + record_id[len("gcg-f2-"):]
        if alt in flags:
            return flags[alt]
    return None


def run_cell(eval_path, embedder, policy, flags, budget, holdout, seed, storage_dtype,
             attack_roles, benign_generators, answers=None, echo_min=1):
    rows, comp = load_rows(eval_path, attack_roles, benign_generators)
    answers = answers or {}
    # Hash every entry text once, and settle the join before the embedding pass rather
    # than after it: a template or corpus mismatch is announced in milliseconds instead
    # of costing an encode of every variant of every row first.
    digests = [entry_digest(row["text"]) if answers else None for row in rows]
    if answers and not any(d in answers for d in digests):
        raise ValueError(
            f"--answers loaded {len(answers)} hashes but none of the {len(rows)} entry "
            f"texts in {eval_path} matched one. The join is the sha256 of the ENTRY "
            f"TEXT, so this is a corpus or prompt-template mismatch, not an empty "
            f"result; scoring it would report DG-only numbers as answer-checked.")

    t0 = time.perf_counter()
    readings = []
    n_encode_calls = 0
    # Insertion-time cost of the second witness, counted over every entry the host would
    # write rather than over the judgeable ones: one extra embedder call per entry whose
    # answer normalises to something. An answer that normalises to nothing is joined and
    # then dropped by build_profile, which is a different fact and gets its own counter.
    n_answer_joined = n_answer_encoded = 0
    for row, sha in zip(rows, digests):
        answer = answers.get(sha) if answers else None
        n_answer_joined += answer is not None
        profile = build_profile(row["text"], embedder, policy,
                                storage_dtype=storage_dtype, answer=answer)
        n_answer_encoded += profile.has_answer
        if not profile.judgeable:
            continue
        anchor = embedder.encode([row["anchor"]])[0]
        n_encode_calls += 1
        readings.append((row, excess(profile, anchor)))
    embed_wall = time.perf_counter() - t0

    benign = [(r, x) for r, x in readings if r["arm"] == "genuine"]
    attack = [(r, x) for r, x in readings if r["arm"] == "attack"]
    if len(benign) < 4 or len(attack) < 2:
        raise ValueError(f"too few rows: {len(benign)} genuine, {len(attack)} attack")

    from collections import namedtuple
    _P = namedtuple("_P", ["intent_id"])
    fit_intents, eval_intents = _intent_holdout(
        [_P(r["intent_id"]) for r, _ in benign], holdout=holdout, seed=seed)
    fit_rows = [(r, x) for r, x in benign if r["intent_id"] in fit_intents]
    eval_rows = [(r, x) for r, x in benign if r["intent_id"] in eval_intents]

    def cal(pairs):
        return [CalibrationRow(x.base_cos, x.words, x.excess_span) for _, x in pairs]

    fence = ExcessFence.fit_flat(cal(fit_rows), budget=budget,
                                 embedder=embedder.model_name, policy=policy.fingerprint())
    scoring_fence = ExcessFence.fit_flat(cal(benign), budget=budget,
                                         embedder=embedder.model_name,
                                         policy=policy.fingerprint())

    benign_excess = np.array([x.excess_span for _, x in benign])
    attack_excess = np.array([x.excess_span for _, x in attack])
    benign_cos = np.array([x.base_cos for _, x in benign])
    attack_cos = np.array([x.base_cos for _, x in attack])

    matched, support = matched_auroc(attack_excess, benign_excess, attack_cos,
                                     benign_cos, width=0.01)
    cosine_fence = float(np.quantile(benign_cos, budget))

    def blocked_rate(pairs):
        return float(np.mean([scoring_fence.blocks(x.base_cos, x.words, x.excess_span)
                              for _, x in pairs]))

    # per-attack-row block decision + poisoned join
    per_attack = []
    for r, x in attack:
        blk = bool(scoring_fence.blocks(x.base_cos, x.words, x.excess_span))
        flag = poisoned_of(r["record_id"], flags)
        per_attack.append({
            "record_id": r["record_id"], "family": r["family"], "blocked": blk,
            "excess_span": float(x.excess_span), "base_cos": float(x.base_cos),
            "words": int(x.words),
            "poisoned": (None if flag is None else bool(flag.get("poisoned"))),
            "poisoned_strict": (None if flag is None else
                                (None if flag.get("poisoned_strict") is None
                                 else bool(flag.get("poisoned_strict")))),
            "flag_found": flag is not None,
        })

    n_flag_missing = sum(1 for a in per_attack if not a["flag_found"])
    pois = [a for a in per_attack if a["poisoned"] is True]
    pois_strict = [a for a in per_attack if a["poisoned_strict"] is True]
    tpr_all = float(np.mean([a["blocked"] for a in per_attack]))
    tpr_pois = (float(np.mean([a["blocked"] for a in pois])) if pois else float("nan"))
    tpr_strict = (float(np.mean([a["blocked"] for a in pois_strict]))
                  if pois_strict else float("nan"))

    families = {}
    for name in sorted({r["family"] for r, _ in attack}):
        fam = [(r, x) for r, x in attack if r["family"] == name]
        fam_excess = np.array([x.excess_span for _, x in fam])
        fam_cos = np.array([x.base_cos for _, x in fam])
        fm, fs = matched_auroc(fam_excess, benign_excess, fam_cos, benign_cos, width=0.01)
        fam_attack_ids = {r["record_id"] for r, _ in fam}
        fam_pa = [a for a in per_attack if a["record_id"] in fam_attack_ids]
        fam_pois = [a for a in fam_pa if a["poisoned"] is True]
        families[name] = {
            "n_attack": len(fam),
            "excess_auroc": auroc(fam_excess, benign_excess),
            "excess_auroc_cos_matched": fm,
            "matched_support": fs,
            "excess_block_rate": blocked_rate(fam),
            "n_poisoned": len(fam_pois),
            "tpr_poisoned": (float(np.mean([a["blocked"] for a in fam_pois]))
                             if fam_pois else float("nan")),
            "cosine_auroc": auroc(-fam_cos, -benign_cos),
            "cosine_block_rate": float((fam_cos < cosine_fence).mean()),
        }

    # ---- the same rows under each answer rule --------------------------------------
    # Nothing above this line reads an answer: `scoring_fence`/`fence` carry
    # answer_rule="none", so every field already published is the DG-only one, and the
    # table below refits its own boundary per rule rather than touching theirs. The
    # equality is checked before this function returns.
    columns = AnswerColumns()
    items = {"attack": [(r["anchor"], x) for r, x in attack],
             "benign": [(r["anchor"], x) for r, x in benign],
             "fit": [(r["anchor"], x) for r, x in fit_rows],
             "eval": [(r["anchor"], x) for r, x in eval_rows]}
    n_loss = sum(1 for _, x in items["benign"] if x.answer_loss is not None)
    n_loss_fit = sum(1 for _, x in items["fit"] if x.answer_loss is not None)
    n_loss_attack = sum(1 for _, x in items["attack"] if x.answer_loss is not None)
    # Partial coverage pulls an answer-checked column back towards dg_only, because a row
    # with no answer fields fires fail-closed; `fitted: false` only catches zero coverage.
    coverage = {"benign": n_loss / len(benign), "attack": n_loss_attack / len(attack)}

    def cal_a(key):
        return [CalibrationRow(x.base_cos, x.words, x.excess_span,
                               answer_loss=columns(a, x)[0]) for a, x in items[key]]

    answer_rules = {}
    for label, rule in ANSWER_RULE_COLUMNS:
        if rule_needs_eta_a(rule) and min(n_loss, n_loss_fit) == 0:
            # No ceiling can be fitted from this arm, and standing one in would invent a
            # boundary and then rescue or block by accident.
            answer_rules[label] = {
                "answer_rule": rule, "fitted": False, "answer_coverage": coverage,
                "reason": f"no benign row carries an answer_loss ({n_loss} scoring, "
                          f"{n_loss_fit} fit-half), so eta_a cannot be fitted"}
            continue
        kw = dict(budget=budget, embedder=embedder.model_name,
                  policy=policy.fingerprint(), echo_min=echo_min)
        scoring = fit_rule_fence(cal_a("benign"), rule, **kw)
        held = fit_rule_fence(cal_a("fit"), rule, **kw)
        # Two calibrations of the same rule, because a conjunction fitted for one half of
        # itself does not spend its budget. `shared` keeps the DG-only height -- the
        # honest "bolt the check onto the deployed fence" number -- and `joint` re-fits
        # the height with the check in place, so the TPRs of two rules are read at the
        # same benign cost. `eta_a` is fitted first, on the benign arm, and is the same
        # in both.
        scoring_joint, joint = joint_fence(scoring, items["benign"], columns, budget)
        held_joint, joint_held = joint_fence(held, items["fit"], columns, budget)

        def rates(fence, prefix=""):
            blocked = blocks_under(fence, items["attack"], columns)
            rows = list(zip(per_attack, blocked))
            p_rows = [b for a, b in rows if a["poisoned"] is True]
            s_rows = [b for a, b in rows if a["poisoned_strict"] is True]
            out = {
                f"excess_block_rate{prefix}": float(np.mean(blocked)),
                f"tpr_all_planted{prefix}": float(np.mean(blocked)),
                f"tpr_poisoned{prefix}": (float(np.mean(p_rows)) if p_rows
                                          else float("nan")),
                f"tpr_poisoned_strict{prefix}": (float(np.mean(s_rows)) if s_rows
                                                 else float("nan")),
                f"benign_block_rate_in_sample{prefix}": float(np.mean(
                    blocks_under(fence, items["benign"], columns))),
            }
            return out, rows

        shared_rates, by_row = rates(scoring)
        joint_rates, joint_by_row = rates(scoring_joint, "_joint_eta")
        rule_pois = [b for a, b in by_row if a["poisoned"] is True]
        rule_strict = [b for a, b in by_row if a["poisoned_strict"] is True]
        answer_rules[label] = {
            "answer_rule": rule, "fitted": True, "echo_min": echo_min,
            "eta_a": scoring.eta_a, "eta_a_holdout": held.eta_a,
            "n_answer_loss_rows": n_loss, "n_answer_loss_rows_holdout": n_loss_fit,
            "n_attack_no_answer_fields": len(items["attack"]) - n_loss_attack,
            "n_benign_no_answer_fields": len(items["benign"]) - n_loss,
            "n_benign_answer_fires": joint.n_fires,
            "answer_coverage": coverage,
            "threshold_scoring": float(scoring.coefficients[0]),
            "threshold_holdout": float(held.coefficients[0]),
            "n_poisoned": len(rule_pois), "n_poisoned_strict": len(rule_strict),
            # --- shared eta: the DG-only height with the check bolted on --------------
            **shared_rates,
            "achieved_benign_block_rate": achieved_under(held, items["eval"], columns),
            "eta_shared": float(scoring.coefficients[0]),
            # --- joint eta: the height re-fitted so the joint rule spends the budget ---
            **joint_rates,
            "achieved_benign_block_rate_joint_eta": achieved_under(
                held_joint, items["eval"], columns),
            "eta_joint": float(scoring_joint.coefficients[0]),
            "eta_joint_holdout": float(held_joint.coefficients[0]),
            "joint_budget_reachable": joint.reachable,
            "joint_budget_reachable_holdout": joint_held.reachable,
            "families": {},
        }
        for name in families:
            fam = [(a, b) for a, b in by_row if a["family"] == name]
            fam_j = [b for a, b in joint_by_row if a["family"] == name]
            fam_pois = [b for a, b in fam if a["poisoned"] is True]
            fam_pois_j = [b for (a, _), b in zip(fam, fam_j) if a["poisoned"] is True]
            answer_rules[label]["families"][name] = {
                "n_attack": len(fam),
                "excess_block_rate": float(np.mean([b for _, b in fam])) if fam
                                     else float("nan"),
                "excess_block_rate_joint_eta": (float(np.mean(fam_j)) if fam_j
                                                else float("nan")),
                "n_poisoned": len(fam_pois),
                "tpr_poisoned": (float(np.mean(fam_pois)) if fam_pois else float("nan")),
                "tpr_poisoned_joint_eta": (float(np.mean(fam_pois_j)) if fam_pois_j
                                           else float("nan")),
            }
    # The cosine-only baseline, in the same table so the comparison sits beside the rules
    # rather than elsewhere in the file. Every value is a top-level one unchanged, read
    # under the cosine floor instead of the fence.
    cos_blocked = [bool(a["base_cos"] < cosine_fence) for a in per_attack]
    # The baseline gets the same two fits as every rule column, so its benign cost is
    # comparable: the block rates are read against the floor fitted on the whole benign
    # arm (which is the published `cosine_block_rate`, unchanged), and the eval-half rate
    # against a floor fitted on the fit half. Reading the eval half against the whole-arm
    # floor -- which is what this reported before -- is an in-sample rate sitting in a row
    # of out-of-sample ones.
    cosine_fence_holdout = float(np.quantile(
        np.array([x.base_cos for _, x in fit_rows]), budget))
    answer_rules["cosine_only"] = {
        "excess_block_rate": float((attack_cos < cosine_fence).mean()),
        "tpr_all_planted": float((attack_cos < cosine_fence).mean()),
        "tpr_poisoned": (float(np.mean([b for a, b in zip(per_attack, cos_blocked)
                                        if a["poisoned"] is True])) if pois
                         else float("nan")),
        "tpr_poisoned_strict": (float(np.mean(
            [b for a, b in zip(per_attack, cos_blocked)
             if a["poisoned_strict"] is True])) if pois_strict else float("nan")),
        "benign_block_rate_in_sample": float(np.mean(
            [x.base_cos < cosine_fence for _, x in benign])),
        "achieved_benign_block_rate": float(np.mean(
            [x.base_cos < cosine_fence_holdout for _, x in eval_rows])),
        "auroc": auroc(-attack_cos, -benign_cos),
        "cosine_threshold": cosine_fence,
        "cosine_threshold_holdout": cosine_fence_holdout}

    # ---- the per-row dump an interval script refits on ------------------------------
    # Everything a replicate needs and nothing it would have to recompute: the echo count
    # is already net of the anchor (the subtraction `answer_check` makes at serving), so a
    # consumer never has to re-tokenise a query and cannot make that subtraction its own
    # way. Written in arm order, which is the order `items["benign"]`/`items["attack"]`
    # were built in, so a rebuild fits the same rows in the same order.
    dumped = []
    for arm_rows, flags_for_arm in ((benign, [None] * len(benign)),
                                    (attack, [a["poisoned"] for a in per_attack])):
        for (row, x), flag in zip(arm_rows, flags_for_arm):
            adl, echo = columns(row["anchor"], x)
            dumped.append({
                "arm": row["arm"], "intent_id": row["intent_id"],
                "record_id": row["record_id"], "family": row["family"],
                "base_cos": float(x.base_cos), "words": int(x.words),
                "excess_span": float(x.excess_span),
                "adl_best": (None if adl is None else float(adl)),
                "echo_best": (None if echo is None else int(echo)),
                "poisoned": flag,
                "has_answer": bool(x.answer_loss is not None
                                   and x.echo_tokens is not None),
            })

    report = {
        "policy": policy.fingerprint(),
        "storage_dtype": storage_dtype,
        "embedder": embedder.model_name,
        "budget": budget, "holdout": holdout, "seed": seed,
        "composition": comp,
        "n_genuine": len(benign), "n_attack": len(attack),
        "n_attack_flag_missing": n_flag_missing,
        "n_poisoned": len(pois), "n_poisoned_strict": len(pois_strict),
        # excess
        "excess_auroc": auroc(attack_excess, benign_excess),
        "excess_auroc_cos_matched": matched, "matched_support": support,
        "excess_block_rate": blocked_rate(attack),          # == tpr_all
        "tpr_all_planted": tpr_all,
        "tpr_poisoned": tpr_pois,
        "tpr_poisoned_strict": tpr_strict,
        # calibration
        "fence_form": "flat",
        "threshold_scoring": float(scoring_fence.coefficients[0]),   # fit on all benign
        "threshold_holdout": float(fence.coefficients[0]),           # fit on fit-half
        "benign_block_rate_in_sample": blocked_rate(benign),
        "achieved_benign_block_rate": achieved_block_rate(fence, cal(eval_rows)),
        "n_fit_intents": len(fit_intents), "n_eval_intents": len(eval_intents),
        # cosine baseline (from same run)
        "cosine_auroc": auroc(-attack_cos, -benign_cos),
        "cosine_block_rate": float((attack_cos < cosine_fence).mean()),
        "cosine_threshold": cosine_fence,
        "families": families,
        "per_attack": per_attack,
        "timing": {"embed_wall_s": embed_wall, "n_encode_calls_anchor": n_encode_calls,
                   "n_entries_profiled": len(readings)},
        "fence_scoring": scoring_fence.to_dict(),
        "fence_holdout": fence.to_dict(),
        "answer_rules": answer_rules,
        "answer_rules_pairing": answer_rule_pairing(
            ("tpr_all_planted", "tpr_poisoned", "tpr_poisoned_strict")),
        "answers": {
            "echo_min": echo_min,
            "hashes_loaded": len(answers),
            "rows_joined": n_answer_joined,
            "rows_joined_unusable": n_answer_joined - n_answer_encoded,
            "insertion_answer_encode_calls": n_answer_encoded,
            "scored": {"genuine": {"with_answer": n_loss,
                                   "without_answer": len(benign) - n_loss},
                       "attack": {"with_answer": n_loss_attack,
                                  "without_answer": len(attack) - n_loss_attack}},
        },
    }
    # The guarantee the published tables rest on, checked rather than promised: the
    # DG-only column IS the top-level number, so adding the rule table cannot move it.
    for key in ("excess_block_rate", "tpr_all_planted", "tpr_poisoned",
                "benign_block_rate_in_sample", "achieved_benign_block_rate",
                "excess_block_rate_joint_eta", "tpr_all_planted_joint_eta",
                "tpr_poisoned_joint_eta", "benign_block_rate_in_sample_joint_eta",
                "achieved_benign_block_rate_joint_eta"):
        # Under rule "none" every row fires, so the joint fit is the DG fit over the same
        # rows: both columns must be the published number.
        published = report[key.replace("_joint_eta", "")]
        column = answer_rules["dg_only"][key]
        if column != published and not (np.isnan(column) and np.isnan(published)):
            raise AssertionError(
                f"dg_only column disagrees with the published {key}: "
                f"{column!r} vs {published!r}")
    return report, [{"text": r["text"], "anchor": r["anchor"], "arm": r["arm"],
                     "family": r["family"]} for r, _ in readings], dumped


# --- per-row dumps: refitting the rule outside run_cell ------------------------------
#: The columns ``--dump-rows`` writes, in order. ``adl_best`` is the winning variant's
#: ``answer_loss``; ``echo_best`` is that variant's echo count **already net of the
#: anchor's content words**, which is the subtraction the serving rule makes. Both are
#: null for an entry profiled without an answer, and such a row fires fail-closed exactly
#: as it does at serving.
DUMP_FIELDS = ("arm", "intent_id", "record_id", "family", "base_cos", "words",
               "excess_span", "adl_best", "echo_best", "poisoned", "has_answer")

DumpedRow = namedtuple("DumpedRow", DUMP_FIELDS)


class DumpedColumns:
    """:class:`AnswerColumns` for rows read back from a dump.

    ``AnswerColumns`` subtracts the anchor's content words from the echo set; a dumped row
    carries the result of that subtraction, so this hands the stored pair back unchanged.
    Doing it any other way -- re-tokenising an anchor here, say -- would be a second
    definition of the echo count, which is precisely what the dump exists to prevent.
    """

    def __call__(self, anchor, reading):
        return reading.adl_best, reading.echo_best


def write_dump_rows(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in DUMP_FIELDS}) + "\n")


def load_dumped_rows(path):
    """(benign, attack) as :class:`DumpedRow` lists, each in the order it was written."""
    benign, attack = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = DumpedRow(*(raw[k] for k in DUMP_FIELDS))
        (benign if row.arm == "genuine" else attack).append(row)
    return benign, attack


def rule_rates_from_rows(benign, attack, rule, budget, echo_min=1,
                         embedder="dumped", policy="dumped"):
    """One answer-rule column, refitted from dumped rows.

    Same three calls ``run_cell`` makes, in the same order: fit the flat DG height and
    ``eta_a`` on the benign rows handed in, re-fit the height so the *conjunction* spends
    ``budget`` on those same rows, then read both heights on the attack rows. Restrict
    ``attack`` to the poisoned rows for the Succ. arm -- the decision is per row, so
    filtering before or after is the same number.
    """
    columns = DumpedColumns()
    b_items = [(None, r) for r in benign]
    a_items = [(None, r) for r in attack]
    cal = [CalibrationRow(r.base_cos, r.words, r.excess_span, answer_loss=r.adl_best)
           for r in benign]
    scoring = fit_rule_fence(cal, rule, budget=budget, embedder=embedder, policy=policy,
                             echo_min=echo_min)
    joint, fit = joint_fence(scoring, b_items, columns, budget)
    return {
        "rule": rule, "budget": budget,
        "n_benign": len(benign), "n_attack": len(attack),
        "eta_a": scoring.eta_a,
        "eta_shared": float(scoring.coefficients[0]),
        "eta_joint": float(joint.coefficients[0]),
        "block_rate": float(np.mean(blocks_under(scoring, a_items, columns))),
        "block_rate_joint_eta": float(np.mean(blocks_under(joint, a_items, columns))),
        "benign_block_rate_in_sample": float(np.mean(
            blocks_under(scoring, b_items, columns))),
        "benign_block_rate_in_sample_joint_eta": float(np.mean(
            blocks_under(joint, b_items, columns))),
        "joint_budget_reachable": fit.reachable,
        "n_benign_answer_fires": fit.n_fires,
    }


def cosine_rates_from_rows(benign, attack, budget):
    """The baseline every rate is reported beside: the cache's own similarity floor."""
    b = np.array([r.base_cos for r in benign], dtype=float)
    a = np.array([r.base_cos for r in attack], dtype=float)
    floor = float(np.quantile(b, budget))
    return {"rule": "cosine_only", "budget": budget,
            "n_benign": len(benign), "n_attack": len(attack),
            "cosine_threshold": floor,
            "block_rate": float((a < floor).mean()),
            "block_rate_joint_eta": float((a < floor).mean()),
            "benign_block_rate_in_sample": float((b < floor).mean())}


def judgeable_rows_no_embed(eval_path, attack_roles, benign_generators, policy):
    """The rows excess would score, decided by segment count alone (no embedding)."""
    from sentry.cache.defense.spans import shortened
    rows, _ = load_rows(eval_path, attack_roles, benign_generators)
    keep = []
    for r in rows:
        sp = shortened(policy, r["text"])
        if sp.segment_count >= policy.min_segments and len(sp.span_names) > 0:
            keep.append({"text": r["text"], "anchor": r["anchor"], "arm": r["arm"],
                         "family": r["family"]})
    return keep


def print_rule_table(report):
    """The per-cell rule table, both calibrations side by side.

    Lifted out of ``main`` so it can be exercised on a report built with a toy embedder;
    a formatting bug here would otherwise only show up on the machine that has the
    models.
    """
    # Both calibrations, each beside its own realised benign rate -- which is the
    # column a reader needs to see that the two TPRs were read at the same benign
    # cost. Under `shared eta` the joint rule under-spends the budget by
    # construction, so its TPR is charged for budget left on the table as well as
    # for the check; `joint eta` re-fits the height with the check in place.
    print(f"  {'':<12}{'':>9}|{'shared eta (DG height)':^36}|"
          f"{'joint eta (matched benign cost)':^45}", flush=True)
    print(f"  {'rule':<12}{'eta_a(sc)':>9}|{'TPR_all':>11}{'TPR_pois':>11}"
          f"{'benign(ev)':>13}|{'eta(sc)':>11}{'TPR_all':>11}{'TPR_pois':>11}"
          f"{'benign(ev)':>12}", flush=True)
    for label, column in report["answer_rules"].items():
        if not column.get("fitted", True):
            print(f"  {label:<12}  not fitted: {column['reason']}", flush=True)
            continue
        eta_a = column.get("eta_a")
        if "eta_joint" not in column:            # cosine_only: one threshold only
            print(f"  {label:<12}{'-':>9}|{column['tpr_all_planted']:>11.4f}"
                  f"{column['tpr_poisoned']:>11.4f}"
                  f"{column['achieved_benign_block_rate']:>13.4f}|"
                  f"{'  (already fitted at the budget)':<45}", flush=True)
            continue
        flag = "" if column["joint_budget_reachable"] else "*"
        print(f"  {label:<12}{('-' if eta_a is None else f'{eta_a:.5f}'):>9}|"
              f"{column['tpr_all_planted']:>11.4f}{column['tpr_poisoned']:>11.4f}"
              f"{column['achieved_benign_block_rate']:>13.4f}|"
              f"{column['eta_joint']:>11.5f}"
              f"{column['tpr_all_planted_joint_eta']:>11.4f}"
              f"{column['tpr_poisoned_joint_eta']:>11.4f}"
              f"{column['achieved_benign_block_rate_joint_eta']:>11.4f}{flag:>1}",
              flush=True)
    if any(not c.get("joint_budget_reachable", True)
           for c in report["answer_rules"].values()):
        print("  * the answer check fires on too few benign rows for any height to "
              "spend the budget; that row blocks every row it fires on", flush=True)
    # The ceilings printed here are the scoring fit; the benign(ev) column is not. A row
    # that pairs them is a boundary nobody ran, so say which is which.
    print("  eta_a(sc)/eta(sc) and both TPR columns are the SCORING fit (whole benign "
          "arm); benign(ev) is the same rule's HOLDOUT fit (eta_a_holdout / "
          "threshold_holdout / eta_joint_holdout) read on the held-out intents -- see "
          "answer_rules_pairing in the report.", flush=True)
    cov = report["answer_rules"]["dg_only"]["answer_coverage"]
    if min(cov.values()) < 1.0:
        print(f"  WARNING: answer coverage benign {cov['benign']:.3f} / attack "
              f"{cov['attack']:.3f}; rows without answer fields fire fail-closed, so the "
              f"adl/echo/either columns are that far towards dg_only", flush=True)


def add_ppl_nli(report, scored_rows, budget):
    """Conditional perplexity + bidirectional NLI on the SAME judgeable rows, timed.

    Same signal representatives as rq2_baselines.py (baselines.defense.{perplexity,binli})
    and the same score(anchor, text, 0.0, None) / block_at_budget definitions, so the
    numbers are directly comparable."""
    from experiments.paper.baselines.defense.binli import BiNLI
    from experiments.paper.baselines.defense.perplexity import PerplexityAsymmetry
    signals = {"perplexity": PerplexityAsymmetry(), "binli": BiNLI()}
    out = {}
    for name, sig in signals.items():
        t0 = time.perf_counter()
        scores = {"attack": [], "genuine": []}
        for r in scored_rows:
            s = float(sig.score(r["anchor"], r["text"], 0.0, None))
            scores[r["arm"]].append(s)
        ms = (time.perf_counter() - t0) * 1000 / max(len(scored_rows), 1)
        a = np.array(scores["attack"]); b = np.array(scores["genuine"])
        line = float(np.quantile(b, 1.0 - budget))
        out[name] = {"auroc": auroc(a, b), "block_at_budget": float((a > line).mean()),
                     "ms_per_row": ms, "n_attack": len(a), "n_benign": len(b)}
    report["baselines_ppl_nli"] = out
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    p.add_argument("--text-prefix", default="", help='e.g. "query: " (e5 model card)')
    p.add_argument("--attack-role", action="append", required=True)
    p.add_argument("--benign-generator", action="append", required=True)
    p.add_argument("--flags", required=True)
    p.add_argument("--policy", required=True,
                   help="span policy fingerprint, e.g. "
                        "'multi[count:4+width:2:cap16]/runs'. REQUIRED: "
                        "spans.deployed_policy() and the cut the paper reports have "
                        "drifted apart, so a default here would let a run silently take "
                        "the other one, and cells cut differently cannot go in one table")
    p.add_argument("--storage-dtype", default="float16")
    p.add_argument("--answers", action="append", default=None,
                   help="repeatable: a gen_answers.py output file, or a directory of "
                        "them. Each entry is profiled with the answer written for its "
                        "own text (joined by sha256 of the text), which is what the "
                        "answer_rules table reads. Omit it and the report carries the "
                        "DG-only column alone.")
    p.add_argument("--echo-min", type=int, default=1,
                   help="content words of the removed run that the entry's answer must "
                        "repeat, net of the arriving query's own vocabulary, for the "
                        "'echo' and 'either' rules to second a DG veto")
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--holdout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dump-rows", default=None,
                   help="JSONL, one row per scored entry (both arms) with base_cos, "
                        "words, excess_span, adl_best and echo_best (net of the "
                        "anchor's content words), intent_id and the poisoned flag: what "
                        "an intent-grouped bootstrap needs to refit eta_a, refit the "
                        "joint height and re-read the rule without re-embedding")
    p.add_argument("--with-ppl-nli", action="store_true")
    p.add_argument("--ppl-nli-only", action="store_true",
                   help="skip excess entirely; only ppl+nli on judgeable rows (no embedder)")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)

    if args.skip_existing and Path(args.out).exists():
        print(f"skip existing {args.out}", flush=True)
        return 0

    policy = parse_policy(args.policy)

    if args.ppl_nli_only:
        print(f"=== ppl+nli only | {args.eval} ===", flush=True)
        rows = judgeable_rows_no_embed(args.eval, args.attack_role,
                                       args.benign_generator, policy)
        report = {"eval": args.eval, "policy": policy.fingerprint(),
                  "n_judgeable": len(rows),
                  "n_attack": sum(1 for r in rows if r["arm"] == "attack"),
                  "n_benign": sum(1 for r in rows if r["arm"] == "genuine")}
        report = add_ppl_nli(report, rows, args.budget)
        for k, v in report["baselines_ppl_nli"].items():
            print(f"  {k}: AUROC {v['auroc']:.4f} block {v['block_at_budget']:.4f} "
                  f"{v['ms_per_row']:.2f} ms/row", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)
        return 0

    from sentry.embeddings import TransformerCLSEmbedder
    embedder = TransformerCLSEmbedder(args.encoder, pooling=args.pooling, text_prefix=args.text_prefix)
    flags = load_flags(args.flags)

    answers, answer_files = ({}, [])
    if args.answers:
        answers, answer_files = load_answers(args.answers)
        if not answers:
            # A --answers path that matched nothing is a typo, not an empty result: the
            # run would otherwise report four identical DG columns as an answer-checked
            # table.
            raise SystemExit(f"--answers {args.answers} loaded 0 answered hashes")
        print(f"answers: {len(answers)} hashes from {len(answer_files)} file(s)",
              flush=True)

    print(f"=== {args.encoder} | {args.eval} | dtype={args.storage_dtype} ===", flush=True)
    report, scored_rows, dumped = run_cell(
        args.eval, embedder, policy, flags, args.budget, args.holdout,
        args.seed, args.storage_dtype, args.attack_role, args.benign_generator,
        answers=answers, echo_min=args.echo_min)
    report["answers"]["files"] = answer_files
    print(f"  AUROC {report['excess_auroc']:.4f}  cos-matched {report['excess_auroc_cos_matched']:.4f}"
          f" (support {report['matched_support']})  TPR_all {report['tpr_all_planted']:.4f}"
          f"  TPR_pois {report['tpr_poisoned']:.4f}  realizedFPR {report['achieved_benign_block_rate']:.4f}"
          f"  cosAUROC {report['cosine_auroc']:.4f}", flush=True)
    if args.answers:
        joined = report["answers"]["scored"]
        print(f"  answers joined: genuine {joined['genuine']['with_answer']}/"
              f"{report['n_genuine']}  attack {joined['attack']['with_answer']}/"
              f"{report['n_attack']}  (unusable {report['answers']['rows_joined_unusable']})",
              flush=True)
        print_rule_table(report)
    # per_attack is large; keep it out of the headline json but stash summary already in.
    report.pop("per_attack", None)
    report["pooling"] = args.pooling; report["text_prefix"] = args.text_prefix
    if args.with_ppl_nli:
        report = add_ppl_nli(report, scored_rows, args.budget)
        for k, v in report["baselines_ppl_nli"].items():
            print(f"  {k}: AUROC {v['auroc']:.4f} block {v['block_at_budget']:.4f} "
                  f"{v['ms_per_row']:.2f} ms/row", flush=True)

    if args.dump_rows:
        write_dump_rows(args.dump_rows, dumped)
        print(f"  wrote {len(dumped)} per-row entries to {args.dump_rows}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
