"""Fit an entry-side fence, and sweep the span policy, from a records file.

The point of this module is that the offline sweep and the online path share
``spans.py`` and ``deletion.py`` *by construction*. Under the previous method the
offline analysis and the serving path each computed their own version of the statistic,
and the way they drifted apart was that nobody could run both from one place. Sweeping
the span policy is a supported operation here, not an edit to a research script.

**Entry side only.** The scored text is the cached entry; the anchor is a benign query for
the same ``intent_id`` -- the pairing ``deletion_test.py --mode a2`` builds. The benign
arm is ``canonical``, the genuine entries of the same corpus the attacks target, which
``docs/METHOD.md`` §5 shows is not an optional detail: an out-of-corpus benign
control produced conclusions that did not survive fixing it.

**The ``canonical`` role is not one corpus.** On the real research corpus it holds
``human_comqa``, ``human_qqp``, and ``human_paws`` rows, and the attacks target
``comqa`` only. ``load_pair_rows`` used to take every ``canonical`` row as the benign arm
and relied on qqp/paws having no ``legal`` anchor to drop them -- correctness by
accident of the data layout. It now reports the ``generator`` composition of both arms
(:class:`CorpusComposition`) unconditionally, and ``main`` warns loudly, rather than
proceeding silently, whenever the benign arm spans more than one generator and
``--benign-generator`` was not given to pick one explicitly.

Every report carries the controls the project's red lines require -- a cosine-only
baseline, because a statistic that does not beat similarity used alone has added
nothing, and a support count beside every matched figure, because a matched AUROC built
on few overlapping rows is indicative rather than established. ``evaluate_policy`` also
reports every figure **per attack family** as well as pooled, so a run is directly
comparable to the per-family table in ``docs/METHOD.md`` §3.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .deletion import (
    DEFAULT_STORAGE_DTYPE, STORAGE_DTYPES, Embedder, build_profile, excess,
)
from .fence import FEATURES, CalibrationRow, ExcessFence, achieved_block_rate
from .spans import DEFAULT_MULTI, SpanPolicy, deployed_policy
from .textnorm import content_tokens

#: Roles that become scored entries. ``canonical`` is the benign arm, ``ndss`` the
#: planted one.
_ENTRY_ROLES = {"canonical": "genuine", "ndss": "attack"}
#: Roles a benign anchor query may be drawn from, in order of preference.
#:
#: comqa intents carry ``legal`` — paraphrases this project generated and validated.
#: QQP and PAWS intents instead carry ``benign_query``: the other half of a
#: human-labelled duplicate pair, which is a legitimate rephrasing of the same question
#: and is exactly what the entry side wants as the arriving query. Preferring ``legal``
#: keeps every
#: existing comqa number bit-identical; the fallback is what lets a second corpus be
#: scored at all.
_ANCHOR_ROLES = ("legal", "benign_query")


@dataclass(frozen=True)
class PairRow:
    """One entry to score, and the benign query that anchors it.

    ``family`` is the record's own ``generator`` -- ``human_comqa``/``human_qqp``/
    ``human_paws`` for a genuine entry, ``ndss_matched_*`` for a planted one. It used to
    be hardcoded to the literal string ``"canonical"`` for every genuine row, which threw
    away exactly the distinction ``CorpusComposition`` and ``--benign-generator`` need.
    """

    text: str
    anchor: str
    arm: str
    family: str
    intent_id: str
    #: The entry's own cached answer, when the corpus carries one. It is the second
    #: witness the answer rules read -- how much the entry's match to it falls when the
    #: winning variant's complement goes, and which of the removed words it repeats --
    #: and nothing else consumes it. A corpus written before the answer check existed
    #: has none, and every row then falls back to the deletion gain alone.
    answer: Optional[str] = None


@dataclass(frozen=True)
class CorpusComposition:
    """What ``load_pair_rows`` found, before any ``--benign-generator`` filter.

    ``docs/METHOD.md`` §5 is the reason this exists: an out-of-corpus benign
    control produced conclusions that did not survive fixing it, and the failure mode was
    silent -- the mixed composition was there in the data the whole time, just never
    reported. ``kept`` counts rows that made it into a :class:`PairRow`; ``dropped_no_anchor``
    counts rows of that role whose intent had no distinct ``legal`` anchor and so could
    not be scored at all, previously invisible because the anchor lookup just dropped them.
    """

    kept: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_no_anchor: dict[str, int] = field(default_factory=dict)


def load_pair_rows(path: Path | str,
                   attack_roles: Sequence[str] = ()) -> tuple[list[PairRow], CorpusComposition]:
    """Entries paired with a benign query for their intent, and the corpus's shape.

    An intent with no benign query yields nothing: there would be no anchor, and the
    entry side is defined by the anchor being the arriving user question. That drop is now counted
    rather than silent -- see :class:`CorpusComposition`.

    ``attack_roles`` names extra ``query_role`` values to score as planted entries, on top
    of ``ndss``. The v3 corpora carry one role per attack family (``scp`` for the Wu-style
    templates, ``gcg`` for the key-collision attacks) instead of reusing ``ndss``, and a
    role this function does not know is skipped *silently* -- which would report a rate
    computed over zero attack rows rather than fail. Left empty the behaviour is exactly
    what it was, so every existing number stays bit-identical.
    """
    records = [json.loads(line) for line in
               Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    entry_roles = dict(_ENTRY_ROLES)
    for role in attack_roles:
        entry_roles[role] = "attack"

    anchors: dict[str, str] = {}
    for role in _ANCHOR_ROLES:
        for record in records:
            if record.get("query_role") == role:
                anchors.setdefault(record["intent_id"], record["text"])

    rows: list[PairRow] = []
    kept: dict[str, Counter] = {"genuine": Counter(), "attack": Counter()}
    dropped: dict[str, int] = {"genuine": 0, "attack": 0}
    for record in records:
        arm = entry_roles.get(record.get("query_role"))
        if arm is None:
            continue
        generator = record.get("generator", "unknown")
        anchor = anchors.get(record["intent_id"])
        if anchor is None or anchor == record["text"]:
            dropped[arm] += 1
            continue
        rows.append(PairRow(
            text=record["text"], anchor=anchor, arm=arm,
            family=generator, intent_id=record["intent_id"],
            answer=record.get("answer")))
        kept[arm][generator] += 1
    composition = CorpusComposition(
        kept={arm: dict(counts) for arm, counts in kept.items()},
        dropped_no_anchor=dropped)
    return rows, composition


def parse_policy(spec: str) -> SpanPolicy:
    """``"count:6"``, ``"width:3"``, ``"width:3:cap12"``, ``"multi"``, or any of those
    with ``"/runs"`` appended.

    ``/runs`` selects §1's definition as written -- the max over *every* contiguous
    sub-span, not only prefixes and suffixes. ``multi`` unions a coarse and a fine cut so
    that a payload shorter than one coarse span still gets isolated. Both were narrowings
    in the deployed code rather than choices in the method; see ``spans._FORMS`` and
    ``spans.DEFAULT_MULTI``.
    """
    form = "span"
    if spec.endswith("/runs"):
        spec, form = spec[: -len("/runs")], "runs"
    elif spec.endswith("/span"):
        spec = spec[: -len("/span")]
    if spec == "multi":
        return SpanPolicy(mode="multi", components=DEFAULT_MULTI, form=form)
    if spec.startswith("multi["):
        inner = spec[len("multi["):].rstrip("]")
        return SpanPolicy(mode="multi", components=tuple(inner.split("+")), form=form)
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"policy {spec!r} must look like 'count:6' or 'width:3'")
    mode, value = parts[0], parts[1]
    if mode == "count":
        return SpanPolicy(mode="count", n=int(value), form=form)
    if mode == "width":
        cap = None
        if len(parts) > 2:
            if not parts[2].startswith("cap"):
                raise ValueError(f"expected 'cap<N>' in {spec!r}, got {parts[2]!r}")
            cap = int(parts[2][3:])
        return SpanPolicy(mode="width", width=int(value), max_segments=cap,
                          form=form)
    raise ValueError(f"unknown policy mode {mode!r}; expected 'count' or 'width'")


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUROC with tie correction. Parity with ``deletion_test.py:auroc``."""
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def matched_auroc(positive: np.ndarray, negative: np.ndarray,
                  positive_field: np.ndarray, negative_field: np.ndarray,
                  width: float, min_per_bin: int = 20) -> tuple[float, int]:
    """AUROC pooled over bins of a matching field, with the support it rests on.

    ``excess`` is a difference of cosines, and a text that starts further from its
    anchor has more room to improve by deletion. Comparing the arms only inside a
    cosine band is what makes the separation attributable to the deletion rather than
    to distance. Bins holding fewer than ``min_per_bin`` rows on either side are
    dropped, and the returned support is how many positives survived -- a matched
    figure without it is indicative, not established.
    """
    if not len(positive) or not len(negative):
        return float("nan"), 0
    low = min(positive_field.min(), negative_field.min())
    high = max(positive_field.max(), negative_field.max())
    areas, weights = [], []
    edge = low
    while edge < high:
        inside_p = positive[(positive_field >= edge) & (positive_field < edge + width)]
        inside_n = negative[(negative_field >= edge) & (negative_field < edge + width)]
        edge += width
        if len(inside_p) < min_per_bin or len(inside_n) < min_per_bin:
            continue
        areas.append(auroc(inside_p, inside_n))
        weights.append(len(inside_p))
    if not weights:
        return float("nan"), 0
    return float(np.average(areas, weights=weights)), int(sum(weights))


#: Which fit every reported rate was read against. A rule column carries two boundaries
#: of each kind -- one fitted on the whole benign arm (so the block rates do not move
#: with the split seed) and one fitted on the fit half (so a false-block rate can be read
#: out of sample) -- and printing a ceiling from one beside a rate from the other is a
#: paper row describing a boundary nobody ran. The pairing is emitted with every report so
#: the assembly cannot be guessed at: for each rate, the exact keys of the DG height and
#: the answer ceiling it was read against.
ANSWER_RULE_PAIRING = {
    "thresholds": {
        "eta_a": {"role": "answer ceiling", "fitted_on": "benign_arm_all"},
        "eta_a_holdout": {"role": "answer ceiling", "fitted_on": "benign_fit_half"},
        "eta_shared": {"role": "DG height, fitted for DG alone",
                       "fitted_on": "benign_arm_all",
                       "same_value_as": "threshold_scoring"},
        "threshold_scoring": {"role": "DG height, fitted for DG alone",
                              "fitted_on": "benign_arm_all"},
        "threshold_holdout": {"role": "DG height, fitted for DG alone",
                              "fitted_on": "benign_fit_half"},
        "eta_joint": {"role": "DG height, re-fitted so the joint rule spends the budget",
                      "fitted_on": "benign_arm_all"},
        "eta_joint_holdout": {
            "role": "DG height, re-fitted so the joint rule spends the budget",
            "fitted_on": "benign_fit_half"},
    },
    "rates": {
        "excess_block_rate": {
            "eta": "eta_shared", "eta_a": "eta_a", "read_on": "attack_rows"},
        "benign_block_rate_in_sample": {
            "eta": "eta_shared", "eta_a": "eta_a", "read_on": "benign_arm_all"},
        "achieved_benign_block_rate": {
            "eta": "threshold_holdout", "eta_a": "eta_a_holdout",
            "read_on": "benign_eval_half"},
        "excess_block_rate_joint_eta": {
            "eta": "eta_joint", "eta_a": "eta_a", "read_on": "attack_rows"},
        "benign_block_rate_in_sample_joint_eta": {
            "eta": "eta_joint", "eta_a": "eta_a", "read_on": "benign_arm_all"},
        "achieved_benign_block_rate_joint_eta": {
            "eta": "eta_joint_holdout", "eta_a": "eta_a_holdout",
            "read_on": "benign_eval_half"},
    },
    "rates_apply_to": ["dg_only", "adl", "echo", "either"],
    #: The baseline column reads no answer, so it has no ``eta_a`` and no joint fit -- but
    #: it does have the same two fits of its own floor, and for the same reason: reading
    #: the eval half against a floor fitted on the whole arm would put an in-sample rate
    #: in a row of out-of-sample ones.
    "cosine_only": {
        "thresholds": {
            "cosine_threshold": {"role": "cosine floor -- the cache's own similarity "
                                         "rule, used alone",
                                 "fitted_on": "benign_arm_all"},
            "cosine_threshold_holdout": {"role": "cosine floor",
                                         "fitted_on": "benign_fit_half"},
        },
        "rates": {
            "excess_block_rate": {"eta": "cosine_threshold",
                                  "read_on": "attack_rows"},
            "benign_block_rate_in_sample": {"eta": "cosine_threshold",
                                            "read_on": "benign_arm_all"},
            "achieved_benign_block_rate": {"eta": "cosine_threshold_holdout",
                                           "read_on": "benign_eval_half"},
        },
    },
    "rows": {
        "benign_arm_all": "every genuine entry scored in this cell",
        "benign_fit_half": "the genuine entries whose intent_id fell in the fit half",
        "benign_eval_half": "the genuine entries whose intent_id fell in the held-out "
                            "half; never used to fit anything",
        "attack_rows": "every planted entry scored in this cell",
    },
    "note": "The out-of-sample false-block rate belongs to the HOLDOUT fit, not to the "
            "scoring one printed beside it: pair `achieved_benign_block_rate` with "
            "`threshold_holdout`/`eta_a_holdout`, and the block rates with "
            "`eta_shared`/`eta_joint` and `eta_a`.",
}


def answer_rule_pairing(attack_rate_aliases: Sequence[str] = ()) -> dict:
    """:data:`ANSWER_RULE_PAIRING`, plus any extra names a tool gives the attack rate.

    ``v3_detect`` reports the same number as ``excess_block_rate``, ``tpr_all_planted``
    and ``tpr_poisoned``; each alias is paired exactly as the rate it duplicates, and a
    rate with no entry here would be a number whose boundary the report does not name.
    """
    pairing = {key: (dict(value) if isinstance(value, dict) else value)
               for key, value in ANSWER_RULE_PAIRING.items()}
    rates = dict(pairing["rates"])
    cosine = dict(ANSWER_RULE_PAIRING["cosine_only"])
    cosine_rates = dict(cosine["rates"])
    for alias in attack_rate_aliases:
        rates[alias] = dict(ANSWER_RULE_PAIRING["rates"]["excess_block_rate"])
        rates[f"{alias}_joint_eta"] = dict(
            ANSWER_RULE_PAIRING["rates"]["excess_block_rate_joint_eta"])
        # The baseline has no joint fit, so it gets the plain alias only.
        cosine_rates[alias] = dict(cosine["rates"]["excess_block_rate"])
    cosine["rates"] = cosine_rates
    pairing["cosine_only"] = cosine
    pairing["rates"] = rates
    return pairing


#: The rule table's columns, in the order the paper's table reads them: the deletion
#: gain alone, then each way the entry's own answer can second the veto. ``dg_only`` is
#: the published column and must reproduce bit-for-bit, which it does by construction --
#: under ``answer_rule="none"`` :meth:`ExcessFence.blocks` never looks at an answer.
ANSWER_RULE_COLUMNS = (("dg_only", "none"), ("adl", "adl"), ("echo", "echo"),
                       ("either", "either"))


class AnswerColumns:
    """The two numbers an answer rule reads, per (anchor, reading).

    ``answer_loss`` comes off the reading; the echo count is the winning variant's echo
    set **net of the arriving query's own content words**, which is the subtraction
    :func:`~sentry.cache.defense.deletion.answer_check` does at serving -- a word the
    query itself asked about is not a word the query did not need. Both are ``None``
    when the entry was profiled without an answer, and :meth:`ExcessFence.blocks` reads
    that pair as fail-closed.

    The anchor's tokens are memoised because one benign query anchors every entry of its
    intent and the table reads the same rows once per rule.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, frozenset[str]] = {}

    def __call__(self, anchor: str, reading) -> tuple[Optional[float], Optional[int]]:
        if reading.answer_loss is None or reading.echo_tokens is None:
            return None, None
        tokens = self._tokens.get(anchor)
        if tokens is None:
            tokens = self._tokens[anchor] = content_tokens(anchor)
        return float(reading.answer_loss), len(reading.echo_tokens - tokens)


def fit_rule_fence(rows: Sequence[CalibrationRow], rule: str, budget: float,
                   embedder: str, policy: str, echo_min: int = 1,
                   fence_form: str = "flat") -> ExcessFence:
    """The DG boundary of ``fence_form``, carrying the answer rule and its ceiling.

    ``eta_a`` is fitted exactly like the flat DG height: the ``1 - budget`` quantile of
    the winning variant's ``answer_loss`` over the rows handed in. The conditional form
    has no quantile step to learn it with, so the flat fit over the same rows supplies
    it -- one definition of the ceiling, whatever the DG boundary's shape.
    """
    flat = ExcessFence.fit_flat(rows, budget=budget, embedder=embedder, policy=policy,
                                answer_rule=rule, echo_min=echo_min)
    if fence_form == "flat":
        return flat
    return ExcessFence.fit(rows, budget=budget, embedder=embedder, policy=policy,
                           answer_rule=rule, eta_a=flat.eta_a, echo_min=echo_min)


def blocks_under(fence: ExcessFence, items, columns: AnswerColumns) -> list[bool]:
    """Per (anchor, reading): blocked under this fence's conjunction of DG and answer."""
    return [bool(fence.blocks(x.base_cos, x.words, x.excess_span, *columns(anchor, x)))
            for anchor, x in items]


def achieved_under(fence: ExcessFence, items, columns: AnswerColumns) -> float:
    """:func:`achieved_block_rate` with the answer fields attached: same filter, same rule.

    The filter is the shared one -- a non-finite reading is a broken vector rather than
    a false block, and charging it to the boundary's budget would be a measurement fault
    reported as over-blocking.
    """
    usable = [(anchor, x) for anchor, x in items
              if x.words >= 1 and np.isfinite(x.excess_span)]
    if not usable:
        return float("nan")
    return float(np.mean(blocks_under(fence, usable, columns)))


def rule_needs_eta_a(rule: str) -> bool:
    return rule in ("adl", "either")


#: A deletion gain no fitted ceiling can sit above. Handing it to
#: :meth:`ExcessFence.blocks` satisfies the DG half of the conjunction, so what comes
#: back is the *answer* verdict alone. Asking the fence rather than reimplementing the
#: rule table is the point: a third copy of "which witness does this rule read" is a
#: third chance for the tools to disagree about what a rule means.
_ABOVE_ANY_FENCE = 1e12


def answer_fires(fence: ExcessFence, anchor: str, reading,
                 columns: AnswerColumns) -> bool:
    """Would the second witness second a veto on this row, the deletion gain aside?

    Rows without answer fields fire, which is the fail-closed reading ``blocks`` already
    takes, and rule ``"none"`` fires always.
    """
    return bool(fence.blocks(reading.base_cos, reading.words, _ABOVE_ANY_FENCE,
                             *columns(anchor, reading)))


#: What :func:`joint_height` found. ``reachable`` is False when the answer check fires on
#: too few benign rows for any DG height to spend the budget (``budget * n >= k``): the
#: boundary then blocks every row the check fires on, which is the most a conjunction can
#: spend, and saying so beats reporting a quantile as if it had converged. ``n_fires`` is
#: how many of ``n_rows`` the check fired on, so a reader can see how close ``k / n`` came
#: to the budget.
JointFit = namedtuple("JointFit", ["height", "reachable", "n_fires", "n_rows"])


def joint_height(fence: ExcessFence, items, columns: AnswerColumns,
                 budget: float) -> JointFit:
    """The DG height at which this fence's **joint** rule spends ``budget`` on ``items``.

    The rule is a conjunction, so attaching an answer check to a height fitted for the
    deletion gain alone can only *lower* the benign block rate: the boundary then
    under-spends its false-block budget and every TPR read against it is charged for
    budget left on the table rather than for the check. Comparing two rules at
    unequal benign cost is exactly what this project's red line forbids, so the second
    calibration re-fits the height with the check in place.

    Exact, not a search. A row is blocked iff its gain clears the height **and** the
    check fires, so only the rows the check fires on can ever be blocked; spending
    ``budget`` of *all* ``n`` rows means blocking ``budget * n`` of the ``k`` firing ones,
    which is their ``1 - budget * n / k`` quantile. With rule ``"none"`` every row fires,
    ``k == n``, and this is ``np.quantile(excess, 1 - budget)`` -- the same call
    :meth:`ExcessFence.fit_flat` makes, over the same rows, after the same filter. The
    DG-only column is therefore identical under both calibrations by construction rather
    than by luck, which is what the test asserts.

    ``eta_a`` is not re-fitted here: it is fitted on the benign arm first, exactly as the
    flat height is, and the height is then fitted given it.
    """
    usable = [(anchor, x) for anchor, x in items
              if x.words >= 1 and np.isfinite(x.excess_span)]
    if not usable:
        raise ValueError("need at least one usable calibration row for the joint fit")
    firing = [x.excess_span for anchor, x in usable
              if answer_fires(fence, anchor, x, columns)]
    n, k = len(usable), len(firing)
    if k == 0 or budget * n / k >= 1.0:
        # Even blocking every row the check fires on spends less than the budget. The
        # honest boundary is the one that blocks all of them; a quantile here would
        # report a fit that did not happen.
        floor = min(x.excess_span for _, x in usable) - 1.0
        return JointFit(float(floor), False, k, n)
    return JointFit(float(np.quantile(firing, 1.0 - budget * n / k)), True, k, n)


def joint_fence(fence: ExcessFence, items, columns: AnswerColumns,
                budget: float) -> tuple[ExcessFence, JointFit]:
    """``fence`` with its height re-fitted so the joint rule spends the whole budget."""
    fit = joint_height(fence, items, columns, budget)
    rebuilt = replace(
        fence,
        coefficients=np.array([fit.height, *fence.coefficients[1:]]),
        metadata={**fence.metadata, "joint_calibrated": True,
                  "joint_budget_reachable": fit.reachable,
                  "n_answer_fires": fit.n_fires, "n_rows_fitted": fit.n_rows})
    return rebuilt, fit


def _intent_holdout(rows: Sequence[PairRow], holdout: float,
                    seed: int) -> tuple[set[str], set[str]]:
    """Split ``intent_id``s into (fit, eval). Never split rows.

    Two entries for the same intent share an anchor and usually most of their words, so
    a row-level split puts near-duplicates on both sides and reports a false-block rate
    the deployment will never see. This is the project's no-leakage red line, enforced
    here rather than assumed. The assignment is a hash of the intent, so it is
    deterministic and does not depend on row order.
    """
    intents = sorted({row.intent_id for row in rows})
    held: set[str] = set()
    for intent in intents:
        digest = hashlib.sha256(f"{seed}:{intent}".encode("utf-8")).digest()
        if int.from_bytes(digest[:8], "big") / 2 ** 64 < holdout:
            held.add(intent)
    return set(intents) - held, held


def evaluate_policy(rows: Sequence[PairRow], embedder: Embedder, policy: SpanPolicy,
                    budget: float = 0.05, holdout: float = 0.5,
                    seed: int = 0,
                    storage_dtype: str = DEFAULT_STORAGE_DTYPE,
                    fence_form: str = "flat", echo_min: int = 1) -> dict:
    """Fit a fence under ``policy`` and report what it achieves, with the controls.

    The fence is fitted on one half of the *intents* and its achieved false-block rate
    is read on the other half. Reporting an in-sample rate would flatter the fence by
    exactly the amount the quantile fit overfits, which is the number a deployment most
    needs to trust.

    Rows carrying an ``answer`` are profiled with it, and ``answer_rules`` then reports
    what each rule would block beside the deletion gain alone. The DG-only column is the
    same arithmetic as the top-level fields and is asserted equal to them before this
    returns: the published numbers cannot move because a second column was added.
    """
    readings = []
    for row in rows:
        profile = build_profile(row.text, embedder, policy,
                                storage_dtype=storage_dtype, answer=row.answer)
        if not profile.judgeable:
            continue
        anchor = embedder.encode([row.anchor])[0]
        readings.append((row, excess(profile, anchor)))

    benign = [(r, x) for r, x in readings if r.arm == "genuine"]
    attack = [(r, x) for r, x in readings if r.arm == "attack"]
    if len(benign) < 4 or len(attack) < 2:
        raise ValueError(
            f"need both arms to report anything: {len(benign)} genuine, "
            f"{len(attack)} attack rows survived the judgeability filter")

    fit_intents, eval_intents = _intent_holdout(
        [r for r, _ in benign], holdout=holdout, seed=seed)
    fit_rows = [(r, x) for r, x in benign if r.intent_id in fit_intents]
    eval_rows = [(r, x) for r, x in benign if r.intent_id in eval_intents]
    if len(fit_rows) < len(FEATURES) + 1 or not eval_rows:
        raise ValueError(
            f"intent-level split left too little to work with: {len(fit_rows)} fit rows "
            f"over {len(fit_intents)} intents, {len(eval_rows)} eval rows over "
            f"{len(eval_intents)}; lower --holdout or supply more intents")

    # One rule decides everything this function reports: the block rate is measured
    # against the same fence that is fitted and shipped.
    fit_fence = ExcessFence.fit_flat if fence_form == "flat" else ExcessFence.fit
    fence = fit_fence(
        [CalibrationRow(x.base_cos, x.words, x.excess_span) for _, x in fit_rows],
        budget=budget, embedder=embedder.model_name, policy=policy.fingerprint())

    benign_excess = np.array([x.excess_span for _, x in benign])
    attack_excess = np.array([x.excess_span for _, x in attack])
    benign_cos = np.array([x.base_cos for _, x in benign])
    attack_cos = np.array([x.base_cos for _, x in attack])

    matched, support = matched_auroc(
        attack_excess, benign_excess, attack_cos, benign_cos, width=0.01)

    # The cosine-only baseline: the cache already thresholds on similarity, so a lower
    # cosine is the attack-like direction and the fence is a floor rather than a ceiling.
    cosine_fence = float(np.quantile(benign_cos, budget))
    # The boundary the block rates below are read against, in the same form as `fence` but
    # fitted on the whole benign arm so the reported catch rate does not move with the
    # split seed. `fence` itself stays fitted on `fit_rows`, because its job is to have an
    # out-of-sample false-block rate to report.
    scoring_fence = fit_fence(
        [CalibrationRow(x.base_cos, x.words, x.excess_span) for _, x in benign],
        budget=budget, embedder=embedder.model_name, policy=policy.fingerprint())

    def blocked(readings) -> float:
        return float(np.mean([scoring_fence.blocks(x.base_cos, x.words, x.excess_span)
                              for x in readings]))

    # Per attack family, so this is comparable to docs/METHOD.md §3.3, which
    # reports one row per `ndss_matched_*` family rather than one pooled number. Every
    # family is scored against the same benign arm the pooled row uses -- the fence, and
    # therefore what "blocked" means, does not change per family.
    families: dict[str, dict] = {}
    for name in sorted({r.family for r, _ in attack}):
        fam_excess = np.array([x.excess_span for r, x in attack if r.family == name])
        fam_cos = np.array([x.base_cos for r, x in attack if r.family == name])
        fam_matched, fam_support = matched_auroc(
            fam_excess, benign_excess, fam_cos, benign_cos, width=0.01)
        families[name] = {
            "n_attack": len(fam_excess),
            "excess_auroc": auroc(fam_excess, benign_excess),
            "excess_auroc_cos_matched": fam_matched,
            "matched_support": fam_support,
            "excess_block_rate": blocked([x for r, x in attack if r.family == name]),
        }

    # The same rows, read again under each answer rule. Nothing above this line changes:
    # `blocked` and the fences it reads are DG-only, and the rule table refits its own
    # boundary per rule rather than mutating theirs.
    columns = AnswerColumns()
    items = {"attack": [(r.anchor, x) for r, x in attack],
             "benign": [(r.anchor, x) for r, x in benign],
             "fit": [(r.anchor, x) for r, x in fit_rows],
             "eval": [(r.anchor, x) for r, x in eval_rows]}
    n_loss = sum(1 for _, x in items["benign"] if x.answer_loss is not None)
    n_loss_fit = sum(1 for _, x in items["fit"] if x.answer_loss is not None)
    n_loss_attack = sum(1 for _, x in items["attack"] if x.answer_loss is not None)
    coverage = {"benign": n_loss / len(benign), "attack": n_loss_attack / len(attack)}
    answer_rules: dict[str, dict] = {}
    for label, rule in ANSWER_RULE_COLUMNS:
        if rule_needs_eta_a(rule) and min(n_loss, n_loss_fit) == 0:
            # No ceiling can be fitted, and standing one in would invent a boundary.
            answer_rules[label] = {
                "answer_rule": rule, "fitted": False, "answer_coverage": coverage,
                "reason": f"no benign row carries an answer_loss ({n_loss} scoring, "
                          f"{n_loss_fit} fit-half), so eta_a cannot be fitted"}
            continue
        scoring = fit_rule_fence([CalibrationRow(x.base_cos, x.words, x.excess_span,
                                                 answer_loss=columns(a, x)[0])
                                  for a, x in items["benign"]],
                                 rule, budget=budget, embedder=embedder.model_name,
                                 policy=policy.fingerprint(), echo_min=echo_min,
                                 fence_form=fence_form)
        held = fit_rule_fence([CalibrationRow(x.base_cos, x.words, x.excess_span,
                                              answer_loss=columns(a, x)[0])
                               for a, x in items["fit"]],
                              rule, budget=budget, embedder=embedder.model_name,
                              policy=policy.fingerprint(), echo_min=echo_min,
                              fence_form=fence_form)
        # Two calibrations, because a conjunction fitted for one half of itself does not
        # spend its budget. `scoring`/`held` keep the DG-only height -- what happens if
        # the check is bolted onto the deployed fence -- and `*_joint` re-fits the height
        # with the check in place, which is the matched-benign-cost comparison the paper
        # reports. `eta_a` is the same in both: it is fitted first, on the benign arm.
        scoring_joint, joint = joint_fence(scoring, items["benign"], columns, budget)
        held_joint, joint_held = joint_fence(held, items["fit"], columns, budget)
        answer_rules[label] = {
            "answer_rule": rule, "fitted": True, "echo_min": echo_min,
            "eta_a": scoring.eta_a, "eta_a_holdout": held.eta_a,
            "n_answer_loss_rows": n_loss, "n_answer_loss_rows_holdout": n_loss_fit,
            "n_benign_answer_fires": joint.n_fires,
            # A row with no answer fields fires fail-closed, so partial coverage pulls an
            # answer-checked column silently back towards dg_only. The `fitted: false`
            # escape only catches coverage of exactly zero; this is the number that says
            # how much of the column is actually answer-checked.
            "answer_coverage": coverage,
            "threshold_scoring": float(scoring.coefficients[0]),
            "threshold_holdout": float(held.coefficients[0]),
            # --- shared eta: the DG-only height, the check bolted on ------------------
            "excess_block_rate": float(np.mean(
                blocks_under(scoring, items["attack"], columns))),
            "benign_block_rate_in_sample": float(np.mean(
                blocks_under(scoring, items["benign"], columns))),
            "achieved_benign_block_rate": achieved_under(held, items["eval"], columns),
            # --- joint eta: the height re-fitted so the joint rule spends the budget ---
            "eta_shared": float(scoring.coefficients[0]),
            "eta_joint": float(scoring_joint.coefficients[0]),
            "eta_joint_holdout": float(held_joint.coefficients[0]),
            "fence_joint_holdout": held_joint.to_dict(),
            "joint_budget_reachable": joint.reachable,
            "joint_budget_reachable_holdout": joint_held.reachable,
            "excess_block_rate_joint_eta": float(np.mean(
                blocks_under(scoring_joint, items["attack"], columns))),
            "benign_block_rate_in_sample_joint_eta": float(np.mean(
                blocks_under(scoring_joint, items["benign"], columns))),
            "achieved_benign_block_rate_joint_eta": achieved_under(
                held_joint, items["eval"], columns),
            "families": {
                name: {"n_attack": sum(1 for r, _ in attack if r.family == name),
                       "excess_block_rate": float(np.mean(blocks_under(
                           scoring, [(r.anchor, x) for r, x in attack
                                     if r.family == name], columns))),
                       "excess_block_rate_joint_eta": float(np.mean(blocks_under(
                           scoring_joint, [(r.anchor, x) for r, x in attack
                                           if r.family == name], columns)))}
                for name in families},
        }
    # The cosine-only baseline, in the same table so the comparison is beside the rules
    # rather than elsewhere in the file. Its attack rate is the top-level one unchanged;
    # its eval-half rate is read against a floor fitted on the fit half, so the benign
    # column is out of sample the way every other column's is.
    cosine_fence_holdout = float(np.quantile(
        np.array([x.base_cos for _, x in fit_rows]), budget))
    answer_rules["cosine_only"] = {
        "excess_block_rate": float((attack_cos < cosine_fence).mean()),
        "benign_block_rate_in_sample": float(np.mean(
            [x.base_cos < cosine_fence for _, x in benign])),
        "achieved_benign_block_rate": float(np.mean(
            [x.base_cos < cosine_fence_holdout for _, x in eval_rows])),
        "auroc": auroc(-attack_cos, -benign_cos),
        "cosine_threshold": cosine_fence,
        "cosine_threshold_holdout": cosine_fence_holdout}

    report = {
        "policy": policy.fingerprint(),
        "storage_dtype": storage_dtype,
        "embedder": embedder.model_name,
        "n_genuine": len(benign),
        "n_attack": len(attack),
        "excess_auroc": auroc(attack_excess, benign_excess),
        "excess_auroc_cos_matched": matched,
        "matched_support": support,
        "excess_block_rate": blocked([x for _, x in attack]),
        "fence_form": fence_form,
        "benign_block_rate_in_sample": blocked([x for _, x in benign]),
        "families": families,
        "achieved_benign_block_rate": achieved_block_rate(
            fence, [CalibrationRow(x.base_cos, x.words, x.excess_span)
                    for _, x in eval_rows]),
        "n_fit_intents": len(fit_intents),
        "n_eval_intents": len(eval_intents),
        "cosine_auroc": auroc(-attack_cos, -benign_cos),
        "cosine_block_rate": float((attack_cos < cosine_fence).mean()),
        "fence": fence.to_dict(),
        "answer_rules": answer_rules,
        "answer_rules_pairing": answer_rule_pairing(),
        "answers": {
            "genuine": {"with_answer": n_loss, "without_answer": len(benign) - n_loss},
            "attack": {"with_answer": n_loss_attack,
                       "without_answer": len(attack) - n_loss_attack}},
    }
    # The guarantee this file's consumers depend on, checked rather than asserted in
    # prose: adding the rule table cannot move the DG-only numbers, because the DG-only
    # column *is* those numbers.
    for key in ("excess_block_rate", "benign_block_rate_in_sample",
                "achieved_benign_block_rate", "excess_block_rate_joint_eta",
                "benign_block_rate_in_sample_joint_eta",
                "achieved_benign_block_rate_joint_eta"):
        # Under rule "none" every row fires, so the joint fit is the DG fit over the
        # same rows and both columns are the published number.
        published = report[key.replace("_joint_eta", "")]
        column = answer_rules["dg_only"][key]
        if column != published and not (np.isnan(column) and np.isnan(published)):
            raise AssertionError(
                f"dg_only column disagrees with the published {key}: "
                f"{column!r} vs {published!r}")
    return report


def _print_composition(composition: CorpusComposition) -> None:
    """The corpus shape, always -- not implicit in the data. See ``CorpusComposition``."""
    for arm in ("genuine", "attack"):
        counts = composition.kept.get(arm, {})
        parts = ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
        print(f"  {arm}: {parts or '(none)'} "
              f"(dropped_no_anchor={composition.dropped_no_anchor.get(arm, 0)})",
              flush=True)


def filter_benign_generators(rows: list[PairRow], generators: Sequence[str]) -> list[PairRow]:
    """Keep every ``attack`` row; keep a ``genuine`` row only if its family is allowed.

    This is the explicit alternative to relying on the anchor lookup to keep the benign
    arm single-corpus by accident -- see the module docstring and ``CorpusComposition``.
    """
    allowed = set(generators)
    return [row for row in rows if row.arm != "genuine" or row.family in allowed]


def main(argv: Optional[Sequence[str]] = None,
         embedder: Optional[Embedder] = None) -> int:
    """Fit one fence per policy, print the comparison table, write both artefacts.

    ``--out`` produces **two** files per policy, and the split is the point. The fence
    goes to ``fence_<policy>.json`` in exactly the shape :meth:`ExcessFence.load` reads,
    because that file is an input to a running cache: ``scripts/run_attack.py --fence``
    and ``scripts/run_victim_attack.py --fence`` hand it straight to the loader. The
    surrounding measurements -- AUROCs, support counts, block rates, the cosine-only
    baseline -- go to ``report_<policy>.json``, because they are evidence for a reader
    and mean nothing to a serving path. Writing the report under the fence's name, which
    an earlier version did, produced a file that no consumer could load at all.

    ``embedder`` is injectable so this can be exercised without a model download; left
    None it builds the CLS embedder named by ``--embedder``, which is what the offline
    corpora are encoded with.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True,
                        help="validated_records.jsonl with query_role and intent_id")
    parser.add_argument("--embedder", default="intfloat/e5-small-v2")
    parser.add_argument("--attack-role", action="append", default=None,
                        help="repeatable: an extra query_role to score as a planted "
                             "entry, on top of 'ndss'. The v3 corpora label each attack "
                             "family with its own role ('scp', 'gcg'), and an unknown "
                             "role is skipped silently -- which would report a rate over "
                             "zero attack rows instead of failing. Omit to keep the "
                             "historical behaviour exactly.")
    parser.add_argument("--policy", action="append", default=None,
                        help="repeatable: count:6, width:3, width:3:cap12")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--fence-form", choices=("flat", "conditional"), default="flat",
                        help="the decision rule. 'flat' is one height for every pair; "
                             "'conditional' lets the height vary with cosine and log word "
                             "count. Flat is the default because it wins the measured "
                             "comparison: over 40 intent-grouped half splits both hold the "
                             "5%% budget out of sample (0.049 vs 0.051), while flat blocks "
                             "0.870 of ndss against conditional's 0.669 and does it with "
                             "half the variance. The conditional form remains the honest "
                             "comparator when the claim is about what the gain adds "
                             "*beyond* cosine, since attacks sit at lower cosine and a flat "
                             "height therefore re-uses some cosine signal.")
    parser.add_argument("--holdout", type=float, default=0.5,
                        help="fraction of INTENTS held out to read the achieved "
                             "false-block rate on; never a fraction of rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--echo-min", type=int, default=1,
                        help="how many content words of the removed run the entry's own "
                             "answer must repeat -- net of the arriving query's own "
                             "vocabulary -- for the 'echo' and 'either' rules to second "
                             "a DG veto. Read only when a record carries an `answer`.")
    parser.add_argument("--benign-generator", action="append", default=None,
                        help="repeatable: restrict the benign (genuine-entry) arm to "
                             "these `generator` values, e.g. human_comqa. Attacks all "
                             "target one source corpus (comqa); leaving this unset when "
                             "the benign arm spans more than one is how "
                             "docs/METHOD.md §5's out-of-corpus-control failure "
                             "happens again.")
    parser.add_argument("--storage-dtype", default=DEFAULT_STORAGE_DTYPE,
                        choices=sorted(STORAGE_DTYPES),
                        help="precision the stored span vectors are rounded to. The "
                             "default is exact; 'float16' halves an entry's cost and "
                             "moves `excess` by ~1e-4 against a fence near 7e-3. int8 is "
                             "not offered -- its error is the fence's own order.")
    parser.add_argument("--out", default="",
                        help="directory to write fence_<policy>.json (loadable by "
                             "ExcessFence.load) and report_<policy>.json (the "
                             "measurements) per policy")
    args = parser.parse_args(argv)

    if embedder is None:
        from sentry.embeddings import TransformerCLSEmbedder
        embedder = TransformerCLSEmbedder(args.embedder)

    rows, composition = load_pair_rows(args.records, args.attack_role or ())
    print(f"{len(rows)} entry-side rows from {args.records}", flush=True)
    if not composition.kept.get("attack"):
        raise SystemExit(
            f"no attack rows in {args.records}: every planted entry was skipped. "
            f"query_role values scored as attacks are 'ndss' plus "
            f"{list(args.attack_role or ())}. Pass --attack-role for this corpus's "
            f"family (v3 uses 'scp' and 'gcg') rather than reporting a rate over "
            f"zero rows.")
    print("corpus composition (before --benign-generator):", flush=True)
    _print_composition(composition)

    benign_generators = set(composition.kept.get("genuine", {}))
    if args.benign_generator:
        rows = filter_benign_generators(rows, args.benign_generator)
        print(f"  --benign-generator restricted the benign arm to "
              f"{sorted(args.benign_generator)}", flush=True)
    elif len(benign_generators) > 1:
        print(f"WARNING: benign arm spans {len(benign_generators)} source corpora "
              f"{sorted(benign_generators)} -- a benign control drawn from a different "
              f"corpus than the one the attacks target produces numbers that do not "
              f"transfer (docs/METHOD.md §5). Pass --benign-generator to pick "
              f"one, e.g. --benign-generator human_comqa.", flush=True)

    # No --policy means "fit what ships", which is `deployed_policy()` rather than a
    # literal here -- a CLI default that drifts from the deployed one produces a fence
    # nothing can use.
    policies = ([parse_policy(spec) for spec in args.policy] if args.policy
                else [deployed_policy()])

    print(f"  {'policy':<16}{'AUROC':>8}{'cos-matched':>13}{'support':>9}"
          f"{'block@budget':>14}{'benign':>9}{'cosine AUROC':>14}{'cosine block':>14}")
    for policy in policies:
        report = evaluate_policy(rows, embedder, policy, budget=args.budget,
                                 storage_dtype=args.storage_dtype,
                                 holdout=args.holdout, seed=args.seed,
                                 fence_form=args.fence_form, echo_min=args.echo_min)
        print(f"  {report['policy']:<16}{report['excess_auroc']:>8.3f}"
              f"{report['excess_auroc_cos_matched']:>13.3f}"
              f"{report['matched_support']:>9}{report['excess_block_rate']:>14.3f}"
              f"{report['achieved_benign_block_rate']:>9.3f}"
              f"{report['cosine_auroc']:>14.3f}{report['cosine_block_rate']:>14.3f}",
              flush=True)
        if report["families"]:
            print(f"    {'family':<32}{'n':>6}{'AUROC':>8}{'cos-matched':>13}"
                  f"{'support':>9}{'block@budget':>14}", flush=True)
            for name, fam in report["families"].items():
                print(f"    {name:<32}{fam['n_attack']:>6}{fam['excess_auroc']:>8.3f}"
                      f"{fam['excess_auroc_cos_matched']:>13.3f}"
                      f"{fam['matched_support']:>9}{fam['excess_block_rate']:>14.3f}",
                      flush=True)
        # The rule table, printed only when the corpus actually carried answers -- a run
        # over records without them would otherwise print four identical DG columns and
        # invite them to be read as a result.
        if report["answers"]["genuine"]["with_answer"]:
            print(f"    {'rule':<12}{'eta_a(sc)':>10}{'block@shared':>14}"
                  f"{'benign(ev)':>14}{'block@joint':>13}{'benign(ev)':>13}",
                  flush=True)
            for label, column in report["answer_rules"].items():
                if not column.get("fitted", True):
                    print(f"    {label:<12}{'-':>10}{'not fitted: ' + column['reason']}",
                          flush=True)
                    continue
                eta = column.get("eta_a")
                shared = (f"{column['excess_block_rate']:>14.3f}"
                          f"{column['achieved_benign_block_rate']:>14.3f}")
                if "eta_joint" not in column:      # cosine_only: one threshold only
                    print(f"    {label:<12}{'-':>10}{shared}"
                          f"{'   (already fitted at the budget)':<27}", flush=True)
                    continue
                flag = "" if column["joint_budget_reachable"] else "*"
                print(f"    {label:<12}{('-' if eta is None else f'{eta:.5f}'):>10}"
                      f"{shared}"
                      f"{column['excess_block_rate_joint_eta']:>13.3f}"
                      f"{column['achieved_benign_block_rate_joint_eta']:>13.3f}{flag}",
                      flush=True)
            if any(not c.get("joint_budget_reachable", True)
                   for c in report["answer_rules"].values()):
                print("    * the answer check fires on too few benign rows for any "
                      "height to spend the budget; that row blocks every row it fires "
                      "on", flush=True)
            # Which fit each column came from, because the ceilings printed here are the
            # scoring ones and the benign(ev) column is read against the holdout fit.
            print("    eta_a(sc) and both block@ columns are the SCORING fit (whole "
                  "benign arm); benign(ev) is the same rule's HOLDOUT fit "
                  "(eta_a_holdout / threshold_holdout / eta_joint_holdout) read on the "
                  "held-out intents. See answer_rules_pairing in the report.", flush=True)
            cov = report["answer_rules"]["dg_only"]["answer_coverage"]
            if min(cov.values()) < 1.0:
                print(f"    WARNING: answer coverage benign {cov['benign']:.3f} / attack "
                      f"{cov['attack']:.3f}; rows without answer fields fire fail-closed, "
                      f"so the answer-checked columns are that far towards dg_only",
                      flush=True)
        if args.out:
            directory = Path(args.out)
            directory.mkdir(parents=True, exist_ok=True)
            # A fingerprint is an identity, not a filename: `multi[...]` and `/runs`
            # both contain characters a path cannot carry, and `/` silently turned into
            # a directory that did not exist. Map every unsafe character, keeping the
            # fingerprint itself untouched inside the JSON where it is the thing that
            # must match a stored profile exactly.
            name = re.sub(r"[^A-Za-z0-9]+", "_", report["policy"]).strip("_")
            # The fence alone, in the shape `ExcessFence.load` reads. A serving path
            # takes this file; it must not be wrapped in anything.
            (directory / f"fence_{name}.json").write_text(
                json.dumps(report["fence"], indent=2), encoding="utf-8")
            # The measurements around it, for a reader rather than a cache.
            (directory / f"report_{name}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def export_runtime_fence(report: dict, path: Path | str,
                         encoder_signature: dict) -> ExcessFence:
    """Export the held-out-fit *joint* either boundary for the serving runtime.

    This uses the existing report's fit-intent calibration, never refits on traffic,
    and never treats its DG-only height as a joint threshold. The caller must supply
    the exact encoder signature used to produce the report (legacy reports lack it).
    """
    if report.get('fence_form') != 'flat' or report.get('storage_dtype') != 'float16':
        raise ValueError('runtime export requires a flat float16 calibration report')
    if encoder_signature.get('model_name') != report.get('embedder'):
        raise ValueError('encoder signature model_name does not match report')
    column = report.get('answer_rules', {}).get('either', {})
    if not column.get('fitted') or column.get('echo_min') != 1:
        raise ValueError('runtime export requires a fitted either rule with echo_min=1')
    fence = ExcessFence.from_dict(column.get('fence_joint_holdout', report['fence']))
    if 'fence_joint_holdout' not in column:
        # Legacy reports carry only the all-benign DG fence's row count. Do not
        # mislabel that count as the fit-half joint calibration's support.
        fence = replace(fence, n_rows=0, metadata={})
    if fence.direction != 'entry' or not fence.is_flat:
        raise ValueError('runtime export requires an entry-side flat fence')
    result = replace(fence,
        coefficients=np.array([column['eta_joint_holdout'], 0., 0.]),
        answer_rule='either', eta_a=column['eta_a_holdout'], echo_min=1,
        metadata={**fence.metadata, 'joint_calibrated':True,
                  'encoder_signature':dict(encoder_signature),
                  'calibration_partition':'fit_intents',
                  'joint_budget_reachable':column['joint_budget_reachable_holdout'],
                  'n_fit_intents':report['n_fit_intents'],
                  'heldout_benign_block_rate':column['achieved_benign_block_rate_joint_eta']})
    result.save(path)
    return result


def _hit_fold(key: str, seed: int) -> int:
    """Fold of a hit, by its cache key, so one key never sits on both sides."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return digest[0] & 1


def calibrate_from_hits(hits: Sequence[dict], embedder: Embedder, *,
                        budget: float = 0.05, policy: Optional[SpanPolicy] = None,
                        min_cosine: float = 0.90,
                        storage_dtype: str = DEFAULT_STORAGE_DTYPE,
                        seed: int = 0, min_hits: int = 20,
                        metadata: Optional[dict] = None) -> tuple[ExcessFence, dict]:
    """Fit a serving fence from benign cache hits alone.

    Each hit is ``{"query", "key", "answer"}``: an arriving query, the cached entry it
    retrieved, and that entry's stored answer. Every hit must be one the cache should
    serve. No attack examples are needed.

    The fit is the paper's rule. The Deletion Gain height and the Answer Check ceiling
    ``eta_a`` are ``1 - budget`` quantiles over the hits, with the ``either`` rule and
    Echo >= 1, and the height is then re-fitted so the joint rule rejects ``budget`` of
    the hits. Hits below ``min_cosine`` are dropped, because the cache never serves
    them and the fence never sees them.

    Returns ``(fence, report)``. The report carries a two-fold held-out estimate of the
    false-rejection rate, with folds split by cache key.
    """
    policy = policy or deployed_policy()
    signature = getattr(embedder, "signature", None)
    if not isinstance(signature, dict):
        raise ValueError("embedder must expose a `signature` dict (see sentry.embeddings)")
    for index, hit in enumerate(hits):
        for field_name in ("query", "key", "answer"):
            if not isinstance(hit.get(field_name), str) or not hit[field_name].strip():
                raise ValueError(f"hit {index} needs a nonempty string {field_name!r}")

    skipped: Counter = Counter()
    rows = []
    anchors = embedder.encode([hit["query"] for hit in hits]) if hits else []
    for hit, anchor in zip(hits, anchors):
        profile = build_profile(hit["key"], embedder, policy, storage_dtype=storage_dtype,
                                answer=hit["answer"])
        if not profile.judgeable:
            skipped["key_too_short"] += 1
            continue
        reading = excess(profile, anchor)
        if reading.base_cos < min_cosine:
            skipped["below_min_cosine"] += 1
            continue
        if reading.words < 1 or not np.isfinite(reading.excess_span):
            skipped["unreadable"] += 1
            continue
        rows.append((hit["key"], hit["query"], reading))
    if len(rows) < min_hits:
        raise ValueError(f"need at least {min_hits} usable hits, got {len(rows)} "
                         f"(skipped: {dict(skipped)})")

    columns = AnswerColumns()

    def fit(subset):
        items = [(query, reading) for _, query, reading in subset]
        base = fit_rule_fence(
            [CalibrationRow(x.base_cos, x.words, x.excess_span,
                            answer_loss=columns(query, x)[0]) for query, x in items],
            "either", budget=budget, embedder=embedder.model_name,
            policy=policy.fingerprint(), echo_min=1, fence_form="flat")
        return joint_fence(base, items, columns, budget)

    heldout = []
    for fold in (0, 1):
        train = [row for row in rows if _hit_fold(row[0], seed) != fold]
        test = [row for row in rows if _hit_fold(row[0], seed) == fold]
        if len(train) >= min_hits and test:
            fold_fence, _ = fit(train)
            heldout.append(achieved_under(fold_fence, [(q, x) for _, q, x in test], columns))
    fence, joint = fit(rows)
    report = {
        "n_hits": len(hits), "n_used": len(rows), "skipped": dict(skipped),
        "budget": budget, "min_cosine": min_cosine, "policy": policy.fingerprint(),
        "eta": float(fence.coefficients[0]), "eta_a": fence.eta_a,
        "joint_budget_reachable": joint.reachable,
        "in_sample_false_rejection": achieved_under(
            fence, [(q, x) for _, q, x in rows], columns),
        "heldout_false_rejection": float(np.mean(heldout)) if heldout else None,
    }
    fence = replace(fence, metadata={
        **fence.metadata, **(metadata or {}), "joint_calibrated": True,
        "encoder_signature": dict(signature), "calibration_partition": "all_hits",
        "min_cosine": min_cosine,
        "heldout_false_rejection": report["heldout_false_rejection"]})
    return fence, report
