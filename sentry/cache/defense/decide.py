"""The entry-side decision ladder, written once for every host that serves it.

Two caches in this repo run the deletion test at serving:
:class:`~sentry.cache.defense.gptcache_plugin.DeletionVetoEvaluation` in front of a
GPTCache evaluator and runtime
in process. They differ in plumbing -- where the query vector comes from, what a "miss"
is expressed as, what gets written to a trace -- and in nothing else. The verdict itself
is one function, and it lives here rather than being written twice.

That is a correctness property, not tidiness. When the ladder existed twice the two
copies drifted: they ordered their checks differently, only one of them enforced the
``cache_threshold`` floor, and a fail-open hole found and fixed on one host was
reintroduced on the other two commits later, because nothing connected them. A finding
about the decision path now has exactly one place to be fixed.

The order is load-bearing, so it is stated once and the reasons travel with it::

    cos unavailable or < cache_threshold  -> miss
    no profile for this entry             -> miss
    profile/fence fingerprints disagree   -> miss
    entry too short to cut                -> miss (unless unjudgeable_serves)
    excess is NaN or infinite             -> miss
    [record the row in the calibration log]
    cos >= safe_accept                    -> accept, near-duplicate bypass, off by default
    no fence                              -> miss
    excess > fence and answer check clean -> accept (rescued)
    excess > fence(cos, words)            -> miss
    otherwise                             -> accept

**Every unknown reduces to a miss**, never to an undefended serve: a missing profile, a
mismatched fingerprint, a missing fence, an unjudgeable entry and an unreadable excess
all mean "serve only near-duplicates".

**The rescue branch only ever un-blocks.** It sits below the fence, so it is asked only
about candidates ``excess`` already vetoed: the joint rule blocks a subset of what DG
alone blocks, on any population. So it can only *lower* the block rate -- on attack
traffic as much as on benign, since an attacker whose payload leaves the entry's match
to its own cached answer untouched is rescued like any benign entry. The bet is that it
lowers the benign side far more, and that is a measured claim rather than a structural
one. It is not an unknown-swallowing branch either -- an entry with no stored answer, or
a fence carrying no answer rule, keeps the DG verdict (a miss) and the fallback is
counted in ``no_answer_fields``.

**The near-duplicate bypass sits below the fingerprint checks.** Firing it earlier would
serve an entry whose spans were built under a different embedder or a different cut --
precisely the silent failure the fingerprints exist to catch.

**The row is logged before the decision is acted on.** A fence refitted on served traffic
alone censors its own upper tail and ratchets downward every epoch until it blocks
everything, so blocked candidates have to reach the log too.

**Entry side only.** ``profile`` is always the *cached entry's* and ``anchor`` is always the
*arriving query's* vector. The attacker owns the entry -- a payload is planted near a
popular question and ordinary users collide into it afterwards -- so the arriving query
is clean by assumption and profiling it would find nothing. A host that found itself
building a profile for the query at serving has drifted into the query side, which this
measures (``docs/METHOD.md`` §3.1) and deliberately does not deploy.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ._log import warn_once
from .deletion import (
    AnswerVerdict,
    DeletionProfile,
    answer_check,
    excess as read_excess,
)
from .fence import CalibrationRow, ExcessFence


def cosine_from_l2_distance(distance: float) -> float:
    """Cosine between unit vectors from their squared-L2 search distance.

    Vector stores (faiss ``IndexFlatL2`` among them) report the *squared* distance, and
    for unit vectors ``d² = 2(1 − cos)``. Clipped to ``[-1, 1]`` because a store's
    arithmetic can land marginally outside it.
    """
    return float(np.clip(1.0 - float(distance) / 2.0, -1.0, 1.0))


def l2_distance_from_cosine(cosine: float) -> float:
    """Inverse of :func:`cosine_from_l2_distance`."""
    return float(2.0 * (1.0 - float(cosine)))


@dataclass(frozen=True)
class DeletionDefenseConfig:
    """Serving policy. Thresholds are cosines, not distances."""

    #: Below this cosine nothing is served, whatever the excess says.
    #:
    #: The host is supposed to reject these on distance, but hosts are configured by
    #: other people: GPTCache's stock ``SearchDistanceEvaluation`` scores a cosine of
    #: 0.5 well above its default hit threshold. Enforcing the floor here keeps the
    #: component correct regardless of how the cache around it is tuned.
    cache_threshold: float = 0.90
    #: At or above this cosine the pair bypasses the deletion test entirely.
    #:
    #: Defaults to 1.0, which means "effectively never". An earlier design set the
    #: equivalent knob to 0.97 to save work on near-duplicates, and measuring it end to
    #: end showed that was a bad trade: the bypass let 35% of in-band attacks through
    #: untested, moving evasion from 22.2% to 50.3%, for two points of hit rate. **That
    #: measurement was made against the paraphrase-subspace residual, not against this
    #: statistic**, so it is suggestive here rather than established. Raise it only with
    #: a measurement of your own.
    safe_accept: float = 1.0
    #: Calibration rows at or above this cosine are marked as poisoning-resistant
    #: anchors. Deliberately independent of ``safe_accept``: these are near-duplicates
    #: an attacker can only produce by genuinely reaching this cosine, and that stays
    #: true whether or not they also bypass the test.
    anchor_cosine: float = 0.97
    #: Which reading to serve on. ``excess_del`` is available for measurement; it is not
    #: what ``docs/METHOD.md`` §1 defines, and no reported number uses it.
    statistic: str = "excess_span"
    #: Whether an entry too short to cut may still be served. False by default, per the
    #: fail-closed rule; the population is texts of fewer than three words.
    unjudgeable_serves: bool = False
    #: Set False to run the host cache untouched, for A/B comparison.
    enabled: bool = True
    #: Convention used by the host vector store; squared L2 on unit vectors by default.
    cosine_from_distance: Callable[[float], float] = cosine_from_l2_distance


@dataclass
class DeletionDecision:
    """What the defense concluded, for tracing and offline analysis."""

    blocked: bool
    reason: str
    cosine: float
    excess: Optional[float] = None
    threshold: Optional[float] = None
    words: int = 0
    best_end: str = ""
    best_kept: Optional[float] = None
    #: True when the excess was read and came back a number -- i.e. the candidate
    #: reached the statistic rather than being turned away by a gate above it, and the
    #: reading is usable. A non-finite excess leaves this False on purpose: the entry
    #: was scored but nothing was learned, and a trace that called that "checked" would
    #: put a broken vector and a genuine verdict in the same bucket.
    checked: bool = False
    #: The winning variant's answer loss, when the answer check ran and the entry
    #: carried an answer to read.
    answer_loss: Optional[float] = None
    #: Content words the winning variant dropped that the answer repeats and the query
    #: never used; None when the check did not run or had nothing to read.
    echo: Optional[int] = None
    #: What the answer check concluded, verbatim from
    #: :func:`~sentry.cache.defense.deletion.answer_check`; empty when it never ran.
    answer_reason: str = ""


class InMemoryCalibrationLog:
    """Bounded log of every evaluated candidate, for refitting the fence.

    Two tiers, and the distinction is the whole point of the class.

    ``anchor`` marks rows at or above ``anchor_cosine``: benign near-duplicates by
    construction, which an attacker can only pollute by genuinely reaching that cosine
    -- the problem they were trying to avoid. They are the poisoning-resistant part of
    the calibration set, and they are the **only** tier a refit may consume.

    Everything else in the log is *evaluated* traffic, attacks included: blocked
    candidates are recorded deliberately, because a log that saw only served traffic
    would censor its own upper tail and ratchet the boundary down every epoch until it
    blocked everything. That makes the full log the right thing to *analyse* and the
    wrong thing to *fit*.
    """

    def __init__(self, maxlen: int = 200_000) -> None:
        self._rows: deque[tuple[CalibrationRow, bool]] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._rows)

    def record(self, row: CalibrationRow, anchor: bool) -> None:
        self._rows.append((row, anchor))

    def rows(self, anchor_only: bool = False) -> list[CalibrationRow]:
        """Logged rows; ``anchor_only`` keeps the poisoning-resistant tier alone.

        **Never pass the default result to** :meth:`ExcessFence.fit` **or**
        :meth:`ExcessFence.bounded_refit`. Fitting is defined on benign rows only, and
        this log contains every evaluated candidate by design -- an attack sits in the
        upper tail, so a quantile fitted over it rises exactly where the attacks are.
        ``bounded_refit``'s drift bound caps movement per epoch, not across epochs, so a
        patient attacker walks the boundary loose one bounded step at a time. Refit
        through :meth:`DeletionVetoEvaluation.refit_fence`, which takes the anchor tier.
        """
        return [row for row, anchor in self._rows if anchor or not anchor_only]


@dataclass
class DefenseCounters:
    """Cheap observability; all monotone."""

    evaluated: int = 0
    below_band: int = 0
    no_profile: int = 0
    config_mismatch: int = 0
    unjudgeable: int = 0
    #: Readings that came back NaN or infinite -- a broken vector somewhere in the
    #: profile. Counted separately from ``blocked`` because it is a fault report, not a
    #: detection: a rising count means the embedder or the stored bundle needs looking
    #: at, and nothing here is evidence about attack traffic.
    non_finite: int = 0
    safe_accepted: int = 0
    no_fence: int = 0
    blocked: int = 0
    #: Vetoes DG raised that the entry's own answer did not back, and which were
    #: therefore served. Read against ``blocked``: this is the false-veto budget the
    #: second witness is buying back.
    rescued: int = 0
    #: Vetoes decided DG-only because the entry carried no answer to check -- an entry
    #: written before the answer fields existed, or by a host that stores none. A count
    #: that stays near ``blocked`` means the answer rule is configured but idle.
    no_answer_fields: int = 0
    accepted: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def decide(profile: Optional[DeletionProfile],
           anchor: Optional[np.ndarray],
           cosine: float,
           fence: Optional[ExcessFence],
           config: DeletionDefenseConfig,
           counters: Optional[DefenseCounters] = None,
           log: Optional[InMemoryCalibrationLog] = None,
           query_text: Optional[str] = None) -> DeletionDecision:
    """Run the entry-side ladder for one retrieved pair. Never raises, never fails open.

    ``profile`` describes the **cached entry** and ``anchor`` is the **arriving query's**
    vector; ``cosine`` is how close the two sit, which the host supplies because only the
    host knows what sources it has (the profile's own whole vector, an embedding the
    cache passed through, or its search distance).

    ``query_text`` is the arriving query's text, which the answer check subtracts from
    the echo set: a word the query itself asked about is not a word the query did not
    need. It is optional so that a host which does not have it keeps working -- ``None``
    is read as the empty string, nothing is subtracted, and the check is then strictly
    more likely to uphold the veto. The plugin always passes it.

    ``counters`` and ``log`` are updated in place when given. The log is a real part of
    the contract rather than instrumentation: §4.4 of the design record requires every
    *evaluated* candidate to reach it before the verdict is acted on, so a host that
    omits it cannot refit its fence honestly.
    """
    counters = counters if counters is not None else DefenseCounters()
    counters.evaluated += 1

    if not math.isfinite(cosine) or cosine < config.cache_threshold:
        counters.below_band += 1
        return DeletionDecision(
            blocked=True, reason="below_cache_threshold", cosine=cosine)

    if profile is None or anchor is None:
        counters.no_profile += 1
        return DeletionDecision(blocked=True, reason="no_profile", cosine=cosine)

    if fence is not None:
        try:
            fence.check_profile(profile)
        except ValueError as exc:
            counters.config_mismatch += 1
            warn_once(
                f"config_mismatch:{fence.embedder}|{fence.policy}"
                f"->{profile.embedder}|{profile.policy}",
                "deletion-test defense: refusing every entry it checks -- %s. Until the "
                "fence and the stored profiles agree, every checked lookup misses and "
                "the cache is indistinguishable from a cold one.", exc)
            return DeletionDecision(blocked=True, reason="config_mismatch",
                                    cosine=cosine, words=profile.words)

    if not profile.judgeable:
        if not config.unjudgeable_serves:
            counters.unjudgeable += 1
            return DeletionDecision(blocked=True, reason="unjudgeable", cosine=cosine,
                                    words=profile.words)
        counters.accepted += 1
        return DeletionDecision(blocked=False, reason="unjudgeable_allowed",
                                cosine=cosine, words=profile.words)

    reading = read_excess(profile, anchor)
    value = (reading.excess_span if config.statistic == "excess_span"
             else reading.excess_del)

    # A NaN or infinite excess is not a small reading, it is no reading at all: some
    # variant vector is broken (an embedder that emitted NaN for a near-empty span, an
    # fp overflow, a corrupt persisted bundle). Comparisons against NaN are all False,
    # so falling through to the fence would *serve* the entry -- and serve it under
    # every fence, however strict, with no counter recording that anything happened. It
    # reduces to a miss like every other unknown. The row is not logged either: the
    # fence is fitted on this log, and a value that is not a number cannot inform a
    # boundary.
    if not math.isfinite(value):
        counters.non_finite += 1
        return DeletionDecision(blocked=True, reason="non_finite_excess", cosine=cosine,
                                words=profile.words)

    # Log in-band rows only, and log before deciding: the fence is fitted over the range
    # it is applied to, and a log that saw only served traffic would censor its own
    # upper tail.
    if log is not None:
        log.record(CalibrationRow(cosine=cosine, words=profile.words, excess=value,
                                  answer_loss=reading.answer_loss),
                   anchor=cosine >= config.anchor_cosine)

    if cosine >= config.safe_accept:
        counters.safe_accepted += 1
        counters.accepted += 1
        return DeletionDecision(blocked=False, reason="safe_accept", cosine=cosine,
                                excess=value, words=reading.words,
                                best_end=reading.best_end, best_kept=reading.best_kept,
                                checked=True)

    if fence is None:
        counters.no_fence += 1
        warn_once(
            "no_fence",
            "deletion-test defense: no fence attached, so every checked entry is "
            "refused. This is the fail-closed reading of 'calibrated for nothing'; fit "
            "one with `python -m sentry.cache.defense.calibrate`.")
        return DeletionDecision(blocked=True, reason="no_fence", cosine=cosine,
                                excess=value, words=profile.words, checked=True)

    ceiling = fence.predict(cosine, profile.words)
    # The second witness, asked only about candidates the fence already vetoed. It reads
    # numbers computed at insertion -- no model call and no second pass over the spans.
    if value > ceiling:
        if fence.answer_rule in ("adl", "either") and fence.eta_a is None:
            # A rule that reads a ceiling the fence does not carry. Standing in a 0.0
            # would invent a boundary and rescue or block by accident; the fail-closed
            # reading is that the check fires and the veto stands.
            verdict = AnswerVerdict(True, None, None, "eta_a_missing")
        else:
            # The 0.0 below is never read: it reaches only the ``none`` and ``echo``
            # rules, neither of which consults a loss ceiling.
            verdict = answer_check(reading, query_text or "", fence.answer_rule,
                                   fence.eta_a if fence.eta_a is not None else 0.0,
                                   fence.echo_min)
        if verdict.fires is None and fence.answer_rule != "none":
            counters.no_answer_fields += 1
        if verdict.fires is False:
            counters.rescued += 1
            counters.accepted += 1
            return DeletionDecision(blocked=False, reason="rescued_by_answer", cosine=cosine,
                                    excess=value, threshold=ceiling, words=reading.words,
                                    best_end=reading.best_end, best_kept=reading.best_kept,
                                    checked=True, answer_loss=verdict.answer_loss,
                                    echo=verdict.echo, answer_reason=verdict.reason)
        counters.blocked += 1
        return DeletionDecision(blocked=True, reason="excess_above_fence", cosine=cosine,
                                excess=value, threshold=ceiling, words=reading.words,
                                best_end=reading.best_end, best_kept=reading.best_kept,
                                checked=True, answer_loss=verdict.answer_loss,
                                echo=verdict.echo, answer_reason=verdict.reason)

    counters.accepted += 1
    return DeletionDecision(blocked=False, reason="excess_within_fence", cosine=cosine,
                            excess=value, threshold=ceiling, words=reading.words,
                            best_end=reading.best_end, best_kept=reading.best_kept,
                            checked=True)
