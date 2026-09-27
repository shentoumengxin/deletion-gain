"""The serving veto, and the order it decides in.

The entry side scores the **cached entry** against the **arriving query**. Every test here builds a
profile for the entry and passes the query only as an anchor vector; a test that
profiled the query would be exercising the query side, which is not what ships.

The instrument is ``tests/collision_embedder.CollisionEmbedder`` rather than a
bag-of-words toy. Under mean-pooled one-hot words a text long enough to carry a payload
has already fallen below the cosine the cache retrieves at, so "retrievable" and
"carries a payload" are disjoint and the planted entry has to shrink to a single
appended word -- at which point the winning prefix reconstructs the question exactly and
the test passes for any statistic that notices a strict superset. The tilt geometry puts
a genuine eleven-word payload above the 0.90 floor with the payload straddling segment
boundaries, which is the shape the statistic actually has to discriminate.

The decision-order tests exist because the order is load-bearing, not cosmetic --
notably that the near-duplicate bypass sits *below* the fingerprint checks, so it can
never serve an entry whose spans were built under a different embedder or a different
cut, and that a reading which is not a number is a miss rather than a serve.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentry.cache.defense.deletion import build_profile, excess
from sentry.cache.defense.entry_store import InMemoryProfileStore, text_key
from sentry.cache.defense.fence import CalibrationRow, ExcessFence
from sentry.cache.defense.gptcache_plugin import (
    DeletionDefenseConfig,
    DeletionVetoEvaluation,
    cosine_from_l2_distance,
)
from sentry.cache.defense.spans import SpanPolicy
from tests.collision_embedder import (
    BENIGN_QUERY,
    GENUINE_ENTRY,
    PARAPHRASE_ENTRY,
    PLANTED_ENTRY,
    UNRELATED_QUERY,
    CollisionEmbedder,
    benign_fence as _benign_fence,
)

POLICY = SpanPolicy(n=6)
EMBEDDER_NAME = CollisionEmbedder.model_name


class InnerEvaluation:
    """Stands in for whatever evaluator the deployment already uses."""

    def __init__(self) -> None:
        self.calls = 0

    def range(self):
        return 0.0, 1.0

    def evaluation(self, src_dict, cache_dict, **kwargs):
        self.calls += 1
        return 0.87


def permissive_fence(**overrides) -> ExcessFence:
    """A fence that blocks nothing, so a test can isolate one other branch."""
    rows = [CalibrationRow(0.95, 12, 10.0) for _ in range(50)]
    return ExcessFence.fit_flat(rows, budget=0.05, embedder=EMBEDDER_NAME,
                                policy="count:6", **overrides)


def strict_fence() -> ExcessFence:
    """A fence at zero: any text with something worth dropping is blocked.

    Both the block case and the serve case are run against *this* fence. Running the
    serve case against a permissive one would demonstrate nothing -- it would pass
    against an implementation that returned an arbitrarily large excess for a genuine
    entry, or one with the sign inverted.
    """
    rows = [CalibrationRow(0.95, 12, 0.0) for _ in range(50)]
    return ExcessFence.fit_flat(rows, budget=0.05, embedder=EMBEDDER_NAME,
                                policy="count:6")


def build(entry_text_value=PLANTED_ENTRY, fence=None, config=None, store=None):
    embedder = CollisionEmbedder()
    # ``is not None`` rather than ``or``: an empty store has len 0 and is falsy, so a
    # store a test passed in would be silently swapped for a fresh one.
    store = store if store is not None else InMemoryProfileStore(
        EMBEDDER_NAME, POLICY.fingerprint())
    store.put(text_key(entry_text_value),
              build_profile(entry_text_value, embedder, POLICY))
    inner = InnerEvaluation()
    defense = DeletionVetoEvaluation(inner, store, fence=fence, config=config)
    return defense, inner, embedder, store


def call(defense, embedder, entry_text_value=PLANTED_ENTRY, query=BENIGN_QUERY,
         distance=None, vector_id="v0"):
    query_vector = embedder.encode([query])[0]
    if distance is None:
        entry_vector = embedder.encode([entry_text_value])[0]
        distance = 2.0 * (1.0 - float(entry_vector @ query_vector))
    return defense.evaluation(
        {"question": query, "embedding": query_vector},
        {"question": entry_text_value, "answer": "cached",
         "search_result": (distance, vector_id)})


def test_the_planted_entry_is_retrievable_in_the_first_place():
    """Otherwise "blocked" would be indistinguishable from "never a candidate"."""
    embedder = CollisionEmbedder()
    cosine = float(embedder.encode([PLANTED_ENTRY])[0] @ embedder.encode([BENIGN_QUERY])[0])
    assert cosine >= DeletionDefenseConfig().cache_threshold


def test_a_planted_entry_is_blocked_when_a_benign_query_retrieves_it():
    """The threat model, in one test: the payload is in the entry, not the query."""
    defense, inner, embedder, _ = build(fence=strict_fence())
    score = call(defense, embedder)

    assert score == 0.0
    assert defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_above_fence"
    assert defense.last_decision.best_end == "prefix"
    assert inner.calls == 0


def test_a_genuine_entry_is_served():
    """Against the same strict fence, so this shows separation and not leniency."""
    defense, inner, embedder, _ = build(entry_text_value=GENUINE_ENTRY,
                                        fence=strict_fence())
    score = call(defense, embedder, entry_text_value=GENUINE_ENTRY)

    assert score == 0.87
    assert not defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_within_fence"
    assert inner.calls == 1


def calibrated_fence() -> ExcessFence:
    """The boundary a 5% budget actually produces on this geometry's benign population.

    ``strict_fence`` is the sharper instrument -- a fence at zero shows separation with
    no calibration to hide behind -- but only ``GENUINE_ENTRY`` clears it, and a suite
    that never runs a fitted fence cannot notice one with no benign headroom. Both are
    kept: the strict one for separation, this one for margin. Shared with both hosts'
    tests via ``tests/collision_embedder.py``.
    """
    return _benign_fence(POLICY, embedder=CollisionEmbedder())


@pytest.mark.parametrize("entry", [GENUINE_ENTRY, PARAPHRASE_ENTRY])
def test_ordinary_benign_entries_clear_a_calibrated_fence(entry):
    """A fence with no benign margin is a fence that blocks ordinary traffic.

    ``PARAPHRASE_ENTRY`` -- thirteen words, cosine 0.99991, a plain paraphrase -- was
    blocked by a fence fitted on a corpus of nothing but the cleanest possible text: the
    5% the budget spends was spent on legitimate queries. Asserting both fixtures serve
    makes that a test failure.
    """
    defense, inner, embedder, _ = build(entry_text_value=entry,
                                        fence=calibrated_fence())
    score = call(defense, embedder, entry_text_value=entry)

    assert score == 0.87
    assert defense.last_decision.reason == "excess_within_fence"
    assert inner.calls == 1


def test_the_planted_entry_is_still_blocked_by_that_same_calibrated_fence():
    """Margin has to cut both ways, or it was bought by switching the defense off."""
    defense, inner, embedder, _ = build(fence=calibrated_fence())
    score = call(defense, embedder)

    assert score == 0.0
    assert defense.last_decision.reason == "excess_above_fence"
    assert defense.last_decision.excess > defense.last_decision.threshold
    assert inner.calls == 0


def test_the_excess_reported_matches_the_statistic_module():
    defense, _, embedder, store = build(fence=permissive_fence())
    call(defense, embedder)

    profile = store.get(text_key(PLANTED_ENTRY))
    expected = excess(profile, embedder.encode([BENIGN_QUERY])[0])
    assert defense.last_decision.excess == pytest.approx(expected.excess_span, abs=1e-12)
    assert defense.last_decision.best_end == expected.best_end


def test_below_the_cache_threshold_nothing_is_served():
    defense, inner, embedder, _ = build(fence=permissive_fence())
    score = call(defense, embedder, query=UNRELATED_QUERY)

    assert score == 0.0
    assert defense.last_decision.reason == "below_cache_threshold"
    assert inner.calls == 0


def test_a_missing_profile_fails_closed():
    embedder = CollisionEmbedder()
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    defense = DeletionVetoEvaluation(InnerEvaluation(), store, fence=permissive_fence())
    score = call(defense, embedder)

    assert score == 0.0
    assert defense.last_decision.reason == "no_profile"


def test_a_missing_fence_fails_closed():
    defense, inner, embedder, _ = build(fence=None)
    score = call(defense, embedder)

    assert score == 0.0
    assert defense.last_decision.reason == "no_fence"
    assert inner.calls == 0


def test_a_non_finite_excess_fails_closed():
    """NaN loses every comparison, so the naive path *serves* a broken profile.

    One unusable variant vector -- an embedder that emitted NaN for a near-empty span,
    an overflow, a corrupt persisted bundle -- is the exact class of unknown the
    fail-closed rule names, and it would otherwise be invisible: the entry would serve
    under every fence however strict, and ``counters.accepted`` would tick as if
    nothing had happened.
    """
    embedder = CollisionEmbedder()
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    profile = build_profile(PLANTED_ENTRY, embedder, POLICY)
    profile.spans[0][:] = np.nan
    store.put(text_key(PLANTED_ENTRY), profile)
    defense = DeletionVetoEvaluation(InnerEvaluation(), store, fence=permissive_fence())

    score = call(defense, embedder)

    assert score == 0.0
    assert defense.last_decision.blocked
    assert defense.last_decision.reason == "non_finite_excess"
    assert defense.counters.non_finite == 1
    assert defense.counters.accepted == 0


def test_a_non_finite_excess_is_not_logged_for_calibration():
    """A value that is not a number cannot inform a boundary fitted on this log."""
    embedder = CollisionEmbedder()
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    profile = build_profile(PLANTED_ENTRY, embedder, POLICY)
    profile.spans[0][:] = np.inf
    store.put(text_key(PLANTED_ENTRY), profile)
    defense = DeletionVetoEvaluation(InnerEvaluation(), store, fence=permissive_fence())

    call(defense, embedder)

    assert len(defense.log) == 0


def test_an_unjudgeable_entry_fails_closed_by_default():
    embedder = CollisionEmbedder()
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    short = "capital france"
    store._profiles[text_key(short)] = build_profile(short, embedder, POLICY)
    defense = DeletionVetoEvaluation(InnerEvaluation(), store, fence=permissive_fence())
    score = call(defense, embedder, entry_text_value=short, query=short)

    assert score == 0.0
    assert defense.last_decision.reason == "unjudgeable"


def test_the_safe_accept_bypass_sits_below_the_fingerprint_checks():
    """A bypass above them would serve an entry cut under another policy, untested."""
    embedder = CollisionEmbedder()
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    store.put(text_key(BENIGN_QUERY), build_profile(BENIGN_QUERY, embedder, POLICY))
    mismatched = ExcessFence.fit_flat(
        [CalibrationRow(0.95, 12, 10.0) for _ in range(50)],
        budget=0.05, embedder=EMBEDDER_NAME, policy="width:3")
    defense = DeletionVetoEvaluation(
        InnerEvaluation(), store, fence=mismatched,
        config=DeletionDefenseConfig(safe_accept=0.5))
    score = call(defense, embedder, entry_text_value=BENIGN_QUERY, query=BENIGN_QUERY)

    assert score == 0.0
    assert defense.last_decision.reason == "config_mismatch"


def test_an_unidentified_fence_fails_closed():
    """A fence that names no embedder and no policy cannot be checked against anything."""
    embedder = CollisionEmbedder()
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    store.put(text_key(GENUINE_ENTRY), build_profile(GENUINE_ENTRY, embedder, POLICY))
    anonymous = ExcessFence.fit_flat(
        [CalibrationRow(0.95, 12, 10.0) for _ in range(50)], budget=0.05)
    defense = DeletionVetoEvaluation(InnerEvaluation(), store, fence=anonymous)

    score = call(defense, embedder, entry_text_value=GENUINE_ENTRY)

    assert score == 0.0
    assert defense.last_decision.reason == "config_mismatch"


def test_disabled_forwards_everything_untouched():
    defense, inner, embedder, _ = build(
        fence=strict_fence(), config=DeletionDefenseConfig(enabled=False))
    score = call(defense, embedder)

    assert score == 0.87
    assert inner.calls == 1


def test_an_internal_error_fails_closed():
    class Exploding:
        def get(self, key):
            raise RuntimeError("store is unreachable")

    defense = DeletionVetoEvaluation(InnerEvaluation(), Exploding(),
                                     fence=permissive_fence())
    score = call(defense, CollisionEmbedder())

    assert score == 0.0
    assert defense.last_decision.reason == "internal_error"
    assert defense.counters.errors == 1


def test_a_fence_fitted_for_a1_is_refused_outright():
    """Not a weaker defense -- an arbitrary one, on a different benign population."""
    store = InMemoryProfileStore(EMBEDDER_NAME, POLICY.fingerprint())
    a1 = ExcessFence.fit_flat([CalibrationRow(0.95, 12, 0.0) for _ in range(50)],
                              budget=0.05, direction="query")
    with pytest.raises(ValueError, match="entry"):
        DeletionVetoEvaluation(InnerEvaluation(), store, fence=a1)


def test_blocked_candidates_still_reach_the_calibration_log():
    """Logging only what was served censors the upper tail and ratchets the fence down."""
    defense, _, embedder, _ = build(fence=strict_fence())
    call(defense, embedder)

    assert len(defense.log) == 1


def test_out_of_band_rows_are_not_logged():
    defense, _, embedder, _ = build(fence=permissive_fence())
    call(defense, embedder, query=UNRELATED_QUERY)

    assert len(defense.log) == 0


def test_the_log_keeps_the_anchor_tier_separable_from_evaluated_traffic():
    """The full log holds attacks on purpose; only the anchor tier may be refitted on."""
    defense, _, _, _ = build(fence=strict_fence())
    defense.log.record(CalibrationRow(0.99, 9, 0.0), anchor=True)
    defense.log.record(CalibrationRow(0.94, 40, 5.0), anchor=False)

    assert len(defense.log.rows()) == 2
    assert [row.excess for row in defense.log.rows(anchor_only=True)] == [0.0]


def test_a_refit_reads_the_anchor_tier_and_ignores_the_attack_rows():
    """The obvious refit -- fit on ``log.rows()`` -- walks the boundary up to the attacks."""
    defense, _, _, _ = build(fence=strict_fence())
    for _ in range(250):
        defense.log.record(CalibrationRow(0.99, 9, 0.0), anchor=True)
    for _ in range(250):
        defense.log.record(CalibrationRow(0.94, 40, 5.0), anchor=False)

    assert defense.refit_fence() is None
    assert defense.fence.predict(0.95, 12) == pytest.approx(0.0, abs=1e-6)


def test_a_refit_that_moves_the_boundary_too_far_is_not_adopted():
    defense, _, _, _ = build(fence=strict_fence())
    original = defense.fence
    for _ in range(250):
        defense.log.record(CalibrationRow(0.99, 9, 5.0), anchor=True)

    assert defense.refit_fence(max_drift=0.05) is not None
    assert defense.fence is original


def test_a_refit_refuses_to_fit_on_a_handful_of_rows():
    defense, _, _, _ = build(fence=strict_fence())
    original = defense.fence
    for _ in range(5):
        defense.log.record(CalibrationRow(0.99, 9, 0.0), anchor=True)

    assert "anchor rows" in defense.refit_fence()
    assert defense.fence is original


def test_range_is_delegated_so_the_hosts_threshold_keeps_its_meaning():
    defense, inner, _, _ = build(fence=permissive_fence())
    assert defense.range() == inner.range()


def test_cosine_from_l2_distance_inverts_the_convention():
    assert cosine_from_l2_distance(2.0 * (1.0 - 0.94)) == pytest.approx(0.94)


from sentry.cache.defense.calibrate import parse_policy
from sentry.cache.defense.decide import (
    DefenseCounters,
    DeletionDefenseConfig,
    InMemoryCalibrationLog,
    decide,
)
from sentry.cache.defense.deletion import build_profile
from sentry.cache.defense.fence import CalibrationRow, ExcessFence
from tests.test_deletion_statistic import ToyEmbedder

_POLICY = parse_policy("multi[count:4+width:2:cap16]/runs")


def _fence(rule, emb, eta_a=0.0):
    rows = [CalibrationRow(cosine=0.95, words=10, excess=-0.01 + i * 1e-4, answer_loss=0.0)
            for i in range(20)]
    fence = ExcessFence.fit_flat(rows, budget=0.05, embedder=emb.model_name,
                                 policy=_POLICY.fingerprint(), answer_rule=rule)
    if rule in ("adl", "either"):
        fence.eta_a = eta_a
    return fence


def _forced_veto(emb, answer):
    # a profile whose excess is certainly above a fence fitted near -0.01
    text = "please tell me who was the first president born in massachusetts"
    return build_profile(text, emb, _POLICY, answer=answer), emb.encode(
        ["who was the first president born in massachusetts"])[0]


def test_rescue_when_answer_check_is_clean():
    emb = ToyEmbedder()
    profile, anchor = _forced_veto(emb, answer="John Adams was born in Braintree.")
    counters = DefenseCounters()
    decision = decide(profile, anchor, 0.95, _fence("echo", emb), DeletionDefenseConfig(),
                      counters=counters, query_text="who was the first president born in massachusetts")
    assert decision.reason == "rescued_by_answer" and not decision.blocked
    assert decision.echo == 0 and counters.rescued == 1


def test_veto_kept_when_answer_echoes_removed_words():
    emb = ToyEmbedder()
    profile, anchor = _forced_veto(emb, answer="Please, I tell you: John Adams.")
    decision = decide(profile, anchor, 0.95, _fence("echo", emb), DeletionDefenseConfig(),
                      query_text="who was the first president born in massachusetts")
    assert decision.blocked and decision.reason == "excess_above_fence"
    assert decision.echo >= 1


def test_no_answer_fields_falls_back_to_dg_only_and_is_counted():
    emb = ToyEmbedder()
    profile, anchor = _forced_veto(emb, answer=None)
    counters = DefenseCounters()
    decision = decide(profile, anchor, 0.95, _fence("either", emb), DeletionDefenseConfig(),
                      counters=counters, query_text="who was the first president born in massachusetts")
    assert decision.blocked and decision.answer_reason == "no_answer_fields"
    assert counters.no_answer_fields == 1


def test_rule_none_is_the_old_behaviour():
    emb = ToyEmbedder()
    profile, anchor = _forced_veto(emb, answer="anything")
    decision = decide(profile, anchor, 0.95, _fence("none", emb), DeletionDefenseConfig(),
                      query_text="who was the first president born in massachusetts")
    assert decision.blocked and decision.reason == "excess_above_fence"


def test_logged_rows_carry_the_answer_loss_a_refit_needs():
    """A fence whose rule reads ``eta_a`` is refitted from the log it filled itself, so
    the row has to carry the loss or the refit raises instead of returning a reason."""
    emb = ToyEmbedder()
    profile, anchor = _forced_veto(emb, answer="John Adams was born in Braintree.")
    fence = _fence("either", emb)
    log = InMemoryCalibrationLog()
    for _ in range(50):
        decide(profile, anchor, 0.95, fence, DeletionDefenseConfig(), log=log,
               query_text="who was the first president born in massachusetts")

    rows = log.rows()
    assert rows and all(r.answer_loss is not None for r in rows)
    refit, rejection = fence.bounded_refit(rows, max_drift=1.0)
    assert rejection is None
    assert refit.answer_rule == "either" and refit.eta_a is not None


def test_a_fence_missing_its_eta_a_blocks_rather_than_rescues():
    emb = ToyEmbedder()
    profile, anchor = _forced_veto(emb, answer="John Adams was born in Braintree.")
    fence = _fence("either", emb, eta_a=1.0)
    query = "who was the first president born in massachusetts"
    assert decide(profile, anchor, 0.95, fence, DeletionDefenseConfig(),
                  query_text=query).reason == "rescued_by_answer"

    fence.eta_a = None
    decision = decide(profile, anchor, 0.95, fence, DeletionDefenseConfig(),
                      query_text=query)
    assert decision.blocked and decision.answer_reason == "eta_a_missing"
