"""The fence: holding a false-block budget, and refusing to be used on the wrong data.

Two families of test. The first is calibration -- a boundary fitted at a 5% budget has
to spend about 5% on held-out benign traffic, including at cosines and lengths the fit
did not see, because that is the only property that makes the number in a paper mean
anything in a deployment.

The second is the mismatch guard, and it is here because the failure it prevents is
silent. A fence fitted under one span policy and applied to another raises nothing,
blocks nothing, and holds no budget -- it just quietly stops being a defense.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentry.cache.defense.deletion import DeletionProfile
from sentry.cache.defense.fence import (
    CalibrationRow,
    ExcessFence,
    achieved_block_rate,
)


def benign_rows(count: int, seed: int = 0) -> list[CalibrationRow]:
    """Benign traffic whose excess grows with the room it has to grow into.

    ``docs/DELETION_TEST.md`` §2: a text already at cosine 0.983 has at most 0.017 of
    room to improve by deletion, one at 0.936 has 0.064. Synthetic rows carry that
    dependence so that a fence which ignored cosine would visibly fail the budget at
    one end of the range.
    """
    rng = np.random.default_rng(seed)
    cosines = rng.uniform(0.90, 0.999, size=count)
    words = rng.integers(5, 40, size=count)
    headroom = 1.0 - cosines
    excess = headroom * rng.uniform(0.0, 0.5, size=count) - 0.002 * np.log(words)
    return [CalibrationRow(float(c), int(w), float(e))
            for c, w, e in zip(cosines, words, excess)]


def test_fit_holds_its_budget_on_held_out_rows():
    fence = ExcessFence.fit(benign_rows(4000, seed=1), budget=0.05)
    achieved = achieved_block_rate(fence, benign_rows(4000, seed=2))
    assert 0.03 <= achieved <= 0.07


def test_fit_holds_its_budget_inside_each_cosine_band():
    """A fence that spends its whole budget at one end is not calibrated, just lucky."""
    fence = ExcessFence.fit(benign_rows(8000, seed=3), budget=0.05)
    held_out = benign_rows(8000, seed=4)
    for low, high in [(0.90, 0.93), (0.93, 0.96), (0.96, 0.999)]:
        band = [r for r in held_out if low <= r.cosine < high]
        assert len(band) > 200, "synthetic band too thin to conclude anything"
        assert 0.01 <= achieved_block_rate(fence, band) <= 0.12


def test_flat_fence_misses_the_budget_at_the_extremes():
    """Why the conditional form is the default, stated as a test rather than a claim."""
    rows = benign_rows(8000, seed=5)
    flat = ExcessFence.fit_flat(rows, budget=0.05)
    held_out = benign_rows(8000, seed=6)
    near = [r for r in held_out if r.cosine >= 0.98]
    far = [r for r in held_out if r.cosine < 0.93]

    assert achieved_block_rate(flat, near) < 0.02      # spends nothing where it is close
    assert achieved_block_rate(flat, far) > 0.10       # over-blocks where it is far


def test_degenerate_calibration_does_not_produce_a_wild_boundary():
    """Every row in one cosine band says nothing about the others; say nothing.

    The minimum-norm least-squares step is what makes this safe: an uninformative
    feature keeps a near-zero coefficient instead of a large one that cancels on the
    training rows and extrapolates absurdly everywhere else.
    """
    rows = [CalibrationRow(0.95, 12, e)
            for e in np.linspace(0.0, 0.02, 200)]
    fence = ExcessFence.fit(rows, budget=0.05)
    heights = [fence.predict(c, w) for c in (0.90, 0.95, 0.99) for w in (5, 20, 60)]
    assert all(-0.5 < h < 0.5 for h in heights), heights


def test_predict_refuses_a_wordless_text():
    fence = ExcessFence.fit(benign_rows(500), budget=0.05)
    with pytest.raises(ValueError):
        fence.predict(0.95, 0)


def test_check_profile_refuses_a_policy_mismatch():
    fence = ExcessFence.fit(benign_rows(500), budget=0.05,
                            embedder="e5", policy="count:6")
    profile = _profile(embedder="e5", policy="width:3")
    with pytest.raises(ValueError, match="policy"):
        fence.check_profile(profile)


def test_check_profile_refuses_an_embedder_mismatch():
    fence = ExcessFence.fit(benign_rows(500), budget=0.05,
                            embedder="e5", policy="count:6")
    with pytest.raises(ValueError, match="embedder"):
        fence.check_profile(_profile(embedder="bge", policy="count:6"))


def test_check_profile_refuses_an_unidentified_fence():
    """A bare ``fit(rows)`` names neither embedder nor policy, so it matches nothing.

    The guard used to wave those through, which meant the red line switched itself off
    in exactly its own failure case: sweep span policies, forget ``--policy``, attach
    the result, and ``count:6`` and ``width:3`` profiles are judged by one boundary.
    """
    fence = ExcessFence.fit(benign_rows(500), budget=0.05)
    with pytest.raises(ValueError, match="unidentified"):
        fence.check_profile(_profile(embedder="e5", policy="count:6"))


def test_check_profile_accepts_a_match():
    fence = ExcessFence.fit(benign_rows(500), budget=0.05,
                            embedder="e5", policy="count:6")
    fence.check_profile(_profile(embedder="e5", policy="count:6"))


def test_a_non_finite_excess_blocks():
    """NaN loses every comparison, so the naive form would serve a broken reading."""
    fence = ExcessFence.fit(benign_rows(500), budget=0.05,
                            embedder="e5", policy="count:6")
    assert fence.blocks(0.95, 12, float("nan"))
    assert fence.blocks(0.95, 12, float("inf"))
    assert not fence.blocks(0.95, 12, -1.0)


def test_the_achieved_block_rate_ignores_non_finite_rows():
    """A broken reading is a measurement fault, not a false block; it is not charged."""
    fence = ExcessFence.fit(benign_rows(500), budget=0.05,
                            embedder="e5", policy="count:6")
    clean = benign_rows(2000, seed=13)
    with_faults = clean + [CalibrationRow(0.95, 12, float("nan")) for _ in range(2000)]

    assert achieved_block_rate(fence, with_faults) == pytest.approx(
        achieved_block_rate(fence, clean))


def test_direction_defaults_to_a2_and_rejects_nonsense():
    assert ExcessFence.fit(benign_rows(500)).direction == "entry"
    with pytest.raises(ValueError):
        ExcessFence.fit(benign_rows(500), direction="sideways")


def test_roundtrips_through_json(tmp_path):
    fence = ExcessFence.fit(benign_rows(500), budget=0.05,
                            embedder="e5", policy="count:6")
    path = tmp_path / "fence.json"
    fence.save(path)
    loaded = ExcessFence.load(path)

    assert loaded.direction == fence.direction
    assert loaded.policy == fence.policy
    assert loaded.embedder == fence.embedder
    assert loaded.statistic == fence.statistic
    assert loaded.predict(0.95, 12) == pytest.approx(fence.predict(0.95, 12), abs=1e-12)


def test_bounded_refit_rejects_a_boundary_that_moved_too_far():
    fence = ExcessFence.fit(benign_rows(2000, seed=7), budget=0.05)
    shifted = [CalibrationRow(r.cosine, r.words, r.excess + 0.5)
               for r in benign_rows(2000, seed=8)]
    refit, rejection = fence.bounded_refit(shifted, max_drift=0.05)

    assert rejection is not None
    assert refit is fence


def test_bounded_refit_accepts_a_boundary_that_barely_moved():
    fence = ExcessFence.fit(benign_rows(4000, seed=9), budget=0.05)
    refit, rejection = fence.bounded_refit(benign_rows(4000, seed=10), max_drift=0.05)

    assert rejection is None
    assert refit is not fence


def test_bounded_refit_keeps_the_form():
    """A flat fence coming back sloped would silently change what is being served."""
    flat = ExcessFence.fit_flat(benign_rows(2000, seed=11), budget=0.05)
    refit, _ = flat.bounded_refit(benign_rows(2000, seed=12), max_drift=1.0)
    assert refit.is_flat


def _profile(embedder: str, policy: str) -> DeletionProfile:
    return DeletionProfile(
        whole=np.zeros(4), spans=np.zeros((2, 4)), deletions=np.zeros((3, 4)),
        span_names=("pre1", "suf1"), deletion_names=("del0", "del1", "del2"),
        words=10, segment_count=3, embedder=embedder, policy=policy, min_segments=3)


def test_a_fence_fitted_before_the_rename_still_loads():
    """Every fence on disk was written with ``direction: "a2"``.

    The labels became ``entry``/``query`` because ``A1``/``A2`` were positional codes
    that said nothing. Refusing the old value would have silently invalidated every
    fitted boundary in ``experiments/paper/results/``, so the old label is accepted
    and canonicalised on the way in.
    """
    fence = ExcessFence.fit(benign_rows(500))
    payload = fence.to_dict()
    payload["direction"] = "a2"
    assert ExcessFence.from_dict(payload).direction == "entry"
    payload["direction"] = "a1"
    assert ExcessFence.from_dict(payload).direction == "query"
    assert ExcessFence(fence.coefficients, 0.05, direction="a2").direction == "entry"


import numpy as np

from sentry.cache.defense.fence import CalibrationRow, ExcessFence


def _rows(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [CalibrationRow(cosine=float(c), words=int(w), excess=float(e), answer_loss=float(a))
            for c, w, e, a in zip(rng.uniform(0.9, 1.0, n), rng.integers(5, 30, n),
                                  rng.normal(0.0, 0.005, n), rng.normal(0.002, 0.003, n))]


def test_fit_flat_learns_eta_a_only_for_answer_rules():
    rows = _rows()
    plain = ExcessFence.fit_flat(rows, budget=0.05, embedder="e", policy="p")
    assert plain.answer_rule == "none" and plain.eta_a is None
    adl = ExcessFence.fit_flat(rows, budget=0.05, embedder="e", policy="p", answer_rule="adl")
    assert adl.eta_a == float(np.quantile([r.answer_loss for r in rows], 0.95))
    echo = ExcessFence.fit_flat(rows, budget=0.05, embedder="e", policy="p", answer_rule="echo",
                                echo_min=2)
    assert echo.eta_a is None and echo.echo_min == 2


def test_blocks_is_conjunctive_and_falls_back_without_answer_fields():
    rows = _rows()
    fence = ExcessFence.fit_flat(rows, budget=0.05, embedder="e", policy="p", answer_rule="either")
    high = fence.predict(0.95, 10) + 0.01
    assert fence.blocks(0.95, 10, high)                                   # no answer fields: DG-only
    assert not fence.blocks(0.95, 10, high, answer_loss=-0.01, echo=0)    # rescued
    assert fence.blocks(0.95, 10, high, answer_loss=fence.eta_a + 1e-3, echo=0)
    assert fence.blocks(0.95, 10, high, answer_loss=-0.01, echo=1)
    assert not fence.blocks(0.95, 10, fence.predict(0.95, 10) - 0.01, answer_loss=1.0, echo=9)


def test_answer_rule_round_trips():
    fence = ExcessFence.fit_flat(_rows(), budget=0.05, embedder="e", policy="p",
                                 answer_rule="adl", echo_min=2)
    again = ExcessFence.from_dict(fence.to_dict())
    assert (again.answer_rule, again.eta_a, again.echo_min) == ("adl", fence.eta_a, 2)
    legacy = ExcessFence.from_dict({k: v for k, v in fence.to_dict().items()
                                    if k not in ("answer_rule", "eta_a", "echo_min")})
    assert legacy.answer_rule == "none"


def test_bounded_refit_keeps_a_conditional_fences_eta_a():
    """``fit`` has no quantile step to relearn ``eta_a`` with, so the refit must carry
    the current one across or the rebuilt fence is an invalid object."""
    fence = ExcessFence.fit(_rows(800, seed=5), budget=0.05, embedder="e", policy="p",
                            answer_rule="adl", eta_a=0.01)
    refit, rejection = fence.bounded_refit(_rows(800, seed=6), max_drift=1.0)

    assert rejection is None
    assert (refit.answer_rule, refit.eta_a, refit.echo_min) == ("adl", 0.01, 1)


def test_a_rule_that_reads_eta_a_blocks_when_the_fence_carries_none():
    """The fence is mutable, so ``eta_a = None`` under ``adl`` is one assignment away.
    A rule that cannot read its own ceiling upholds the veto rather than waiving it."""
    fence = ExcessFence.fit_flat(_rows(), budget=0.05, embedder="e", policy="p",
                                 answer_rule="adl")
    high = fence.predict(0.95, 10) + 0.01
    fence.eta_a = 1.0
    assert not fence.blocks(0.95, 10, high, answer_loss=-1.0, echo=0)
    fence.eta_a = None
    assert fence.blocks(0.95, 10, high, answer_loss=-1.0, echo=0)
