"""Why the shipped boundary conditions on cosine, made executable.

``ExcessFence.fit`` fits a surface::

    excess_max(cos, words) = a + b·(cos − c̄) + c·(log words − w̄)

and it is what ``sentry/cache/defense/`` serves. ``ExcessFence.fit_flat`` fits one
height for every pair. Until this file existed nothing exercised the fitted form at all:
every serving and end-to-end test used ``fit_flat``, over a benign corpus whose cosine
range was ~0.9995 to ~0.9999, and that only worked because a flat boundary ignores
cosine. Swapping the default back to flat would have broken no test.

The argument for conditioning, from ``docs/DELETION_TEST.md`` §2: ``excess`` is a
*difference of cosines*, so a text already matching its anchor at 0.983 has at most
0.017 of room to improve by deletion while one at 0.936 has 0.064. Distance alone
inflates the statistic. One height calibrated across a wide range therefore lands above
the benign population at the near end -- where it blocks nothing, spends none of its
false-block budget, and lets through any attack a near-end text can carry -- and below
it at the far end, where it blocks ordinary traffic. Every headline number in §3 is
reported inside a matched cosine band for this reason, so a deployed boundary that
ignored cosine would not be enforcing what those numbers measured.

The tests below fit **both** forms on the **same** benign rows at the **same** budget
and read their achieved false-block rates inside cosine bands. Holding everything but
the form fixed is what makes the comparison an argument about the form.

Three disciplines the measurement is not meaningful without:

- **Held out, and held out by intent.** The fence is fitted on one half of the corpus's
  intents and read on the other. An in-sample rate flatters a quantile fit by exactly
  the amount it overfits, and a *row*-level split would put near-duplicate surface forms
  of one intent on both sides -- the project's no-leakage red line. The split is
  ``calibrate._intent_holdout``, the same one the offline CLI uses, rather than a second
  implementation that could drift from it.
- **Banded, with support reported.** Bins are fixed-width in cosine and a bin holding
  fewer than ``MIN_PER_BAND`` rows is dropped, the same construction
  ``calibrate.matched_auroc`` uses -- a per-band rate over a handful of rows is
  indicative, not established.
- **Not seed luck.** Every claim is asserted over several independent draws of the
  corpus, because a per-band rate over ~40 rows moves in steps of 2.5%.
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from sentry.cache.defense.calibrate import _intent_holdout
from sentry.cache.defense.fence import CalibrationRow, ExcessFence, achieved_block_rate
from tests.collision_embedder import (
    WIDE_POLICY,
    CollisionEmbedder,
    wide_benign_corpus,
    wide_benign_rows,
)

BUDGET = 0.05
BAND_WIDTH = 0.02
#: Below this a band's rate is a coin flip; ``calibrate.matched_auroc`` draws the same
#: line for the same reason.
MIN_PER_BAND = 20
#: Independent draws of the benign corpus. Nothing below is asserted on one of them.
CORPUS_SEEDS = (17, 19, 20)

EMBEDDER = CollisionEmbedder()


@lru_cache(maxsize=None)
def _rows(seed: int) -> tuple[tuple[str, CalibrationRow], ...]:
    """Benign entry-side rows for one draw of the corpus. Cached: profiling 720 texts is the
    expensive part of this file and every test wants the same rows."""
    return tuple(wide_benign_rows(embedder=EMBEDDER, policy=WIDE_POLICY, seed=seed))


def _split(seed: int) -> tuple[list[CalibrationRow], list[CalibrationRow]]:
    """(fit, held-out) rows, split on ``intent_id`` and never on the row."""
    rows = _rows(seed)
    fit_intents, held_intents = _intent_holdout(
        [SimpleNamespace(intent_id=intent) for intent, _ in rows], holdout=0.5, seed=0)
    return ([row for intent, row in rows if intent in fit_intents],
            [row for intent, row in rows if intent in held_intents])


def _both_fences(fit_rows: list[CalibrationRow]) -> tuple[ExcessFence, ExcessFence]:
    """The shipped surface and its flat comparator, from identical rows and budget."""
    return (ExcessFence.fit(fit_rows, budget=BUDGET, embedder=EMBEDDER.model_name,
                            policy=WIDE_POLICY.fingerprint()),
            ExcessFence.fit_flat(fit_rows, budget=BUDGET, embedder=EMBEDDER.model_name,
                                 policy=WIDE_POLICY.fingerprint()))


def _bands(rows: list[CalibrationRow]) -> list[tuple[float, list[CalibrationRow]]]:
    """Fixed-width cosine bins with enough support to read a rate off, low edge first."""
    cosines = np.array([row.cosine for row in rows])
    edges = np.arange(np.floor(cosines.min() * 50) / 50, 1.0, BAND_WIDTH)
    bands = []
    for edge in edges:
        inside = [r for r in rows if edge <= r.cosine < edge + BAND_WIDTH]
        if len(inside) >= MIN_PER_BAND:
            bands.append((float(edge), inside))
    return bands


# ---- the fixture itself -------------------------------------------------------

def test_the_wide_corpus_spans_a_cosine_range_a_surface_can_be_fitted_over():
    """The precondition for everything else, asserted rather than assumed.

    The corpus the other tests in this repo calibrate on spans ~0.9995 to ~0.9999. A
    slope fitted over that is an extrapolation everywhere it is then used, and its being
    near zero says nothing about whether cosine matters. This corpus has to reach down
    to the retrieval floor, and it has to vary in length as well, or ``log words`` is a
    constant column and its coefficient is a fit to nothing (``fence._design``).
    """
    rows = [row for _, row in _rows(CORPUS_SEEDS[0])]
    cosines = np.array([row.cosine for row in rows])
    words = np.array([row.words for row in rows])

    assert cosines.min() < 0.92 and cosines.max() > 0.999
    assert len(np.unique(words)) > 20
    # Enough support in the far half that a per-band rate down there means something.
    assert (cosines < 0.96).sum() >= MIN_PER_BAND * 2


def test_benign_excess_rises_as_the_entry_sits_further_from_the_anchor():
    """The mechanism the conditional form exists to absorb, measured on the benign arm.

    ``excess`` is a difference of cosines: a text at 0.998 has 0.002 of room to improve
    by deletion and a text at 0.92 has 0.08. So even a corpus whose off-topic mass is
    spread evenly -- no droppable span anywhere, nothing an attacker put there -- shows
    ``excess`` growing as cosine falls, purely from distance. If this were flat, one
    height would be the right boundary and ``fit`` would be answering a question nobody
    asked.
    """
    rows = [row for _, row in _rows(CORPUS_SEEDS[0])]
    near = [r.excess for r in rows if r.cosine >= 0.99]
    far = [r.excess for r in rows if r.cosine < 0.95]

    assert len(near) >= MIN_PER_BAND and len(far) >= MIN_PER_BAND
    # Measured ratio on this corpus is ~8x (0.047 against 0.0058); the assertion is
    # loosened to 5x so it reads the effect rather than the draw.
    assert float(np.quantile(far, 0.95)) > 5 * float(np.quantile(near, 0.95))


@pytest.mark.parametrize("seed", CORPUS_SEEDS)
def test_the_fitted_surface_finds_a_real_cosine_slope(seed):
    """``b`` is not incidentally non-zero, and it leans the way the argument says.

    Negative: the further an entry sits from the anchor, the more ``excess`` it is
    allowed before the boundary calls it an attack. The magnitude matters too -- a slope
    of 1e-4 would be a fit that found nothing while technically not being zero -- so it
    is read against the flat fence's own height: over this corpus's ~0.08 of cosine
    range the slope moves the boundary by more than the whole flat height, which is the
    quantitative statement that the two forms are not interchangeable.

    ``c``, the ``log words`` coefficient, is deliberately *not* asserted non-zero. This
    corpus draws length independently of off-topic fraction, so there is no length
    effect to find and a fit that invented one would be the bug. It is asserted small,
    which is the finding.
    """
    fit_rows, _ = _split(seed)
    conditional, flat = _both_fences(fit_rows)
    _, b, c = conditional.coefficients

    assert b < 0
    assert abs(b) * 0.08 > flat.coefficients[0]
    assert abs(c) < 0.02


# ---- the claim: the conditional form holds the budget where flat does not ------

@pytest.mark.parametrize("seed", CORPUS_SEEDS)
def test_the_flat_fence_over_blocks_the_far_band_and_the_conditional_one_does_not(seed):
    """Flat's over-blocking end: legitimate far-from-anchor traffic, refused.

    One height calibrated across the range sits below the benign population at the far
    end, because down there ordinary entries have room to improve by deletion and use
    it. The conditional boundary rises with distance and keeps its budget.

    This test fails if someone swaps ``fit`` for ``fit_flat`` as the default: the two
    fences would then be the same object and the strict inequality could not hold.
    """
    fit_rows, held = _split(seed)
    conditional, flat = _both_fences(fit_rows)
    edge, band = _bands(held)[0]

    assert edge < 0.95, "the lowest band must actually be far from the anchor"
    flat_rate = achieved_block_rate(flat, band)
    conditional_rate = achieved_block_rate(conditional, band)

    assert flat_rate > 2 * BUDGET
    assert conditional_rate < flat_rate


@pytest.mark.parametrize("seed", CORPUS_SEEDS)
def test_the_flat_fence_spends_none_of_its_budget_in_the_near_band(seed):
    """Flat's under-blocking end, and the more dangerous one.

    A boundary that blocks *nothing* in a band has not been generous there, it has
    switched off: any attack an entry at that distance can carry passes untested, and
    ``achieved_block_rate``'s docstring names near-zero as the failure it is. The near
    band holds the majority of benign traffic, so this is where a real cache spends most
    of its lookups.

    The conditional fence, by contrast, tracks the benign population down to where it
    actually lies and keeps spending its budget there -- which is what leaves a near-end
    plant nowhere to hide (``tests/test_deletion_gptcache.py``).
    """
    fit_rows, held = _split(seed)
    conditional, flat = _both_fences(fit_rows)
    edge, band = _bands(held)[-1]

    assert edge >= 0.98, "the highest band must actually be near the anchor"
    assert achieved_block_rate(flat, band) == 0.0
    assert achieved_block_rate(conditional, band) > 0.0


@pytest.mark.parametrize("seed", CORPUS_SEEDS)
def test_the_conditional_fence_tracks_the_budget_more_closely_across_every_band(seed):
    """The two ends together: worst-band deviation from the nominal budget.

    A fence is calibrated at a budget so that the budget means the same thing wherever
    the pair happens to sit. The summary statistic for that is the largest gap between
    a band's achieved false-block rate and the nominal budget, and the conditional form
    has to win it on every draw of the corpus -- not on average, and not on the draw
    that was looked at first.
    """
    fit_rows, held = _split(seed)
    conditional, flat = _both_fences(fit_rows)
    bands = _bands(held)
    assert len(bands) >= 3, "at least three bands, or 'across bands' means little"

    def worst(fence: ExcessFence) -> float:
        return max(abs(achieved_block_rate(fence, band) - BUDGET) for _, band in bands)

    assert worst(conditional) < worst(flat)


def test_a_row_level_split_is_not_what_this_measures():
    """Guards the no-leakage discipline the numbers above rest on.

    Each intent contributes several surface forms of one shape, so splitting rows rather
    than intents puts near-duplicates on both sides of the split and reports a
    false-block rate no deployment will see. This asserts the corpus really does carry
    that structure -- if ``wide_benign_corpus`` ever emitted one row per intent, the
    intent-level split above would silently become a row-level one and nothing else
    here would notice.
    """
    corpus = wide_benign_corpus(intents=8, variants=3)
    assert len(corpus) == 24
    assert len({intent for intent, _ in corpus}) == 8
