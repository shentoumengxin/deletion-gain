"""Offline checks for the appendix statistics (no model, no network)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.paper.rq1_detection import supp_rules as R


def test_joint_matches_recompute_e2e_asr_decisions():
    rng = np.random.default_rng(0)
    dg, adl, echo = rng.normal(0, .01, 499), rng.normal(0, .01, 499), rng.integers(0, 3, 499)
    eta, eta_a = R.joint_thresholds(dg, adl, echo)
    # Reference arithmetic of paper/data/recompute_e2e_asr.py::decisions.
    ref_eta_a = float(np.quantile(adl, .95))
    fires = (adl > ref_eta_a) | (echo >= 1)
    ref_eta = float(np.quantile(dg[fires], 1 - .05 * len(fires) / fires.sum()))
    assert eta_a == ref_eta_a and eta == ref_eta


def test_joint_when_check_alone_fits_budget():
    dg = np.linspace(-1, 1, 100)
    adl = np.zeros(100)
    echo = np.zeros(100, dtype=int)
    echo[:3] = 1                       # the check fires on 3% < 5%
    eta, _ = R.joint_thresholds(dg, adl, echo)
    assert eta == float('-inf')


def test_ac_only_is_benign_calibrated_and_within_budget():
    rng = np.random.default_rng(1)
    adl, echo = rng.normal(0, .01, 499), rng.integers(0, 4, 499)
    eta_a, e = R.ac_only_thresholds(adl, echo, .05)
    fpr = ((adl > eta_a) | (echo >= e)).mean()
    assert fpr <= .05 + 1e-12 and (echo >= e).mean() <= .05
    # the ADL part spends the remaining room: one more row would exceed the budget
    assert fpr > .05 - 1.5 / 499


def test_discrete_threshold_never_exceeds_budget():
    b = np.array([0] * 300 + [1] * 150 + [2] * 40 + [3] * 9)
    t, fpr = R.discrete_threshold(b, .05)
    assert t == 3 and fpr == pytest.approx(9 / 499)


def test_randomized_br_interpolates_between_operating_points():
    b = np.array([0] * 90 + [1] * 10)      # FPR 0.10 at t=1, 0.00 at t=2
    a = np.array([0] * 20 + [1] * 80)      # BR 0.80 at t=1, 0.00 at t=2
    assert R.randomized_br(b, a, .05) == pytest.approx(.40)


def test_auc_ties_and_envelope_bounds():
    assert R.auc([1, 1], [1, 1]) == .5
    assert R.auc([2, 3], [0, 1]) == 1.0
    rng = np.random.default_rng(2)
    ba, be = rng.normal(0, 1, 200), rng.integers(0, 3, 200)
    aa, ae = rng.normal(1, 1, 200), rng.integers(1, 4, 200)
    env = R.disjunction_envelope_auc(ba, be, aa, ae)
    assert R.auc(aa, ba) - 1e-9 <= env <= 1.0


def test_asr_requires_all_three_events():
    cos = np.array([.95, .95, .85, .95])
    pois = np.array([1, 1, 1, 0], bool)
    blocked = np.array([False, True, False, False])
    assert R.asr(cos, pois, blocked) == (1, 4, .25)


def test_full_key_echo_counts_answer_words_absent_from_query():
    k = "Who wrote Hamlet? Introduce Marlowe."
    y = "**Marlowe** wrote it."
    q = "who is the author of hamlet"
    assert R.full_key_echo(k, y, q) == 2   # {'wrote', 'marlowe'}; 'hamlet' is in q


def test_intent_bootstrap_is_deterministic_and_brackets_point():
    rng = np.random.default_rng(3)
    b = {'intent_id': np.repeat(np.arange(100), 2), 'x': rng.normal(0, 1, 200)}
    a = {'intent_id': np.repeat(np.arange(80), 2), 'x': rng.normal(2, 1, 160)}

    def stat(bb, aa):
        t = np.quantile(bb['x'], .95)
        return float((aa['x'] > t).mean())

    lo, hi = R.intent_bootstrap(b, a, stat, replicates=200, seed='unit')
    assert (lo, hi) == R.intent_bootstrap(b, a, stat, replicates=200, seed='unit')
    assert lo <= stat(b, a) <= hi


_DATA = os.environ.get('SENTRY_DATA_ROOT')


@pytest.mark.skipif(not _DATA, reason='needs SENTRY_DATA_ROOT with runs/paper/perrow')
def test_main_table_ours_row_reproduces():
    path = Path(_DATA) / 'runs/paper/perrow/answer_check_rows/e5-small-v2__lmp.jsonl'
    b, a = R.split_arms([json.loads(l) for l in path.read_text().splitlines()])
    eta, eta_a = R.joint_thresholds(b['excess_span'], b['adl_best'], b['echo_best'])
    assert eta == pytest.approx(0.002645233293430393, abs=1e-15)
    assert eta_a == pytest.approx(0.01965637207031249, abs=1e-15)
    blocked = R.joint_blocks(a, eta, eta_a)
    assert round(float(blocked.mean()), 3) == 0.820
