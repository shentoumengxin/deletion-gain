"""Held-out segmentation selection (sweep_full_rule) never fits or chooses on test rows.

Synthetic tables only: no model, no data files.
"""
import numpy as np
import pytest

from experiments.paper.rq1_detection import supp_rules as R
from experiments.paper.rq3_mechanism import sweep_full_rule as S

N_SPECS = 5


def _arm(rng, ids, benign, n_specs):
    n = len(ids)
    return {'intent': np.array(ids, dtype=object), 'benign': np.full(n, benign),
            'poisoned': (np.zeros(n, bool) if benign else rng.random(n) < .6),
            'cos': rng.uniform(.85, 1.0, n),
            'dg': rng.normal(.02 if not benign else 0, .01, (n, n_specs)),
            'adl': rng.normal(.02 if not benign else 0, .01, (n, n_specs)),
            'echo': rng.integers(0, 3, (n, n_specs)).astype(float),
            'j3': np.ones((n, n_specs), bool), 'j2': np.ones((n, n_specs), bool),
            'n_spans': rng.integers(5, 40, (n, n_specs)).astype(float)}


def _cat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def synthetic(seed=0, n_intents=60):
    rng = np.random.default_rng(seed)
    comqa = [f'cq-{i}' for i in range(n_intents)]
    nq = [f'nq-{i}' for i in range(n_intents)]
    comqa_benign = _arm(rng, comqa, True, N_SPECS)       # CAP and SCP share these rows
    tables = {}
    for c, _, _, pool in S.SETS:
        ids = comqa if pool == 'ComQA' else nq
        benign = comqa_benign if pool == 'ComQA' else _arm(rng, nq, True, N_SPECS)
        tables[c] = _cat(benign, _arm(rng, [i for i in ids for _ in range(2)], False, N_SPECS))
    return tables


def intents_of(tables):
    return sorted({i for t in tables.values() for i in t['intent']})


def perturb(tables, ids, seed):
    """Overwrite every row whose intent is in ``ids`` with fresh random values."""
    rng = np.random.default_rng(seed)
    out = {}
    for c, t in tables.items():
        t = {k: v.copy() for k, v in t.items()}
        m = np.isin(t['intent'], list(ids))
        k = int(m.sum())
        for key in ('dg', 'adl'):
            t[key][m] = rng.normal(0, .05, (k, N_SPECS))
        t['echo'][m] = rng.integers(0, 5, (k, N_SPECS))
        t['n_spans'][m] = rng.integers(1, 500, (k, N_SPECS))
        t['cos'][m] = rng.uniform(.8, 1, k)
        t['poisoned'][m] = rng.random(k) < .5
        out[c] = t
    return out


def test_split_is_disjoint_complete_and_seeded():
    ids = intents_of(synthetic())
    a, b = S.split_intents(ids, 3)
    assert not (a & b) and (a | b) == set(ids) and abs(len(a) - len(b)) <= 1
    assert S.split_intents(ids, 3) == (a, b) and S.split_intents(ids, 4) != (a, b)


@pytest.mark.parametrize('rule', ['joint', 'dg'])
@pytest.mark.parametrize('seed', [0, 1, 7])
def test_test_half_never_reaches_fit_or_choice(rule, seed):
    tables = synthetic()
    intents = intents_of(tables)
    cols, jref = list(range(N_SPECS)), 2
    base = S.run_seed(tables, cols, jref, rule, seed, intents)
    _, test_ids = S.split_intents(intents, seed)
    moved = S.run_seed(perturb(tables, test_ids, 99), cols, jref, rule, seed, intents)
    assert moved['pick'] == base['pick'] and moved['tied'] == base['tied']
    assert moved['sel_worst'] == base['sel_worst']
    assert moved['fits'] == base['fits']
    # the perturbation did reach the test half, so the equality above is not vacuous
    assert any(moved['held'][j]['worst'] != base['held'][j]['worst'] for j in cols)


@pytest.mark.parametrize('rule', ['joint', 'dg'])
def test_selection_half_does_drive_the_choice(rule):
    tables = synthetic()
    intents = intents_of(tables)
    sel_ids, _ = S.split_intents(intents, 0)
    for t in tables.values():                      # make column 4 perfect on the selection half
        m = np.isin(t['intent'], list(sel_ids)) & ~t['benign']
        t['dg'][m, 4] = t['adl'][m, 4] = 1.0
    r = S.run_seed(tables, list(range(N_SPECS)), 2, rule, 0, intents)
    assert r['pick'] == 4 and r['sel_worst'][4] == 1.0


def test_fits_are_selection_half_benign_thresholds_and_test_reads_them():
    tables = synthetic()
    intents = intents_of(tables)
    cols, jref = list(range(N_SPECS)), 2
    r = S.run_seed(tables, cols, jref, 'joint', 5, intents)
    sel_ids, test_ids = S.split_intents(intents, 5)
    for c, t in tables.items():
        b = t['benign'] & np.isin(t['intent'], list(sel_ids))
        exp = R.joint_thresholds(t['dg'][b, jref], t['adl'][b, jref], t['echo'][b, jref])
        assert r['fits'][jref][c] == exp
        tb = t['benign'] & np.isin(t['intent'], list(test_ids))
        x = {'excess_span': t['dg'][tb, jref], 'adl_best': t['adl'][tb, jref],
             'echo_best': t['echo'][tb, jref]}
        fpr = float(R.joint_blocks(x, *exp).mean())
        pool = dict((n, p) for n, _, _, p in S.SETS)[c]
        assert r['held'][jref]['fpr'][pool] == pytest.approx(fpr, abs=0)


def test_unjudgeable_rows_count_as_blocked_in_effective_rates():
    t = synthetic()['CAP']
    t['j3'][:5, 0] = False                         # five benign rows cannot be cut
    att = np.flatnonzero(~t['benign'])[:3]
    t['j3'][att, 0] = False                        # three attack rows cannot be cut
    jud = t['j3'][:, 0]
    out = S.read('joint', t, 0, S.fit('joint', t, 0, jud), jud)
    assert out['n_benign_unjudgeable'] == 5 and out['n_attack_unjudgeable'] == 3
    nb, na = t['benign'].sum(), (~t['benign']).sum()
    assert out['fpr_effective'] == pytest.approx((out['fpr'] * (nb - 5) + 5) / nb)
    assert out['br_effective'] == pytest.approx((out['br'] * (na - 3) + 3) / na)
