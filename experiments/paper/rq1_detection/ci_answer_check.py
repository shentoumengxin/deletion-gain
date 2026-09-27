"""Intent-grouped bootstrap 95% CIs for the answer-checked Deletion Gain cells.

Same discipline as ``ci_table1.py`` -- cluster bootstrap over intents, B=2000, one
generator per cell keyed by (set, encoder, rule, budget, arm) so adding a cell leaves the
others unchanged -- with the one thing the answer check adds: the rule is a *conjunction*,
so a replicate cannot reuse a threshold. Each replicate

  1. refits ``eta_a`` on the resampled benign rows (their ``1 - budget`` answer-loss
     quantile), then
  2. refits the Deletion Gain height so the conjunction spends the whole budget on those
     same rows (``calibrate.joint_height``), then
  3. reads the rule on the resampled attack rows.

Nothing is fitted on attack rows and nothing is fitted on held-out intents.

Input: the per-row dumps ``v3_detect.py --dump-rows`` writes, one file per
(encoder, set), named ``<encoder>__<set>.jsonl`` -- the same naming the cells use. A row
carries ``base_cos``, ``words``, ``excess_span``, the winning variant's ``adl_best`` and
its ``echo_best`` already net of the anchor's content words, plus ``intent_id`` and the
poisoned flag.

Parity, asserted rather than eyeballed: at budget 0.05 every point estimate must equal the
published cell in ``experiments/paper/results/answer_check_20260909/main_table.json``
exactly -- ``dg_only`` against ``excess_block_rate``/``tpr_poisoned`` and ``either``
against the ``*_joint_eta`` pair. The point estimates themselves are produced by the
shipped rule code (``v3_detect.rule_rates_from_rows`` -> ``calibrate.fit_rule_fence`` /
``joint_fence`` / ``blocks_under``); the 2,000 replicates use a vectorised equivalent that
is checked against that code, on the full sample and on the first replicates of every
cell, before any interval is reported.

Every rate is reported beside its cosine-only baseline (the cache's own similarity floor
at the same budget), which is the control this project's red lines require.

Usage:
    python ci_answer_check.py --rows DIR [--out ci_answer_check.json]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

CD_REPO = os.environ.get('CD_REPO', str(REPO_ROOT))
sys.path.insert(0, CD_REPO)
sys.path.insert(0, os.path.join(CD_REPO, 'experiments/paper/rq1_detection'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.paper.rq1_detection import v3_detect  # noqa: E402
from experiments.paper.rq1_detection.boot import B, rng_for, _groups  # noqa: E402  (same generator discipline as Table 1)

#: eval-file stem -> the paper's name for that attack set.
SETS = {'lmp': 'CAP', 'scp': 'SCP', 'kca': 'KCA'}
#: Pooling recipes a dump may name. Only ``cls`` is the published one.
POOLINGS = ('cls', 'mean')
#: The columns the paper reads: Deletion Gain alone, and the answer-checked conjunction.
RULES = (('dg_only', 'none'), ('either', 'either'))
BUDGETS = (0.10, 0.05, 0.02, 0.01)
ARMS = ('All', 'Succ.')
ECHO_MIN = 1


# --- the arrays a replicate is read off ---------------------------------------------

class Arm:
    """One arm of one cell as flat arrays, plus the intent clusters to resample."""

    def __init__(self, rows):
        self.rows = rows
        self.cos = np.array([r.base_cos for r in rows], float)
        self.words = np.array([r.words for r in rows], int)
        self.exc = np.array([r.excess_span for r in rows], float)
        self.adl = np.array([np.nan if r.adl_best is None else r.adl_best for r in rows],
                            float)
        self.echo = np.array([-1 if r.echo_best is None else r.echo_best for r in rows],
                             int)
        self.has = np.array([r.adl_best is not None for r in rows], bool)
        self.intents = np.array([r.intent_id for r in rows])
        self.poisoned = np.array([r.poisoned is True for r in rows], bool)
        # `blocks` reads `predict(cos, words)`, which refuses a text with no words; and a
        # row carrying one answer field but not the other would take the fail-closed
        # branch here and a different one in the dump. Both are corpus faults, not rates.
        assert (self.words >= 1).all(), 'a dumped row has no words'
        assert all((r.adl_best is None) == (r.echo_best is None) for r in rows), \
            'a dumped row carries one answer field but not the other'

    def take(self, idx):
        return _Sub(self, idx)

    def all(self):
        return _Sub(self, np.arange(len(self.rows)))


class _Sub:
    """A resampled (or whole) arm: the four columns the rule reads."""

    def __init__(self, arm, idx):
        self.arm, self.idx = arm, idx
        self.cos = arm.cos[idx]
        self.words = arm.words[idx]
        self.exc = arm.exc[idx]
        self.adl = arm.adl[idx]
        self.echo = arm.echo[idx]
        self.has = arm.has[idx]

    def rows(self):
        return [self.arm.rows[i] for i in self.idx]


def _fires(sub, rule, eta_a):
    """Would the answer second a veto on each row? Rows with no answer fields fire.

    Mirrors ``ExcessFence.blocks`` branch for branch, including the corner where a rule
    that reads ``eta_a`` has none: the fence upholds the veto there rather than waiving
    it, so ``by_loss`` is all-True and not all-False.
    """
    if rule == 'none':
        return np.ones(len(sub.exc), bool)
    by_loss = (np.ones(len(sub.exc), bool) if eta_a is None else sub.adl > eta_a)
    by_echo = sub.echo >= ECHO_MIN
    fired = {'adl': by_loss, 'echo': by_echo, 'either': by_loss | by_echo}[rule]
    return (~sub.has) | (sub.has & fired)


def _fit(benign, rule, budget):
    """(eta_a, shared DG height, joint DG height) on these benign rows. See calibrate."""
    usable = (benign.words >= 1) & np.isfinite(benign.exc)
    exc = benign.exc[usable]
    eta_a = None
    if rule in ('adl', 'either'):
        losses = benign.adl[usable & benign.has]
        eta_a = float(np.quantile(losses, 1.0 - budget))
    shared = float(np.quantile(exc, 1.0 - budget))
    fires = _fires(benign, rule, eta_a)[usable]
    n, k = int(usable.sum()), int(fires.sum())
    if k == 0 or budget * n / k >= 1.0:
        joint = float(exc.min() - 1.0)
    else:
        joint = float(np.quantile(exc[fires], 1.0 - budget * n / k))
    return eta_a, shared, joint


def _blocked(sub, height, rule, eta_a):
    """``ExcessFence.blocks`` over an arm: non-finite blocks; otherwise gain AND answer."""
    nonfinite = ~np.isfinite(sub.exc)
    over = sub.exc > height
    if rule == 'none':
        return nonfinite | over
    return nonfinite | (over & _fires(sub, rule, eta_a))


def _rates(benign, attack, rule, budget):
    """The vectorised replicate: fit on the benign rows, read on the attack rows."""
    eta_a, shared, joint = _fit(benign, rule, budget)
    return {'eta_a': eta_a, 'eta_shared': shared, 'eta_joint': joint,
            'block_rate': float(_blocked(attack, shared, rule, eta_a).mean()),
            'block_rate_joint_eta': float(_blocked(attack, joint, rule, eta_a).mean()),
            'benign_block_rate_in_sample_joint_eta': float(
                _blocked(benign, joint, rule, eta_a).mean())}


def _cosine_rates(benign, attack, budget):
    """The baseline every rate is reported beside: the cache's own similarity floor."""
    floor = float(np.quantile(benign.cos, budget))
    return {'cosine_threshold': floor,
            'block_rate': float((attack.cos < floor).mean()),
            'block_rate_joint_eta': float((attack.cos < floor).mean()),
            'benign_block_rate_in_sample_joint_eta': float((benign.cos < floor).mean())}


def _fast(benign, attack, rule, budget):
    return (_cosine_rates(benign, attack, budget) if rule == 'cosine_only'
            else _rates(benign, attack, rule, budget))


def _shipped(benign, attack, rule, budget):
    """The same numbers through the shipped rule code, for the parity checks."""
    if rule == 'cosine_only':
        return v3_detect.cosine_rates_from_rows(benign.rows(), attack.rows(), budget)
    return v3_detect.rule_rates_from_rows(benign.rows(), attack.rows(), rule, budget,
                                          echo_min=ECHO_MIN)


def _agree(fast, slow, where):
    for key in ('block_rate', 'block_rate_joint_eta'):
        assert fast[key] == slow[key], (where, key, fast[key], slow[key])
    for key in ('eta_a', 'eta_shared', 'eta_joint'):
        if key in slow and key in fast:
            assert fast[key] == slow[key], (where, key, fast[key], slow[key])


def boot_rule(key, benign, attack, rule, budget, arm, checks=3):
    """Point estimate from the shipped code; interval from B resampled refits.

    ``Succ.`` restricts the attack arm to the rows the judge marked poisoned, exactly as
    ``ci_table1.boot_br``'s mask does: the decision is per row, so the restriction is a
    choice of which rows to average, never a change to the fit. The first ``checks``
    replicates are also evaluated through the shipped code and must agree exactly, so the
    vectorised path is pinned on resampled inputs and not only on the whole sample.
    """
    pos = (np.arange(len(attack.rows)) if arm == 'All'
           else np.flatnonzero(attack.poisoned))
    full_b, full_a = benign.all(), attack.take(pos)
    point_slow = _shipped(full_b, full_a, rule, budget)
    _agree(_fast(full_b, full_a, rule, budget), point_slow, (key, 'point'))

    b_idx = _groups(benign.intents)
    a_idx = [pos[g] for g in _groups(attack.intents[pos])]
    rng = rng_for(key)
    out = np.empty(B)
    for b in range(B):
        bs = np.concatenate([b_idx[j] for j in rng.integers(0, len(b_idx), len(b_idx))])
        as_ = np.concatenate([a_idx[j] for j in rng.integers(0, len(a_idx), len(a_idx))])
        rb, ra = benign.take(bs), attack.take(as_)
        rep = _fast(rb, ra, rule, budget)
        if b < checks:
            _agree(rep, _shipped(rb, ra, rule, budget), (key, f'replicate {b}'))
        out[b] = rep['block_rate_joint_eta']
    lo, hi = np.percentile(out, [2.5, 97.5])
    return (point_slow['block_rate_joint_eta'], float(lo), float(hi)), point_slow


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--rows', default=None,
                   help='directory of <encoder>__<set>.jsonl dumps from '
                        'v3_detect.py --dump-rows')
    p.add_argument('--main-table',
                   default=os.path.join(
                       CD_REPO,
                       'experiments/paper/results/answer_check_20260909/'
                       'main_table.json'),
                   help='the published cells the 5%% point estimates must reproduce')
    p.add_argument('--out', default=None)
    p.add_argument('--checks', type=int, default=3,
                   help='replicates per cell also evaluated through the shipped code')
    args = p.parse_args(argv)
    if args.rows is None:
        args.rows = str(perrow_root() / 'answer_check_rows')
    if args.out is None:
        args.out = str(perrow_root() / 'ci_answer_check.json')

    published = json.load(open(args.main_table))['cells']
    # Recursive: the mean-pooled dumps sit in a `mean/` subdirectory so they cannot
    # be mistaken for the published CLS cells, and a non-recursive glob silently
    # scored twelve cells where the run produced fifteen.
    files = sorted(Path(args.rows).rglob('*__*.jsonl'))
    if not files:
        raise SystemExit(f'no <encoder>__<set>.jsonl dumps under {args.rows}')

    results, parity, unchecked = {}, [], []
    for path in files:
        # <encoder>__<set>[__<pooling>]. The pooling suffix is how the run names the
        # mean-pooled cells of the pooling table; they are scored like any other cell but
        # no published cell covers them, which the parity block below reports rather than
        # passes over.
        parts = path.stem.split('__')
        if len(parts) == 3 and parts[2] in POOLINGS:
            encoder, setstem, pooling = parts
        elif len(parts) == 2:
            (encoder, setstem), pooling = parts, 'cls'
        else:
            raise SystemExit(f'{path.name}: expected <encoder>__<set>[__<pooling>]')
        if setstem not in SETS:
            raise SystemExit(f'{path.name}: unknown attack set {setstem!r}')
        setname = SETS[setstem]
        # Only the CLS cells are the published ones; a mean-pooled cell of the same
        # encoder must not be checked against, or silently stand in for, a CLS cell.
        label = encoder if pooling == 'cls' else f'{encoder}/{pooling}'
        benign_rows, attack_rows = v3_detect.load_dumped_rows(path)
        benign, attack = Arm(benign_rows), Arm(attack_rows)
        n_pois = int(attack.poisoned.sum())
        cell = {'encoder': encoder, 'pooling': pooling, 'set': setname,
                'file': path.name,
                'n_benign': len(benign_rows), 'n_attack': len(attack_rows),
                'n_poisoned': n_pois, 'budgets': {}}
        ref = published.get(f'{encoder}|{setstem}') if pooling == 'cls' else None
        if ref is None:
            # Not a published cell -- an exploratory encoder or pooling recipe. It is
            # scored, but no parity assertion covers it, and a reader of the output must
            # be able to see which cells that is rather than assume all of them checked.
            unchecked.append(f'{setname}|{label}')
        for budget in BUDGETS:
            per_rule = {}
            for rule_label, rule in (*RULES, ('cosine_only', 'cosine_only')):
                entry = {}
                for arm in ARMS:
                    key = f'{setname}|{label}|{rule_label}|{budget}|{arm}'
                    ci, full = boot_rule(key, benign, attack, rule, budget, arm,
                                         checks=args.checks)
                    entry[arm] = list(ci)
                    if arm == 'All':
                        entry.update({k: full[k] for k in
                                      ('eta_a', 'eta_shared', 'eta_joint',
                                       'cosine_threshold',
                                       'benign_block_rate_in_sample_joint_eta',
                                       'joint_budget_reachable',
                                       'n_benign_answer_fires')
                                      if k in full})
                per_rule[rule_label] = entry
                print(f'{setname:4s} {label:22s} {rule_label:11s} b={budget:<5} '
                      f"All {entry['All'][0]:.4f} [{entry['All'][1]:.4f},"
                      f"{entry['All'][2]:.4f}]  "
                      f"Succ. {entry['Succ.'][0]:.4f} [{entry['Succ.'][1]:.4f},"
                      f"{entry['Succ.'][2]:.4f}]", flush=True)
            cell['budgets'][f'{budget}'] = per_rule
            if budget == 0.05 and ref is not None:
                for rule_label, keys in (
                        ('dg_only', ('excess_block_rate', 'tpr_poisoned')),
                        ('either', ('excess_block_rate_joint_eta',
                                    'tpr_poisoned_joint_eta'))):
                    want = ref['answer_rules'][rule_label]
                    got = per_rule[rule_label]
                    for arm, key in zip(ARMS, keys):
                        assert got[arm][0] == want[key], (
                            f'{setname}/{label} {rule_label} {arm}: recomputed '
                            f'{got[arm][0]!r} but {args.main_table} publishes '
                            f'{want[key]!r}')
                    assert got['eta_joint'] == want['eta_joint'], (setname, label,
                                                                   rule_label)
                    parity.append(f'{setname}|{label}|{rule_label}')
        results[f'{setname}|{label}'] = cell

    if not parity:
        raise SystemExit('no cell matched a published cell: the parity assertion that '
                         'makes these intervals trustworthy never ran')
    if unchecked:
        print(f'NOTE: no published cell to check against for {sorted(unchecked)}; those '
              f'rows carry intervals but no parity assertion', flush=True)
    out = {'what': 'Intent-grouped bootstrap 95% CIs for the answer-checked Deletion '
                   'Gain, with the cosine-only baseline, at four FPR budgets.',
           'B': B, 'echo_min': ECHO_MIN, 'budgets': list(BUDGETS),
           'rows_dir': str(args.rows), 'main_table': args.main_table,
           'parity_checked_at_0.05': sorted(parity),
           'no_published_cell_to_check': sorted(unchecked),
           'read_this_first': [
               'The `either` rate is the joint-height (matched benign cost) one: eta_a '
               'is fitted on the benign arm, then the DG height is refitted so the '
               'conjunction spends the whole budget. Reading it at DG\'s own height '
               'compares two rules at unequal benign cost.',
               'Point estimates come from the shipped rule code; the replicates use a '
               'vectorised equivalent checked against it on the full sample and on the '
               'first replicates of every cell.',
               'Every rule is reported beside the cosine-only baseline at the same '
               'budget, on the same rows.'],
           'cells': results}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'wrote {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
