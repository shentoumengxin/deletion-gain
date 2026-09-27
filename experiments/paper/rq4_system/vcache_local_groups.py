"""FPR by key and answer length on the vCache benchmarks under local calibration.

Reads the per-hit dumps of ``vcache_replay.py`` (``<dataset>_hits.jsonl``) and repeats its
same-dataset recalibration (200 halves of the key equivalence classes, seed 20260923,
joint rule fitted on the valid hits of one half at the 5% budget). Each split's thresholds
are read on the other half; rejected and total valid hits are summed over the splits per
key-length and answer-length bin. The overall mean FPR must reproduce
``recalibrated_joint_reference.fpr.mean`` of the replay summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from experiments.paper.rq4_system.vcache_replay import (
    ANSWER_BINS, BUDGET, KEY_BINS, bin_label, rule_rejects)

DATASETS = ('lmarena', 'search', 'classification')


def load(path: Path, cls_type) -> dict:
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    h = {k: np.array([r[k] for r in rows]) for k in
         ('valid', 'judgeable', 'has_answer', 'dg', 'adl', 'echo', 'key_words', 'answer_words')}
    h['judgeable'] = h['judgeable'].astype(bool)
    h['has_answer'] = h['has_answer'].astype(bool)
    h['valid'] = h['valid'].astype(bool)
    h['adl'] = np.array([np.nan if v is None else v for v in h['adl']], float)
    h['dg'] = np.array([np.nan if v is None else v for v in h['dg']], float)
    h['key_cls'] = np.array([cls_type(r['key_class']) for r in rows])
    return h


def grouped(h: dict, reps: int, seed: int) -> dict:
    kb = np.array([bin_label(w, KEY_BINS) for w in h['key_words']])
    ab = np.array([bin_label(w, ANSWER_BINS) for w in h['answer_words']])
    uniq = np.unique(h['key_cls'])
    rng = np.random.default_rng(seed)
    fit_ok = h['judgeable'] & h['has_answer']
    labels = {'key': [b[0] for b in KEY_BINS], 'answer': [b[0] for b in ANSWER_BINS]}
    sums = {g: {b: [0, 0, 0, 0] for b in labels[g]} for g in labels}   # rej_valid, valid, rej, hits
    fprs, losses = [], []
    for _ in range(reps):
        half = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
        in_a = np.array([g in half for g in h['key_cls']])
        fit = in_a & h['valid'] & fit_ok
        if not fit.any():
            continue
        eta, eta_a = R.joint_thresholds(h['dg'][fit], h['adl'][fit], h['echo'][fit], BUDGET)
        rej = rule_rejects(h['dg'], h['adl'], h['echo'], h['judgeable'], h['has_answer'],
                           {'kind': 'joint', 'eta': eta, 'eta_a': eta_a})
        ev = ~in_a
        v = ev & h['valid']
        if v.any():
            fprs.append(float(np.mean(rej[v])))
        losses.append(float(np.mean(rej[ev])))
        for g, bins in (('key', kb), ('answer', ab)):
            for b in labels[g]:
                m = bins == b
                s = sums[g][b]
                s[0] += int(np.sum(m & v & rej))
                s[1] += int(np.sum(m & v))
                s[2] += int(np.sum(m & ev & rej))
                s[3] += int(np.sum(m & ev))
    table = {g: [{'bin': b, 'fpr': (s[0] / s[1]) if s[1] else None,
                  'valid_hits_per_split': s[1] / reps,
                  'hit_rate_loss': (s[2] / s[3]) if s[3] else None}
                 for b, s in sums[g].items() if s[3]]
             for g in sums}
    return {'fpr_mean': float(np.mean(fprs)), 'hit_rate_loss_mean': float(np.mean(losses)),
            'by_key_words': table['key'], 'by_answer_words': table['answer']}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hits-dir', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True, help='vcache_replay.json (tau 0.90)')
    p.add_argument('--reps', type=int, default=200)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    ref = json.loads(a.summary.read_text())['datasets']
    out = {'protocol': 'recalibrated_joint_reference of vcache_replay.py, grouped by length',
           'datasets': {}}
    for name in DATASETS:
        target = ref[name]['recalibrated_joint_reference']['fpr']['mean']
        for cls_type in (int, str):
            try:
                h = load(a.hits_dir / f'{name}_hits.jsonl', cls_type)
            except ValueError:
                continue
            res = grouped(h, a.reps, a.seed)
            if abs(res['fpr_mean'] - target) < 1e-12:
                break
        assert abs(res['fpr_mean'] - target) < 1e-12, (name, res['fpr_mean'], target)
        res['reproduces_summary_fpr_mean'] = target
        out['datasets'][name] = res
        print(name, json.dumps(res), flush=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
