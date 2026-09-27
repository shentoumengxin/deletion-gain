"""Flat versus conditional DG threshold under the deployed 4+2 cut (no model calls).

The conditional form lets the DG height vary with key-query cosine and log word count
(pinball-loss linear quantile regression, ``ExcessFence.fit``); the flat form is one
height. Both are fitted on benign calibration intents and read on held-out intents with
the protocol of ``rq1_detection/main_table_holdout.py`` (200 intent-grouped half splits,
same split function), for DG only and for the deployed rule (DG and the answer check,
joint height refit on the calibration half). Rows are the answer-aligned row tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection.main_table_holdout import aggregate, split_masks
from experiments.paper.rq1_detection.v3_detect import DUMP_FIELDS, DumpedColumns, DumpedRow
from sentry.cache.defense.calibrate import blocks_under, fit_rule_fence, joint_fence
from sentry.cache.defense.fence import CalibrationRow

POLICY = 'multi[count:4+width:2:cap16]/runs'
EMBEDDER = 'intfloat/e5-small-v2'
BUDGET = 0.05
SETS = (('CAP', 'lmp'), ('SCP', 'scp'), ('KCA', 'kca'))
RULES = {'DG only': 'none', 'DG + answer check': 'either'}
FORMS = ('flat', 'conditional')


def load(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    # every row-table entry carries its cached answer (checked when the tables were built)
    as_dumped = [DumpedRow(*(True if k == 'has_answer' else r[k] for k in DUMP_FIELDS))
                 for r in rows]
    benign = [r for r in as_dumped if r.arm == 'genuine']
    attack = [r for r in as_dumped if r.arm == 'attack']
    return benign, attack


def decisions(benign, attack, fit_b, rule: str, form: str):
    calibration = [r for r, keep in zip(benign, fit_b) if keep]
    rows = [CalibrationRow(r.base_cos, r.words, r.excess_span, answer_loss=r.adl_best)
            for r in calibration]
    fence = fit_rule_fence(rows, rule, budget=BUDGET, embedder=EMBEDDER, policy=POLICY,
                           echo_min=1, fence_form=form)
    columns = DumpedColumns()
    if rule == 'either':
        fence, _ = joint_fence(fence, [(None, r) for r in calibration], columns, BUDGET)
    return tuple(np.asarray(blocks_under(fence, [(None, r) for r in arm], columns), bool)
                 for arm in (benign, attack))


def run(rows_dir: Path, splits: int) -> dict:
    out = {'protocol': {'policy': POLICY, 'budget': BUDGET, 'n_splits': splits,
                        'split_function': 'main_table_holdout.split_masks (sha256(seed:intent_id) < 0.5)',
                        'fit_population': 'benign calibration intents only',
                        'evaluation': 'benign and attack rows of held-out intents',
                        'conditional_covariates': 'key-query cosine, log word count (pinball loss)',
                        'answer_rule': 'either, echo_min=1, joint height refit per split',
                        'model_calls': 0},
           'sets': {}}
    for tag, stem in SETS:
        path = rows_dir / f'rows_{stem}.jsonl'
        benign, attack = load(path)
        bi, ai = [r.intent_id for r in benign], [r.intent_id for r in attack]
        poisoned = np.array([bool(r.poisoned) for r in attack])
        cell = {'rows_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'n_benign': len(benign), 'n_attack': len(attack), 'results': {}}
        full = np.ones(len(benign), bool)
        for rule_name, rule in RULES.items():
            for form in FORMS:
                bb, ba = decisions(benign, attack, full, rule, form)
                split_rates = []
                for seed in range(splits):
                    fit_b, test_b, test_a = split_masks(bi, ai, seed)
                    sb, sa = decisions(benign, attack, fit_b, rule, form)
                    split_rates.append({'fpr': float(sb[test_b].mean()),
                                        'br_all': float(sa[test_a].mean()),
                                        'br_success': float(sa[test_a & poisoned].mean())})
                cell['results'][f'{rule_name} | {form}'] = {
                    'in_sample': {'fpr': float(bb.mean()), 'br_all': float(ba.mean()),
                                  'br_success': float(ba[poisoned].mean())},
                    'held_out': {k: aggregate([r[k] for r in split_rates])
                                 for k in ('fpr', 'br_all', 'br_success')}}
                h = cell['results'][f'{rule_name} | {form}']['held_out']
                print(tag, rule_name, form, f"FPR {h['fpr']['mean']:.4f}±{h['fpr']['sd_across_splits']:.4f}",
                      f"BR {h['br_all']['mean']:.4f}±{h['br_all']['sd_across_splits']:.4f}", flush=True)
        out['sets'][tag] = cell
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows-dir', type=Path, required=True)
    p.add_argument('--splits', type=int, default=200)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    result = run(a.rows_dir, a.splits)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
