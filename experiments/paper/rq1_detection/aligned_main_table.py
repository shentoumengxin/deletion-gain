"""Every main-table quantity of Ours before and after the CAP/SCP answer alignment.

No model calls. For each class it reads the published dump (before) and the
answer-aligned dump (after; KCA's was aligned on 2026-09-20 and is read from the SC-IPI
scores), and reports what the paper prints for Ours: BR over all planted entries and over
poisoned ones with the intent-bootstrap intervals of ``ci_answer_check.boot_rule`` (same
generator keys, so unchanged cells reproduce exactly), ASR with the interval procedure of
``paper/data/recompute_e2e_asr.py`` (same seed), the per-construction BR and
ASR of the families table, and BR at the four budgets of the budgets table. Thresholds are
fitted on benign rows, which the alignment does not touch, so they cannot move.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from experiments.paper.rq1_detection.ci_answer_check import Arm, boot_rule
from experiments.paper.rq1_detection.v3_detect import load_dumped_rows

BUDGETS = (0.10, 0.05, 0.02, 0.01)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def summarize(name: str, path: Path) -> dict:
    b, a = R.split_arms(read_rows(path))
    eta, eta_a = R.joint_thresholds(b['excess_span'], b['adl_best'], b['echo_best'])
    blocked = R.joint_blocks(a, eta, eta_a)
    floor = float(np.quantile(b['base_cos'], .05))

    def asr_all(bb, aa):
        e, ea = R.joint_thresholds(bb['excess_span'], bb['adl_best'], bb['echo_best'])
        fl = float(np.quantile(bb['base_cos'], .05))
        joint = (aa['base_cos'] >= .90) & aa['poisoned']
        return [float((joint & ~R.joint_blocks(aa, e, ea)).mean()),
                float((joint & (aa['base_cos'] >= fl)).mean())]

    lo, hi = R.intent_bootstrap(b, a, asr_all, 2000, seed=name)
    benign_rows, attack_rows = load_dumped_rows(path)
    arm_b, arm_a = Arm(benign_rows), Arm(attack_rows)
    br_ci = {arm: boot_rule(f'{name}|e5-small-v2|either|0.05|{arm}', arm_b, arm_a, 'either', .05, arm)[0]
             for arm in ('All', 'Succ.')}
    k, n, rate = R.asr(a['base_cos'], a['poisoned'], blocked)
    joint = (a['base_cos'] >= .90) & a['poisoned']
    families = {}
    for fam in sorted(set(a['family'])):
        m = a['family'] == fam
        families[fam] = {'n': int(m.sum()), 'br': float(blocked[m].mean()),
                         'asr_ours': float((joint & ~blocked)[m].mean()),
                         'asr_none': float(joint[m].mean()),
                         'asr_cosine': float((joint & (a['base_cos'] >= floor))[m].mean())}
    budgets = {}
    for budget in BUDGETS:
        e, ea = R.joint_thresholds(b['excess_span'], b['adl_best'], b['echo_best'], budget)
        budgets[str(budget)] = float(R.joint_blocks(a, e, ea).mean())
    return {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'thresholds': {'eta': eta, 'eta_a': eta_a},
            'br_all': br_ci['All'], 'br_poisoned': br_ci['Succ.'],
            'asr': {'successes': k, 'n': n, 'rate': rate, 'ci95': [lo[0], hi[0]]},
            'asr_cosine_ci95': [lo[1], hi[1]],
            'families': families, 'br_by_budget': budgets}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--published-dir', type=Path, required=True,
                   help='runs/paper/perrow/answer_check_rows')
    p.add_argument('--aligned-dir', type=Path, required=True)
    p.add_argument('--kca-aligned', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    out = {'rule': 'DG > eta and (ADL > eta_A or Echo >= 1), joint calibration, 5% unless stated',
           'cells': {}}
    for name, stem in (('CAP', 'lmp'), ('SCP', 'scp'), ('KCA', 'kca')):
        before = a.published_dir / f'e5-small-v2__{stem}.jsonl'
        after = a.kca_aligned if stem == 'kca' else a.aligned_dir / f'e5-small-v2__{stem}.jsonl'
        out['cells'][name] = {'before': summarize(name, before), 'after': summarize(name, after)}
        bf, af = out['cells'][name]['before'], out['cells'][name]['after']
        print(name, 'BR', round(bf['br_all'][0], 4), '->', round(af['br_all'][0], 4),
              'ASR', bf['asr']['successes'], '->', af['asr']['successes'], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
