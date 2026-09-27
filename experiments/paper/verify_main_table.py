"""Recompute the headline numbers of Table 1 from the released per-row tables.

No model, embedding or network calls. For each attack class the script reads
``runs/supp_appendix_20260923/inputs/rows_<set>.jsonl`` under the data root (``lmp`` is
CAP), fits the thresholds of our defense on the benign rows of the same corpus at a 5%
false-positive budget, and prints

* AUC of Deletion Gain (attack vs. benign entries),
* BR: the share of poisoned entries our defense rejects,
* end-to-end ASR: retrieved (cosine >= 0.90), accepted, and judged poisoned,
  for our defense and without a defense,
* the realized FPR on the benign rows used for calibration.

Usage::

    export SENTRY_DATA_ROOT=$PWD/data
    python -m experiments.paper.verify_main_table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from sentry.artifacts import data_root

CLASSES = (('CAP', 'lmp'), ('SCP', 'scp'), ('KCA', 'kca'))
ROWS = Path('runs/supp_appendix_20260923/inputs')


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def evaluate(rows: list[dict], budget: float) -> dict:
    benign, attack = R.split_arms(rows)
    eta, eta_a = R.joint_thresholds(benign['excess_span'], benign['adl_best'],
                                    benign['echo_best'], budget)
    blocked = R.joint_blocks(attack, eta, eta_a)
    k, n, asr = R.asr(attack['base_cos'], attack['poisoned'], blocked)
    k0, _, asr0 = R.asr(attack['base_cos'], attack['poisoned'], np.zeros(n, dtype=bool))
    return {'n_attack': n, 'n_benign': len(benign['excess_span']),
            'auc_dg': R.auc(attack['excess_span'], benign['excess_span']),
            'br': float(blocked.mean()), 'blocked': int(blocked.sum()),
            'asr': asr, 'successes': k, 'asr_no_defense': asr0, 'successes_no_defense': k0,
            'benign_fpr': float(R.joint_blocks(benign, eta, eta_a).mean()),
            'eta': eta, 'eta_a': eta_a}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--budget', type=float, default=0.05)
    parser.add_argument('--json', action='store_true', help='print the full result as JSON')
    args = parser.parse_args(argv)
    root = data_root(args.data_root)
    results = {name: evaluate(read_rows(root / ROWS / f'rows_{stem}.jsonl'), args.budget)
               for name, stem in CLASSES}
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    print(f"{'class':5} {'n':>4} {'AUC(DG)':>8} {'BR':>6} {'ASR':>6} {'ASR none':>9} {'FPR':>6}")
    for name, r in results.items():
        print(f"{name:5} {r['n_attack']:4d} {r['auc_dg']:8.3f} {r['br']:6.3f} {r['asr']:6.3f} "
              f"{r['asr_no_defense']:9.3f} {r['benign_fpr']:6.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
