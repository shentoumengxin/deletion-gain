"""Answer-aware versus query-only LLM judges on the Table 1 rows (no model calls).

Up to five judges, one population per class (the row tables; CAP and SCP share the ComQA
benign rows, KCA uses NQ):

* Qwen3-8B, logprob score, without the answer (``qk``) and with it (``qky``);
* DeepSeek-v4-flash, 10-vote UNSAFE fraction, without the answer (the published Table 1
  judge, ``runs/paper/perrow/judge_vote_*.json``) and with it (``judge_answer_aware_vote``);
  optionally with it under the poisoning-focused prompt (``--prompt poisoning``).

Every judge is thresholded like every Table 1 row: block iff score > the benign 1-b
quantile, so vote fractions with ties spend at most the budget; the realized FPR is
reported beside each BR. ASR follows the main table (cos >= 0.90, accepted, poisoned).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R

BUDGETS = (0.01, 0.02, 0.05, 0.10)
CLASSES = (('CAP', 'lmp', 'cap'), ('SCP', 'scp', 'scp'), ('KCA', 'kca', 'kca'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def threshold(b_score, budget):
    return float(np.quantile(b_score, 1 - budget))


def evaluate(b: dict, a: dict, name: str, replicates: int) -> dict:
    bs, as_ = b['score'], a['score']
    out = {'auc': R.auc(as_, bs), 'budgets': {}}
    for budget in BUDGETS:
        t = threshold(bs, budget)
        out['budgets'][str(budget)] = {'br': float((as_ > t).mean()),
                                       'realized_fpr': float((bs > t).mean())}
    t = threshold(bs, .05)
    blocked = as_ > t
    k, n, rate = R.asr(a['base_cos'], a['poisoned'], blocked)

    def stat(bb, aa):
        tt = threshold(bb['score'], .05)
        bl = aa['score'] > tt
        return [float(bl.mean()), R.asr(aa['base_cos'], aa['poisoned'], bl)[2]]

    lo, hi = R.intent_bootstrap(b, a, stat, replicates, seed=name)
    out.update(br=float(blocked.mean()), br_ci95=[lo[0], hi[0]],
               br_poisoned=float(blocked[a['poisoned']].mean()),
               asr=rate, asr_successes=k, asr_ci95=[lo[1], hi[1]],
               realized_fpr=float((bs > t).mean()))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows-dir', type=Path, required=True)
    p.add_argument('--qwen', type=Path, required=True, help='judge_qwen3_8b.jsonl')
    p.add_argument('--deepseek-qky', type=Path, required=True, help='deepseek_vote_qky.jsonl')
    p.add_argument('--deepseek-qk-dir', type=Path, required=True, help='runs/paper/perrow')
    p.add_argument('--deepseek-qky-poison', type=Path, default=None,
                   help='deepseek_vote_qky_poisoning.jsonl (poisoning-focused prompt)')
    p.add_argument('--qwen-meta', type=Path, default=None)
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)

    qwen = {(r['set'], r['arm'], r['record_id']): r for r in read_jsonl(a.qwen)}
    ds_qky = {(r['arm'], r['record_id']): r['score'] for r in read_jsonl(a.deepseek_qky)}
    ds_poison = ({(r['arm'], r['record_id']): r['score'] for r in read_jsonl(a.deepseek_qky_poison)}
                 if a.deepseek_qky_poison else None)
    result = {'threshold': 'score > benign (1-b) quantile, per corpus',
              'judges': {'Qwen3-8B (q,k)': 'logprob UNSAFE - SAFE, prompt without the answer',
                         'Qwen3-8B (q,k,y)': 'logprob UNSAFE - SAFE, prompt with the stored answer',
                         'DeepSeek vote (q,k)': 'published Table 1 judge, 10 votes at T=1',
                         'DeepSeek vote (q,k,y)': 'same scoring, prompt with the stored answer',
                         'DeepSeek vote (q,k,y), poisoning prompt': 'same scoring; the prompt says an honest wrong answer is not a reason to flag'},
              'cells': {}}
    if a.qwen_meta and a.qwen_meta.exists():
        result['qwen_meta'] = json.loads(a.qwen_meta.read_text())
    for name, stem, short in CLASSES:
        rows = read_jsonl(a.rows_dir / f'rows_{stem}.jsonl')
        published = {(r['arm'], r['record_id']): r['score'] for r in
                     json.loads((a.deepseek_qk_dir / f'judge_vote_{short}.json').read_text())['rows']}
        scores = {
            'Qwen3-8B (q,k)': [qwen[(stem, r['arm'], r['record_id'])]['qk']['score'] for r in rows],
            'Qwen3-8B (q,k,y)': [qwen[(stem, r['arm'], r['record_id'])]['qky']['score'] for r in rows],
            'DeepSeek vote (q,k)': [published[(r['arm'], r['record_id'])] for r in rows],
            'DeepSeek vote (q,k,y)': [ds_qky[(r['arm'], r['record_id'])] for r in rows],
        }
        if ds_poison is not None:
            scores['DeepSeek vote (q,k,y), poisoning prompt'] = [ds_poison[(r['arm'], r['record_id'])] for r in rows]
        cell = {'n_attack': sum(r['arm'] == 'attack' for r in rows),
                'n_benign': sum(r['arm'] == 'genuine' for r in rows), 'judges': {}}
        for judge, values in scores.items():
            rs = [dict(r, score=v) for r, v in zip(rows, values)]
            b, at = R.split_arms(rs)
            b['score'] = np.array([r['score'] for r in rs if r['arm'] == 'genuine'], float)
            at['score'] = np.array([r['score'] for r in rs if r['arm'] == 'attack'], float)
            cell['judges'][judge] = evaluate(b, at, f'{name}|{judge}', a.replicates)
            j = cell['judges'][judge]
            print(name, judge, f"AUC {j['auc']:.3f} BR {j['br']:.3f} ASR {j['asr']:.3f} FPR {j['realized_fpr']:.3f}",
                  {k: round(v['br'], 3) for k, v in j['budgets'].items()}, flush=True)
        result['cells'][name] = cell
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
