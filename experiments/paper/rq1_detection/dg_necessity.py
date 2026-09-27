"""Is Deletion Gain necessary? Answer-Check-only and full-key Echo beside DG and Ours.

Same records, benign controls, 5% budget and ASR definition as the main table (e5,
multi[count:4+width:2:cap16]/runs, answer-aligned rows). Rows of the output table:

* Cosine        block if cos(k, q) < benign 5% quantile.
* DG only       block if DG > benign 95% quantile.
* AC only       block if ADL(s*) > eta_A or Echo(s*) >= e, both fitted on benign rows only
                (``supp_rules.ac_only_thresholds``); s* is still DG's arg-max variant.
* Full-key Echo block if |(W(k) ∩ W(y)) \\ W(q)| >= t, t the smallest integer that keeps
                benign FPR <= 5%; no deletion search at all.
* Ours          DG > eta and (ADL > eta_A or Echo >= 1), joint calibration.

AUC ranks by the rule's own score: cosine, DG, the integer full-key Echo, DG for Ours
(the paper's convention), and for AC only the upper ROC envelope of its two-threshold
family (an optimistic bound for the ablation). No model calls; runs on the laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from experiments.paper.rq1_detection.v3_detect import load_rows

BUDGETS = (0.01, 0.02, 0.05, 0.10)
CELLS = {  # display name -> (stem, attack role, benign generator)
    'CAP': ('lmp', 'ndss', 'human_comqa'),
    'SCP': ('scp', 'scp', 'human_comqa'),
    'KCA': ('kca', 'gcg', 'cacheattack_cleaned_qa'),
}
METHODS = ('Cosine', 'DG only', 'AC only', 'Full-key Echo', 'Ours')


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


# ---------------------------------------------------------------- decision rules

def fit(method: str, b: dict, budget: float) -> dict:
    """Thresholds from benign rows only."""
    if method == 'Cosine':
        return {'floor': float(np.quantile(b['base_cos'], budget))}
    if method == 'DG only':
        return {'eta': R.dg_only_threshold(b['excess_span'], budget)}
    if method == 'AC only':
        eta_a, e = R.ac_only_thresholds(b['adl_best'], b['echo_best'], budget)
        return {'eta_a': eta_a, 'echo_min': e}
    if method == 'Full-key Echo':
        t, _ = R.discrete_threshold(b['fk_echo'], budget)
        return {'t': t}
    if method == 'Ours':
        eta, eta_a = R.joint_thresholds(b['excess_span'], b['adl_best'], b['echo_best'], budget)
        return {'eta': eta, 'eta_a': eta_a}
    raise KeyError(method)


def blocks(method: str, x: dict, th: dict) -> np.ndarray:
    if method == 'Cosine':
        return x['base_cos'] < th['floor']
    if method == 'DG only':
        return x['excess_span'] > th['eta']
    if method == 'AC only':
        return (x['adl_best'] > th['eta_a']) | (x['echo_best'] >= th['echo_min'])
    if method == 'Full-key Echo':
        return x['fk_echo'] >= th['t']
    if method == 'Ours':
        return R.joint_blocks(x, th['eta'], th['eta_a'])
    raise KeyError(method)


def ranking_auc(method: str, b: dict, a: dict) -> float:
    if method == 'Cosine':
        return R.auc(-a['base_cos'], -b['base_cos'])
    if method in ('DG only', 'Ours'):
        return R.auc(a['excess_span'], b['excess_span'])
    if method == 'Full-key Echo':
        return R.auc(a['fk_echo'], b['fk_echo'])
    if method == 'AC only':
        return R.disjunction_envelope_auc(b['adl_best'], b['echo_best'],
                                          a['adl_best'], a['echo_best'])
    raise KeyError(method)


# ---------------------------------------------------------------- inputs

def cached_answers(stem: str, rows: list[dict], texts: dict, args) -> dict:
    """record_id -> the answer the cache stores (the one the success label judged)."""
    out = {}
    if stem == 'kca':
        by_sha = {r['prompt_sha']: r['response'] for r in read_jsonl(args.kca_answers)}
        for r in rows:
            out[r['record_id']] = by_sha[sha_text(texts[r['record_id']]['text'])]
        return out
    from sentry.research.pipeline.instruction_benign import load_answer_files
    bare, _ = load_answer_files([str(args.bare_answers)])
    judged_name = {'lmp': 'judged_lmp_all_full.jsonl', 'scp': 'judged_scp_full.jsonl'}[stem]
    judged = {}
    for j in read_jsonl(args.judged_dir / judged_name):
        for rid in [j['record_id'], *j.get('alt_ids', [])]:
            judged[rid] = j
    for r in rows:
        k = texts[r['record_id']]['text']
        if r['arm'] == 'attack':
            j = judged[r['record_id']]
            assert j['prompt'] == k
            if 'answer_sha256' in r:
                assert sha_text(j['response']) == r['answer_sha256'], r['record_id']
            out[r['record_id']] = j['response']
        else:
            out[r['record_id']] = bare[sha_text(k)]
    return out


def load_cell(name: str, args) -> tuple[dict, dict, dict]:
    stem, role, gen = CELLS[name]
    path = args.kca_rows if stem == 'kca' else args.aligned_dir / f'e5-small-v2__{stem}.jsonl'
    rows = read_jsonl(path)
    loaded, _ = load_rows(args.eval_dir / f'{stem}_eval.jsonl', [role], [gen])
    texts = {r['record_id']: r for r in loaded}
    answers = cached_answers(stem, rows, texts, args)
    for r in rows:
        t = texts[r['record_id']]
        r['fk_echo'] = R.full_key_echo(t['text'], answers[r['record_id']], t['anchor'])
    b, a = R.split_arms(rows)
    b['fk_echo'] = np.array([r['fk_echo'] for r in rows if r['arm'] == 'genuine'], float)
    a['fk_echo'] = np.array([r['fk_echo'] for r in rows if r['arm'] == 'attack'], float)
    source = {'rows': str(path), 'rows_sha256': sha_file(path)}
    return b, a, source


# ---------------------------------------------------------------- summary

def summarize(name: str, b: dict, a: dict, replicates: int) -> dict:
    out = {'n_attack': int(len(a['base_cos'])), 'n_benign': int(len(b['base_cos'])),
           'n_poisoned_backend': int(a['poisoned'].sum()), 'methods': {}}
    for m in METHODS:
        th = fit(m, b, .05)
        blocked, b_blocked = blocks(m, a, th), blocks(m, b, th)
        k, n, rate = R.asr(a['base_cos'], a['poisoned'], blocked)

        def stat(bb, aa, m=m):
            t = fit(m, bb, .05)
            bl = blocks(m, aa, t)
            return [float(bl.mean()), R.asr(aa['base_cos'], aa['poisoned'], bl)[2]]

        lo, hi = R.intent_bootstrap(b, a, stat, replicates, seed=f'{name}|{m}')
        cell = {'thresholds': th, 'auc': ranking_auc(m, b, a),
                'br': float(blocked.mean()), 'br_ci95': [lo[0], hi[0]],
                'br_poisoned': float(blocked[a['poisoned']].mean()),
                'asr': rate, 'asr_successes': k, 'asr_ci95': [lo[1], hi[1]],
                'realized_fpr_in_sample': float(b_blocked.mean()),
                'br_by_budget': {}}
        for budget in BUDGETS:
            tb = fit(m, b, budget)
            cell['br_by_budget'][str(budget)] = {
                'br': float(blocks(m, a, tb).mean()),
                'realized_fpr': float(blocks(m, b, tb).mean())}
        if m == 'Full-key Echo':
            cell['br_randomized_exact_budget'] = R.randomized_br(b['fk_echo'], a['fk_echo'], .05)
        if m == 'AC only':
            cell['auc_components'] = {'adl': R.auc(a['adl_best'], b['adl_best']),
                                      'echo_s_star': R.auc(a['echo_best'], b['echo_best'])}
        out['methods'][m] = cell
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', type=Path, required=True)
    p.add_argument('--aligned-dir', type=Path, required=True)
    p.add_argument('--kca-rows', type=Path, required=True)
    p.add_argument('--kca-answers', type=Path, required=True)
    p.add_argument('--judged-dir', type=Path, required=True)
    p.add_argument('--bare-answers', type=Path, required=True)
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    result = {'policy': 'multi[count:4+width:2:cap16]/runs', 'encoder': 'intfloat/e5-small-v2',
              'budget': .05, 'retrieval_floor': R.RETRIEVAL_FLOOR,
              'asr_definition': 'mean(cos>=0.90 AND accepted AND poisoned) over all attack rows',
              'word_set': 'textnorm.content_tokens: strip markdown, lowercase, '
                          "[a-z0-9][a-z0-9'-]*, strip '/- ends, drop 127 function words, no stemming",
              'bootstrap': f'{args.replicates} intent-grouped replicates, thresholds refit',
              'cells': {}}
    for name in CELLS:
        b, a, source = load_cell(name, args)
        cell = summarize(name, b, a, args.replicates)
        cell['source'] = source
        result['cells'][name] = cell
        print(name, {m: (round(v['auc'], 3), round(v['br'], 3), round(v['asr'], 3),
                         round(v['realized_fpr_in_sample'], 3))
                     for m, v in cell['methods'].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
