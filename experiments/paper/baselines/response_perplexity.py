"""SCP's own countermeasure: perplexity of the cached answer given the user query.

Wu et al. (NDSS 2026, Sec. VII-B) check each hit after retrieval: compute the perplexity
of the cached response conditioned on the user query with distilgpt2 (their Sec. VII-A
model) and flag high values as poisoned. Here:

* Score. exp(S(y | q)) with S(A | B) the mean token NLL of A given the prefix ``B + " | "``,
  exactly the conditioning of ``baselines/defense/perplexity.py::PerplexityAsymmetry``
  (same model, tokenizer, separator, max_length 512, context left-truncated). y is the
  row's ``answer`` (the judged answer the cache stores), q its ``query``. The helper cuts
  an answer longer than max_length - 1 tokens; this scorer instead covers every answer
  token with overlapping 512-token windows and counts the rows that needed it.
* Reference score. exp(S(y)) with no query: y given only ``<|endoftext|>``. The gap
  S(y | q) - S(y) is reported beside it as a diagnostic; it is not Wu's method.
* Threshold. The benign (1 - budget) quantile per corpus (ComQA for CAP and SCP, NQ for
  KCA), block iff score > t: the 5% budget every baseline gets, not Wu's ROC point.
* Cost. ms per hit is the wall time of one conditional score (tokenise, one forward,
  NLL), single-threaded on cpu-server (OMP_NUM_THREADS=1), median over rows.

Subcommands: ``score`` and ``timing`` (distilgpt2, cpu-server), ``summarize`` (NumPy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R

MODEL = 'distilgpt2'
SEPARATOR = ' | '          # PerplexityAsymmetry._conditional_surprisal
MAX_LENGTH = 512
CLASSES = {'CAP': ('lmp', 'comqa'), 'SCP': ('scp', 'comqa'), 'KCA': ('kca', 'nq')}
BUDGETS = (0.01, 0.02, 0.05, 0.10)
SCORES = {  # name -> (per-row field, transform)
    'PPL(y|q)': ('nll_cond', np.exp),
    'PPL(y)': ('nll_uncond', np.exp),
    'S(y|q)-S(y)': ('nll_gap', lambda v: v),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- scoring core (NumPy)

def windows(context_ids: list[int], target_ids: list[int], max_length: int = MAX_LENGTH):
    """[(ids, first_scored)] covering every target token exactly once.

    A target that fits keeps it whole and left-truncates the context to the room left
    (at least one token), as ``PerplexityAsymmetry`` does. A longer target is scored in
    chunks of max_length // 2 tokens, each read with the max_length tokens that end it.
    """
    assert context_ids, 'the first target token needs a prefix'
    if len(target_ids) <= max_length - 1:
        keep = max(1, max_length - len(target_ids))
        ctx = context_ids[-keep:]
        return [(ctx + target_ids, len(ctx))]
    stream, first, stride = context_ids + target_ids, len(context_ids), max_length // 2
    out = []
    for c in range(first, len(stream), stride):
        e = min(c + stride, len(stream))
        b = max(0, e - max_length)
        out.append((stream[b:e], c - b))
    return out


def mean_target_nll(logprob_fn, context_ids: list[int], target_ids: list[int],
                    max_length: int = MAX_LENGTH) -> tuple[float, dict]:
    """Mean NLL over the target tokens only. ``logprob_fn(ids)`` returns
    log p(ids[i] | ids[:i]) for i = 1 .. len(ids) - 1."""
    wins = windows(context_ids, target_ids, max_length)
    nll = []
    for ids, start in wins:
        lp = np.asarray(logprob_fn(ids), float)
        nll.append(-lp[start - 1:])
    nll = np.concatenate(nll)
    assert len(nll) == len(target_ids)
    kept = len(wins[0][0]) - len(target_ids) if len(wins) == 1 else None
    info = {'n_target': len(target_ids),
            'context_truncated': bool(kept is not None and kept < len(context_ids)),
            'windows': len(wins)}
    return float(nll.mean()), info


class ResponseScorer:
    """Wraps a ``PerplexityAsymmetry`` instance: its model, tokenizer, device, max_length."""

    def __init__(self, helper):
        helper._load_model()
        self.helper = helper
        self.tok, self.model, self.device = helper._tokenizer, helper._model, helper._device
        self.max_length = helper._max_length

    def _logprobs(self, ids: list[int]) -> np.ndarray:
        import torch

        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            logits = self.model(x).logits[0, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0]
        return lp.cpu().numpy().astype(float)

    def conditional(self, answer: str, query: str) -> tuple[float, dict]:
        ctx = self.tok.encode(query + SEPARATOR, add_special_tokens=False)
        tgt = self.tok.encode(answer, add_special_tokens=False)
        nll, info = mean_target_nll(self._logprobs, ctx, tgt, self.max_length)
        return nll, {**info, 'n_context': len(ctx)}

    def unconditional(self, answer: str) -> tuple[float, dict]:
        tgt = self.tok.encode(answer, add_special_tokens=False)
        return mean_target_nll(self._logprobs, [self.tok.eos_token_id], tgt, self.max_length)


# ---------------------------------------------------------------- inputs

def load_rows(inputs: Path) -> list[dict]:
    """Every attack row plus each benign row once (ComQA benign is shared by CAP and SCP)."""
    rows, seen = [], set()
    for cls, (stem, corpus) in CLASSES.items():
        for r in read_jsonl(inputs / f'rows_{stem}.jsonl'):
            if r['arm'] == 'genuine':
                if r['record_id'] in seen:
                    continue
                seen.add(r['record_id'])
            rows.append({**r, 'cls': cls if r['arm'] == 'attack' else '', 'corpus': corpus})
    return rows


# ---------------------------------------------------------------- model subcommands

def _worker_init():
    import torch
    torch.set_num_threads(1)
    from experiments.paper.baselines.defense.perplexity import PerplexityAsymmetry
    global _SCORER
    _SCORER = ResponseScorer(PerplexityAsymmetry(model_name=MODEL, max_length=MAX_LENGTH))


def _score_row(job):
    r, cross_check = job
    t0 = time.perf_counter()
    nll_c, info_c = _SCORER.conditional(r['answer'], r['query'])
    t1 = time.perf_counter()
    nll_u, info_u = _SCORER.unconditional(r['answer'])
    t2 = time.perf_counter()
    out = {'record_id': r['record_id'], 'set': r['set'], 'arm': r['arm'],
           'nll_cond': nll_c, 'nll_uncond': nll_u, 'n_target': info_c['n_target'],
           'n_context': info_c['n_context'], 'context_truncated': info_c['context_truncated'],
           'windows': info_c['windows'], 'windows_uncond': info_u['windows'],
           'ms_cond': 1e3 * (t1 - t0), 'ms_uncond': 1e3 * (t2 - t1)}
    if cross_check:
        out['helper_nll_cond'] = _SCORER.helper._conditional_surprisal(r['answer'], r['query'])
    return out


def cmd_score(args) -> int:
    import multiprocessing as mp

    rows = load_rows(args.inputs)
    jobs = [(r, i % args.cross_check_every == 0) for i, r in enumerate(rows)]
    t0 = time.time()
    with mp.get_context('spawn').Pool(args.workers, initializer=_worker_init) as pool:
        out = pool.map(_score_row, jobs, chunksize=8)
    print(f'scored {len(out)} rows in {time.time() - t0:.0f}s with {args.workers} '
          f'single-threaded workers', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as fh:
        for o in out:
            fh.write(json.dumps(o) + '\n')
    return 0


def cmd_timing(args) -> int:
    """Isolated single-threaded pass over a stratified sample: the headline ms/hit."""
    assert os.environ.get('OMP_NUM_THREADS') == '1', 'run with OMP_NUM_THREADS=1'
    _worker_init()
    rows = load_rows(args.inputs)
    rng = np.random.default_rng(0)
    groups = {}
    for r in rows:
        groups.setdefault((r['set'], r['arm']) if r['arm'] == 'attack' else r['corpus'], []).append(r)
    sample = [g[i] for g in groups.values() for i in rng.choice(len(g), args.per_group, replace=False)]
    for r in sample[:5]:                                   # warm-up, not timed
        _SCORER.conditional(r['answer'], r['query'])
    ms = []
    for r in sample:
        t0 = time.perf_counter()
        _SCORER.conditional(r['answer'], r['query'])
        ms.append(1e3 * (time.perf_counter() - t0))
    ms = np.asarray(ms)
    res = {'n': int(len(ms)), 'per_group': args.per_group, 'groups': sorted(map(str, groups)),
           'median_ms': float(np.median(ms)), 'p10_ms': float(np.percentile(ms, 10)),
           'p90_ms': float(np.percentile(ms, 90)), 'mean_ms': float(ms.mean()),
           'torch_threads': 1, 'host': os.uname().nodename}
    print(res, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + '\n')
    return 0


# ---------------------------------------------------------------- summary (NumPy)

def cell_arrays(rows: list[dict], scored: dict, cls: str) -> tuple[dict, dict]:
    corpus = CLASSES[cls][1]
    sub_a = [r for r in rows if r['arm'] == 'attack' and r['cls'] == cls]
    sub_b = [r for r in rows if r['arm'] == 'genuine' and r['corpus'] == corpus]

    def pack(sub):
        d = {k: np.array([r[k] for r in sub], dtype=float)
             for k in ('base_cos', 'excess_span', 'adl_best', 'echo_best')}
        d['intent_id'] = np.array([r['intent_id'] for r in sub], dtype=object)
        d['family'] = np.array([r['family'] for r in sub], dtype=object)
        d['poisoned'] = np.array([bool(r['poisoned']) for r in sub])
        for name, (field, fn) in SCORES.items():
            d[name] = fn(np.array([scored[r['record_id']][field] for r in sub], float))
        for k in ('n_target', 'windows', 'context_truncated', 'ms_cond'):
            d[k] = np.array([scored[r['record_id']][k] for r in sub], float)
        return d
    return pack(sub_b), pack(sub_a)


def score_cell(b: dict, a: dict, name: str, replicates: int, seed: str) -> dict:
    def at(bb, aa, budget):
        t = R.dg_only_threshold(bb[name], budget)          # benign (1 - budget) quantile
        return t, aa[name] > t, float((bb[name] > t).mean())
    t5, blocked, fpr5 = at(b, a, .05)
    k, n, rate = R.asr(a['base_cos'], a['poisoned'], blocked)

    def stat(bb, aa):
        _, bl, _ = at(bb, aa, .05)
        return [R.auc(aa[name], bb[name]), float(bl.mean()),
                R.asr(aa['base_cos'], aa['poisoned'], bl)[2]]
    lo, hi = R.intent_bootstrap(b, a, stat, replicates, seed=seed)
    cell = {'auc': R.auc(a[name], b[name]), 'auc_ci95': [lo[0], hi[0]],
            'threshold_5pct': t5, 'br': float(blocked.mean()), 'br_ci95': [lo[1], hi[1]],
            'realized_fpr': fpr5, 'br_poisoned': float(blocked[a['poisoned']].mean()),
            'asr': rate, 'asr_successes': k, 'asr_ci95': [lo[2], hi[2]],
            'br_by_budget': {}, 'br_by_family': {},
            'n_attack': n, 'n_benign': int(len(b[name])), 'n_poisoned': int(a['poisoned'].sum())}
    for budget in BUDGETS:
        _, bl, fpr = at(b, a, budget)
        cell['br_by_budget'][str(budget)] = {'br': float(bl.mean()), 'realized_fpr': fpr}
    for fam in sorted(set(a['family'])):
        m = a['family'] == fam
        cell['br_by_family'][fam] = {'n': int(m.sum()), 'br': float(blocked[m].mean())}
    cell['median_score'] = {'attack': float(np.median(a[name])), 'benign': float(np.median(b[name]))}
    return cell


def cmd_summarize(args) -> int:
    from experiments.paper.rq1_detection import dg_necessity as D

    rows = load_rows(args.inputs)
    scored = {}
    for o in read_jsonl(args.scores):
        o['nll_gap'] = o['nll_cond'] - o['nll_uncond']
        scored[o['record_id']] = o
    assert set(scored) == {r['record_id'] for r in rows}
    checked = [o for o in scored.values() if 'helper_nll_cond' in o and o['windows'] == 1]
    helper_diff = max(abs(o['helper_nll_cond'] - o['nll_cond']) for o in checked)
    assert helper_diff < 1e-4, helper_diff
    timing = json.loads(args.timing.read_text())
    allrows = list(scored.values())
    result = {
        'source': 'Wu et al., NDSS 2026, Sec. VII-B (post-retrieval PPL of the cached '
                  'response conditioned on the user query; distilgpt2)',
        'model': MODEL, 'conditioning': f'prefix query + {SEPARATOR!r}, as '
                  'baselines/defense/perplexity.py::PerplexityAsymmetry',
        'max_length': MAX_LENGTH, 'answer': "row 'answer' (the judged answer the cache stores)",
        'threshold': 'benign (1 - budget) quantile per corpus (ComQA for CAP/SCP, NQ for KCA); '
                     'block iff score > t; not the ROC point of Wu et al.',
        'asr_definition': 'mean(cos>=0.90 AND accepted AND poisoned) over all attack rows',
        'bootstrap': f'{args.replicates} intent-grouped replicates, thresholds refit',
        'inputs_sha256': {s: sha_file(args.inputs / f'rows_{s}.jsonl') for s, _ in CLASSES.values()},
        'scores_file_sha256': sha_file(args.scores),
        'n_rows_scored': len(allrows),
        'truncation': {
            'context_left_truncated': int(sum(o['context_truncated'] for o in allrows)),
            'answer_longer_than_max_length_minus_1': int(sum(o['windows'] > 1 for o in allrows)),
            'max_answer_tokens': int(max(o['n_target'] for o in allrows)),
            'max_context_tokens': int(max(o['n_context'] for o in allrows))},
        'helper_cross_check': {'rows': len(checked), 'max_abs_nll_diff': helper_diff},
        'cost': {'ms_per_hit_isolated_single_thread': timing,
                 'ms_per_hit_under_load': {
                     'median_ms': float(np.median([o['ms_cond'] for o in allrows])),
                     'note': 'all rows, 32 concurrent single-threaded workers on one host'}},
        'cells': {},
    }
    for cls in CLASSES:
        b, a = cell_arrays(rows, scored, cls)
        cell = {'scores': {n: score_cell(b, a, n, args.replicates, f'ppl|{cls}|{n}')
                           for n in SCORES}, 'baselines': {}}
        for m in ('Cosine', 'Ours'):
            th = D.fit(m, b, .05)
            bl = D.blocks(m, a, th)
            k, n, rate = R.asr(a['base_cos'], a['poisoned'], bl)
            cell['baselines'][m] = {'thresholds': th, 'auc': D.ranking_auc(m, b, a),
                                    'br': float(bl.mean()), 'asr': rate, 'asr_successes': k,
                                    'realized_fpr': float(D.blocks(m, b, th).mean())}
        result['cells'][cls] = cell
        print(cls, {n: (round(v['auc'], 3), round(v['br'], 3), round(v['asr'], 3))
                    for n, v in cell['scores'].items()},
              {m: (round(v['auc'], 3), round(v['br'], 3), round(v['asr'], 3))
               for m, v in cell['baselines'].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--inputs', type=Path, required=True)
    s.add_argument('--workers', type=int, default=32)
    s.add_argument('--cross-check-every', type=int, default=10)
    s.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('timing')
    t.add_argument('--inputs', type=Path, required=True)
    t.add_argument('--per-group', type=int, default=60)
    t.add_argument('--out', type=Path, required=True)
    m = sub.add_parser('summarize')
    m.add_argument('--inputs', type=Path, required=True)
    m.add_argument('--scores', type=Path, required=True)
    m.add_argument('--timing', type=Path, required=True)
    m.add_argument('--replicates', type=int, default=2000)
    m.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    return {'score': cmd_score, 'timing': cmd_timing, 'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
