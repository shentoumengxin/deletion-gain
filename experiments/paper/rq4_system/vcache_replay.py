"""vCache replay: how often the deployed hit filter rejects valid hits on real prompts.

Question. With no attacker in the stream, how many valid cache hits does the deployed
filter reject (FPR), and how much hit rate does that cost? Three public vCache benchmarks
(Apache-2.0), each replayed in file order:

* ``lmarena``        SemBenchmarkLmArena/train.parquet: chat prompts, class ``ID_Set``,
                     cached answer ``response_gpt-4o-mini``.
* ``search``         SemBenchmarkSearchQueries/other_60k_with_responses.parquet: search
                     queries, class ``id_set``, cached answer ``response_gpt-4o-mini``.
* ``classification`` SemBenchmarkClassification/train.parquet: classification prompts,
                     cached answer = the one-word label ``response_llama_3_8b``; a hit is
                     valid iff the two labels agree (vCache's own correctness rule).

Cache. e5-small-v2 CLS, no prefix, L2 norm, the tokenizer's 512-token truncation. The
candidate is the nearest cached key by cosine (ties within 1e-9 go to the earliest
insertion); cos >= tau is a hit (0.90 main run, ``--tau 0.95`` follow-up); a miss inserts
(prompt, answer). Decision-level numbers
are read on the undefended stream, whose insertions the filter does not change. The exact
defended replay (Algorithm 1: a rejected hit goes to the backend and its (prompt, answer)
is inserted) is run as well and its hit rate reported beside the decision-level one.

Filter. The cached key's 4+2 profile with its cached answer (``build_profile``, float16),
read against the incoming prompt: DG = ``excess_span``, ADL at DG's arg-max variant,
Echo = |O_s* \\ W(q)| (``AnswerColumns``). Rules: the frozen ComQA joint thresholds
(primary, no retuning), the frozen ComQA DG-only threshold, and a same-dataset reference
(200 random half-splits by equivalence class, joint rule fitted on the valid hits of one
half at 5%, FPR read on the other half). A key with fewer than 3 segments (1-2 words) has
no judgeable profile and is a fail-closed miss; it is counted apart from rule rejections.

Efficiency. The replay is block-wise: one matrix product per block against the cache as
it stood before the block, then sequential resolution inside the block. It equals a naive
sequential replay (``tests/test_supp_vcache.py``; re-checked on real prompts at run
time). Only keys that receive a hit are profiled, in worker processes that share a
32-thread cap. Model work runs on cpu-server only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from sentry.cache.defense import textnorm
from sentry.cache.defense.calibrate import AnswerColumns, parse_policy
from sentry.cache.defense.deletion import excess
from sentry.cache.defense.spans import shortened
from sentry.cache.defense.textnorm import normalise_answer

ENCODER = 'intfloat/e5-small-v2'
POLICY = 'multi[count:4+width:2:cap16]/runs'
TAU = 0.90
TIE_EPS = 1e-9
BUDGET = 0.05
RULES = {  # frozen ComQA main-table thresholds (plan, Global Constraints)
    'primary_joint': {'kind': 'joint', 'eta': 0.002645233293430393,
                      'eta_a': 0.01965637207031249},
    'dg_only': {'kind': 'dg_only', 'eta': 0.006797628467845549},
}
DATASETS = {
    'lmarena': {'repo': 'vCache/SemBenchmarkLmArena', 'file': 'SemBenchmarkLmArena/train.parquet',
                'cls': 'ID_Set', 'answer': 'response_gpt-4o-mini', 'valid_by': 'ID_Set',
                'bootstrap_cluster': 'query_class'},
    'search': {'repo': 'vCache/SemBenchmarkSearchQueries',
               'file': 'SemBenchmarkSearchQueries/other_60k_with_responses.parquet',
               'cls': 'id_set', 'answer': 'response_gpt-4o-mini', 'valid_by': 'id_set',
               'bootstrap_cluster': 'query_class'},
    'classification': {'repo': 'vCache/SemBenchmarkClassification',
                       'file': 'SemBenchmarkClassification/train.parquet',
                       'cls': 'response_llama_3_8b', 'answer': 'response_llama_3_8b',
                       'valid_by': 'label (response_llama_3_8b) equality',
                       # 36 labels are too few clusters for a bootstrap; the cached entry
                       # (whose one profile every hit on it reads) is the cluster instead.
                       'bootstrap_cluster': 'cached_entry'},
}
KEY_BINS = (('1-8', 1, 8), ('9-16', 9, 16), ('17-32', 17, 32), ('33-64', 33, 64),
            ('65+', 65, None))
ANSWER_BINS = (('0', 0, 0), ('1', 1, 1), ('2-20', 2, 20), ('21-100', 21, 100),
               ('101-300', 101, 300), ('300+', 301, None))
LOW_SUPPORT = 30


def sha_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- data

def load_stream(name: str, root: Path, limit: Optional[int] = None) -> dict:
    """Only the needed columns, in file order. ``limit`` keeps a stream prefix (pilot only)."""
    import pyarrow.parquet as pq
    spec = DATASETS[name]
    cols = ['id', 'prompt', spec['answer']] + ([spec['cls']] if spec['cls'] != spec['answer'] else [])
    if name == 'classification':
        cols.append('dataset_name')
    table = pq.read_table(root / spec['file'], columns=cols)
    if limit:
        table = table.slice(0, limit)
    d = table.to_pydict()
    prompts = [str(p) for p in d['prompt']]
    answers = [str(a) for a in d[spec['answer']]]
    classes = np.array([str(c) for c in d[spec['cls']]], dtype=object)
    return {'name': name, 'row_id': np.asarray(d['id'], dtype=np.int64), 'prompt': prompts,
            'answer': answers, 'cls': classes,
            'source': np.array(d.get('dataset_name', [name] * len(prompts)), dtype=object),
            'key_words': np.array([len(p.split()) for p in prompts]),
            'answer_words': np.array([len(normalise_answer(a).split()) for a in answers])}


def bin_label(value: int, bins) -> str:
    for label, lo, hi in bins:
        if value >= lo and (hi is None or value <= hi):
            return label
    raise ValueError(f'{value} falls in no bin')


# ---------------------------------------------------------------- replay

def naive_replay(emb: np.ndarray, tau: float = TAU,
                 reject: Optional[Callable[[int, int], bool]] = None,
                 eps: float = TIE_EPS) -> dict:
    """Reference: one query at a time against the whole cache."""
    emb = np.asarray(emb, dtype=np.float64)
    n = len(emb)
    out = _empty_result(n)
    cache, owners = np.empty_like(emb), []
    for i in range(n):
        if owners:
            c = cache[:len(owners)] @ emb[i]
            k = int(np.argmax(c >= c.max() - eps))
            j, cos = owners[k], float(c[k])
            out['nn'][i], out['cos'][i] = j, cos
            if cos >= tau:
                out['hit'][i] = True
                if reject is None or not reject(i, j):
                    continue
                out['rejected'][i] = True
        cache[len(owners)] = emb[i]
        owners.append(i)
        out['inserted'][i] = True
    return out


def replay(emb: np.ndarray, tau: float = TAU,
           reject: Optional[Callable[[int, int], bool]] = None,
           block: int = 1024, eps: float = TIE_EPS) -> dict:
    """Block-wise replay, equal to :func:`naive_replay`.

    Per block: cosines against the cache as it stood before the block (one matrix
    product), then the block's own rows in order against the entries inserted earlier in
    the same block. The nearest key is the earliest insertion within ``eps`` of the
    maximum cosine, so the tie rule does not depend on how the products were grouped.
    ``reject(i, j)`` is asked only on hits; a rejected hit is inserted like a miss.
    """
    emb = np.asarray(emb, dtype=np.float64)
    n, dim = emb.shape
    cache, owner, size = np.empty((n, dim)), np.empty(n, dtype=np.int64), 0
    out = _empty_result(n)
    for s in range(0, n, block):
        e = min(n, s + block)
        q = emb[s:e]
        if size:
            sims = q @ cache[:size].T
            top = sims.max(axis=1)
            first = np.argmax(sims >= (top - eps)[:, None], axis=1)
        gram = q @ q.T
        local: list[int] = []
        for t in range(e - s):
            i = s + t
            j, cos = -1, -np.inf
            if size:
                j, cos = int(owner[first[t]]), float(sims[t, first[t]])
            if local:
                g = gram[t, local]
                g_max = float(g.max())
                if size and top[t] >= g_max - eps:
                    if g_max > top[t]:   # the global max is in-block but an earlier key ties
                        k = int(np.argmax(sims[t] >= g_max - eps))
                        j, cos = int(owner[k]), float(sims[t, k])
                else:
                    k = int(np.argmax(g >= g_max - eps))
                    j, cos = s + local[k], float(g[k])
            if j >= 0:
                out['nn'][i], out['cos'][i] = j, cos
                if cos >= tau:
                    out['hit'][i] = True
                    if reject is None or not reject(i, j):
                        continue
                    out['rejected'][i] = True
            local.append(t)
            out['inserted'][i] = True
        for t in local:
            cache[size], owner[size] = emb[s + t], s + t
            size += 1
    return out


def _empty_result(n: int) -> dict:
    return {'nn': np.full(n, -1, dtype=np.int64), 'cos': np.full(n, -np.inf),
            'hit': np.zeros(n, bool), 'rejected': np.zeros(n, bool),
            'inserted': np.zeros(n, bool)}


def same_replay(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in ('nn', 'hit', 'rejected', 'inserted')) \
        and np.allclose(a['cos'], b['cos'], atol=1e-12, rtol=0)


# ---------------------------------------------------------------- the filter on one hit

def read_hit(profile, anchor: np.ndarray, query: str, key: str, columns: AnswerColumns,
             policy, text: bool = True) -> dict:
    """DG, ADL, Echo of a cached key's profile against the arriving prompt. ``text`` adds
    the echoed words and the deleted words d* (diagnostics; the decision never reads them)."""
    base = {'judgeable': bool(profile.judgeable), 'has_answer': bool(profile.has_answer),
            'dg': float('nan'), 'adl': float('nan'), 'echo': -1, 'base_cos_profile': float('nan'),
            'best_name': None, 'deleted': None, 'echo_words': []}
    if not profile.judgeable:
        return base
    reading = excess(profile, anchor)
    adl, echo = columns(query, reading)
    words = sorted(reading.echo_tokens - textnorm.content_tokens(query)) \
        if text and reading.echo_tokens is not None else []
    out = dict(base, dg=reading.excess_span, base_cos_profile=reading.base_cos,
               adl=float('nan') if adl is None else adl, echo=-1 if echo is None else echo,
               best_name=reading.best_name)
    if text:
        out.update(echo_words=words,
                   deleted=shortened(policy, key).removed_texts[reading.best_index])
    return out


def rule_rejects(dg, adl, echo, judgeable, has_answer, rule: dict) -> np.ndarray:
    """Rejected hits under ``rule``. Unjudgeable or non-finite DG is a fail-closed miss;
    a profile without answer fields keeps the DG veto (Algorithm 1)."""
    dg, judgeable = np.asarray(dg, float), np.asarray(judgeable, bool)
    ok = judgeable & np.isfinite(dg)
    over = np.where(ok, dg, -np.inf) > rule['eta']
    if rule['kind'] == 'joint':
        fires = ~np.asarray(has_answer, bool) | (np.nan_to_num(adl, nan=-np.inf) > rule['eta_a']) \
            | (np.asarray(echo) >= 1)
        over &= fires
    return over | ~ok


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def cluster_bootstrap(num: np.ndarray, den: np.ndarray, clusters: np.ndarray,
                      reps: int = 2000, seed: int = 0) -> list:
    """95% percentile CI of sum(num)/sum(den), resampling whole clusters."""
    _, inv = np.unique(clusters, return_inverse=True)
    cn = np.bincount(inv, weights=np.asarray(num, float))
    cd = np.bincount(inv, weights=np.asarray(den, float))
    rng = np.random.default_rng(seed)
    c = len(cn)
    counts = rng.multinomial(c, np.full(c, 1.0 / c), size=reps)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratios = (counts @ cn) / (counts @ cd)
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return [float(lo), float(hi)]


def rate(k: int, n: int) -> dict:
    return {'k': int(k), 'n': int(n), 'rate': (k / n) if n else float('nan'),
            'wilson95': wilson(int(k), int(n))}


def rule_metrics(h: dict, rejected: np.ndarray, n_prompts: int, clusters: np.ndarray,
                 rule: dict) -> dict:
    """FPR, hit-rate loss and invalid-hit rejection for one rule on the undefended stream."""
    valid, judg = h['valid'], h['judgeable']
    fail = ~judg
    by_rule = rejected & judg
    ones = np.ones(len(valid))
    out = {
        'fpr': dict(rate(np.sum(rejected & valid), np.sum(valid)),
                    cluster_bootstrap95=cluster_bootstrap(rejected & valid, valid, clusters)),
        'fpr_rule_part': rate(np.sum(by_rule & valid), np.sum(valid)),
        'fpr_fail_closed_part': rate(np.sum(fail & valid), np.sum(valid)),
        'fpr_on_judgeable_valid': rate(np.sum(by_rule & valid), np.sum(judg & valid)),
        'hit_rate_loss': dict(rate(np.sum(rejected), len(valid)),
                              cluster_bootstrap95=cluster_bootstrap(rejected, ones, clusters)),
        'invalid_hit_rejection': rate(np.sum(rejected & ~valid), np.sum(~valid)),
        'served_hit_rate_decision_level': rate(len(valid) - np.sum(rejected), n_prompts),
        'served_valid_share': rate(np.sum(valid & ~rejected), len(valid) - np.sum(rejected)),
        'served_valid_hit_rate': rate(np.sum(valid & ~rejected), n_prompts),
    }
    if rule['kind'] == 'joint':   # which witness seconded the veto on rejected valid hits
        adl = np.nan_to_num(h['adl'], nan=-np.inf)
        out['rule_rejected_valid_trigger'] = {
            'n': int(np.sum(by_rule & valid)),
            'adl_above_eta_a': int(np.sum(by_rule & valid & (adl > rule['eta_a']))),
            'echo_ge_1': int(np.sum(by_rule & valid & (h['echo'] >= 1))),
            'no_answer_fields': int(np.sum(by_rule & valid & ~h['has_answer']))}
    return out


def by_source(h: dict, rejected: np.ndarray) -> dict:
    """Classification mixes three source datasets; report each (one source elsewhere)."""
    out = {}
    for src in np.unique(h['source']):
        m = h['source'] == src
        v = m & h['valid']
        out[str(src)] = {'hits': int(m.sum()), 'valid_hits': int(v.sum()),
                         'fpr': rate(int((v & rejected).sum()), int(v.sum())),
                         'hit_rate_loss': rate(int((m & rejected).sum()), int(m.sum()))}
    return out


def rejection_attribution(h: dict, rej_joint: np.ndarray, rej_dg: np.ndarray,
                          rule: dict = RULES['primary_joint']) -> dict:
    """Which part of the joint rule decides, on valid hits and on all hits.

    ``dg_over_eta``: DG clears the joint eta. Of those, the answer check either fires
    (rejected; split by Echo >= 1 only, ADL > eta_A only, both, or no answer fields) or
    stays silent (rescued). ``vs_dg_only`` compares the joint verdict with the frozen DG-only
    verdict on the same hits. Fail-closed (unjudgeable) hits are counted apart.
    """
    judg = np.asarray(h['judgeable'], bool)
    dg = np.where(judg, np.nan_to_num(np.asarray(h['dg'], float), nan=-np.inf), -np.inf)
    adl = np.nan_to_num(np.asarray(h['adl'], float), nan=-np.inf) > rule['eta_a']
    echo = np.asarray(h['echo']) >= 1
    no_ans = judg & ~np.asarray(h['has_answer'], bool)
    over = judg & (dg > rule['eta'])
    fires = adl | echo | no_ans

    def split(m):
        return {'hits': int(m.sum()), 'dg_over_eta': int((m & over).sum()),
                'rejected_by_rule': int((m & over & fires).sum()),
                'echo_only': int((m & over & echo & ~adl).sum()),
                'adl_only': int((m & over & adl & ~echo).sum()),
                'echo_and_adl': int((m & over & adl & echo).sum()),
                'no_answer_fields': int((m & over & no_ans).sum()),
                'rescued_by_answer_check': int((m & over & ~fires).sum()),
                'fail_closed': int((m & ~judg).sum()),
                'joint_only': int((m & rej_joint & ~rej_dg).sum()),
                'dg_only_only': int((m & rej_dg & ~rej_joint).sum()),
                'both_rules': int((m & rej_joint & rej_dg).sum())}

    valid = np.asarray(h['valid'], bool)
    return {'valid_hits': split(valid), 'invalid_hits': split(~valid),
            'all_hits': split(np.ones(len(valid), bool)), 'rule': rule}


def first_order(h: dict, cos_min: float) -> dict:
    """Frozen-rule FPR on the hits of an existing stream whose cosine clears ``cos_min``.

    A first-order reading of a stricter retrieval threshold: the nearest keys are those of
    the looser stream, so insertions a stricter cache would make are ignored."""
    keep = np.asarray(h['cos'], float) >= cos_min
    valid = np.asarray(h['valid'], bool)
    out = {'cos_min': cos_min, 'hits_kept': rate(int(keep.sum()), len(keep)),
           'valid_share_of_kept': rate(int((keep & valid).sum()), int(keep.sum()))}
    for name in RULES:
        rej = np.asarray(h[f'rejected_{name}'], bool)
        out[name] = {'fpr': rate(int((keep & valid & rej).sum()), int((keep & valid).sum())),
                     'hit_rate_loss': rate(int((keep & rej).sum()), int(keep.sum())),
                     'invalid_hit_rejection': rate(int((keep & ~valid & rej).sum()),
                                                   int((keep & ~valid).sum()))}
    return out


def hits_table(path: Path) -> dict:
    """Per-row hit dump (``<dataset>_hits.jsonl``) as aligned arrays."""
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    cols = ('cos', 'valid', 'judgeable', 'has_answer', 'dg', 'adl', 'echo',
            'rejected_primary_joint', 'rejected_dg_only')
    return {k: np.array([r[k] for r in rows]) for k in cols}


def posthoc(hits_dir: Path, cos_min: float) -> dict:
    """First-order stricter-threshold reading and rejection attribution from saved hits."""
    out = {}
    for name in DATASETS:
        path = hits_dir / f'{name}_hits.jsonl'
        if not path.exists():
            continue
        h = hits_table(path)
        out[name] = {'hits_jsonl_sha256': sha_file(path),
                     'first_order': first_order(h, cos_min),
                     'rejection_attribution': rejection_attribution(
                         h, h['rejected_primary_joint'], h['rejected_dg_only'])}
    return out


def attach(main: Path, key: str, part: dict) -> None:
    """Add ``part`` under ``key`` of the task summary; existing keys are never rewritten."""
    report = json.loads(main.read_text())
    assert key not in report, f'{key} already present in {main}'
    report[key] = part
    main.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')


def group_table(h: dict, rejected: np.ndarray) -> dict:
    """FPR and hit-rate loss per key-length x answer-length cell, plus both margins."""
    kb = np.array([bin_label(w, KEY_BINS) for w in h['key_words']])
    ab = np.array([bin_label(w, ANSWER_BINS) for w in h['answer_words']])
    valid, fail = h['valid'], ~h['judgeable']

    def cell(mask):
        nv, nh = int(np.sum(mask & valid)), int(np.sum(mask))
        return {'hits': nh, 'valid_hits': nv,
                'rejected_valid': int(np.sum(mask & valid & rejected)),
                'rejected_hits': int(np.sum(mask & rejected)),
                'fail_closed_hits': int(np.sum(mask & fail)),
                'fpr': (np.sum(mask & valid & rejected) / nv) if nv else None,
                'fpr_wilson95': wilson(int(np.sum(mask & valid & rejected)), nv) if nv else None,
                'hit_rate_loss': (np.sum(mask & rejected) / nh) if nh else None,
                'low_support_fpr': nv < LOW_SUPPORT, 'low_support_loss': nh < LOW_SUPPORT}

    cells = [dict(key_words=k, answer_words=a, **cell((kb == k) & (ab == a)))
             for k, _, _ in KEY_BINS for a, _, _ in ANSWER_BINS]
    return {'cells': [c for c in cells if c['hits'] or c['answer_words'] != '0'],
            'by_key_words': [dict(key_words=k, **cell(kb == k)) for k, _, _ in KEY_BINS],
            'by_answer_words': [dict(answer_words=a, **cell(ab == a))
                                for a, _, _ in ANSWER_BINS if a != '0' or np.any(ab == a)]}


def recalibrate(h: dict, reps: int = 200, seed: int = 20260923, budget: float = BUDGET) -> dict:
    """Same-dataset reference: split equivalence classes (key's class) into halves, fit the
    joint rule on the judgeable valid hits of one half, read the other half."""
    groups = h['key_cls']
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fit_ok = h['judgeable'] & h['has_answer']
    rows = {k: [] for k in ('eta', 'eta_a', 'fpr', 'fpr_on_judgeable_valid', 'hit_rate_loss',
                            'invalid_hit_rejection', 'n_fit', 'n_eval_valid')}
    unreachable = 0
    for _ in range(reps):
        half = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
        in_a = np.array([g in half for g in groups])
        fit = in_a & h['valid'] & fit_ok
        if not fit.any():   # a half without valid hits has nothing to calibrate on
            for k in rows:
                rows[k].append(np.nan)
            continue
        eta, eta_a = R.joint_thresholds(h['dg'][fit], h['adl'][fit], h['echo'][fit], budget)
        unreachable += eta == float('-inf')
        rej = rule_rejects(h['dg'], h['adl'], h['echo'], h['judgeable'], h['has_answer'],
                           {'kind': 'joint', 'eta': eta, 'eta_a': eta_a})
        ev = ~in_a
        v = ev & h['valid']
        rows['eta'].append(eta)
        rows['eta_a'].append(eta_a)
        rows['fpr'].append(np.mean(rej[v]) if v.any() else np.nan)
        vj = v & h['judgeable']
        rows['fpr_on_judgeable_valid'].append(np.mean(rej[vj]) if vj.any() else np.nan)
        rows['hit_rate_loss'].append(np.mean(rej[ev]) if ev.any() else np.nan)
        iv = ev & ~h['valid']
        rows['invalid_hit_rejection'].append(np.mean(rej[iv]) if iv.any() else np.nan)
        rows['n_fit'].append(int(fit.sum()))
        rows['n_eval_valid'].append(int(v.sum()))
    summary = {k: _spread(v) for k, v in rows.items()}
    return {'reps': reps, 'seed': seed, 'budget': budget, 'group': 'key equivalence class',
            'n_groups': int(len(uniq)), 'splits_with_unreachable_budget': int(unreachable),
            **summary}


def _spread(values) -> Optional[dict]:
    """Mean/sd over the splits where the value is finite; -inf (the answer check alone
    already fits the budget, so eta = -inf) and NaN (empty half) are counted, not averaged."""
    v = np.asarray(values, float)
    fin = v[np.isfinite(v)]
    out = {'n_splits': int(len(fin)), 'n_neg_inf': int(np.sum(v == -np.inf)),
           'n_nan': int(np.sum(np.isnan(v)))}
    if not len(fin):
        return out
    return dict(out, mean=float(fin.mean()), sd=float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
                min=float(fin.min()), max=float(fin.max()))


def quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.95, 0.99)) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'q{int(q * 100)}': float(np.quantile(x, q)) for q in qs} if len(x) else {}


# ---------------------------------------------------------------- model workers

_WORKER: dict = {}


def _worker_init(encoder: str, threads: int) -> None:
    # RAYON_* caps the fast tokenizer's pool, which otherwise sizes itself to all cores.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'RAYON_NUM_THREADS', 'RAYON_RS_NUM_CPUS'):
        os.environ[var] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from sentry.embeddings import TransformerCLSEmbedder
    _WORKER['embedder'] = TransformerCLSEmbedder(encoder)
    _WORKER['policy'] = parse_policy(POLICY)


#: Tokens per forward batch. Long texts in batches of 128 are ~1.8x slower on these CPUs
#: than in batches of 8 (measured on cpu-server); the budget picks the size per batch.
TOKEN_BUDGET = 2048


def budget_encode(emb, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``emb.encode`` over batches sized to :data:`TOKEN_BUDGET`; also the untruncated
    token count of each text. Batching changes padding only, not what a text encodes to."""
    if not hasattr(emb, 'tokenizer'):     # tokenizer-free test embedders
        return np.asarray(emb.encode(texts), dtype=np.float32), np.zeros(len(texts), int)
    full = [len(t) for t in emb.tokenizer(texts, add_special_tokens=True,
                                          truncation=False)['input_ids']]
    used = np.minimum(full, int(emb.tokenizer.model_max_length))
    order = np.argsort(-used, kind='stable')
    out = np.zeros((len(texts), emb.dimension), dtype=np.float32)
    k = 0
    while k < len(order):
        size = int(min(128, max(8, TOKEN_BUDGET // max(1, used[order[k]]))))
        part = order[k:k + size]
        emb.batch_size = len(part)
        out[part] = emb.encode([texts[q] for q in part])
        k += size
    return out, np.asarray(full)


def _encode_chunk(item):
    idx, texts = item
    vecs, tok = budget_encode(_WORKER['embedder'], texts)
    return idx, vecs, tok


def _profile_chunk(items):
    from sentry.cache.defense.deletion import build_profile
    emb, policy = _WORKER['embedder'], _WORKER['policy']
    return [(j, build_profile(key, emb, policy, storage_dtype='float16', answer=answer))
            for j, key, answer in items]


def _worker_meta(_):
    emb = _WORKER['embedder']
    return {'model_max_length': int(emb.tokenizer.model_max_length),
            'commit': getattr(emb.model.config, '_commit_hash', None),
            'dimension': emb.dimension, 'batch_size': emb.batch_size, 'pooling': emb.pooling}


def encode_all(pool, prompts: list[str], chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings in stream order; batches are formed over length-sorted prompts so a
    batch pads to similar lengths (CLS output does not depend on the padding)."""
    order = np.argsort([len(p) for p in prompts], kind='stable')[::-1]
    jobs = [(order[s:s + chunk], [prompts[k] for k in order[s:s + chunk]])
            for s in range(0, len(order), chunk)]
    emb, ntok = None, np.zeros(len(prompts), dtype=np.int64)
    for idx, vec, tok in pool.imap_unordered(_encode_chunk, jobs):
        if emb is None:
            emb = np.zeros((len(prompts), vec.shape[1]), dtype=np.float32)
        emb[idx], ntok[idx] = vec, tok
    return emb, ntok


def _encode_texts(texts):
    return texts, budget_encode(_WORKER['embedder'], texts)[0]


class BankEmbedder:
    """``encode`` reads vectors computed beforehand in one length-sorted, de-duplicated
    pass. ``build_profile`` runs unchanged on top of it; only the batching differs from
    calling the model per key (padding changes a CLS vector by float noise, far below the
    float16 storage rounding). :func:`bank_parity` measures the difference at run time."""

    def __init__(self, model_name: str, bank: dict):
        self.model_name, self.bank = model_name, bank

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([self.bank[t] for t in texts])


def profile_texts(key: str, answer: str, policy) -> list[str]:
    """Every text ``build_profile(key, ..., answer=answer)`` asks the embedder for."""
    texts = [key, *shortened(policy, key).all_texts]
    cleaned = normalise_answer(answer)
    return texts + [cleaned] if cleaned else texts


def compute_profiles(pool, keys: list[int], prompts, answers, policy,
                     max_chunk: int = 256) -> dict:
    """4+2 profiles (float16, with the cached answer) for ``keys``."""
    from sentry.cache.defense.deletion import build_profile
    if not keys:
        return {}
    texts: set = set()
    for j in keys:
        texts.update(profile_texts(prompts[j], answers[j], policy))
    ordered = sorted(texts, key=len, reverse=True)
    chunk = int(min(max_chunk, max(16, -(-len(ordered) // 32))))   # small fetches still spread
    jobs = [ordered[s:s + chunk] for s in range(0, len(ordered), chunk)]
    bank = {}
    for part, vecs in pool.imap_unordered(_encode_texts, jobs):
        bank.update(zip(part, vecs))
    emb = BankEmbedder(ENCODER, bank)
    return {j: build_profile(prompts[j], emb, policy, storage_dtype='float16',
                             answer=answers[j]) for j in keys}


def bank_parity(pool, pairs, prompts, answers, emb, policy) -> dict:
    """Largest DG/ADL/cos difference between bank profiles and ``build_profile`` called
    with the model itself, over (query, key) hit pairs."""
    keys = sorted({j for _, j in pairs})
    direct = {}
    items = [(j, prompts[j], answers[j]) for j in keys]
    for part in pool.imap_unordered(_profile_chunk, [items[s:s + 4] for s in range(0, len(items), 4)]):
        direct.update(part)
    banked = compute_profiles(pool, keys, prompts, answers, policy)
    cols = AnswerColumns()
    worst = {'dg': 0.0, 'adl': 0.0, 'base_cos': 0.0, 'echo_mismatch': 0, 'decision_mismatch': 0}
    for i, j in pairs:
        a = read_hit(direct[j], emb[i], prompts[i], prompts[j], cols, policy, text=False)
        b = read_hit(banked[j], emb[i], prompts[i], prompts[j], cols, policy, text=False)
        if not a['judgeable']:
            continue
        worst['dg'] = max(worst['dg'], abs(a['dg'] - b['dg']))
        worst['adl'] = max(worst['adl'], abs(a['adl'] - b['adl']))
        worst['base_cos'] = max(worst['base_cos'], abs(a['base_cos_profile'] - b['base_cos_profile']))
        worst['echo_mismatch'] += int(a['echo'] != b['echo'])
        ra, rb = (rule_rejects([x['dg']], [x['adl']], [x['echo']], [True], [x['has_answer']],
                               RULES['primary_joint'])[0] for x in (a, b))
        worst['decision_mismatch'] += int(ra != rb)
    return {'n_pairs': len(pairs), 'n_keys': len(keys), 'max_abs_diff': worst}


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- one dataset

class Decider:
    """``reject(i, j)`` for the exact defended replay (Algorithm 1).

    Every decision reads the real profile of the key that was hit; nothing is decided
    provisionally. When a hit lands on a key without a profile the replay pauses and
    profiles, in one batch, (a) every unprofiled entry the defended cache holds at that
    moment and (b) the prompts of the next ``window`` that the current cache predicts will
    be inserted (a miss, or a hit the rule rejects). (b) is only prefetching: a wrong
    prediction costs a later pause, never a different decision.
    """

    def __init__(self, pool, profiles, emb, s, rule, policy, window: int = 2048,
                 tau: float = TAU):
        self.pool, self.profiles, self.emb, self.s, self.tau = pool, profiles, emb, s, tau
        self.rule, self.policy, self.columns, self.window = rule, policy, AnswerColumns(), window
        n = len(emb)
        self.served = np.zeros(n, bool)
        self.profiled = np.zeros(n, bool)
        self.profiled[list(profiles)] = True
        self.fetches = self.fetched = self.fail_closed = 0

    def _profile(self, todo) -> None:
        todo = sorted(set(int(k) for k in todo) - set(np.flatnonzero(self.profiled).tolist()))
        if not todo:
            return
        self.profiles.update(compute_profiles(self.pool, todo, self.s['prompt'],
                                              self.s['answer'], self.policy))
        self.profiled[todo] = True
        self.fetched += len(todo)

    def _decide(self, i: int, j: int) -> tuple[dict, bool]:
        prompts = self.s['prompt']
        r = read_hit(self.profiles[j], self.emb[i], prompts[i], prompts[j], self.columns,
                     self.policy, text=False)
        return r, bool(rule_rejects([r['dg']], [r['adl']], [r['echo']], [r['judgeable']],
                                    [r['has_answer']], self.rule)[0])

    def _predict_inserts(self, i: int, cache: np.ndarray) -> list[int]:
        """Replay the next ``window`` prompts against the current cache with the profiles
        at hand: a miss or a rejected hit is predicted to be inserted and can capture later
        prompts in the window; a hit on an entry without a profile is predicted served."""
        win = np.arange(i, min(len(self.emb), i + self.window))
        q = self.emb[win].astype(np.float64)
        sims = q @ self.emb[cache].astype(np.float64).T
        best = np.argmax(sims, axis=1)
        gram = q @ q.T
        local, likely = [], []
        for t, qi in enumerate(win):
            j, c = int(cache[best[t]]), float(sims[t, best[t]])
            if local:
                g = gram[t, local]
                k = int(np.argmax(g))
                if g[k] > c:
                    j, c = int(win[local[k]]), float(g[k])
            if c >= self.tau and not (self.profiled[j] and self._decide(int(qi), j)[1]):
                continue
            local.append(t)
            likely.append(int(qi))
        return likely

    def _fetch(self, i: int) -> None:
        cache = np.flatnonzero(~self.served[:i])
        self._profile(cache[~self.profiled[cache]])
        self._profile(self._predict_inserts(i, cache))
        self.fetches += 1
        if self.fetches % 20 == 0:
            log(f'  defended replay at prompt {i}: {self.fetches} pauses, '
                f'{self.fetched} extra profiles')

    def __call__(self, i: int, j: int) -> bool:
        if not self.profiled[j]:
            self._fetch(i)
        r, rejected = self._decide(i, j)
        self.fail_closed += not r['judgeable']
        self.served[i] = not rejected
        return rejected


def defended_replay(pool, emb, s, profiles, rule, policy, block,
                    window: int = 2048, tau: float = TAU) -> tuple[dict, dict]:
    """Algorithm 1 exactly: a rejected hit goes to the backend and its (prompt, answer) is
    inserted, so later nearest keys (and their decisions) follow the defended cache."""
    dec = Decider(pool, profiles, emb, s, rule, policy, window, tau)
    out = replay(emb, tau, reject=dec, block=block)
    valid = s['cls'][np.maximum(out['nn'], 0)] == s['cls']
    served = out['hit'] & ~out['rejected']
    n, hits = len(emb), int(out['hit'].sum())
    return {'profile_pauses': dec.fetches, 'extra_profiles_computed': dec.fetched,
            'retrieval_hits': hits, 'retrieval_hit_rate': hits / n,
            'rejected_hits': int(out['rejected'].sum()),
            'fail_closed_hits': int(dec.fail_closed),
            'served_hits': int(served.sum()), 'served_hit_rate': rate(int(served.sum()), n),
            'valid_hits': int((out['hit'] & valid).sum()),
            'rejected_valid_hits': int((out['rejected'] & valid).sum()),
            'fpr_in_defended_stream': rate(int((out['rejected'] & valid).sum()),
                                           int((out['hit'] & valid).sum())),
            'served_valid_share': rate(int((served & valid).sum()), int(served.sum())),
            'cache_entries': int(out['inserted'].sum())}, out


def replay_checks(emb: np.ndarray, block: int, seed: int = 7, tau: float = TAU) -> dict:
    """Block replay equals naive replay on real vectors: the first 5,000 prompts and a
    random 500-prompt subsample kept in stream order."""
    rng = np.random.default_rng(seed)
    out = {}
    n = min(5000, len(emb))
    out['prefix'] = {'n': n, 'equal': same_replay(naive_replay(emb[:n], tau),
                                                  replay(emb[:n], tau, block=block))}
    sub = np.sort(rng.choice(len(emb), size=min(500, len(emb)), replace=False))
    a, b = naive_replay(emb[sub], tau), replay(emb[sub], tau, block=97)
    out['random500'] = {'n': int(len(sub)), 'seed': seed, 'block': 97,
                        'equal': same_replay(a, b), 'hits': int(a['hit'].sum())}
    return out


def run_dataset(name: str, args, pool, policy) -> tuple[dict, list]:
    t0 = time.perf_counter()
    wall = {}
    s = load_stream(name, args.root, args.limit)
    n = len(s['prompt'])
    wall['load'] = time.perf_counter() - t0

    emb_path = args.out_dir / f'{name}_emb.npy'
    tok_path = args.out_dir / f'{name}_ntok.npy'
    digest = sha_text('\n'.join(s['prompt']) + f'|{ENCODER}|{n}')
    meta_path = args.out_dir / f'{name}_emb.sha'
    t = time.perf_counter()
    if emb_path.exists() and meta_path.exists() and meta_path.read_text() == digest:
        emb, ntok = np.load(emb_path), np.load(tok_path)
    else:
        log(f'{name}: encoding {n} prompts')
        emb, ntok = encode_all(pool, s['prompt'])
        np.save(emb_path, emb)
        np.save(tok_path, ntok)
        meta_path.write_text(digest)
    wall['encode'] = time.perf_counter() - t

    t = time.perf_counter()
    und = replay(emb, args.tau, block=args.block)
    wall['replay'] = time.perf_counter() - t
    t = time.perf_counter()
    checks = replay_checks(emb, args.block, tau=args.tau)
    wall['replay_checks'] = time.perf_counter() - t
    assert checks['prefix']['equal'] and checks['random500']['equal'], checks
    log(f"{name}: undefended replay {int(und['hit'].sum())}/{n} hits; checks {checks}")

    hit_i = np.flatnonzero(und['hit'])
    hit_j = und['nn'][hit_i]
    t = time.perf_counter()
    prof_path = args.out_dir / f'{name}_profiles.pkl'
    profiles = {}
    if prof_path.exists() and args.reuse_profiles:
        profiles = pickle.loads(prof_path.read_bytes())
    todo = sorted(set(hit_j.tolist()) - set(profiles))
    log(f'{name}: profiling {len(todo)} hit keys')
    profiles.update(compute_profiles(pool, todo, s['prompt'], s['answer'], policy))
    wall['profiles_undefended'] = time.perf_counter() - t
    if args.save_profiles and todo:
        prof_path.write_bytes(pickle.dumps(profiles, protocol=5))
    n_profiles_undefended = len(set(hit_j.tolist()))
    t = time.perf_counter()
    sample = np.random.default_rng(13).choice(len(hit_i), size=min(200, len(hit_i)), replace=False)
    parity = bank_parity(pool, [(int(hit_i[k]), int(hit_j[k])) for k in sample],
                         s['prompt'], s['answer'], emb, policy)
    wall['bank_parity'] = time.perf_counter() - t
    log(f'{name}: bank parity {parity}')

    t = time.perf_counter()
    columns = AnswerColumns()
    reads = [read_hit(profiles[j], emb[i], s['prompt'][i], s['prompt'][j], columns, policy)
             for i, j in zip(hit_i, hit_j)]
    wall['read_hits'] = time.perf_counter() - t
    h = {'i': hit_i, 'j': hit_j, 'cos': und['cos'][hit_i],
         'valid': s['cls'][hit_j] == s['cls'][hit_i],
         'key_cls': s['cls'][hit_j], 'query_cls': s['cls'][hit_i],
         'key_words': s['key_words'][hit_j], 'answer_words': s['answer_words'][hit_j],
         'key_tokens': ntok[hit_j], 'source': s['source'][hit_i],
         **{k: np.array([r[k] for r in reads]) for k in
            ('judgeable', 'has_answer', 'dg', 'adl', 'echo', 'base_cos_profile')}}
    h['judgeable'] = h['judgeable'].astype(bool)
    h['has_answer'] = h['has_answer'].astype(bool)
    clusters = h['j'] if DATASETS[name]['bootstrap_cluster'] == 'cached_entry' else h['query_cls']
    rejected = {r: rule_rejects(h['dg'], h['adl'], h['echo'], h['judgeable'], h['has_answer'],
                                spec) for r, spec in RULES.items()}

    t = time.perf_counter()
    recal = recalibrate(h, reps=args.recal_reps)
    wall['recalibrate'] = time.perf_counter() - t

    defended = {}
    t = time.perf_counter()
    for r in RULES:
        log(f'{name}: defended replay under {r}')
        defended[r], dout = defended_replay(pool, emb, s, profiles, RULES[r], policy, args.block,
                                            tau=args.tau)
        np.savez_compressed(args.out_dir / f'{name}_defended_{r}.npz', **dout)
        log(f'{name}: defended {r} {json.dumps(defended[r], default=_json_default)}')
        if args.save_profiles:   # checkpoint: a later crash does not redo this work
            prof_path.write_bytes(pickle.dumps(profiles, protocol=5))
        decision = int(len(hit_i) - rejected[r].sum())
        defended[r]['served_hit_rate_decision_level'] = decision / n
        defended[r]['served_hit_rate_gap_exact_minus_decision'] = \
            defended[r]['served_hit_rate']['rate'] - decision / n
    wall['defended_replays'] = time.perf_counter() - t

    np.savez_compressed(args.out_dir / f'{name}_undefended.npz', **und, ntok=ntok,
                        row_id=s['row_id'])
    rows_path = args.out_dir / f'{name}_hits.jsonl'
    with open(rows_path, 'w', encoding='utf-8') as f:
        for k, (i, j) in enumerate(zip(hit_i, hit_j)):
            r = reads[k]
            f.write(json.dumps({
                'i': int(i), 'row_id': int(s['row_id'][i]), 'j': int(j),
                'key_row_id': int(s['row_id'][j]), 'cos': float(h['cos'][k]),
                'valid': bool(h['valid'][k]), 'query_class': str(h['query_cls'][k]),
                'key_class': str(h['key_cls'][k]), 'key_words': int(h['key_words'][k]),
                'key_tokens': int(h['key_tokens'][k]), 'answer_words': int(h['answer_words'][k]),
                'judgeable': r['judgeable'], 'has_answer': r['has_answer'],
                'dg': r['dg'], 'adl': r['adl'], 'echo': int(r['echo']),
                'echo_words': r['echo_words'], 'best_name': r['best_name'],
                'deleted': r['deleted'], 'base_cos_profile': r['base_cos_profile'],
                **{f'rejected_{rn}': bool(rejected[rn][k]) for rn in RULES}}) + '\n')

    valid = h['valid']
    judg = h['judgeable']
    long_mask = h['key_words'] > 32
    uniq_keys = np.unique(hit_j)
    kw = s['key_words'][uniq_keys]
    summary = {
        'n_prompts': n, 'n_classes': int(len(np.unique(s['cls']))),
        'stream': {'hit_rate': rate(len(hit_i), n), 'cache_entries': int(und['inserted'].sum()),
                   'valid_hit_share': rate(int(valid.sum()), len(hit_i)),
                   'valid_hit_rate': rate(int(valid.sum()), n),
                   'unique_hit_keys': int(len(uniq_keys)),
                   'profiles_built_for_undefended_hits': n_profiles_undefended},
        'cosine_only': {'fpr': 0.0, 'hit_rate_loss': 0.0,
                        'served_hit_rate': len(hit_i) / n,
                        'served_valid_share': float(valid.mean()) if len(valid) else None,
                        'note': 'undefended cache; the filter only removes hits'},
        'rules': {r: dict(rule_metrics(h, rejected[r], n, clusters, RULES[r]),
                          thresholds=RULES[r], by_query_source=by_source(h, rejected[r]))
                  for r in RULES},
        'recalibrated_joint_reference': recal,
        'defended_replay': defended,
        'rejection_attribution': rejection_attribution(h, rejected['primary_joint'],
                                                       rejected['dg_only']),
        'groups_primary_joint': group_table(h, rejected['primary_joint']),
        'groups_dg_only': group_table(h, rejected['dg_only']),
        'long_keys_gt32_words': {
            'hits': rate(int(long_mask.sum()), len(hit_i)),
            'valid_hits': rate(int((long_mask & valid).sum()), int(valid.sum())),
            'unique_hit_keys': rate(int((kw > 32).sum()), len(uniq_keys)),
            'hits_key_over_512_tokens': rate(int((h['key_tokens'] > 512).sum()), len(hit_i))},
        'unjudgeable_fail_closed': {
            'hits': rate(int((~judg).sum()), len(hit_i)),
            'valid_hits': rate(int((~judg & valid).sum()), int(valid.sum())),
            'unique_hit_keys': int(len(np.unique(hit_j[~judg]))),
            'key_words_of_unjudgeable': sorted(set(int(w) for w in h['key_words'][~judg]))},
        'no_answer_fields_hits': int((~h['has_answer'] & judg).sum()),
        'valid_hit_scores': {
            'dg': quantiles(h['dg'][valid & judg]), 'adl': quantiles(h['adl'][valid & judg]),
            'echo_ge_1_share': float(np.mean(h['echo'][valid & judg] >= 1)) if (valid & judg).any() else None,
            'dg_above_primary_eta_share': float(np.mean(h['dg'][valid & judg] > RULES['primary_joint']['eta']))
            if (valid & judg).any() else None},
        'replay_equivalence_checks': checks,
        'bank_vs_direct_build_profile': parity,
        'bootstrap': {'reps': 2000, 'cluster': DATASETS[name]['bootstrap_cluster']},
        'wall_seconds': {k: round(v, 1) for k, v in wall.items()},
    }
    candidates = [k for k in range(len(hit_i))
                  if rejected['primary_joint'][k] and valid[k] and judg[k]]
    examples = []
    rng = np.random.default_rng(11)
    for k in rng.permutation(candidates)[:3] if candidates else []:
        r, i, j = reads[k], int(hit_i[k]), int(hit_j[k])
        examples.append({'dataset': name, 'query_row_id': int(s['row_id'][i]),
                         'key_row_id': int(s['row_id'][j]), 'key': s['prompt'][j],
                         'query': s['prompt'][i], 'answer_head': s['answer'][j][:300],
                         'cos': float(h['cos'][k]), 'dg': r['dg'], 'adl': r['adl'],
                         'echo': int(r['echo']), 'echo_words': r['echo_words'],
                         'deleted_words': r['deleted'], 'best_variant': r['best_name']})
    summary['per_row_outputs'] = {p.name: sha_file(p) for p in sorted(args.out_dir.glob(f'{name}_*'))
                                  if p.suffix in ('.jsonl', '.npz', '.npy')}
    return summary, examples


def w_definition() -> dict:
    return {
        'function': 'sentry.cache.defense.textnorm.content_tokens',
        'steps': ['strip markdown characters matching [*_`#>]+ (normalise_answer)',
                  'collapse whitespace', 'lowercase',
                  "tokens = regex [a-z0-9][a-z0-9'\\-]* (ASCII letters and digits only)",
                  "strip leading/trailing ' and - from each token",
                  f'drop {len(textnorm.FUNCTION_WORDS)} general English function words',
                  'no stemming, no lemmatisation, set semantics (no counts)'],
        'function_words_count': len(textnorm.FUNCTION_WORDS),
        'function_words': sorted(textnorm.FUNCTION_WORDS),
        'function_words_sha256': sha_text(' '.join(sorted(textnorm.FUNCTION_WORDS))),
        'echo': '|(W(d*) ∩ W(y)) \\ W(q)|, d* = words of k the DG arg-max variant dropped',
        'applied_to': 'key deletion d*, normalised cached answer y, incoming prompt q',
    }


def pick_examples(candidates: list) -> list:
    """Three rejected valid hits for the report: one per dataset first, in dataset order."""
    by_ds = {}
    for e in candidates:
        by_ds.setdefault(e['dataset'], []).append(e)
    picked = [by_ds[d][0] for d in DATASETS if d in by_ds][:3]
    rest = [e for e in candidates if e not in picked]
    return picked + rest[: 3 - len(picked)]


def merge(paths: list, out: Path) -> None:
    """Combine per-dataset summaries (one run per dataset) into the task summary."""
    parts = [json.loads(Path(q).read_text()) for q in paths]
    base = dict(parts[0])
    for key in ('code_sha256', 'W'):
        assert all(q[key] == base[key] for q in parts), f'{key} differs between runs'
    for key in ('encoder', 'tau', 'policy', 'rules', 'limit'):
        assert all(q['config'][key] == base['config'][key] for q in parts), key
    base['datasets'] = {k: v for q in parts for k, v in q['datasets'].items()}
    base['inputs'] = {k: v for q in parts for k, v in q['inputs'].items()}
    base['config'] = dict(base['config'], runs={
        name: {k: q['config'][k] for k in ('workers', 'threads_per_worker', 'block')}
        for q in parts for name in q['datasets']})
    cands = [e for q in parts for e in q['example_candidates']]
    base['example_candidates'] = cands
    base['examples_rejected_valid_primary'] = pick_examples(cands)
    base['wall_seconds_total'] = {name: q['wall_seconds_total'] for q in parts
                                  for name in q['datasets']}
    base['merged_from'] = {Path(q).name: sha_file(Path(q)) for q in paths}
    base['created'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, indent=2) + '\n')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merge', nargs='+', type=Path, default=None,
                   help='per-dataset summaries to combine into --summary (no model work)')
    p.add_argument('--attach', nargs=2, metavar=('KEY', 'JSON'), default=None,
                   help='add a merged summary under KEY of --summary (no model work)')
    p.add_argument('--posthoc', nargs=2, metavar=('KEY', 'HITS_DIR'), default=None,
                   help='add the first-order reading and attribution of saved hits under KEY')
    p.add_argument('--posthoc-cos', type=float, default=0.95)
    p.add_argument('--root', type=Path, help='.../inputs/vcache')
    p.add_argument('--out-dir', type=Path, help='server per-row output dir')
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=4, help='torch threads per worker')
    p.add_argument('--block', type=int, default=1024)
    p.add_argument('--tau', type=float, default=TAU, help='retrieval threshold (cos >= tau hits)')
    p.add_argument('--recal-reps', type=int, default=200)
    p.add_argument('--limit', type=int, default=None, help='stream prefix, pilots only')
    p.add_argument('--save-profiles', action='store_true')
    p.add_argument('--reuse-profiles', action='store_true')
    args = p.parse_args(argv)
    if args.merge:
        merge(args.merge, args.summary)
        print('wrote', args.summary, flush=True)
        return 0
    if args.attach:
        key, path = args.attach
        part = json.loads(Path(path).read_text())
        attach(args.summary, key, {k: v for k, v in part.items() if k not in ('W', 'inputs')}
               | {'source_sha256': sha_file(Path(path))})
        print('attached', key, 'to', args.summary, flush=True)
        return 0
    if args.posthoc:
        key, hits_dir = args.posthoc
        attach(args.summary, key, {
            'reading': (f'first order: tau=0.90 stream, valid hits with cos >= '
                        f'{args.posthoc_cos}; nearest keys are the tau=0.90 ones, so a '
                        f'stricter cache\'s extra insertions are not modelled'),
            'code_sha256': sha_file(Path(__file__).resolve()),
            'datasets': posthoc(Path(hits_dir), args.posthoc_cos)})
        print('attached', key, 'to', args.summary, flush=True)
        return 0
    assert args.root and args.out_dir, '--root and --out-dir are required for a run'
    assert args.workers * args.threads <= 32, '32-thread cap on cpu-server'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp
    policy = parse_policy(POLICY)
    t0 = time.perf_counter()
    here = Path(__file__).resolve()
    repo = here.parents[3]
    code = {str(q.relative_to(repo)): sha_file(q) for q in [
        here, repo / 'sentry/cache/defense/deletion.py', repo / 'sentry/cache/defense/spans.py',
        repo / 'sentry/cache/defense/textnorm.py', repo / 'sentry/cache/defense/calibrate.py',
        repo / 'sentry/embeddings.py', repo / 'experiments/paper/rq1_detection/supp_rules.py']}
    inputs = {}
    for name in args.datasets:
        f = args.root / DATASETS[name]['file']
        meta = f.parent / '.cache/huggingface/download' / (f.name + '.metadata')
        lines = meta.read_text().split('\n') if meta.exists() else []
        digest = sha_file(f)
        inputs[name] = {'repo': DATASETS[name]['repo'], 'file': DATASETS[name]['file'],
                        'revision': lines[0] if lines else None, 'sha256': digest,
                        'hf_etag': lines[1] if len(lines) > 1 else None,
                        'sha256_matches_hf_etag': len(lines) > 1 and lines[1] == digest,
                        'readme_sha256': sha_file(f.parent / 'README.md'),
                        'class_column': DATASETS[name]['cls'],
                        'answer_column': DATASETS[name]['answer'],
                        'valid_hit_rule': DATASETS[name]['valid_by']}
    ctx = mp.get_context('spawn')
    results, examples = {}, []
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(ENCODER, args.threads)) as pool:
        encoder_meta = pool.map(_worker_meta, range(args.workers))[0]
        for name in args.datasets:
            log(f'{name} start')
            results[name], ex = run_dataset(name, args, pool, policy)
            examples += ex
            print(f'[{time.strftime("%H:%M:%S")}] {name} done', json.dumps({
                'stream': results[name]['stream'],
                'fpr': results[name]['rules']['primary_joint']['fpr'],
                'wall': results[name]['wall_seconds']}), flush=True)
    report = {
        'task': 'appendix Task 6: vCache replay (FPR and hit-rate cost on real prompts)',
        'created': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'code_sha256': code,
        'config': {'encoder': ENCODER, 'encoder_meta': encoder_meta, 'text_prefix': '',
                   'normalisation': 'l2', 'tau': args.tau, 'nn_tie_rule': f'earliest within {TIE_EPS}',
                   'policy': policy.fingerprint(), 'storage_dtype': 'float16',
                   'profile_call': "build_profile(key, embedder, policy, storage_dtype='float16', answer=y)",
                   'rules': RULES, 'echo_min': 1, 'unjudgeable': 'fail-closed miss (<3 segments)',
                   'no_answer_fields': 'DG veto stands (Algorithm 1)',
                   'answer_words': 'len(normalise_answer(y).split())', 'key_words': 'len(k.split())',
                   'key_bins': [b[0] for b in KEY_BINS], 'answer_bins': [b[0] for b in ANSWER_BINS],
                   'low_support_threshold': LOW_SUPPORT, 'limit': args.limit,
                   'workers': args.workers, 'threads_per_worker': args.threads,
                   'block': args.block},
        'W': w_definition(),
        'inputs': inputs,
        'datasets': results,
        'example_candidates': examples,
        'examples_rejected_valid_primary': pick_examples(examples),
        'wall_seconds_total': round(time.perf_counter() - t0, 1),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    print('wrote', args.summary, flush=True)
    return 0


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


if __name__ == '__main__':
    raise SystemExit(main())
