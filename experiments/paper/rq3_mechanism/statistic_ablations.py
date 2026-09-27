"""Statistic ablations under the deployed 4+2 cut: does DG need the query and the deletion?

DG compares a shortened key with the arriving query and subtracts the whole key's match.
Two ablations each remove one ingredient; both are ported from the six-segment script
``<server-home>/project-workdir/ablations_v3.py`` (cpu-server) to the 4+2 cut:

* anchor-free split coherence (no query): score = -min over cut points c of
  cos(E(k[:c]), E(k[c:])). The cut points are the union of every component cut's interior
  boundaries (4 near-equal segments and 2-word segments, 16 equal segments past 32 words,
  the cap16 rule of ``spans.py``), which are exactly the prefix/suffix pairs D(k) holds.
* isolated span (no deletion difference): every distinct segment of either cut is scored
  alone against q, g = clip((cos - floor) / (1 - floor), 0), and the statistic is the
  coefficient of variation or the range of g. floor is the original's anisotropy floor:
  the median pairwise cosine of the first 400 benign keys in eval-file order. CV and range
  of the raw cosines are reported beside.

DG(4+2) is the row table's ``excess_span``; it is recomputed here from float16-rounded
vectors and checked. Cosine-only (score -cos(k, q)) is reported on the same rows. The same
code run with the ``count:6`` cut, prefix/suffix spans only, must reproduce the published
six-segment numbers (0.423 / 0.578 / 0.626 / DG 0.960 on CAP); that run is the check that
the port is faithful. AUC ranks attack above benign; CIs are the 2,000-replicate intent
bootstrap of ``supp_rules``.

Runs where the embedder is available (cpu-server). Per-row scores stay there.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from experiments.paper.rq3_mechanism.rewrite_recovery import (
    CLASS, ENCODER, POLICY, component, components, encode_texts, f16, read_jsonl, sha_file,
    unit, variant_ranges,
)
from sentry.cache.defense.calibrate import parse_policy
from sentry.cache.defense.spans import SpanPolicy

COUNT6 = SpanPolicy(mode='count', n=6)
FLOOR_SAMPLE = 400
PARITY_TOL = 1e-4
PUBLISHED = {  # ablations_v3.py outputs, out/v3/lacache/ablations_{lmp,scp}.json
    'CAP': {'dg_count6': 0.9600450901803608, 'anchor_free_count6': 0.4233491983967936,
            'span_cv_count6': 0.5779158316633266, 'span_range_count6': 0.6259757014028056},
    'SCP': {'dg_count6': 0.9795103992446045, 'anchor_free_count6': 0.11096880477747476,
            'span_cv_count6': 0.36140451328722606, 'span_range_count6': 0.666934118864295},
}
PUBLISHED_FLOOR = 0.8556309342384338
SCORES = ('dg', 'cosine', 'anchor_free', 'span_cv', 'span_range', 'span_cv_raw',
          'span_range_raw', 'dg_count6', 'anchor_free_count6', 'span_cv_count6',
          'span_range_count6')


# ---------------------------------------------------------------- cut geometry

def _boundaries(comp: SpanPolicy, key: str) -> np.ndarray:
    return np.cumsum([0] + [len(p.split()) for p in comp.segments(key)])


def cut_points(policy: SpanPolicy, key: str) -> list[int]:
    """Interior word boundaries of every component cut, unioned and sorted."""
    n = len(key.split())
    cuts = {int(c) for comp in components(policy) for c in _boundaries(comp, key)}
    return sorted(c for c in cuts if 0 < c < n)


def segments(policy: SpanPolicy, key: str) -> list[tuple[int, int]]:
    """Distinct segment word ranges of every component cut, in order of first appearance."""
    out: list[tuple[int, int]] = []
    for comp in components(policy):
        b = _boundaries(comp, key)
        for s in zip(b[:-1].tolist(), b[1:].tolist()):
            if s not in out:
                out.append(s)
    return out


# ---------------------------------------------------------------- statistics

def spread(seg_scores, floor: float) -> tuple[float, float]:
    """(CV, range) of g = clip((cos - floor) / (1 - floor), 0), as ablations_v3.spread_stats."""
    g = np.clip((np.asarray(seg_scores, float) - floor) / max(1.0 - floor, 1e-9), 0.0, None)
    return float(g.std() / max(g.mean(), 1e-9)), float(g.max() - g.min())


def spread_raw(seg_scores) -> tuple[float, float]:
    s = np.asarray(seg_scores, float)
    return float(s.std() / max(abs(s.mean()), 1e-9)), float(s.max() - s.min())


def anchor_free(prefix_vecs: np.ndarray, suffix_vecs: np.ndarray) -> float:
    return -float(np.min(np.sum(prefix_vecs * suffix_vecs, axis=1)))


def block_at_budget(a, b, budget: float = .05) -> float:
    return float((np.asarray(a) > np.quantile(np.asarray(b), 1 - budget)).mean())


# ---------------------------------------------------------------- one row

def plan_row(row: dict, policy: SpanPolicy) -> dict:
    key = row['key']
    words = key.split()
    join = lambda a, b: ' '.join(words[a:b])
    out = {'row': row, 'variants': [t for t, _, _ in variant_ranges(policy, key)]}
    for tag, pol in (('', policy), ('_count6', COUNT6)):
        cuts = cut_points(pol, key)
        out[f'pre{tag}'] = [join(0, c) for c in cuts]
        out[f'suf{tag}'] = [join(c, len(words)) for c in cuts]
        out[f'seg{tag}'] = [join(a, b) for a, b in segments(pol, key)]
    return out


def plan_texts(p: dict) -> list[str]:
    return ([p['row']['key'], p['row']['query']] + p['variants'] + p['pre'] + p['suf']
            + p['seg'] + p['pre_count6'] + p['suf_count6'] + p['seg_count6'])


def score_row(p: dict, vec, floor: float) -> dict:
    row = p['row']
    M = lambda texts: unit(np.vstack([vec(t) for t in texts]))
    q, k = unit(vec(row['query'])), unit(vec(row['key']))
    base16 = float(f16(k) @ q)
    dg16 = float((f16(M(p['variants'])) @ q).max()) - base16
    base = float(k @ q)
    out = {'record_id': row['record_id'], 'intent_id': row['intent_id'], 'arm': row['arm'],
           'family': row['family'], 'words': len(row['key'].split()),
           'dg': row['excess_span'], 'dg_recomputed': dg16, 'base_cos_recomputed': base16,
           'cosine': -row['base_cos']}
    for tag in ('', '_count6'):
        seg = M(p[f'seg{tag}']) @ q
        out[f'anchor_free{tag}'] = anchor_free(M(p[f'pre{tag}']), M(p[f'suf{tag}']))
        out[f'span_cv{tag}'], out[f'span_range{tag}'] = spread(seg, floor)
        if not tag:
            out['span_cv_raw'], out['span_range_raw'] = spread_raw(seg)
            out['n_cut_points'], out['n_segments'] = len(p['pre']), len(p['seg'])
    # six-segment DG, fp32 as the original: prefix/suffix spans of count:6 are its cut pairs
    out['dg_count6'] = float((M(p['pre_count6'] + p['suf_count6']) @ q).max()) - base
    return out


# ---------------------------------------------------------------- main

def anisotropy_floor(benign_keys: list[str], vec) -> float:
    S = unit(np.vstack([vec(t) for t in benign_keys[:FLOOR_SAMPLE]]))
    gram = S @ S.T
    return float(np.median(gram[np.triu_indices(len(S), k=1)]))


def class_report(scored: list[dict], seed: str, replicates: int) -> dict:
    b = {s: np.array([r[s] for r in scored if r['arm'] == 'genuine']) for s in SCORES}
    a = {s: np.array([r[s] for r in scored if r['arm'] == 'attack']) for s in SCORES}
    b['intent_id'] = np.array([r['intent_id'] for r in scored if r['arm'] == 'genuine'], object)
    a['intent_id'] = np.array([r['intent_id'] for r in scored if r['arm'] == 'attack'], object)
    lo, hi = R.intent_bootstrap(b, a, lambda bb, aa: np.array(
        [R.auc(aa[s], bb[s]) for s in SCORES]), replicates=replicates, seed=seed)
    return {'n_attack': int(len(a['dg'])), 'n_benign': int(len(b['dg'])),
            'n_attack_intents': int(len(set(a['intent_id']))),
            'n_benign_intents': int(len(set(b['intent_id']))),
            'scores': {s: {'auc': R.auc(a[s], b[s]), 'auc_ci95': [lo[i], hi[i]],
                           'block_at_5pct': block_at_budget(a[s], b[s])}
                       for i, s in enumerate(SCORES)}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', type=Path, required=True)
    p.add_argument('--rows-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=None)
    p.add_argument('--policy', default=POLICY)
    p.add_argument('--encoder', default=ENCODER)
    p.add_argument('--replicates', type=int, default=2000)
    args = p.parse_args(argv)
    policy = parse_policy(args.policy)
    t0 = time.perf_counter()

    data = {}
    for stem in ('lmp', 'scp', 'kca'):
        rows_path = args.rows_dir / f'rows_{stem}.jsonl'
        eval_path = args.eval_dir / f'{stem}_eval.jsonl'
        rows = read_jsonl(rows_path)
        order = {r['record_id']: i for i, r in enumerate(read_jsonl(eval_path))}
        benign_keys = [r['key'] for r in sorted((r for r in rows if r['arm'] == 'genuine'),
                                                key=lambda r: order[r['record_id']])]
        data[stem] = {'rows': rows, 'plans': [plan_row(r, policy) for r in rows],
                      'benign_keys': benign_keys,
                      'inputs': {'rows': str(rows_path), 'rows_sha256': sha_file(rows_path),
                                 'eval': str(eval_path), 'eval_sha256': sha_file(eval_path)}}

    from sentry.embeddings import TransformerCLSEmbedder
    embedder = TransformerCLSEmbedder(args.encoder)
    texts = [t for d in data.values() for pl in d['plans'] for t in plan_texts(pl)]
    index, matrix = encode_texts(texts, embedder, args.cache)
    vec = lambda t: matrix[index[t]].astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {'task': 'appendix Task 8: statistic ablations under 4+2',
              'policy': policy.fingerprint(), 'encoder': args.encoder, 'pooling': 'cls',
              'port_check_policy': COUNT6.fingerprint() + ' (prefix/suffix spans)',
              'published_six_segment': PUBLISHED, 'published_floor': PUBLISHED_FLOOR,
              'classes': {}, 'n_texts_encoded': len(index)}
    for stem, d in data.items():
        cls = CLASS[stem]
        floor = anisotropy_floor(d['benign_keys'], vec)
        scored = [score_row(pl, vec, floor) for pl in d['plans']]
        path = args.out_dir / f'statistic_ablations_{stem}.jsonl'
        with path.open('w', encoding='utf-8') as fh:
            for r in scored:
                fh.write(json.dumps(r) + '\n')
        parity = {
            'max_abs_dg_minus_excess_span': max(abs(r['dg_recomputed'] - r['dg']) for r in scored),
            'max_abs_base_cos': max(abs(r['base_cos_recomputed'] + r['cosine']) for r in scored),
            'tolerance': PARITY_TOL}
        rep = class_report(scored, f'ablations:{cls}', args.replicates)
        rep.update({'anisotropy_floor': floor, 'parity': parity, 'inputs': d['inputs'],
                    'per_row_output': {'path': str(path), 'sha256': sha_file(path)},
                    'cut_points_quantiles': {str(q): float(v) for q, v in zip(
                        (5, 50, 95), np.percentile([r['n_cut_points'] for r in scored],
                                                   (5, 50, 95)))},
                    'segments_quantiles': {str(q): float(v) for q, v in zip(
                        (5, 50, 95), np.percentile([r['n_segments'] for r in scored],
                                                   (5, 50, 95)))}})
        if cls in PUBLISHED:
            rep['port_check'] = {s: {'published': v, 'reproduced': rep['scores'][s]['auc'],
                                     'abs_diff': abs(rep['scores'][s]['auc'] - v)}
                                 for s, v in PUBLISHED[cls].items()}
        report['classes'][cls] = rep
        print(cls, json.dumps({s: round(v['auc'], 4) for s, v in rep['scores'].items()}),
              json.dumps(parity), flush=True)
    report['runtime_s'] = time.perf_counter() - t0
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'wrote {args.summary}', flush=True)
    bad = [c for c, r in report['classes'].items()
           if max(r['parity']['max_abs_dg_minus_excess_span'],
                  r['parity']['max_abs_base_cos']) > PARITY_TOL]
    if bad:
        raise SystemExit(f'DG/cosine parity failure in {bad}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
