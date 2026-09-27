"""GPTCache workload replay under the deployed rule (DG and the answer check).

``rq4_gptcache_workload.py`` stores marker answers ("POISONED answer for …"), which is
enough for a DG-only filter but would disarm the answer check: a marker never repeats the
residual. This replay keeps that trace (same intents, plants, stream, seed and host
settings) and stores each entry's real cached answer from the row table instead (for
planted entries, the answer the success label judged). Planted hits are identified by the
returned answer through a side table, and hits whose planted answer was judged poisoned are
counted separately. Arms: no profile writer; writer without the filter; DG only at its
main-table threshold; DG and the answer check at the joint main-table thresholds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.paper.rq4_system.rq4_gptcache_workload import Encoder, build_trace

JOINT = {'eta': 0.002645233293430393, 'eta_a': 0.01965637207031249}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def full_rule_fence(rows_path: Path, policy_fp: str):
    from experiments.paper.rq1_detection.v3_detect import DUMP_FIELDS, DumpedColumns, DumpedRow
    from sentry.cache.defense.calibrate import fit_rule_fence, joint_fence
    from sentry.cache.defense.fence import CalibrationRow
    benign = [DumpedRow(*(True if k == 'has_answer' else r[k] for k in DUMP_FIELDS))
              for r in read_jsonl(rows_path) if r['arm'] == 'genuine']
    rows = [CalibrationRow(r.base_cos, r.words, r.excess_span, answer_loss=r.adl_best) for r in benign]
    fence = fit_rule_fence(rows, 'either', budget=0.05, embedder='intfloat/e5-small-v2',
                           policy=policy_fp, echo_min=1)
    fence, _ = joint_fence(fence, [(None, r) for r in benign], DumpedColumns(), 0.05)
    assert abs(float(fence.coefficients[0]) - JOINT['eta']) < 1e-12, fence.coefficients
    assert abs(float(fence.eta_a) - JOINT['eta_a']) < 1e-12, fence.eta_a
    return fence


def run_arm(entries, stream, kinds, encoder, policy, fence, defended, writer, threshold, workdir):
    from gptcache import Cache
    from gptcache.adapter.api import get as cache_get, put as cache_put
    from gptcache.manager import manager_factory
    from gptcache.processor.pre import get_prompt
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    from sentry.cache.defense.decide import DeletionDefenseConfig
    from sentry.cache.defense.entry_store import InMemoryProfileStore
    from sentry.cache.defense.gptcache_plugin import DeletionVetoEvaluation
    from sentry.cache.defense.insertion import install_profile_writer

    store = InMemoryProfileStore(encoder.model_name, policy.fingerprint())
    manager = manager_factory('sqlite,faiss', data_dir=str(workdir),
                              vector_params={'dimension': encoder.dimension})
    evaluation = DeletionVetoEvaluation(
        SearchDistanceEvaluation(max_distance=2.0, positive=False), store, fence=fence,
        config=DeletionDefenseConfig(enabled=defended, cache_threshold=threshold))
    cache = Cache()
    cache.init(pre_embedding_func=get_prompt, embedding_func=encoder, data_manager=manager,
               similarity_evaluation=evaluation)
    if writer:
        install_profile_writer(cache, store, encoder, policy)
    t0 = time.perf_counter()
    for text, answer, _ in entries:
        cache_put(text, answer, cache_obj=cache)
    insert_wall = time.perf_counter() - t0
    got, latencies = [], []
    t0 = time.perf_counter()
    for query, _, _ in stream:
        start = time.perf_counter()
        got.append(cache_get(query, cache_obj=cache))
        latencies.append(time.perf_counter() - start)
    serve_wall = time.perf_counter() - t0
    hits = [a for a in got if a is not None]
    planted = [a for a in hits if kinds.get(a, ('genuine', False))[0] == 'planted']
    poisoned = [a for a in planted if kinds[a][1]]
    lat = np.array(latencies) * 1000.0
    return {'defended': defended, 'profile_writer': writer,
            'n_entries': len(entries), 'n_requests': len(stream),
            'insertion_ms_per_entry': insert_wall / len(entries) * 1000,
            'hit_rate': len(hits) / len(stream),
            'planted_hit_rate': len(planted) / len(stream),
            'poisoned_answer_hit_rate': len(poisoned) / len(stream),
            'qps': len(stream) / serve_wall,
            'latency_ms_p50': float(np.median(lat)), 'latency_ms_p95': float(np.quantile(lat, .95)),
            'counters': evaluation.counters.as_dict() if hasattr(evaluation.counters, 'as_dict') else {}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval', required=True)
    p.add_argument('--rows', type=Path, required=True, help='rows_lmp.jsonl')
    p.add_argument('--dg-fence-json', required=True, help='cand42_lmp.json (DG-only main-table fence)')
    p.add_argument('--intents', type=int, default=400)
    p.add_argument('--poison-frac', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=20260901)
    p.add_argument('--threshold', type=float, default=0.90)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--workdir', default=None, help='parent of the per-arm GPTCache dirs (kept until exit)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.fence import ExcessFence
    policy = parse_policy('multi[count:4+width:2:cap16]/runs')
    encoder = Encoder('intfloat/e5-small-v2')
    entries, stream, n_planted = build_trace(a.eval, {'ndss'}, 'human_comqa', a.intents,
                                             a.poison_frac, a.seed)
    table = {(r['arm'], r['key']): r for r in read_jsonl(a.rows)}
    real, kinds, missing = [], {}, []
    for text, _, kind in entries:
        r = table.get(('genuine' if kind == 'genuine' else 'attack', text))
        if r is None:
            missing.append((kind, text))
            continue
        real.append((text, r['answer'], kind))
        previous = kinds.get(r['answer'])
        assert previous is None or previous[0] == kind, 'an answer shared by a genuine and a planted entry'
        kinds[r['answer']] = (kind, bool(r['poisoned']) if kind == 'planted' else False)
    assert not missing, missing[:3]
    dg_fence = ExcessFence.from_dict(json.loads(Path(a.dg_fence_json).read_text())['fence_scoring'])
    full_fence = full_rule_fence(a.rows, policy.fingerprint())
    report = {'policy': policy.fingerprint(), 'n_intents': a.intents, 'poison_frac': a.poison_frac,
              'seed': a.seed, 'n_planted_intents': n_planted, 'n_entries': len(real),
              'n_requests': len(stream), 'cache_threshold': a.threshold,
              'fences': {'dg_only': float(dg_fence.coefficients[0]), 'joint': JOINT},
              'runs': {}}
    arms = [('off', dg_fence, False, False), ('undefended', dg_fence, False, True),
            ('dg_only', dg_fence, True, True), ('dg_answer_check', full_fence, True, True)]
    if a.workdir:
        Path(a.workdir).mkdir(parents=True, exist_ok=True)
    encoder.encode([text for text, _, _ in real[:64]])          # warm the encoder up
    timing = ('insertion_ms_per_entry', 'qps', 'latency_ms_p50', 'latency_ms_p95')
    rounds = {label: [] for label, *_ in arms}
    workdirs = []
    for rep in range(a.repeats):                                 # interleaved, order rotated
        order = arms[rep % len(arms):] + arms[:rep % len(arms)]
        for label, fence, defended, writer in order:
            workdir = Path(tempfile.mkdtemp(prefix='gcwl-full-', dir=a.workdir))
            workdirs.append(workdir)
            row = run_arm(real, stream, kinds, encoder, policy, fence, defended, writer,
                          a.threshold, workdir)
            rounds[label].append(row)
            print(rep, label, {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in row.items() if k != 'counters'}, flush=True)
    for label, rows in rounds.items():
        for key in ('hit_rate', 'planted_hit_rate', 'poisoned_answer_hit_rate'):
            assert len({r[key] for r in rows}) == 1, (label, key)   # decisions are deterministic
        merged = dict(rows[0])
        for key in timing:
            merged[key] = float(np.median([r[key] for r in rows]))
            merged[key + '_repeats'] = [r[key] for r in rows]
        report['runs'][label] = merged
    report['repeats'] = a.repeats
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    # GPTCache flushes its faiss index at interpreter exit, so the dirs are removed by the
    # caller afterwards rather than here.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
