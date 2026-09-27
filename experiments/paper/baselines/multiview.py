"""Multi-encoder agreement baseline (better matching), per row, on the Table 1 rows.

Re-embeds the (arriving query, cached entry) pair with the primary encoder (e5-small-v2,
CLS pooling, the cache's own) and two auxiliary sentence encoders (all-MiniLM-L6-v2,
bge-small-en-v1.5, via SentenceTransformer as in baselines/defense/multiview.py) and
records the three cosines. Reads derived downstream (ci_table1.py / budgets.py):
  disagreement  = max(0, cos_primary - mean(cos_aux))   (baselines.defense.multiview)
  min-cosine    = -min over the three encoders          (require agreement everywhere)
Serving cost: the entry's auxiliary vectors are stored at insertion, so a hit pays the two
auxiliary encodes of the query; ms/hit is measured unbatched on --time-rows rows.
Usage (cpu-server):
  PYTHONPATH=<repo> HF_HOME=... python multiview.py --set cap --repo R --eval-dir E --out F
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

SETS = {'cap': ('lmp_eval.jsonl', 'ndss', 'human_comqa'),
        'scp': ('scp_eval.jsonl', 'scp', 'human_comqa'),
        'kca': ('kca_eval.jsonl', 'gcg', 'cacheattack_cleaned_qa')}
AUX = ['sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set', required=True, choices=SETS)
    p.add_argument('--primary', default='intfloat/e5-small-v2')
    p.add_argument('--repo', required=True); p.add_argument('--eval-dir', required=True)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--time-rows', type=int, default=300)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, 'experiments/paper/rq1_detection'))
    import torch; torch.set_num_threads(a.threads)
    from experiments.paper.rq1_detection.v3_detect import load_rows
    from sentry.embeddings import TransformerCLSEmbedder
    from sentence_transformers import SentenceTransformer
    ef, role, gen = SETS[a.set]
    rows, comp = load_rows(os.path.join(a.eval_dir, ef), [role], [gen])
    print(f'{a.set}: {len(rows)} rows {comp}', flush=True)
    q = [r['anchor'] for r in rows]; e = [r['text'] for r in rows]
    prim = TransformerCLSEmbedder(a.primary, batch_size=128)
    Q, E = prim.encode(q).astype(np.float64), prim.encode(e).astype(np.float64)
    cos = {'primary': (Q * E).sum(1)}
    aux_models = [SentenceTransformer(m) for m in AUX]
    for name, m in zip(AUX, aux_models):
        Qa = m.encode(q, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
        Ea = m.encode(e, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
        cos[name] = (Qa * Ea).sum(1).astype(np.float64)
    # serving cost: unbatched auxiliary encodes of the query, one row at a time
    t = []
    for text in q[:a.time_rows]:
        t0 = time.perf_counter()
        for m in aux_models: m.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        t.append((time.perf_counter() - t0) * 1000)
    recs = [{'record_id': r['record_id'], 'arm': r['arm'], 'family': r['family'], 'intent_id': r['intent_id'],
             'cos_primary': float(cos['primary'][i]), **{f'cos_aux{j}': float(cos[n][i]) for j, n in enumerate(AUX)}}
            for i, r in enumerate(rows)]
    Path(a.out).write_text(json.dumps({'set': a.set, 'primary': a.primary, 'aux': AUX, 'threads': a.threads,
                                       'ms_per_hit_median': float(np.median(t)), 'ms_per_hit_mean': float(np.mean(t)),
                                       'timed_rows': len(t), 'rows': recs}))
    print(f'wrote {a.out}  ms/hit median {np.median(t):.2f} mean {np.mean(t):.2f} (2 aux query encodes, {a.threads} threads)', flush=True)


if __name__ == '__main__':
    main()
