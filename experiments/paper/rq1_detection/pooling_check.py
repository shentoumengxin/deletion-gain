"""Cosine under the cache's pooling (CLS, no prefix) against the encoder's native recipe.

For every Table 1 row, embeds (arriving query, cached entry) twice with the same checkpoint:
  cls     last_hidden_state[:, 0] + l2, no prefix   (the cache's recipe; what the attacks
                                                     were optimized / screened against)
  native  --pooling / --text-prefix as the model card specifies (e5: mean + "query: ";
          gte, MiniLM: mean; bge: cls)
and reports, per set: median cosine of attacks and of genuine entries under each recipe,
the benign hit rate at tau = 0.90 under cls, the native threshold tau' that keeps that benign
hit rate, and the fraction of attacks that still collide (cos >= tau') under native.
Per-row cosines are written so the DG runs under --pooling can be joined to them.

Usage (cpu-server):
  PYTHONPATH=<repo> HF_HOME=... python pooling_check.py --set cap --repo R --eval-dir E \
      --encoder intfloat/e5-small-v2 --pooling mean --text-prefix "query: " --out F
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import argparse, json, os, sys
from pathlib import Path
import numpy as np

SETS = {'cap': ('lmp_eval.jsonl', 'ndss', 'human_comqa'),
        'scp': ('scp_eval.jsonl', 'scp', 'human_comqa'),
        'kca': ('kca_eval.jsonl', 'gcg', 'cacheattack_cleaned_qa')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set', required=True, choices=SETS)
    p.add_argument('--encoder', default='intfloat/e5-small-v2')
    p.add_argument('--pooling', default='mean'); p.add_argument('--text-prefix', default='query: ')
    p.add_argument('--tau', type=float, default=0.90)
    p.add_argument('--repo', required=True); p.add_argument('--eval-dir', required=True)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, 'experiments/paper/rq1_detection'))
    import torch; torch.set_num_threads(a.threads)
    from experiments.paper.rq1_detection.v3_detect import load_rows
    from sentry.embeddings import TransformerCLSEmbedder
    ef, role, gen = SETS[a.set]
    rows, comp = load_rows(os.path.join(a.eval_dir, ef), [role], [gen])
    print(f'{a.set}: {len(rows)} rows {comp}', flush=True)
    q = [r['anchor'] for r in rows]; e = [r['text'] for r in rows]
    att = np.array([r['arm'] == 'attack' for r in rows])
    reads = {}
    for name, kw in (('cls', {}), ('native', {'pooling': a.pooling, 'text_prefix': a.text_prefix})):
        emb = TransformerCLSEmbedder(a.encoder, batch_size=128, **kw)
        Q = emb.encode(q).astype(np.float64); E = emb.encode(e).astype(np.float64)
        reads[name] = (Q * E).sum(1)
    c, n = reads['cls'], reads['native']
    benign_hit = float(np.mean(c[~att] >= a.tau))
    tau_native = float(np.quantile(n[~att], 1 - benign_hit)) if 0 < benign_hit < 1 else float('nan')
    summary = {'encoder': a.encoder, 'native': {'pooling': a.pooling, 'text_prefix': a.text_prefix}, 'tau': a.tau,
               'median_cos': {k: {'attack': float(np.median(v[att])), 'benign': float(np.median(v[~att]))} for k, v in reads.items()},
               'benign_hit_rate_cls': benign_hit, 'tau_native_matched': tau_native,
               'attack_collide_cls': float(np.mean(c[att] >= a.tau)),
               'attack_collide_native': float(np.mean(n[att] >= tau_native)),
               'n_attack': int(att.sum()), 'n_benign': int((~att).sum())}
    print(json.dumps(summary, indent=1), flush=True)
    recs = [{'record_id': r['record_id'], 'arm': r['arm'], 'family': r['family'], 'intent_id': r['intent_id'],
             'cos_cls': float(c[i]), 'cos_native': float(n[i])} for i, r in enumerate(rows)]
    Path(a.out).write_text(json.dumps({'set': a.set, **summary, 'rows': recs}))
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
