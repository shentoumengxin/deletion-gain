"""Key salting (Zhang et al. 2026, CacheAttack Sec. 7.1) per row, on the Table 1 rows.

A user-agnostic secret salt s, "a random 5-token string" sampled once per cache instance, is
written into the text before encoding at both insertion and lookup,
    k_s = E(A_s(k)),  q_s = E(A_s(q)),
and the cache's own cosine is read on the salted keys. The paper's augmentations:
    prefix    A_s(p) = s || p
    suffix    A_s(p) = p || s
    template  A_s(p) = "[SALT=s]" || p
Five salts (seeds 0-4) per augmentation; salt tokens are drawn from the encoder's own
vocabulary (whole-word alphabetic entries), so a salt is five ordinary words. The score is the
salted cosine (low = suspicious) and the operating point is the salted retrieval threshold
that costs benign entries the same 5% of their hits (read downstream, ci_table1.py).
Serving cost: none beyond the cache's own query encode (five extra tokens), as for cosine.

Usage (cpu-server):
  PYTHONPATH=<repo> HF_HOME=... python salting_text.py --set cap --repo R --eval-dir E --out F
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
AUG = {'prefix': lambda s, p: f'{s} {p}',
       'suffix': lambda s, p: f'{p} {s}',
       'template': lambda s, p: f'[SALT={s}] {p}'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set', required=True, choices=SETS)
    p.add_argument('--encoder', default='intfloat/e5-small-v2')
    p.add_argument('--repo', required=True); p.add_argument('--eval-dir', required=True)
    p.add_argument('--seeds', type=int, default=5); p.add_argument('--salt-tokens', type=int, default=5)
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
    emb = TransformerCLSEmbedder(a.encoder, batch_size=128)
    special = set(emb.tokenizer.all_special_tokens)
    words = sorted(w for w in emb.tokenizer.get_vocab() if w.isalpha() and len(w) >= 3 and w not in special)
    salts = {seed: ' '.join(np.random.default_rng(seed).choice(words, a.salt_tokens, replace=False))
             for seed in range(a.seeds)}
    print(f'  salts: {salts}', flush=True)

    def cos(qs, es):
        Q = emb.encode(qs).astype(np.float64); E = emb.encode(es).astype(np.float64)
        return (Q * E).sum(1)

    reads = {'plain': cos(q, e)}
    for aug, fn in AUG.items():
        for seed, s in salts.items():
            reads[f'{aug}:{seed}'] = cos([fn(s, x) for x in q], [fn(s, x) for x in e])
            print(f'  {aug}:{seed} done', flush=True)
    recs = [{'record_id': r['record_id'], 'arm': r['arm'], 'family': r['family'], 'intent_id': r['intent_id'],
             **{k: float(v[i]) for k, v in reads.items()}} for i, r in enumerate(rows)]
    att = np.array([x['arm'] == 'attack' for x in recs])
    for k, v in reads.items():
        print(f'  {k:12s} median cos attack {np.median(v[att]):.3f} benign {np.median(v[~att]):.3f}', flush=True)
    Path(a.out).write_text(json.dumps({'set': a.set, 'encoder': a.encoder, 'salt_tokens': a.salt_tokens,
                                       'salts': salts, 'augmentations': list(AUG), 'rows': recs}))
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
