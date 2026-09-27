"""One row table per attack class for every appendix experiment.

Each row carries the cached entry k, the incoming benign query q, the answer the cache
stores (for attacks, the answer the success label judged), the label, and the published
4+2 scores from the answer-aligned dumps. Every downstream experiment reads k, q and y
from here, so no two experiments can pair an entry with different answers.
Text-only; runs on the laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.paper.rq1_detection.dg_necessity import CELLS, cached_answers, read_jsonl, sha_file, sha_text
from experiments.paper.rq1_detection.v3_detect import load_rows

FIELDS = ('record_id', 'intent_id', 'arm', 'family', 'poisoned', 'base_cos', 'words',
          'excess_span', 'adl_best', 'echo_best')


def build(name: str, args) -> tuple[list[dict], dict]:
    stem, role, gen = CELLS[name]
    path = args.kca_rows if stem == 'kca' else args.aligned_dir / f'e5-small-v2__{stem}.jsonl'
    rows = read_jsonl(path)
    loaded, _ = load_rows(args.eval_dir / f'{stem}_eval.jsonl', [role], [gen])
    texts = {r['record_id']: r for r in loaded}
    answers = cached_answers(stem, rows, texts, args)
    out = []
    for r in rows:
        t = texts[r['record_id']]
        y = answers[r['record_id']]
        out.append({**{k: r[k] for k in FIELDS}, 'set': stem, 'key': t['text'],
                    'query': t['anchor'], 'answer': y, 'answer_sha256': sha_text(y)})
    return out, {'rows_source': str(path), 'rows_source_sha256': sha_file(path)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', type=Path, required=True)
    p.add_argument('--aligned-dir', type=Path, required=True)
    p.add_argument('--kca-rows', type=Path, required=True)
    p.add_argument('--kca-answers', type=Path, required=True)
    p.add_argument('--judged-dir', type=Path, required=True)
    p.add_argument('--bare-answers', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name in CELLS:
        rows, info = build(name, args)
        path = args.out_dir / f'rows_{CELLS[name][0]}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        info.update(n=len(rows), n_attack=sum(r['arm'] == 'attack' for r in rows),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        receipt[name] = info
        print(name, info, flush=True)
    (args.out_dir / 'rows_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
