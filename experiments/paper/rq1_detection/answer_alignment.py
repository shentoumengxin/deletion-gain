"""Compute the CAP/SCP answer-check fields on the cached answers that carry the outcome labels.

Each attack entry's cached answer is the Qwen3-8B response that the outcome judge labels
(``judged_*_full.jsonl``). This script profiles every CAP/SCP attack entry with that
answer through the code path ``v3_detect.run_cell`` uses and writes ``adl_best`` and
``echo_best``. Benign rows, DG, cosine, words, identities and labels are copied and
checked unchanged.

Runs where the embedder is available. No generation or judge call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection.v3_detect import load_rows
from sentry.cache.defense.calibrate import AnswerColumns, parse_policy
from sentry.cache.defense.deletion import build_profile, excess
from sentry.cache.defense.textnorm import normalise_answer

POLICY = 'multi[count:4+width:2:cap16]/runs'
SETS = {'lmp': ('ndss', 'judged_lmp_all_full.jsonl'),
        'scp': ('scp', 'judged_scp_full.jsonl')}
TOL = 1e-5


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def load_old_answers(root: Path) -> dict:
    """The answers the published dump read (last non-empty per prompt hash wins)."""
    from sentry.research.pipeline.instruction_benign import load_answer_files
    answers, _ = load_answer_files([str(root)])
    return answers


def align(stem, role, judged_path, eval_dir, dump_path, embedder, policy, old_answers, tol=TOL):
    rows, _ = load_rows(eval_dir / f'{stem}_eval.jsonl', [role], ['human_comqa'])
    by_id = {r['record_id']: r for r in rows}
    dump = read_jsonl(dump_path)
    judged = {}
    for j in read_jsonl(judged_path):
        for rid in [j['record_id'], *j.get('alt_ids', [])]:
            judged[rid] = j
    columns = AnswerColumns()
    out, worst = [], {'excess_span': 0.0, 'base_cos': 0.0, 'unchanged_answer_adl': 0.0}
    n_changed_text = n_unchanged_echo_mismatch = 0
    for d in dump:
        if d['arm'] == 'genuine':
            out.append(dict(d))
            continue
        r, j = by_id[d['record_id']], judged[d['record_id']]
        assert j['prompt'] == r['text'], d['record_id']
        assert bool(j['judge_success']) == d['poisoned'], d['record_id']
        answer = j['response']
        profile = build_profile(r['text'], embedder, policy, storage_dtype='float16',
                                answer=answer)
        reading = excess(profile, embedder.encode([r['anchor']])[0])
        adl, echo = columns(r['anchor'], reading)
        assert adl is not None, f"answer normalised to nothing: {d['record_id']}"
        worst['excess_span'] = max(worst['excess_span'], abs(reading.excess_span - d['excess_span']))
        worst['base_cos'] = max(worst['base_cos'], abs(reading.base_cos - d['base_cos']))
        old = old_answers.get(sha(r['text']))
        changed = old is None or normalise_answer(old) != normalise_answer(answer)
        n_changed_text += changed
        if not changed:   # same text: the recomputation must reproduce the published fields
            worst['unchanged_answer_adl'] = max(worst['unchanged_answer_adl'],
                                                abs(adl - d['adl_best']))
            n_unchanged_echo_mismatch += int(echo != d['echo_best'])
        out.append(dict(d, adl_best=float(adl), echo_best=int(echo),
                        answer_sha256=sha(answer), answer_changed=bool(changed)))
    assert worst['excess_span'] < tol and worst['base_cos'] < tol, worst
    assert worst['unchanged_answer_adl'] < tol, worst
    return out, {'n_rows': len(out), 'n_attack': sum(r['arm'] == 'attack' for r in out),
                 'n_attack_answer_changed': n_changed_text,
                 'max_abs_diff': worst, 'tolerance': tol,
                 'unchanged_answer_echo_mismatches': n_unchanged_echo_mismatch}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', type=Path, required=True,
                   help='datasets/final500/eval (the frozen v3 eval sets)')
    p.add_argument('--dump-template', required=True,
                   help="published e5 dump path with {stem}, e.g. .../e5-small-v2__{stem}.jsonl")
    p.add_argument('--judged-dir', type=Path, required=True)
    p.add_argument('--old-answers', type=Path, required=True,
                   help='answers directory behind the published dumps')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--encoder', default='intfloat/e5-small-v2')
    p.add_argument('--pooling', default='cls', choices=('cls', 'mean'))
    p.add_argument('--text-prefix', default='')
    p.add_argument('--sets', nargs='+', default=list(SETS), choices=list(SETS))
    p.add_argument('--tol', type=float, default=TOL,
                   help='max |recomputed - published| for DG, cosine and unchanged-answer ADL')
    args = p.parse_args(argv)

    from sentry.embeddings import TransformerCLSEmbedder
    embedder = TransformerCLSEmbedder(args.encoder, pooling=args.pooling,
                                      text_prefix=args.text_prefix)
    policy = parse_policy(POLICY)
    old_answers = load_old_answers(args.old_answers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {'policy': policy.fingerprint(), 'encoder': args.encoder, 'pooling': args.pooling,
               'text_prefix': args.text_prefix, 'storage_dtype': 'float16', 'sets': {}}
    tag = Path(args.encoder).name + ('' if args.pooling == 'cls' else f'__{args.pooling}')
    for stem in args.sets:
        role, judged_name = SETS[stem]
        t0 = time.perf_counter()
        dump_path = Path(args.dump_template.format(stem=stem))
        rows, info = align(stem, role, args.judged_dir / judged_name, args.eval_dir,
                           dump_path, embedder, policy, old_answers, tol=args.tol)
        info['published_dump_sha256'] = hashlib.sha256(dump_path.read_bytes()).hexdigest()
        path = args.out_dir / f'{tag}__{stem}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        info.update(seconds=round(time.perf_counter() - t0, 1),
                    output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    judged_sha256=hashlib.sha256((args.judged_dir / judged_name).read_bytes()).hexdigest())
        receipt['sets'][stem] = info
        print(stem, json.dumps(info), flush=True)
    (args.out_dir / f'receipt_{tag}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
