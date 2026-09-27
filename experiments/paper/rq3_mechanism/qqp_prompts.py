"""Prompts that need a Qwen3-8B victim answer for the 4+2 QQP transfer (text only).

* every CAP attack entry of ``datasets/qqp/c01_records.jsonl`` (3,586 rows, deduplicated
  by text at generation time);
* every QQP canonical entry that has a human-labelled duplicate query
  (``benign_query`` rows of ``validated_records.jsonl``), which is the benign cache entry;
* a reproducibility sample: 100 CAP and 100 SCP attack prompts whose labelled answers
  exist, regenerated on the new GPU to measure agreement with the original victim run.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qqp-validated', type=Path, required=True)
    p.add_argument('--c01', type=Path, required=True)
    p.add_argument('--judged-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(argv)

    val = read_jsonl(a.qqp_validated)
    c01 = read_jsonl(a.c01)
    canon = {r['record_id']: r for r in val if r['query_role'] == 'canonical'}
    pairs = [r for r in val if r['query_role'] == 'benign_query']
    parents = sorted({r['parent_id'] for r in pairs})
    attacks = [r for r in c01 if r['query_role'] == 'ndss']
    prompts = [{'prompt': r['text'], 'kind': 'qqp_attack'} for r in attacks]
    prompts += [{'prompt': canon[pid]['text'], 'kind': 'qqp_benign_entry'} for pid in parents]

    rng = random.Random(20260923)
    reference = []
    for name in ('judged_lmp_all_full.jsonl', 'judged_scp_full.jsonl'):
        for j in rng.sample(read_jsonl(a.judged_dir / name), 100):
            reference.append({'prompt': j['prompt'], 'kind': 'validation_' + name.split('_')[1],
                              'reference_response': j['response']})
    prompts += [{'prompt': r['prompt'], 'kind': r['kind']} for r in reference]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'gen_prompts.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in prompts))
    (a.out_dir / 'validation_reference.json').write_text(json.dumps(reference))
    c01_canon = [r for r in c01 if r['query_role'] == 'canonical' and r['generator'] == 'human_qqp']
    info = {'benign_pairs': len(pairs), 'benign_entries': len(parents),
            'c01_canonical_inside_pairs': sum(r['record_id'] in set(parents) for r in c01_canon),
            'attack_rows': len(attacks), 'attack_distinct_texts': len({r['text'] for r in attacks}),
            'attack_intents': len({r['intent_id'] for r in attacks}),
            'prompts': len(prompts), 'distinct_prompts': len({x['prompt'] for x in prompts}),
            'by_kind': collections.Counter(x['kind'] for x in prompts)}
    (a.out_dir / 'gen_prompts_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
