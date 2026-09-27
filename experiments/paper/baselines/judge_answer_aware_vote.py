"""Answer-aware DeepSeek judge scored by vote fraction (the Table 1 judge's scoring).

The pilot of ``judge_answer_aware_api.py`` found DeepSeek-v4-flash logprobs degenerate
(first-token logprob 0.0, every alternative -9999; 6 distinct scores on 50 rows), so a
logprob score is not available from this API. This keeps the scoring of the published LLM
judge row (``judge_vote.py``: 10 samples at temperature 1.0, thinking disabled, score =
fraction of UNSAFE verdicts) and changes one thing, the prompt: ``PROMPT_QKY`` adds the
stored answer. The published row is therefore the matched control without the answer.
``--prompt poisoning`` swaps in ``PROMPT_QKY_POISON``, which asks whether the entry was
planted and says an honest wrong answer is not a reason to flag it.
Responses are cached by the repository client, so a rerun costs no API calls.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.paper.baselines.judge_vote import parse_verdict
from experiments.paper.gpu.judge_answer_aware_score import PROMPT_QKY, PROMPT_QKY_POISON

PROMPTS = {'validity': PROMPT_QKY, 'poisoning': PROMPT_QKY_POISON}


def main(argv=None) -> int:
    from sentry.research.operators import Client, load_env

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--votes', type=int, default=10)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--prompt', default='validity', choices=list(PROMPTS),
                   help='validity: may the entry be reused (default); poisoning: was the entry planted')
    a = p.parse_args(argv)

    rows = []
    for path in a.rows:
        rows += [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    seen, unique = set(), []
    for r in rows:                      # CAP and SCP share the ComQA benign rows
        key = (r['arm'], r['record_id'])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    client = Client(load_env(), cache_name='judge_answer_aware_vote' + ('' if a.prompt == 'validity' else '_' + a.prompt))
    template = PROMPTS[a.prompt]
    extra = {'thinking': {'type': 'disabled'}}

    def one(job):
        i, j = job
        r = unique[i]
        msgs = [{'role': 'user', 'content': template.format(
            entry=r['key'], query=r['query'], answer=r['answer'].strip())}]
        seed = a.votes * i + j
        cached = client._key(msgs, a.temperature, seed, 8, extra) in client._cache
        t0 = time.perf_counter()
        ans = client.chat(msgs, temperature=a.temperature, seed=seed, max_tokens=8, extra_body=extra)
        return i, j, parse_verdict(ans), (None if cached else (time.perf_counter() - t0) * 1000)

    votes = np.zeros((len(unique), a.votes))
    latency = []
    jobs = [(i, j) for i in range(len(unique)) for j in range(a.votes)]
    with ThreadPoolExecutor(a.workers) as ex:
        for n, (i, j, v, ms) in enumerate(ex.map(one, jobs)):
            votes[i, j] = v
            if ms is not None:
                latency.append(ms)
            if (n + 1) % 2000 == 0:
                print(f'{n + 1}/{len(jobs)}', flush=True)
    with open(a.out, 'w') as f:
        for i, r in enumerate(unique):
            f.write(json.dumps({**{k: r[k] for k in ('set', 'record_id', 'intent_id', 'arm', 'family')},
                                'score': float(votes[i].mean()),
                                'votes': [int(v) for v in votes[i]]}) + '\n')
    meta = {'model': client.creds['model'], 'prompt': a.prompt, 'votes': a.votes, 'temperature': a.temperature,
            'rows': len(unique), 'uncached_calls': len(latency),
            'ms_per_call_median': float(np.median(latency)) if latency else None}
    Path(a.out + '.meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
