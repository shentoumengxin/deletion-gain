"""Victim answers with vLLM offline, same settings as the main-table victim (GPU box).

Standalone. Qwen3-8B, bf16, max_model_len 4096, one user message per prompt, thinking
disabled through the chat template, temperature 0 (greedy), max_tokens 200 -- the serving
flags recorded in ``gpu/serve_qwen3.sh`` and ``results/v3/asr/README.md``. Output rows are
keyed by the sha256 of the prompt text, the key ``gen_answers.py`` uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main(argv=None) -> int:
    from vllm import LLM, SamplingParams

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', required=True, help='jsonl with a "prompt" field')
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=200)
    a = p.parse_args(argv)

    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    seen, prompts = set(), []
    for r in rows:
        if r['prompt'] not in seen:
            seen.add(r['prompt'])
            prompts.append(r['prompt'])
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=4096, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    t0 = time.perf_counter()
    outs = llm.chat([[{'role': 'user', 'content': q}] for q in prompts], params,
                    chat_template_kwargs={'enable_thinking': False})
    wall = time.perf_counter() - t0
    with open(a.out, 'w') as f:
        for q, o in zip(prompts, outs):
            f.write(json.dumps({'prompt': q, 'prompt_sha': hashlib.sha256(q.encode()).hexdigest(),
                                'response': o.outputs[0].text,
                                'finish_reason': o.outputs[0].finish_reason,
                                'victim_model': 'Qwen/Qwen3-8B', 'temperature': 0.0,
                                'max_tokens': a.max_tokens}) + '\n')
    meta = {'n_prompts': len(prompts), 'wall_seconds': wall, 'model': a.model}
    Path(a.out + '.meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
