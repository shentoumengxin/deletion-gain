"""Per-hit latency of the answer-aware Qwen3-8B judge at batch size 1 (GPU box).

A serving deployment needs one prefill of the judge prompt and the next-token
distribution (the SAFE/UNSAFE first tokens differ), so each hit is timed as one forward
pass over the prompt, CUDA-synchronised, after warm-up.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from judge_answer_aware_score import PROMPT_QKY


def main(argv=None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to('cuda').eval()
    rows = []
    for path in a.rows:
        rows += [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = random.Random(0).sample(rows, a.n)
    times, lengths = [], []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            text = tok.apply_chat_template([{'role': 'user', 'content': PROMPT_QKY.format(
                entry=r['key'], query=r['query'], answer=r['answer'].strip())}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=ids).logits[:, -1].float().softmax(-1)
            torch.cuda.synchronize()
            if i >= 10:                       # warm-up excluded
                times.append((time.perf_counter() - t0) * 1000)
                lengths.append(ids.shape[1])
    import numpy as np
    meta = {'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'batch': 1,
            'n_timed': len(times), 'ms_median': float(np.median(times)),
            'ms_p95': float(np.percentile(times, 95)), 'prompt_tokens_median': float(np.median(lengths))}
    Path(a.out).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
