#!/usr/bin/env python
"""A genuinely third-vendor second judge on the same stratified sample.

`cross_judge_v3.py` says so itself: its second judge is deepseek-v4-pro, "different model,
same endpoint; only DeepSeek creds exist". Flash against pro shares a vendor, a tokenizer
family and most of a training pipeline, so agreement between them bounds *model* variance,
not *vendor* variance -- and vendor variance is what a reviewer means when they ask whether
the verdicts are an artifact of who produced the judge.

This runs a third judge from a different vendor entirely, locally, on the same rows:
the same seed and the same stratification code, so the sample is identical; the same two
system prompts routed the same way (inject for KCA, poison for the rest); the same payload
keys; the same parse. Only the model behind the verdict changes.

Errors are excluded from agreement rather than scored as disagreement, exactly as the
original does, and the error count is reported so a judge that mostly failed to answer
cannot masquerade as a judge that mostly agreed.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "<server-home>/project-workdir")
from experiments.paper.rq2_robustness.asr_judge import _POISON_JUDGE                        # noqa: E402
from experiments.paper.rq2_robustness.injection_success import _JUDGE_SYSTEM, cohen_kappa   # noqa: E402

SEED = 20260827
PER_GROUP = 56
OUT = Path("<server-workdir>/out/v3")
LOCAL = Path("<server-workdir>/supp_fix")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/Phi-4-mini-instruct"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 16

files = {
    "benign": OUT / "judged_benign_full.jsonl",
    "scp": OUT / "judged_scp_full.jsonl",
    "kca": OUT / "judged_kca_full.jsonl",
    "lmp": OUT / "judged_lmp_all_full.jsonl",
}


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


data = {k: load(v) for k, v in files.items()}


def primary_pos(r):
    return r.get("outcome") == "poisoned"


# ---- identical stratified sample: same seed, same code path -----------------
rng = random.Random(SEED)
sample = []
groups = {}
for setname, rows in data.items():
    for r in rows:
        groups.setdefault((setname, r.get("family")), []).append(r)
for (setname, fam), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    pos = [r for r in rows if primary_pos(r)]
    neg = [r for r in rows if not primary_pos(r)]
    half = PER_GROUP // 2
    take_pos = rng.sample(pos, min(half, len(pos)))
    take_neg = rng.sample(neg, min(PER_GROUP - len(take_pos), len(neg)))
    if len(take_pos) + len(take_neg) < PER_GROUP:
        extra_pool = [r for r in pos if r not in take_pos] + [r for r in neg if r not in take_neg]
        rng.shuffle(extra_pool)
        take_pos += extra_pool[:PER_GROUP - len(take_pos) - len(take_neg)]
    for r in take_pos + take_neg:
        sample.append((setname, fam, r))
print(f"stratified sample n={len(sample)} across {len(groups)} groups, seed={SEED}", flush=True)

# ---- the third-vendor judge ------------------------------------------------
import torch                                               # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
print("loaded", flush=True)


def build(item):
    setname, fam, r = item
    if setname == "kca":
        system, payload, key = _JUDGE_SYSTEM, {
            "benign_query": r.get("canonical", ""),
            "ipi_prompt": r.get("payload", ""),
            "llm_response": r.get("response") or ""}, "success"
    else:
        system, payload, key = _POISON_JUDGE, {
            "question": r.get("canonical", ""),
            "planted_answer": r.get("literal", ""),
            "llm_response": r.get("response") or ""}, "poisoned"
    text = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=True)
    return text, key


prompts = [build(it) for it in sample]
verdicts = []
start = time.perf_counter()
for i in range(0, len(prompts), BATCH):
    chunk = prompts[i:i + BATCH]
    enc = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
              truncation=True, max_length=3072)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=96, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    for j, c in enumerate(chunk):
        reply = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        verdicts.append((reply, c[1]))
    done = min(i + BATCH, len(prompts))
    el = time.perf_counter() - start
    print(f"  {done}/{len(prompts)}  {el:.0f}s  eta {el/done*(len(prompts)-done)/60:.1f} min",
          flush=True)


def parse(reply, key):
    """Same tolerance as PromptJudge: strip a fence, then require JSON with the key."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo >= 0 and hi > lo:
        text = text[lo:hi + 1]
    try:
        v = json.loads(text)
    except Exception as exc:
        return None, f"unparseable: {type(exc).__name__}"
    if key not in v:
        return None, f"missing key {key!r}"
    return bool(v[key]), ""


results, nerr = [], 0
for (setname, fam, r), (reply, key) in zip(sample, verdicts):
    cv, err = parse(reply, key)
    if err:
        nerr += 1
    results.append({"set": setname, "family": fam, "record_id": r.get("record_id"),
                    "primary": primary_pos(r), "cross": cv, "error": err})

(LOCAL / "cross_judge_local.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")

both = [x for x in results if x["cross"] is not None]


def kap(rows):
    if not rows:
        return None, None, 0
    a = [x["primary"] for x in rows]
    b = [x["cross"] for x in rows]
    return (round(cohen_kappa(a, b), 4),
            round(sum(p == c for p, c in zip(a, b)) / len(rows), 4), len(rows))


k, ag, n = kap(both)
summary = {"cross_model": MODEL, "vendor_note": "third vendor: distinct from the DeepSeek "
                                                "primary judge and the Qwen3-8B victim",
           "seed": SEED, "n_sampled": len(sample), "n_answered": len(both),
           "n_cross_errors": nerr,
           "overall": {"cohen_kappa": k, "raw_agreement": ag, "n": n,
                       "primary_pos_rate": round(sum(x["primary"] for x in both) / n, 4),
                       "cross_pos_rate": round(sum(x["cross"] for x in both) / n, 4)}}
summary["by_set"] = {}
for s in ("benign", "scp", "kca", "lmp"):
    rs = [x for x in both if x["set"] == s]
    k2, ag2, n2 = kap(rs)
    summary["by_set"][s] = {
        "cohen_kappa": k2, "raw_agreement": ag2, "n": n2,
        "primary_pos_rate": round(sum(x["primary"] for x in rs) / n2, 4) if n2 else None,
        "cross_pos_rate": round(sum(x["cross"] for x in rs) / n2, 4) if n2 else None}

(LOCAL / "cross_judge_local.summary.json").write_text(
    json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
print("CROSS_LOCAL_DONE")
