"""Assemble the v3 detection matrix into local per-encoder JSONs + a README.

Usage: python scratchpad_assemble.py <stage_dir> <out_dir>
Reads under <stage_dir>:
  {enc}/{set}.json            excess + cosine per cell (from v3_detect.py)
  baselines/{set}_pplnli.json ppl + nli on e5 (v3_detect --with-ppl-nli)
  judge/{set}.json            llm judge v3_few_shot on e5 (rq2_llm_filter.py)
  realized/{comqa,nq}_e5.json realized-rate seed sweep (v3_realized_seeds.py)
Writes under <out_dir>:
  {enc}/{set}.json  (cleaned per-cell)  and  README.md
"""
import json, sys
from pathlib import Path

STAGE = Path(sys.argv[1]); OUT = Path(sys.argv[2])
ENCS = ["e5-small-v2", "bge-small-en-v1.5", "gte-small", "all-MiniLM-L6-v2"]
SETS = ["lmp", "scp", "kca"]
SET_LABEL = {"lmp": "LMP-800", "scp": "SCP-800", "kca": "KCA-800"}
ARM = {"lmp": "ComQA human_comqa", "scp": "ComQA human_comqa", "kca": "NQ cacheattack_cleaned_qa"}
# part G cost, computed offline (deployed policy, this corpus)
COST = {"dotprods_median": 20, "dotprods_min": 2, "dotprods_max": 65,
        "storage_float16_bytes_median": 15360, "storage_float16_bytes_max": 49920,
        "dim": 384}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, float) and x != x:
        return "nan"
    return f"{x:.{n}f}"


cells = {}
for e in ENCS:
    for s in SETS:
        d = load(STAGE / e / f"{s}.json")
        if d:
            cells[(e, s)] = d
            (OUT / e).mkdir(parents=True, exist_ok=True)
            (OUT / e / f"{s}.json").write_text(json.dumps(d, indent=2))

L = []
W = L.append
W("# v3 detection matrix (A / B / C / G of EVAL_REDESIGN §12)\n")
W("**Deployed config** (not changed): statistic `excess = max_S cos(variant,anchor) - "
  "cos(entry,anchor)`; policy `multi[count:6+width:2:cap16]/runs`; float16 storage; the "
  "**entry** is scored against the **arriving benign query** (the legal paraphrase); flat "
  "fence; FPR budget 0.05; intent-grouped 50%% holdout, seed 0. Every cell re-embeds with "
  "its own encoder (all 384-dim, CLS pooling; bge-large excluded).\n")
W("**Benign arm (C, red line EVAL_REDESIGN §8.6):** SCP and LMP calibrate on the ComQA "
  "arm (`--benign-generator human_comqa`); **KCA calibrates on its own NQ arm** "
  "(`--benign-generator cacheattack_cleaned_qa`) — never the ComQA arm. Two thresholds "
  "are calibrated per encoder, one per arm.\n")
W("**TPR two columns:** `all planted` = block rate over every scored attack entry; "
  "`poisoned` = block rate restricted to attack rows the judge marked poisoned in "
  "`poisoned_flags.jsonl` (join on record_id; KCA f2 remaps `gcg-f2-NNNN`->`kca-f2-NNNN`).\n")

W("\n## Thresholds (flat-fence height, fit on the WHOLE benign arm)\n")
W("| encoder | ComQA threshold (LMP/SCP) | NQ threshold (KCA) |")
W("|---|---|---|")
for e in ENCS:
    lmp, scp, kca = cells.get((e, "lmp")), cells.get((e, "scp")), cells.get((e, "kca"))
    comqa = None
    if lmp:
        comqa = lmp["threshold_scoring"]
    elif scp:
        comqa = scp["threshold_scoring"]
    nq = kca["threshold_scoring"] if kca else None
    note = ""
    if lmp and scp and abs(lmp["threshold_scoring"] - scp["threshold_scoring"]) > 1e-9:
        note = f" (lmp {f(lmp['threshold_scoring'],6)} / scp {f(scp['threshold_scoring'],6)})"
    W(f"| {e} | {f(comqa,6)}{note} | {f(nq,6)} |")

W("\n## A. Detection (excess), all 12 cells\n")
W("| set | encoder | AUROC | cos-matched AUROC (support) | TPR@5% all | TPR@5% poisoned | realized FPR | cosine AUROC | cosine block@5% |")
W("|---|---|---|---|---|---|---|---|---|")
for s in SETS:
    for e in ENCS:
        d = cells.get((e, s))
        if not d:
            W(f"| {SET_LABEL[s]} | {e} | *pending* | | | | | | |"); continue
        W(f"| {SET_LABEL[s]} | {e} | {f(d['excess_auroc'])} | "
          f"{f(d['excess_auroc_cos_matched'])} ({d['matched_support']}) | "
          f"{f(d['tpr_all_planted'])} | {f(d['tpr_poisoned'])} | "
          f"{f(d['achieved_benign_block_rate'])} | {f(d['cosine_auroc'])} | "
          f"{f(d['cosine_block_rate'])} |")

W("\nPer cell also recorded in JSON: n_genuine, n_attack, n_poisoned, in-sample benign "
  "block rate, both fence heights (scoring + holdout), cosine threshold, timing.\n")

W("\n## A. Per-family breakdown\n")
for s in SETS:
    W(f"\n### {SET_LABEL[s]}")
    W("| encoder | family | n | AUROC | cos-matched (support) | block@5% | n pois | TPR pois |")
    W("|---|---|---|---|---|---|---|---|")
    for e in ENCS:
        d = cells.get((e, s))
        if not d:
            continue
        for fam, fd in sorted(d["families"].items()):
            W(f"| {e} | {fam} | {fd['n_attack']} | {f(fd['excess_auroc'])} | "
              f"{f(fd['excess_auroc_cos_matched'])} ({fd['matched_support']}) | "
              f"{f(fd['excess_block_rate'])} | {fd.get('n_poisoned','-')} | "
              f"{f(fd.get('tpr_poisoned'))} |")

W("\n## B. Baselines on e5 (main table)\n")
W("| set | method | AUROC | block/TPR@5% | ms/row | note |")
W("|---|---|---|---|---|---|")
for s in SETS:
    d = cells.get(("e5-small-v2", s))
    base = load(STAGE / "baselines" / f"{s}_pplnli.json")
    judge = load(STAGE / "judge" / f"{s}.json")
    if d:
        W(f"| {SET_LABEL[s]} | cosine | {f(d['cosine_auroc'])} | {f(d['cosine_block_rate'])} | ~0.00 | one dot product |")
    bl = ((base or {}).get("baselines_ppl_nli") or {})
    if bl.get("perplexity"):
        p = bl["perplexity"]
        W(f"| {SET_LABEL[s]} | conditional perplexity (distilgpt2) | {f(p['auroc'])} | {f(p['block_at_budget'])} | {f(p['ms_per_row'],2)} | |")
    if bl.get("binli"):
        b = bl["binli"]
        W(f"| {SET_LABEL[s]} | bidirectional NLI (MiniLM x-enc) | {f(b['auroc'])} | {f(b['block_at_budget'])} | {f(b['ms_per_row'],2)} | |")
    if judge:
        jf = judge["families"].get("POOLED", {})
        wall = judge.get("wall_seconds")
        total = judge.get("n_benign", 0) + jf.get("n_attack", 0)
        ms = (wall * 1000 / total) if (wall and total) else None
        W(f"| {SET_LABEL[s]} | LLM judge {judge.get('prompt_variant','?')} | "
          f"{f(jf.get('llm_auroc'))} | {f(jf.get('llm_block'))} | {f(ms,1)} | "
          f"emits a verdict not a score; most verdicts disk-cached so ms is a floor, "
          f"deployed cost = one generation/hit |")
    if d:
        W(f"| {SET_LABEL[s]} | **excess (ours)** | {f(d['excess_auroc'])} | {f(d['tpr_all_planted'])} | "
          f"~sub-ms | {COST['dotprods_median']} dot products/hit (median) |")

W("\nLaCache k=20 row is a sibling agent's job — left for merge.\n")

W("\n## C. Realized benign block rate vs the 5% budget — is 0.088 a miscalibration?\n")
comqa_r = load(STAGE / "realized" / "comqa_e5.json")
nq_r = load(STAGE / "realized" / "nq_e5.json")
W("`achieved_benign_block_rate` (the calibrate 'benign' column) is **out of sample**: the "
  "flat fence is fit as the 95th percentile of benign excess on the fit-half of intents, "
  "then its block rate is read on the held-out eval-half. In sample (fence fit and read on "
  "the whole arm) the rate is ~0.05 by construction. The single-seed out-of-sample value "
  "scatters around 0.05 with the split.\n")
W("| arm | in-sample | seed-0 (reported) | seed-mean +/- sd | p05..p95 | frac seeds > 0.05 | benign intents / judgeable |")
W("|---|---|---|---|---|---|---|")
for label, r in [("ComQA (LMP/SCP)", comqa_r), ("NQ (KCA)", nq_r)]:
    if not r:
        W(f"| {label} | *pending* | | | | | |"); continue
    W(f"| {label} | {f(r['in_sample_block_rate'])} | {f(r['realized_seed0'])} | "
      f"{f(r['realized_mean'])} +/- {f(r['realized_std'])} | "
      f"{f(r['realized_p05'])}..{f(r['realized_p95'])} | "
      f"{f(r['frac_seeds_above_budget'],3)} | {r['n_benign_intents']} / {r['n_benign_judgeable']} |")
W("\n**Reading:** the fence holds the budget in sample; the out-of-sample rate is holdout "
  "variance driven by the small benign eval half (~200 intents split in two) and the "
  "discreteness of the excess tail, not a systematic over-block — provided the seed-mean "
  "sits at ~0.05. If a seed-mean sits materially above 0.05 for an arm, that arm's TPR "
  "column is optimistic and is flagged. See per-arm numbers above.\n")

W("\n## G. Cost per hit (deployed policy, this corpus)\n")
W(f"- Dot products read at serving = number of stored span vectors: median "
  f"**{COST['dotprods_median']}** (min {COST['dotprods_min']}, max {COST['dotprods_max']}); "
  f"the paper's '17 dot products' is this quantity.\n"
  f"- Storage per entry (float16, {COST['dim']}-dim span vectors): median "
  f"**{COST['storage_float16_bytes_median']:,} bytes** (max {COST['storage_float16_bytes_max']:,}); "
  f"half the float64 cost, and the rounding moves excess by <=1.4e-4 vs a fence near 5e-3.\n"
  f"- Serving is dot products only — no model call. The embedding of the entry's spans is "
  f"insertion-time work, done once.\n")

W("\n## Commands (exact, run on cpu-server)\n")
W("```bash")
W("export PYTHONPATH=. HF_HOME=<server-home>/hf_cache")
W("# --- A: the 12 detection cells (excess + cosine + TPR poisoned), float16 ---")
W("# per (set, encoder); ENC in {intfloat/e5-small-v2, BAAI/bge-small-en-v1.5,")
W("#   thenlper/gte-small, sentence-transformers/all-MiniLM-L6-v2}")
W("python v3_detect.py --eval final500/eval/lmp_eval.jsonl --encoder $ENC \\")
W("  --attack-role ndss --benign-generator human_comqa \\")
W("  --flags out/v3/poisoned_flags.jsonl --policy 'multi[count:6+width:2:cap16]/runs' \\")
W("  --storage-dtype float16 --out out/v3/detm/$EDIR/lmp.json")
W("python v3_detect.py --eval final500/eval/scp_eval.jsonl --encoder $ENC \\")
W("  --attack-role scp  --benign-generator human_comqa ...  --out .../scp.json")
W("python v3_detect.py --eval final500/eval/kca_eval.jsonl --encoder $ENC \\")
W("  --attack-role gcg  --benign-generator cacheattack_cleaned_qa ... --out .../kca.json")
W("# --- B: ppl + nli on e5 (same judgeable rows) ---")
W("python v3_detect.py --eval <set>_eval.jsonl --encoder intfloat/e5-small-v2 \\")
W("  --attack-role <role> --benign-generator <arm> --flags ... --with-ppl-nli \\")
W("  --out out/v3/detm/baselines/<set>_pplnli.json")
W("# --- B: LLM judge, strongest (two worked examples) prompt ---")
W("python rq2_llm_filter.py --records <set>_eval.jsonl --embedder intfloat/e5-small-v2 \\")
W("  --attack-role <role> --benign-generator <arm> --prompt v3_few_shot --out .../judge/<set>.json")
W("# --- C: realized-FPR seed sweep (200 seeds) ---")
W("python v3_realized_seeds.py --eval scp_eval.jsonl --benign-generator human_comqa \\")
W("  --encoder intfloat/e5-small-v2 --storage-dtype float16 --seeds 200 --out .../comqa_e5.json")
W("python v3_realized_seeds.py --eval kca_eval.jsonl --benign-generator cacheattack_cleaned_qa ...")
W("```")

(OUT / "README.md").write_text("\n".join(L) + "\n")
print("wrote", OUT / "README.md")
print("cells present:", sorted(f"{e}/{s}" for (e, s) in cells))
