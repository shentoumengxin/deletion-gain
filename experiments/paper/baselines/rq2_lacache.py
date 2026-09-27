"""LaCache (arXiv 2608.01718) as a competing defense, reimplemented from its description.

**This is a reimplementation of the mechanism as the paper describes it, not the authors'
code.** Numbers here should be read as "the described mechanism, built by us" and not as
a reproduction of their reported results.

The mechanism: a cache hit is not accepted on embedding similarity alone. The system also
decodes the first *k* tokens of an answer to the **arriving query** and compares them to
the **cached answer's** first *k* tokens. Agreement means the stored answer is the one
this query would have got; divergence means it is not, and the hit is refused.

Applied to our threat model:

- A genuine entry and the arriving query mean the same thing, so the two answers should
  open the same way and the hit is served.
- A planted entry's cached answer *is* the poison — that is how it got there. The arriving
  query is clean and would be answered truthfully. The two openings diverge, and the hit
  is refused.

Two questions matter:

**1. Which constructions does it catch?** It is scored on the same poisoned entries and
benign controls as every other defense, at the same budget.

**2. What does it cost?** It has to decode to check, and decoding is the thing a cache
exists to avoid. Even k tokens pays the prefill and the time-to-first-token, which is the
expensive part of a short request.

Scoring is continuous rather than binary — the fraction of the first k words that match,
turned around so higher means more suspicious — so it can be read at the same 5% budget
as everything else instead of only at its own operating point.

**Two grains.** ``--unit words`` (the default, and what the first pass ran) decodes here,
through an API model, and compares the first k *words*. ``--unit tokens --k 20
--prefixes <jsonl>`` is the faithful reading: the prefixes were decoded ahead of time by
``experiments/paper/gpu/lacache_prefix.py`` on the same Qwen3-8B that answers for the
ASR table, cut to 20 tokens by that model's own tokenizer, which is what the paper
specifies. Nothing is generated in this file on that path — it only embeds and scores, so
it runs on cpu-server with no victim model present. ``--embedder2`` adds the paper's own
encoder (bge-large-en-v1.5) as a second column beside our e5 main.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import (
    auroc, filter_benign_generators, load_pair_rows, parse_policy,
)
from sentry.cache.defense.deletion import build_profile, excess

ANSWER_PROMPT = "{text}"


def opening(text: str, k: int) -> list[str]:
    """The first k words of an answer, lowercased and stripped of punctuation.

    Words stand in for tokens. The paper counts tokens; we do not share its tokenizer, and
    a word-level prefix is the same comparison at a coarser grain. Coarser favours the
    baseline — it forgives small differences that a token comparison would flag — so this
    does not understate LaCache.
    """
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in (text or ""))
    return cleaned.split()[:k]


def divergence_lexical(cached_answer: str, fresh_answer: str, k: int) -> float:
    """Share of the first k words that fail to match in position.

    **Kept only as a secondary reading, because it is too brittle to be the primary.**
    Two correct answers to the same question diverge at word one whenever they are worded
    differently -- "Lincoln High School, in Warren" against "Eminem attended Lincoln High
    School" agree on the fact and share no prefix. Measured, that puts the benign 95th
    percentile at complete divergence, which drives the block rate to zero and would
    understate the baseline badly.
    """
    a, b = opening(cached_answer, k), opening(fresh_answer, k)
    if not a and not b:
        return 0.0
    width = max(len(a), len(b), 1)
    agree = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
    return 1.0 - agree / width


def divergence_semantic(cached_vec, fresh_vec) -> float:
    """1 - cosine between the two answer openings. The primary reading.

    What LaCache is asking is whether the stored answer is the one this query would have
    produced. That is a question about meaning, not about wording, so the comparison is
    made between the openings' embeddings. It is the form of the mechanism most likely to
    match what the paper intends, and it is far kinder to the baseline than exact prefix
    matching.
    """
    return 1.0 - float(cached_vec @ fresh_vec)


def _poison_literal(row) -> str:
    """The string the attacker wanted stored, pulled out of the attack text."""
    import re
    match = re.search(r'"([^"]{2,})"', row.text)
    return match.group(1) if match else ""


def block_at_budget(attack, benign, budget=0.05):
    if len(attack) < 2 or len(benign) < 2:
        return float("nan")
    return float((np.asarray(attack) > float(np.quantile(benign, 1.0 - budget))).mean())


# ---- the precomputed-prefix path (--unit tokens) ---------------------------

def _embed_unique(embedder, texts: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Normalized rows for the distinct strings, plus the string -> row index."""
    uniq = sorted({t or "" for t in texts})
    chunks = [embedder.encode([u or "empty" for u in uniq[i:i + 256]])
              for i in range(0, len(uniq), 256)]
    matrix = np.concatenate(chunks, axis=0)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)
    return matrix, {t: i for i, t in enumerate(uniq)}


def _cells(scored, key, families, benign, budget):
    """One ``{family: cell}`` block for a score key, plus POOLED."""
    out = {}
    for family in families + ["POOLED"]:
        attacks = [r for r in scored if r["arm"] == "attack"
                   and (family == "POOLED" or r["family"] == family)]
        if len(attacks) < 2:
            continue
        a = np.array([r[key] for r in attacks])
        b = np.array([r[key] for r in benign])
        out[family] = {"n_attack": len(attacks), "auroc": auroc(a, b),
                       "block": block_at_budget(a, b, budget)}
    return out


def run_from_prefixes(args) -> int:
    """Score LaCache from prefixes decoded elsewhere. No victim model is contacted.

    The arm is read off the set: ``benign`` rows are the genuine entries the 5% budget is
    fitted on, everything else is a plant. ``excess`` is recomputed here on the same rows
    from the entry and query text the prefix file carries, so the head-to-head column is
    the same rows for both defenses.
    """
    from sentry.embeddings import TransformerCLSEmbedder

    rows = [json.loads(line) for line in
            Path(args.prefixes).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{args.prefixes}: no rows")
    file_k = {r.get("k") for r in rows}
    if file_k != {args.k}:
        raise SystemExit(f"--k {args.k} but the prefix file holds k in {sorted(file_k)}; "
                         "regenerate the prefixes or pass the matching --k")

    poisoned_ids: set[str] | None = None
    if args.poisoned_ids:
        blob = json.loads(Path(args.poisoned_ids).read_text(encoding="utf-8"))
        poisoned_ids = set(blob["record_ids"] if isinstance(blob, dict) else blob)
        print(f"poisoned subset: {len(poisoned_ids)} record_ids from "
              f"{args.poisoned_ids}", flush=True)

    scored = []
    for r in rows:
        literal = (r.get("literal") or "").strip()
        if poisoned_ids is not None:
            poisoned = r["record_id"] in poisoned_ids
        else:
            poisoned = bool(literal) and literal.lower() in (r.get("entry_response")
                                                             or "").lower()
        scored.append({
            "record_id": r["record_id"],
            "arm": "genuine" if r.get("set") == "benign" else "attack",
            "family": f'{r.get("set")}/{r.get("family")}',
            "entry_text": r["entry_text"], "query_text": r["query_text"],
            "entry_prefix": r["entry_prefix"], "query_prefix": r["query_prefix"],
            "poisoned": poisoned,
        })
    print(f"{len(scored)} rows | k = {args.k} tokens | "
          f"{sum(1 for r in scored if r['arm'] == 'genuine')} benign", flush=True)

    encoders = [("main", args.embedder)] + ([("second", args.embedder2)]
                                            if args.embedder2 else [])
    embedders = {}
    for slot, name in encoders:
        print(f"embedding prefixes with {name} ({slot})", flush=True)
        embedder = TransformerCLSEmbedder(name)
        embedders[slot] = embedder
        matrix, index = _embed_unique(
            embedder, [r["entry_prefix"] for r in scored]
                      + [r["query_prefix"] for r in scored])
        for r in scored:
            r[f"lacache_{slot}"] = divergence_semantic(
                matrix[index[r["entry_prefix"] or ""]],
                matrix[index[r["query_prefix"] or ""]])

    # ours, on exactly these rows, with the main encoder
    policy = parse_policy(args.policy)
    main_embedder = embedders["main"]
    keep = []
    for r in scored:
        profile = build_profile(r["entry_text"], main_embedder, policy)
        r["judgeable"] = profile.judgeable
        if profile.judgeable:
            keep.append((r, profile))
    anchors = sorted({r["query_text"] for r, _ in keep})
    amatrix, aindex = _embed_unique(main_embedder, anchors)
    for r, profile in keep:
        r["excess"] = excess(profile, amatrix[aindex[r["query_text"]]]).excess_span
    unjudgeable = [r for r in scored if not r["judgeable"]]
    if unjudgeable:
        print(f"  {len(unjudgeable)} rows too short for the span policy; excess columns "
              f"drop them, LaCache columns keep them", flush=True)

    families = sorted({r["family"] for r in scored if r["arm"] == "attack"})
    benign = [r for r in scored if r["arm"] == "genuine"]
    report = {"unit": "tokens", "k": args.k, "budget": args.budget,
              "prefixes": str(args.prefixes), "policy": args.policy,
              "encoders": {slot: name for slot, name in encoders},
              "n_rows": len(scored), "n_benign": len(benign),
              "n_attack": sum(1 for r in scored if r["arm"] == "attack"),
              "n_poisoned": sum(1 for r in scored
                                if r["arm"] == "attack" and r["poisoned"]),
              "poisoned_from": args.poisoned_ids or "literal-in-stored-answer heuristic",
              "note": "reimplementation from the paper's description, not the authors' "
                      "code; prefixes decoded by the ASR victim, cut with its tokenizer"}

    for label, subset in (("all_planted", scored),
                          ("poisoned", [r for r in scored
                                        if r["arm"] == "genuine" or r["poisoned"]])):
        sub_benign = [r for r in subset if r["arm"] == "genuine"]
        block = {}
        for slot, _ in encoders:
            block[f"lacache_{slot}"] = _cells(subset, f"lacache_{slot}", families,
                                              sub_benign, args.budget)
        block["excess"] = _cells([r for r in subset if r["judgeable"]], "excess", families,
                                 [r for r in sub_benign if r["judgeable"]], args.budget)
        report[label] = block

        print(f"\n  {label}")
        header = f"  {'family':<28}{'n':>6}"
        for slot, _ in encoders:
            header += f"{'LaCache/' + slot + ' AUROC':>22}{'block':>8}"
        header += f"{'excess AUROC':>14}{'block':>8}"
        print(header)
        for family in families + ["POOLED"]:
            cell0 = block["lacache_main"].get(family)
            if cell0 is None:
                continue
            line = f"  {family:<28}{cell0['n_attack']:>6}"
            for slot, _ in encoders:
                cell = block[f"lacache_{slot}"].get(family, {})
                line += f"{cell.get('auroc', float('nan')):>22.3f}" \
                        f"{cell.get('block', float('nan')):>8.3f}"
            ours = block["excess"].get(family, {})
            line += f"{ours.get('auroc', float('nan')):>14.3f}" \
                    f"{ours.get('block', float('nan')):>8.3f}"
            print(line)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="",
                        help="required for --unit words; unused with --prefixes, which "
                             "already carries the entry and query text")
    parser.add_argument("--embedder", default="intfloat/e5-small-v2")
    parser.add_argument("--embedder2", default="",
                        help="second encoder for the prefix comparison, e.g. "
                             "BAAI/bge-large-en-v1.5 (LaCache's own). Appendix column.")
    parser.add_argument("--policy", default="count:6")
    parser.add_argument("--benign-generator", action="append", default=["human_comqa"])
    parser.add_argument("--unit", choices=("words", "tokens"), default="words",
                        help="'words' decodes here and compares k words (the first pass); "
                             "'tokens' scores k model tokens decoded ahead of time")
    parser.add_argument("--k", type=int, default=8, help="prefix length compared")
    parser.add_argument("--prefixes", default="",
                        help="JSONL from gpu/lacache_prefix.py. Implies --unit tokens.")
    parser.add_argument("--poisoned-ids", default="",
                        help="JSON list (or {'record_ids': [...]}) of the plants that "
                             "actually poisoned, from the judge. Without it the poisoned "
                             "subset falls back to literal-in-stored-answer.")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--limit-per-arm", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    if args.prefixes:
        args.unit = "tokens"
        return run_from_prefixes(args)
    if args.unit == "tokens":
        raise SystemExit("--unit tokens needs --prefixes: this file does not tokenize "
                         "with the victim's tokenizer, gpu/lacache_prefix.py does")
    if not args.records:
        raise SystemExit("--records is required for --unit words")

    from collections import defaultdict

    from sentry.research.operators import Client, load_env
    from sentry.embeddings import TransformerCLSEmbedder

    rows, _ = load_pair_rows(args.records)
    rows = filter_benign_generators(rows, args.benign_generator)
    if args.limit_per_arm:
        per = defaultdict(list)
        for r in rows:
            per[r.family].append(r)
        rows = [r for fam in per for r in per[fam][:args.limit_per_arm]]

    policy = parse_policy(args.policy)
    embedder = TransformerCLSEmbedder(args.embedder)
    client = Client(load_env(), cache_name="rq2_lacache")
    print(f"{len(rows)} entry-side rows | k = {args.k} words", flush=True)

    # ours, on the same rows
    pool: dict[str, int] = {}

    def intern(t):
        if t not in pool:
            pool[t] = len(pool)
        return pool[t]

    keep = []
    for row in rows:
        profile = build_profile(row.text, embedder, policy)
        if not profile.judgeable:
            continue
        keep.append((row, profile, intern(row.anchor)))
    texts = [None] * len(pool)
    for t, i in pool.items():
        texts[i] = t
    chunks = [embedder.encode(texts[i:i + 256]) for i in range(0, len(texts), 256)]
    matrix = np.concatenate(chunks, axis=0)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)

    scored = []
    for row, profile, ai in keep:
        reading = excess(profile, matrix[ai])
        scored.append({"row": row, "excess": reading.excess_span})

    # Every distinct text that needs an answer: entries supply the CACHED answer,
    # anchors supply the freshly decoded one.
    wanted = sorted({r["row"].text for r in scored} | {r["row"].anchor for r in scored})
    print(f"answers needed for {len(wanted)} distinct texts", flush=True)
    answers: dict[str, str] = {}
    start = time.perf_counter()
    for index, text in enumerate(wanted):
        answers[text] = client.chat(
            [{"role": "user", "content": ANSWER_PROMPT.format(text=text)}],
            temperature=0.0, seed=index, max_tokens=48,
            extra_body={"thinking": {"type": "disabled"}},
        ) or ""
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(wanted)}", flush=True)
    wall = time.perf_counter() - start
    fresh = len(wanted) - getattr(client, "cache_hits", 0)
    print(f"  {wall:.0f}s wall, {fresh} fresh generations "
          f"({wall / max(fresh, 1):.2f}s each)", flush=True)

    # Embed the first k words of every answer so the comparison can be semantic.
    openings = {t: " ".join(opening(a, args.k)) or "empty" for t, a in answers.items()}
    uniq = sorted(set(openings.values()))
    ochunks = [embedder.encode(uniq[i:i + 256]) for i in range(0, len(uniq), 256)]
    omat = np.concatenate(ochunks, axis=0)
    omat /= np.linalg.norm(omat, axis=1, keepdims=True).clip(1e-12)
    oindex = {t: i for i, t in enumerate(uniq)}

    for record in scored:
        row = record["row"]
        cached = answers[row.text]
        record["lacache"] = divergence_semantic(
            omat[oindex[openings[row.text]]], omat[oindex[openings[row.anchor]]])
        record["lacache_lexical"] = divergence_lexical(cached, answers[row.anchor], args.k)
        # Did the plant actually poison anything? LaCache compares the STORED answer
        # against a fresh one, so it can only object when the stored answer is wrong.
        # An attack whose instruction the model refused leaves a truthful answer in the
        # cache -- there is nothing to reject, and declining to reject it is correct, not
        # a miss. Scoring LaCache over attempts that poisoned nothing would charge it for
        # doing the right thing.
        literal = _poison_literal(row)
        record["poisoned"] = bool(literal) and literal.lower() in (cached or "").lower()

    families = sorted({r["row"].family for r in scored if r["row"].arm == "attack"})
    benign = [r for r in scored if r["row"].arm == "genuine"]
    report = {"k": args.k, "n_benign": len(benign), "seconds_per_generation":
              wall / max(fresh, 1), "families": {},
              "note": "reimplementation from the paper's description, not the authors' code"}

    n_pois = sum(1 for r in scored if r["row"].arm == "attack" and r["poisoned"])
    n_atk = sum(1 for r in scored if r["row"].arm == "attack")
    print(f"\n  of {n_atk} attacks, {n_pois} actually left a poisoned answer "
          f"({n_pois / max(n_atk, 1):.1%})", flush=True)
    report["n_attack"] = n_atk
    report["n_poisoned"] = n_pois

    print(f"\n  ALL attack attempts")
    print(f"  {'family':<30}{'n':>6}{'LaCache AUROC':>15}{'LaCache block':>15}"
          f"{'excess AUROC':>14}{'excess block':>14}")
    for family in families + ["POOLED"]:
        attacks = [r for r in scored if r["row"].arm == "attack"
                   and (family == "POOLED" or r["row"].family == family)]
        if len(attacks) < 2:
            continue
        la = np.array([r["lacache"] for r in attacks])
        lb = np.array([r["lacache"] for r in benign])
        ea = np.array([r["excess"] for r in attacks])
        eb = np.array([r["excess"] for r in benign])
        cell = {"n_attack": len(attacks),
                "lacache_auroc": auroc(la, lb),
                "lacache_block": block_at_budget(la, lb, args.budget),
                "excess_auroc": auroc(ea, eb),
                "excess_block": block_at_budget(ea, eb, args.budget)}
        report["families"][family] = cell
        print(f"  {family:<30}{len(attacks):>6}{cell['lacache_auroc']:>15.3f}"
              f"{cell['lacache_block']:>15.3f}{cell['excess_auroc']:>14.3f}"
              f"{cell['excess_block']:>14.3f}")

    # The comparison that is fair to LaCache: only the plants that poisoned.
    print(f"\n  ONLY the attempts that actually poisoned the cache")
    print(f"  {'family':<30}{'n':>6}{'LaCache AUROC':>15}{'LaCache block':>15}"
          f"{'excess AUROC':>14}{'excess block':>14}")
    report["poisoned_only"] = {}
    for family in families + ["POOLED"]:
        attacks = [r for r in scored if r["row"].arm == "attack" and r["poisoned"]
                   and (family == "POOLED" or r["row"].family == family)]
        if len(attacks) < 2:
            continue
        la = np.array([r["lacache"] for r in attacks])
        lb = np.array([r["lacache"] for r in benign])
        ea = np.array([r["excess"] for r in attacks])
        eb = np.array([r["excess"] for r in benign])
        cell = {"n_attack": len(attacks),
                "lacache_auroc": auroc(la, lb),
                "lacache_block": block_at_budget(la, lb, args.budget),
                "excess_auroc": auroc(ea, eb),
                "excess_block": block_at_budget(ea, eb, args.budget)}
        report["poisoned_only"][family] = cell
        print(f"  {family:<30}{len(attacks):>6}{cell['lacache_auroc']:>15.3f}"
              f"{cell['lacache_block']:>15.3f}{cell['excess_auroc']:>14.3f}"
              f"{cell['excess_block']:>14.3f}")

    print("\n  The two answer different questions. excess refuses the ENTRY, whether or")
    print("  not the poisoning worked, so a failed plant is still kept out of the cache.")
    print("  LaCache compares the STORED answer, so it can only object once the poisoning")
    print("  succeeded -- and it pays a decode on every hit to find out.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
