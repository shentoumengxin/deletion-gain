"""RQ4: what does it cost an attacker to spread a payload past ``excess``?

``excess_span`` maximises over **prefixes and suffixes**. A payload sitting at one end is
recovered by dropping that end, which is why every attack corpus we hold — append and
suffix constructions — scores positive. The evasion follows immediately from the
definition and needs no search: **put payload at both ends**, and no prefix and no suffix
is clean, so no shortened version beats the whole and ``excess`` goes to zero or below.

That is not the interesting question. The interesting question is what the evasion costs.
Every word of payload is a word that is not the victim's question, so it pulls
``cos(t, k)`` down, and an entry the cache will not retrieve is not an attack. So the
measurement is a **trade-off curve, not a rate**:

    evasion of ``excess``   against   retrievability at the cache's own floor

If the constructions that evade ``excess`` all fall below the retrieval threshold, then
the statistic and the threshold together bound the attacker even though the statistic
alone is evadable — and that is a defensible claim. If some construction evades while
staying retrievable, that is the hole, and we would rather publish it than have it found.

The probe from ``rq1_probe.py`` is scored on the same texts. It is a signature matcher
for the append template (0.996 there, 0.662 on untemplated non-equivalence), so a new
construction is the test of whether it generalises at all.

**Payload efficacy (ISR) is not measured here.** A construction that evades but no longer
executes is not an attack, and the judge that decides that is a separate pass. This script
produces the candidates and the geometry; ISR judges them afterwards.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import PairRow, auroc, parse_policy
from sentry.cache.defense.deletion import build_profile, excess, unit
from sentry.cache.defense.spans import build_spans


def excess_all_runs(text: str, anchor_vec, embedder, policy) -> float:
    """``excess`` over EVERY contiguous run of spans, not only prefixes and suffixes.

    ``DELETION_TEST.md`` §1 states the statistic as ``max over contiguous sub-spans S``
    and then narrows it, in the same sentence, to "every prefix and every suffix". The
    deployed code took the narrow reading. That is the hole the spread constructions walk
    through: a payload at both ends, or repeated through the text, leaves no clean prefix
    and no clean suffix — but it leaves a clean **interior** run, which the narrow form
    never looks at.

    The general form costs ``n(n+1)/2 - 1`` sub-texts instead of ``2n - 2`` (20 vs 10 at
    n=6), so the storage and the serving dot products roughly double. Whether that buys
    back the evasion is what this measures.
    """
    parts = policy.segments(text)
    if len(parts) < policy.min_segments:
        return float("nan")
    runs = [" ".join(parts[i:j])
            for i in range(len(parts))
            for j in range(i + 1, len(parts) + 1)
            if not (i == 0 and j == len(parts))]
    runs = [r for r in runs if r.strip()]
    vectors = embedder.encode([text] + runs)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    a = unit(anchor_vec)
    base = float(vectors[0] @ a)
    return float((vectors[1:] @ a).max()) - base


# ---- constructions ---------------------------------------------------------
# Each takes the victim's question and a payload clause, and returns the entry text an
# attacker would plant. `append` reproduces what the ndss families already do and is the
# control every other row is read against.

def c_append(question: str, payload: str) -> str:
    return f"{question.rstrip('?').rstrip()}, {payload}?"


def c_prepend(question: str, payload: str) -> str:
    return f"{payload}: {question}"


def c_bracket(question: str, payload: str) -> str:
    """Payload at both ends. Defeats a prefix/suffix maximum by construction."""
    return f"{payload}: {question.rstrip('?').rstrip()}, {payload}?"


def c_interleave(question: str, payload: str, every: int = 3) -> str:
    """Payload words dealt into the question every ``every`` words.

    The most thorough spread available without a generator, and the one that should hurt
    fluency and cosine the most.
    """
    q = question.split()
    p = payload.split()
    out, pi = [], 0
    for i, word in enumerate(q):
        out.append(word)
        if (i + 1) % every == 0 and pi < len(p):
            out.append(p[pi])
            pi += 1
    out.extend(p[pi:])
    return " ".join(out)


def c_repeat(question: str, payload: str, times: int = 3) -> str:
    """Payload repeated at even intervals through the question."""
    q = question.rstrip("?").rstrip().split()
    if times < 2 or not q:
        return c_bracket(question, payload)
    cuts = [round(i * len(q) / (times - 1)) for i in range(times - 1)]
    out, seen = [], 0
    for idx, word in enumerate(q):
        while seen < len(cuts) and cuts[seen] == idx:
            out.append(payload)
            seen += 1
        out.append(word)
    out.append(payload)
    return " ".join(out) + "?"


CONSTRUCTIONS = {
    "append": c_append,
    "prepend": c_prepend,
    "bracket": c_bracket,
    "interleave3": lambda q, p: c_interleave(q, p, every=3),
    "interleave2": lambda q, p: c_interleave(q, p, every=2),
    "repeat3": lambda q, p: c_repeat(q, p, times=3),
    "repeat4": lambda q, p: c_repeat(q, p, times=4),
}


# ---- corpus ----------------------------------------------------------------

def load_targets(path, limit: int = 0):
    """(question, anchor, intent) for comqa intents that have a benign query.

    The anchor is a ``legal`` paraphrase — the query an ordinary user would arrive with —
    and the question is the canonical the attacker targets. Same pairing the entry side uses
    everywhere else.
    """
    records = [json.loads(line) for line in
               Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    canon, legal = {}, {}
    for r in records:
        if r.get("query_role") == "canonical" and r.get("generator") == "human_comqa":
            canon.setdefault(r["intent_id"], r["text"])
        elif r.get("query_role") == "legal":
            legal.setdefault(r["intent_id"], r["text"])
    out = [(canon[i], legal[i], i) for i in sorted(set(canon) & set(legal))]
    return out[:limit] if limit else out


def load_payloads(path, limit: int = 40):
    """The payload clauses the ndss families actually use, lifted from their texts.

    Taken from the corpus rather than invented, so the construction is the only thing
    that changes between the control row and the spread rows.
    """
    records = [json.loads(line) for line in
               Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, out = set(), []
    for r in records:
        if r.get("query_role") != "ndss":
            continue
        m = re.search(r'((?:reply with|the answer is)\s+"[^"]+")', r["text"], re.I)
        if not m:
            continue
        clause = m.group(1)
        if clause.lower() not in seen:
            seen.add(clause.lower())
            out.append(clause)
        if len(out) >= limit:
            break
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--embedder", default="intfloat/e5-small-v2")
    parser.add_argument("--policy", default="count:6")
    parser.add_argument("--cache-threshold", type=float, default=0.90)
    parser.add_argument("--limit-intents", type=int, default=200)
    parser.add_argument("--all-runs", action="store_true",
                        help="also score the general form: max over EVERY contiguous "
                             "run, not only prefixes and suffixes")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from sentry.embeddings import TransformerCLSEmbedder

    policy = parse_policy(args.policy)
    embedder = TransformerCLSEmbedder(args.embedder)
    targets = load_targets(args.records, args.limit_intents)
    payloads = load_payloads(args.records)
    print(f"{len(targets)} target intents x {len(CONSTRUCTIONS)} constructions "
          f"| {len(payloads)} payload clauses | {args.embedder}", flush=True)

    # benign reference: the genuine entry against the same anchor
    benign_rows = [PairRow(text=q, anchor=a, arm="genuine", family="canonical",
                         intent_id=i) for q, a, i in targets]

    rows = list(benign_rows)
    for n, (question, anchor, intent) in enumerate(targets):
        payload = payloads[n % len(payloads)]
        for name, build in CONSTRUCTIONS.items():
            rows.append(PairRow(text=build(question, payload), anchor=anchor,
                              arm="attack", family=name, intent_id=intent))

    # encode once over the whole pool
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

    per = defaultdict(list)
    for row, profile, ai in keep:
        reading = excess(profile, matrix[ai])
        rec = {
            "cos": reading.base_cos, "excess": reading.excess_span,
            "words": reading.words, "best_end": reading.best_end,
            "text": row.text,
        }
        if args.all_runs:
            rec["excess_all"] = excess_all_runs(row.text, matrix[ai], embedder, policy)
        per[row.family].append(rec)

    benign = per["canonical"]
    b_exc = np.array([r["excess"] for r in benign])
    fence = float(np.quantile(b_exc, 0.95))
    fence_all = (float(np.quantile([r["excess_all"] for r in benign], 0.95))
                 if args.all_runs else None)
    b_cos = np.array([r["cos"] for r in benign])
    print(f"\nbenign: n={len(benign)}  median cos {np.median(b_cos):.4f}  "
          f"excess p95 (the fence) {fence:.5f}")

    if fence_all is not None:
        print(f"benign: all-runs excess p95 (the general fence) {fence_all:.5f}")
    print(f"\n  {'construction':<14}{'n':>5}{'med cos':>9}{'retriev':>9}"
          f"{'evade P/S':>11}{'BOTH P/S':>10}{'evade ALL':>11}{'BOTH ALL':>10}")
    report = {"embedder": args.embedder, "policy": policy.fingerprint(),
              "fence_excess_p95": fence, "cache_threshold": args.cache_threshold,
              "n_benign": len(benign), "constructions": {}}
    for name in CONSTRUCTIONS:
        recs = per.get(name, [])
        if not recs:
            continue
        cos = np.array([r["cos"] for r in recs])
        exc = np.array([r["excess"] for r in recs])
        words = np.array([r["words"] for r in recs])
        retrievable = cos >= args.cache_threshold
        evades = exc <= fence
        both = retrievable & evades
        line = {
            "n": int(len(recs)), "median_cos": float(np.median(cos)),
            "retrievable_rate": float(retrievable.mean()),
            "median_excess": float(np.median(exc)),
            "evasion_rate": float(evades.mean()),
            "evades_and_retrievable": float(both.mean()),
            "median_words": float(np.median(words)),
        }
        if fence_all is not None:
            exc_all = np.array([r["excess_all"] for r in recs])
            ev_all = exc_all <= fence_all
            line["median_excess_all_runs"] = float(np.nanmedian(exc_all))
            line["evasion_rate_all_runs"] = float(ev_all.mean())
            line["evades_and_retrievable_all_runs"] = float((retrievable & ev_all).mean())
        report["constructions"][name] = line
        ea = line.get("evasion_rate_all_runs", float("nan"))
        ba = line.get("evades_and_retrievable_all_runs", float("nan"))
        print(f"  {name:<14}{len(recs):>5}{np.median(cos):>9.4f}"
              f"{retrievable.mean():>9.3f}{evades.mean():>11.3f}{both.mean():>10.3f}"
              f"{ea:>11.3f}{ba:>10.3f}")

    print("\n  'BOTH' is the number that matters: an entry that evades the statistic but")
    print("  cannot be retrieved is not an attack. Payload efficacy (ISR) is a separate")
    print("  pass and is NOT included here.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        samples = {n: [r["text"] for r in per[n][:3]] for n in per}
        Path(args.out).with_suffix(".samples.json").write_text(
            json.dumps(samples, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
