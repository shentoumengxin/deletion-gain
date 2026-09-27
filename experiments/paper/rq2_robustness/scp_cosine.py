"""E2 acceptance — embed the SCP set with e5-small-v2 and report the numbers.

Reuses the project encoder verbatim (``TransformerCLSEmbedder``: CLS-token pooling
+ L2 normalisation, raw text, no e5 query/passage prefix — matching how the
corpus in ``embeddings/e5/`` was built). For every SCP row it computes:

  * ``cos(x*, canonical)`` — the SCP text vs its own ComQA canonical (Wu's Qtarget);
  * ``cos(x*, legal)``     — the SCP text vs the *nearest* legal paraphrase of the
    same intent (the closest benign anchor a cache would actually hit).

Reports, per template (Z/I/P): the fraction with cosine >= 0.90 for both anchors
(Wu reports 0.82-0.93 similarity for his constructions), and the word/char length
distribution vs the legal paraphrases. Also folds in the payload rejection counts
and per-template intent coverage, and emits example rows. Writes one JSON with all
of it (default: ``scp_build.json`` next to this script's output dir).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np

_HERE = Path(__file__).resolve()
for _root in (_HERE.parents[3], _HERE.parents[0]):
    if (_root / "sentry").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from sentry.embeddings import TransformerCLSEmbedder  # noqa: E402

TEMPLATES = ["scp_z", "scp_i", "scp_p"]
LETTER = {"scp_z": "Z", "scp_i": "I", "scp_p": "P"}


def _decode_true_answer(answer: str) -> str:
    """Same decode rule as scp_payloads.py: Wikipedia URL -> the entity it names."""
    if not isinstance(answer, str):
        return ""
    text = answer.strip()
    if not text:
        return ""
    if text.lower().startswith(("http://", "https://")):
        slug = unquote(text.rstrip("/").split("/")[-1]).replace("_", " ").strip()
        text = " ".join(
            w if any(c.isupper() for c in w) else (w[:1].upper() + w[1:])
            for w in slug.split(" ")
            if w
        )
    return text


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _matches_true(y_star: str, true_answers: set[str]) -> bool:
    ys = _norm(y_star)
    if not ys:
        return False
    for raw in true_answers:
        t = _norm(raw)
        if not t:
            continue
        if ys == t or (len(t) >= 3 and t in ys) or (len(ys) >= 3 and ys in t):
            return True
    return False


def _word_hist(counts: list[int]) -> dict:
    if not counts:
        return {"n": 0}
    from collections import Counter as _C
    hist = _C(counts)
    return {
        "n": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": round(statistics.fmean(counts), 3),
        "median": statistics.median(counts),
        "histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _len_stats(texts: list[str]) -> dict:
    words = [len(t.split()) for t in texts]
    chars = [len(t) for t in texts]
    return {
        "n": len(texts),
        "words_mean": (statistics.fmean(words) if words else float("nan")),
        "words_median": (statistics.median(words) if words else float("nan")),
        "words_p10": _pctl(words, 10),
        "words_p90": _pctl(words, 90),
        "chars_mean": (statistics.fmean(chars) if chars else float("nan")),
        "chars_median": (statistics.median(chars) if chars else float("nan")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default="<server-workdir>/final500/datasets/validated_records.jsonl",
    )
    parser.add_argument(
        "--scp",
        default="<server-workdir>/final500/datasets/scp_records.jsonl",
    )
    parser.add_argument(
        "--payloads",
        default="<server-workdir>/final500/datasets/scp_payloads.jsonl",
    )
    parser.add_argument("--model", default="intfloat/e5-small-v2")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument(
        "--out",
        default="<server-home>/project-workdir/results/v3/scp_build.json",
    )
    args = parser.parse_args(argv)

    base = _read_jsonl(Path(args.records))
    scp = _read_jsonl(Path(args.scp))
    payloads = _read_jsonl(Path(args.payloads))

    # Canonical (human_comqa) text per intent, every legal paraphrase per intent,
    # and the decoded true-answer set per intent (for the y*-leak check).
    canonical_text: dict[str, str] = {}
    legal_by_intent: dict[str, list[str]] = {}
    legal_texts: list[str] = []
    true_by_intent: dict[str, set[str]] = {}
    for r in base:
        role = r.get("query_role")
        if role == "canonical" and r.get("generator") == "human_comqa":
            canonical_text.setdefault(r["intent_id"], r["text"])
            true_by_intent[r["intent_id"]] = {
                _decode_true_answer(a) for a in (r.get("canonical_answer") or []) if a
            }
        elif role == "legal":
            legal_by_intent.setdefault(r["intent_id"], []).append(r["text"])
            legal_texts.append(r["text"])

    # ---- embed everything once with the project's e5 CLS encoder --------------
    embedder = TransformerCLSEmbedder(args.model)
    scp_texts = [r["text"] for r in scp]
    canon_intents = list(canonical_text.keys())
    canon_vecs = embedder.encode([canonical_text[i] for i in canon_intents])
    canon_idx = {iid: k for k, iid in enumerate(canon_intents)}

    legal_flat_intents: list[str] = []
    legal_flat_texts: list[str] = []
    for iid, texts in legal_by_intent.items():
        for t in texts:
            legal_flat_intents.append(iid)
            legal_flat_texts.append(t)
    legal_vecs = (
        embedder.encode(legal_flat_texts)
        if legal_flat_texts
        else np.zeros((0, canon_vecs.shape[1]), dtype=np.float32)
    )
    legal_rows_by_intent: dict[str, list[int]] = {}
    for k, iid in enumerate(legal_flat_intents):
        legal_rows_by_intent.setdefault(iid, []).append(k)

    scp_vecs = embedder.encode(scp_texts) if scp_texts else np.zeros((0, 1))

    # ---- per-row cosines ------------------------------------------------------
    per_template: dict[str, dict] = {
        t: {
            "cos_canonical": [],
            "cos_legal": [],
            "texts": [],
            "intents": set(),
            "examples": [],
        }
        for t in TEMPLATES
    }
    missing_canonical = 0
    missing_legal = 0
    for k, r in enumerate(scp):
        gen = r.get("generator")
        if gen not in per_template:
            continue
        iid = r["intent_id"]
        vec = scp_vecs[k]
        bucket = per_template[gen]
        bucket["texts"].append(r["text"])
        bucket["intents"].add(iid)
        if iid in canon_idx:
            c = float(np.dot(vec, canon_vecs[canon_idx[iid]]))
            bucket["cos_canonical"].append(c)
        else:
            c = float("nan")
            missing_canonical += 1
        rows = legal_rows_by_intent.get(iid, [])
        if rows:
            sims = legal_vecs[rows] @ vec
            lc = float(np.max(sims))
            bucket["cos_legal"].append(lc)
        else:
            lc = float("nan")
            missing_legal += 1
        if len(bucket["examples"]) < args.n_examples:
            bucket["examples"].append(
                {
                    "intent_id": iid,
                    "record_id": r["record_id"],
                    "text": r["text"],
                    "payload_y_star": r.get("payload"),
                    "cos_canonical": round(c, 4) if c == c else None,
                    "cos_nearest_legal": round(lc, 4) if lc == lc else None,
                }
            )

    thr = args.threshold

    def frac_ge(values: list[float]) -> float:
        vals = [v for v in values if v == v]
        return (sum(1 for v in vals if v >= thr) / len(vals)) if vals else float("nan")

    def summ(values: list[float]) -> dict:
        vals = [v for v in values if v == v]
        return {
            "n": len(vals),
            "mean": (statistics.fmean(vals) if vals else float("nan")),
            "median": (statistics.median(vals) if vals else float("nan")),
            "p10": _pctl(vals, 10),
            "p90": _pctl(vals, 90),
            f"frac_ge_{thr:g}": frac_ge(vals),
        }

    template_report = {}
    for t in TEMPLATES:
        b = per_template[t]
        template_report[LETTER[t]] = {
            "generator": t,
            "n_rows": len(b["texts"]),
            "n_intents": len(b["intents"]),
            "cos_to_canonical": summ(b["cos_canonical"]),
            "cos_to_nearest_legal": summ(b["cos_legal"]),
            "length": _len_stats(b["texts"]),
            "examples": b["examples"],
        }

    # ---- y*-leak check: how many SCP rows carry y* == a decoded true answer ----
    # Acceptance number for the URL decode; target ~0.
    leak_rows = 0
    leak_by_template: dict[str, int] = {LETTER[t]: 0 for t in TEMPLATES}
    leak_examples: list[dict] = []
    y_star_words: list[int] = []
    for r in scp:
        ys = r.get("payload") or ""
        y_star_words.append(len(ys.split()))
        true = true_by_intent.get(r["intent_id"], set())
        if _matches_true(ys, true):
            leak_rows += 1
            tmpl = (r.get("metadata") or {}).get("scp_template", "?")
            if tmpl in leak_by_template:
                leak_by_template[tmpl] += 1
            if len(leak_examples) < 10:
                leak_examples.append(
                    {"intent_id": r["intent_id"], "y_star": ys,
                     "true_answers": sorted(true)[:4]}
                )

    # y* length dist from payloads too (per-intent, non-reject), and ctx word dist.
    payload_y_words = [
        len((r.get("y_star") or "").split()) for r in payloads if not r.get("reject")
    ]
    ctx_word_counts = [
        r.get("ctx_words", len((r.get("ctx") or "").split()))
        for r in payloads
        if not r.get("reject") and (r.get("ctx") or "").strip()
    ]

    # ---- payload rejection accounting ----------------------------------------
    reject_reasons: dict[str, int] = {}
    n_reject = 0
    n_judge_flag = 0
    n_regenerated = 0
    n_ctx_ok = 0
    for r in payloads:
        if r.get("reject"):
            n_reject += 1
            reject_reasons[r.get("reject_reason", "?")] = (
                reject_reasons.get(r.get("reject_reason", "?"), 0) + 1
            )
        if r.get("judge_flag"):
            n_judge_flag += 1
        if r.get("regenerated"):
            n_regenerated += 1
        if (r.get("ctx") or "").strip():
            n_ctx_ok += 1

    all_scp_intents = {r["intent_id"] for r in scp}
    report = {
        "model": args.model,
        "threshold": thr,
        "counts": {
            "scp_rows": len(scp),
            "per_template_rows": {
                LETTER[t]: len(per_template[t]["texts"]) for t in TEMPLATES
            },
            "intents_covered": len(all_scp_intents),
            "intents_per_template": {
                LETTER[t]: len(per_template[t]["intents"]) for t in TEMPLATES
            },
            "canonical_intents": len(canonical_text),
            "missing_canonical_anchor": missing_canonical,
            "missing_legal_anchor": missing_legal,
        },
        "y_star_true_answer_leak": {
            "rows_with_true_ystar": leak_rows,
            "per_template": leak_by_template,
            "examples": leak_examples,
        },
        "y_star_words_over_records": _word_hist(y_star_words),
        "y_star_words_over_payloads": _word_hist(payload_y_words),
        "ctx_words_over_payloads": _word_hist(ctx_word_counts),
        "payloads": {
            "rows": len(payloads),
            "hard_rejects": n_reject,
            "reject_reasons": reject_reasons,
            "judge_flagged_kept": n_judge_flag,
            "regenerated": n_regenerated,
            "ctx_present": n_ctx_ok,
        },
        "templates": template_report,
        "legal_paraphrase_length": _len_stats(legal_texts),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:3500], flush=True)
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
