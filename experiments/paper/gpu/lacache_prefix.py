#!/usr/bin/env python
"""LaCache's k-token response prefixes, decoded once on the GPU box.

LaCache (arXiv 2608.01718) refuses a cache hit when the stored answer is not the answer
the arriving query would have got. It decides that by decoding the first **k tokens** of
an answer to the arriving query and comparing them against the first k tokens of the
**cached** answer. Our reimplementation used k = 8 *words* and an API model, which is the
mechanism at a coarser grain and on a different victim than our own ASR numbers. This file
produces the faithful object instead: k = 20 tokens of the same victim (Qwen3-8B on vLLM,
greedy, thinking off), counted with the victim's own tokenizer.

Two prefixes per row, because LaCache compares a pair:

* the **entry** prefix — the answer to the text sitting in the cache. For an attack row
  that is the attack text, whose answer is the poison; for a benign row it is the
  canonical question.
* the **query** prefix — the answer to the arriving query, which is the row's ``anchor``:
  the legal paraphrase an ordinary user shows up with.

Rows are built by ``asr_generate.build_rows`` rather than re-derived here, so the
``record_id``s join to the answers file and the two files cannot drift apart.

Scoring is not done here — it needs the e5 and BGE encoders, which live on cpu-server.
``rq2_lacache.py --unit tokens --k 20 --prefixes <this file>`` reads it.

Cost note. With ``--answers`` pointed at the E5 answers file, the entry prefix is taken
from the stored answer instead of re-decoded. At temperature 0 against the same server
the first 20 tokens of a 200-token answer *are* the 20-token generation, so this is the
same object, one generation cheaper per attack row, and it is the *stored* answer by
construction — which is what LaCache actually compares. Without it every text is decoded
here.

Resumable twice over: the output JSONL is appended and existing ``record_id``s are
skipped, and ``Client`` caches every completion on disk under ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Two layouts have to work: the repo (this file under experiments/paper/gpu, its
# siblings under .../analysis, the package at the repo root) and the flat working
# directories on cpu-server and the GPU box, where every script sits next to ``sentry/``.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parents[2], _HERE.parent / "analysis", _HERE):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from experiments.paper.rq2_robustness import asr_generate as ag  # noqa: E402

from sentry.research.operators import Client, load_env  # noqa: E402


def prefix_ids(tokenizer, text: str, k: int) -> list[int]:
    """The first k token ids of ``text`` under the victim's own tokenizer.

    ``add_special_tokens=False``: LaCache compares generated content, and a leading
    BOS/chat marker would be k-1 tokens of agreement handed to the baseline for free.
    """
    return list(tokenizer.encode(text or "", add_special_tokens=False))[:k]


def _sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    ag.add_row_arguments(parser)
    parser.add_argument("--out", required=True, help="JSONL, appended and resumable")
    parser.add_argument("--tokenizer", required=True,
                        help="local Qwen3-8B weights dir (tokenizer.json / vocab.json / "
                             "merges.txt live there). Nothing is downloaded.")
    parser.add_argument("--k", type=int, default=20, help="tokens compared (LaCache: 20)")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="decode budget per prefix; equals --k for the faithful run")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--answers", default="",
                        help="E5 answers JSONL. Given, the entry prefix is cut from the "
                             "stored answer instead of re-decoded — the same object at "
                             "temperature 0, and the answer LaCache actually compares.")
    parser.add_argument("--cache-name", default="lacache_prefix")
    parser.add_argument("--cache-dir", default="",
                        help="on-disk completion cache; defaults to ORBIT_CACHE_DIR")
    parser.add_argument("--env-dir", default="",
                        help="directory holding the victim's .env (url/openai_key/model)")
    parser.add_argument(
        "--extra-body",
        default='{"chat_template_kwargs": {"enable_thinking": false}}',
        help="JSON passed through to the endpoint. vLLM/Qwen3 wants "
             '\'{"chat_template_kwargs": {"enable_thinking": false}}\'; DeepSeek wants '
             '\'{"thinking": {"type": "disabled"}}\'. Left on, a 20-token budget buys 20 '
             "tokens of reasoning and no answer at all.")
    args = parser.parse_args(argv)

    rows, set_reports = ag.build_rows(args)
    for report in set_reports:
        print("set list: " + json.dumps(report), flush=True)
    if args.limit_per_set:
        rows = ag.limit_per_set(rows, args.limit_per_set)
    if args.limit:
        rows = rows[:args.limit]

    # An entry with no arriving query has no pair to compare, and LaCache is undefined on
    # it. Counted, not dropped in silence.
    no_anchor = [r for r in rows if not (r.get("anchor") or "").strip()]
    rows = [r for r in rows if (r.get("anchor") or "").strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["record_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in rows if r["record_id"] not in done]

    def entry_text(row: dict) -> str:
        return row["prompt"] if row["set"] != "benign" else (row["canonical"] or row["prompt"])

    # Stored answers, when the E5 pass already produced them.
    stored: dict[str, str] = {}
    if args.answers:
        for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("response"):
                stored[record["record_id"]] = record["response"]
        print(f"stored answers available for {len(stored)} record_ids", flush=True)

    # One generation per distinct text. Anchors collapse hard — 2,400 attack rows share
    # at most 500 arriving queries — so this is roughly half the naive call count.
    wanted: set[str] = set()
    for row in todo:
        wanted.add(row["anchor"])
        if not (args.answers and row["record_id"] in stored):
            wanted.add(entry_text(row))
    wanted.discard("")
    order = sorted(wanted)

    print(json.dumps({"rows_total": len(rows), "dropped_no_anchor": len(no_anchor),
                      "already_done": len(done), "to_emit": len(todo),
                      "distinct_texts_to_decode": len(order), "k": args.k}, indent=1),
          flush=True)
    if not todo:
        return 0

    extra_body = json.loads(args.extra_body) if args.extra_body.strip() else None
    creds = load_env(Path(args.env_dir)) if args.env_dir else load_env()
    client = (Client(creds, cache_name=args.cache_name, cache_dir=Path(args.cache_dir))
              if args.cache_dir else Client(creds, cache_name=args.cache_name))
    print(f"victim endpoint: {creds['base_url']}  model: {creds['model']}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(f"tokenizer: {type(tokenizer).__name__} from {args.tokenizer}", flush=True)

    answers: dict[str, str] = {}
    failures: list[tuple[str, str]] = []

    def decode(item: tuple[int, str]) -> tuple[str, str | None]:
        index, text = item
        try:
            return text, client.chat(
                [{"role": "user", "content": text}],
                temperature=args.temperature, seed=args.seed + index,
                max_tokens=args.max_tokens, extra_body=extra_body)
        except Exception as exc:  # noqa: BLE001
            failures.append((_sha(text), f"{type(exc).__name__}: {str(exc)[:160]}"))
            return text, None

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for done_count, (text, answer) in enumerate(pool.map(decode, enumerate(order)), 1):
            if answer is not None:
                answers[text] = answer
            if done_count % 200 == 0:
                print(f"  decoded {done_count}/{len(order)}", flush=True)

    handle = out_path.open("a", encoding="utf-8")
    empties: list[str] = []
    written = 0
    for row in todo:
        entry_raw = stored.get(row["record_id"]) if args.answers else None
        if entry_raw is None:
            entry_raw = answers.get(entry_text(row))
        query_raw = answers.get(row["anchor"])
        if entry_raw is None or query_raw is None:
            continue                                    # its generation failed; retry later
        entry_i = prefix_ids(tokenizer, entry_raw, args.k)
        query_i = prefix_ids(tokenizer, query_raw, args.k)
        if not entry_i or not query_i:
            empties.append(row["record_id"])
        handle.write(json.dumps({
            "record_id": row["record_id"], "set": row["set"], "family": row["family"],
            "intent_id": row.get("intent_id", ""), "k": args.k,
            "literal": row.get("literal", ""),
            "entry_text": entry_text(row), "query_text": row["anchor"],
            "entry_source": "stored" if (args.answers and row["record_id"] in stored)
                            else "decoded",
            "entry_response": entry_raw, "query_response": query_raw,
            "entry_prefix": tokenizer.decode(entry_i),
            "entry_prefix_ids": entry_i,
            "query_prefix": tokenizer.decode(query_i),
            "query_prefix_ids": query_i,
            "victim_model": creds["model"], "tokenizer": str(args.tokenizer),
        }, ensure_ascii=False) + "\n")
        written += 1
    handle.close()

    print(json.dumps({"written": written, "failed_generations": len(failures),
                      "empty_prefixes": len(empties)}, indent=1), flush=True)
    if failures:
        print("first failures:", failures[:5], flush=True)
    if empties:
        # An empty prefix compares equal to every other empty prefix, which reads as
        # LaCache agreeing and serving the poisoned hit. Loud, and a nonzero exit.
        print(f"ERROR: {len(empties)} rows have an empty k-token prefix. Thinking mode is "
              f"probably still on; check --extra-body. Examples: {empties[:5]}", flush=True)
        return 2
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
