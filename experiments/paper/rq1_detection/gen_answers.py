#!/usr/bin/env python
"""Victim answers for an arbitrary prompt list, keyed by the prompt's sha256.

``asr_generate.py`` builds its rows from the attack corpora; this file takes rows that
someone else built (wrapped benign entries, composed attacks, non-echo payloads) and
does only the generation, with the same victim discipline: the prompt is the entry text
and nothing else, thinking is disabled through ``--extra-body``, answers are appended
as they arrive, identical prompts are generated once, and an empty answer is an error
rather than a datum.

Input rows are JSONL and must carry a ``prompt`` field; every other field is copied
through to the output row untouched. Output rows are
``{**row, "prompt_sha", "response", "victim_model", "temperature", "seed"}``.
Re-running against an existing ``--out`` skips the prompt hashes already answered
there, so a killed run resumes where it stopped.

**An empty answer is not an answer.** It exits 2 *and* leaves the prompt unanswered, so
the rerun after fixing ``--extra-body`` regenerates exactly those prompts. ``--out`` is
append-only, so a regenerated row follows its empty predecessor and a prompt hash can
appear more than once: **consumers must take the last row per** ``prompt_sha``. Use
``load_answers(path)``, which does that (last non-empty response per hash wins, and a
hash that never got a non-empty response is absent), rather than reading the file
directly.

Usage (GPU box, local vLLM serving Qwen3-8B; ``--env-dir`` holds a ``.env`` with
``url`` / ``openai_key`` / ``model``)::

    python gen_answers.py --prompts prompts.jsonl --out answers.jsonl --env-dir .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from sentry.research.operators import Client, load_env  # noqa: E402


def prompt_sha(text: str) -> str:
    """Stable identity of a prompt: the sha256 of its UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_answers(path) -> dict[str, dict]:
    """The answers in an ``--out`` file, one per prompt hash.

    The file is append-only and a prompt whose answer came back empty is regenerated on
    the next run, so a hash can carry several rows. The last row with a non-empty
    ``response`` wins; a hash whose rows are all empty is **absent** from the result,
    because it has no answer yet. This is the only place answers should be read from.
    """
    path = Path(path)
    if not path.exists():
        return {}
    answers: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:  # tolerate a partial line left by a killed process
            row = json.loads(line)
            sha = row["prompt_sha"]
        except (json.JSONDecodeError, KeyError):
            continue
        if (row.get("response") or "").strip():
            answers[sha] = row
    return answers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="JSONL, one row per prompt (field: prompt)")
    parser.add_argument("--out", required=True, help="JSONL to append answers to (resumable)")
    parser.add_argument("--env-dir", required=True, help="directory holding the victim .env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--cache-name", default="answer_check_victim")
    parser.add_argument("--extra-body", default='{"chat_template_kwargs": {"enable_thinking": false}}',
                        help="JSON merged into the request body; the default disables thinking")
    parser.add_argument("--limit", type=int, default=0, help="cap the to-do list (0 = no cap)")
    args = parser.parse_args(argv)

    rows = _read_jsonl(Path(args.prompts))
    for index, row in enumerate(rows):
        if "prompt" not in row:
            print(f"ERROR: row {index} of {args.prompts} has no 'prompt' field", flush=True)
            return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Only a non-empty response counts as done; a hash whose row came back empty is
    # re-queued, so the rerun after fixing --extra-body regenerates exactly those.
    done = set(load_answers(out_path))

    todo: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        sha = prompt_sha(row["prompt"])
        if sha in done or sha in seen:
            continue
        seen.add(sha)
        todo.append({**row, "prompt_sha": sha})
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} rows, {len(done)} already answered, {len(todo)} to generate", flush=True)
    if not todo:
        return 0

    extra_body = json.loads(args.extra_body) if args.extra_body.strip() else None
    creds = load_env(Path(args.env_dir))
    client = Client(creds, cache_name=args.cache_name)
    print(f"victim endpoint: {creds['base_url']}  model: {creds['model']}", flush=True)

    empties: list[str] = []
    failures: list[tuple[str, str]] = []

    def run(item: tuple[int, dict]) -> dict | None:
        index, row = item
        try:
            answer = client.chat(
                [{"role": "user", "content": row["prompt"]}],
                temperature=args.temperature, seed=args.seed + index,
                max_tokens=args.max_tokens, extra_body=extra_body,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((row["prompt_sha"], f"{type(exc).__name__}: {str(exc)[:160]}"))
            return None
        if not (answer or "").strip():
            empties.append(row["prompt_sha"])
        return {**row, "response": answer, "victim_model": creds["model"],
                "temperature": args.temperature, "seed": args.seed + index}

    written = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for result in pool.map(run, enumerate(todo)):
            if result is None:
                continue
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                handle.flush()
                print(f"  {written}/{len(todo)}", flush=True)

    print(json.dumps({"written": written, "failed": len(failures), "empty": len(empties)}), flush=True)
    if failures:
        print("first failures:", failures[:5], flush=True)
    if empties:
        print(f"ERROR: {len(empties)} empty responses; check --extra-body", flush=True)
        return 2
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
