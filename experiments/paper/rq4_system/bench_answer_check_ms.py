#!/usr/bin/env python
"""What the answer check costs at serving, and what it costs to store.

Extends ``experiments/paper/results/v3/detection/excess_serving_ms.json``. That file
timed the DG serving path only -- the entry's stored span vectors dotted with the query
embedding the cache already computed for retrieval, then a max -- and its methodology is
kept here verbatim so the two columns are comparable: single-thread numpy, dim 384,
float16 storage, 2000 repeats, median and p95. The script that produced it lived on
cpu-server (``bench_excess_ms.py``) and is not in this checkout; ``rq3_granularity.py
--mode bench`` reimplements the same ``time_hit`` and this file reimplements it again,
identically, so the "before" column reproduces rather than being quoted.

**What is added to the timed path.** Under an answer rule the veto is a conjunction, and
the second half of it reads numbers computed at insertion:

    sims = spans @ q ; best = argmax(sims)          # DG, as before (argmax, not max)
    answer_loss[best] > eta_a                       # one float16 read and a compare
    len(echo_tokens[best] - content_tokens(query))  # tokenise the query, subtract, count

The query tokenisation is the only part that is not O(1), and it is the reason this
measurement exists: it is a regex pass over the arriving query string, once per hit, and
it happens on the serving path rather than at insertion. It is timed inside the hit and
also on its own, so the attribution is visible rather than inferred.

**Storage.** ``before`` charges the whole vector plus the span block at float16 -- the
convention ``excess_serving_ms.json`` and the v3 cost table use, where the deletion
block is diagnostic and reported separately. ``after`` adds what the answer check keeps:
one float16 scalar per span variant (``answer_loss``) and the per-variant echo token
sets, which ``entry_store`` writes as JSON lists of sorted tokens. The token sets are
the only part that is not a formula, so they are **measured** on real entries when
``--eval``/``--answers`` are given -- which costs no model call at all, since an echo set
is ``content_tokens(complement) & content_tokens(answer)``.

Usage (single thread, as the input JSON was measured)::

    OMP_NUM_THREADS=1 python3 experiments/paper/rq4_system/bench_answer_check_ms.py \\
      --in experiments/paper/results/v3/detection/excess_serving_ms.json \\
      --eval <server-workdir>/out/v3/lmp_eval.jsonl --set-name LMP \\
      --answers <server-workdir>/answer_check_20260909/answers \\
      --policy 'multi[count:4+width:2:cap16]/runs' \\
      --out experiments/paper/results/answer_check_20260909/cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from sentry.cache.defense.calibrate import parse_policy  # noqa: E402
from sentry.cache.defense.spans import shortened  # noqa: E402
from sentry.cache.defense.textnorm import content_tokens, normalise_answer  # noqa: E402

DIM = 384
#: float16, the precision an entry's vectors are stored at (deletion.STORAGE_DTYPES).
BYTES_PER_SCALAR = 2
#: A ComQA-shaped arriving query. The tokenisation cost is a function of this string's
#: length, so it is reported alongside every number it produced rather than assumed.
DEFAULT_QUERY = "when did mariah carey and nick cannon get married in the caribbean"


def time_hit(n_spans: int, repeats: int, query: str, echo_size: int,
             eta_a: float, echo_min: int, answer_checked: bool) -> list[float]:
    """One hit, in ms. ``answer_checked=False`` is the published methodology verbatim.

    Setup is outside the loop because insertion pays for it: the span block, the
    per-variant ``answer_loss`` scalars and the per-variant echo sets are all built when
    the entry is written. What the loop measures is what arrives with the query.
    """
    rng = np.random.default_rng(0)
    block = rng.standard_normal((n_spans, DIM)).astype(np.float16)
    block /= np.linalg.norm(block.astype(np.float32), axis=1,
                            keepdims=True).astype(np.float16)
    anchor = rng.standard_normal(DIM).astype(np.float32)
    anchor /= np.linalg.norm(anchor)
    losses = rng.standard_normal(n_spans).astype(np.float16) * np.float16(0.02)
    echo_sets = [frozenset(f"tok{rng.integers(0, 5000)}" for _ in range(echo_size))
                 for _ in range(n_spans)]

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sims = block.astype(np.float32) @ anchor
        if not answer_checked:
            _ = float(sims.max())
        else:
            best = int(np.argmax(sims))
            _ = float(sims[best])
            by_loss = float(losses[best]) > eta_a
            echo = len(echo_sets[best] - content_tokens(query))
            _ = by_loss or echo >= echo_min
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def time_tokenise(repeats: int, query: str) -> list[float]:
    """The query tokenisation on its own, so the added cost can be attributed."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = content_tokens(query)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def summarise(samples: list[float]) -> dict:
    return {"ms_median": round(statistics.median(samples), 5),
            "ms_p95": round(sorted(samples)[int(0.95 * len(samples))], 5)}


def bytes_for(n_spans: int, echo_json_bytes: int) -> dict:
    """Stored bytes per entry, before and after the answer fields.

    ``before`` is the whole vector plus the span block; the one-segment deletion block is
    diagnostic and charged separately by whoever reports it (rq3_granularity's
    convention, kept). ``after`` adds one float16 per span variant and the echo sets as
    ``entry_store`` writes them.
    """
    before = (1 + n_spans) * DIM * BYTES_PER_SCALAR
    loss_bytes = n_spans * BYTES_PER_SCALAR
    return {"n_spans": n_spans,
            "before_bytes": before,
            "answer_loss_bytes": loss_bytes,
            "echo_tokens_bytes": echo_json_bytes,
            "after_bytes": before + loss_bytes + echo_json_bytes,
            "overhead_fraction": round((loss_bytes + echo_json_bytes) / before, 5)}


def measure_entries(eval_paths, answer_paths, policy, limit: int) -> dict:
    """Per-entry span counts and echo-set sizes on real entries. No model call.

    An echo set is set arithmetic over the complement's content words and the answer's,
    so this is exactly what insertion would store, computed without embedding anything.
    """
    # The join rule lives in one place -- see v3_detect.load_answers for why it is
    # imported rather than restated.
    from sentry.research.pipeline.instruction_benign import (
        digest, load_answer_files)

    answers, files = load_answer_files(answer_paths)
    texts: list[str] = []
    for path in eval_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text") or record.get("prompt")
            if text:
                texts.append(text)
    seen, rows = set(), []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        answer = answers.get(digest(text))
        if answer is None:
            continue
        cleaned = normalise_answer(answer)
        if not cleaned:
            continue
        spans = shortened(policy, text)
        answer_words = content_tokens(cleaned)
        echo = [sorted(content_tokens(gone) & answer_words)
                for gone in spans.removed_texts]
        rows.append({
            "n_spans": len(spans.span_names),
            "n_deletions": len(spans.deletion_names),
            "echo_json_bytes": len(json.dumps(echo).encode("utf-8")),
            "echo_tokens_total": sum(len(s) for s in echo),
        })
        if limit and len(rows) >= limit:
            break

    if not rows:
        raise SystemExit("--eval/--answers matched no entry with a usable answer")

    def stat(field):
        values = sorted(r[field] for r in rows)
        return {"median": values[len(values) // 2],
                "mean": round(sum(values) / len(values), 2),
                "p95": values[int(0.95 * (len(values) - 1))],
                "max": values[-1]}

    per_entry = {field: stat(field) for field in
                 ("n_spans", "n_deletions", "echo_json_bytes", "echo_tokens_total")}
    sizes = {}
    for label in ("median", "mean", "p95"):
        n = max(1, int(round(per_entry["n_spans"][label])))
        sizes[label] = bytes_for(n, int(round(per_entry["echo_json_bytes"][label])))
    return {"n_entries": len(rows), "answer_files": files, "per_entry": per_entry,
            "bytes_per_entry": sizes,
            "note": "echo sets measured on real entries and their real answers; no "
                    "embedder call is involved in computing them"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="source", default=str(
        _REPO / "experiments/paper/results/v3/detection/excess_serving_ms.json"),
        help="the published serving-ms JSON this extends; its n_spans cells are the "
             "ones re-timed with the answer check")
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=2000)
    p.add_argument("--query", default=DEFAULT_QUERY,
                   help="the arriving query whose tokenisation is inside the timed path")
    p.add_argument("--echo-set-size", type=int, default=3,
                   help="tokens in the winning variant's echo set (the set the query's "
                        "own tokens are subtracted from)")
    p.add_argument("--eta-a", type=float, default=0.02)
    p.add_argument("--echo-min", type=int, default=1)
    p.add_argument("--n-spans", type=int, action="append", default=None,
                   help="time these span counts instead of the ones in --in")
    p.add_argument("--eval", action="append", default=None,
                   help="repeatable: an eval corpus to measure real echo-set bytes on")
    p.add_argument("--answers", action="append", default=None,
                   help="repeatable: the answers for those entries (file or directory)")
    p.add_argument("--policy", default="multi[count:4+width:2:cap16]/runs")
    p.add_argument("--set-name", default="", help="label for the --eval measurement")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    print(f"EFFECTIVE ARGS: in={args.source} out={args.out} repeats={args.repeats} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    sets = source.get("sets", {})
    if args.n_spans:
        wanted = sorted(set(args.n_spans))
    else:
        wanted = sorted({cell["n_spans"] for group in sets.values()
                         for cell in group.values()})

    timings, cache = {}, {}
    for n in wanted:
        dg = summarise(time_hit(n, args.repeats, args.query, args.echo_set_size,
                                args.eta_a, args.echo_min, answer_checked=False))
        checked = summarise(time_hit(n, args.repeats, args.query, args.echo_set_size,
                                     args.eta_a, args.echo_min, answer_checked=True))
        cache[n] = {"n_spans": n, "dg_only": dg, "answer_checked": checked,
                    "delta_ms_median": round(checked["ms_median"] - dg["ms_median"], 5)}
        timings[str(n)] = cache[n]
        print(f"  n_spans={n:>4}  dg {dg['ms_median']:.5f} ms  "
              f"answer-checked {checked['ms_median']:.5f} ms  "
              f"(+{cache[n]['delta_ms_median']:.5f})", flush=True)

    tokenise = summarise(time_tokenise(args.repeats, args.query))
    print(f"  query tokenisation alone: {tokenise['ms_median']:.5f} ms "
          f"({len(args.query.split())} words)", flush=True)

    # The published cells, with the answer-checked column beside them. The original
    # `ms_median`/`ms_p95` are copied through untouched -- and they were measured on
    # another host, so they must NOT be subtracted from `answer_checked`. The
    # host-consistent comparison is `dg_only_remeasured` vs `answer_checked`, which is
    # what `delta_ms_median` is.
    extended = {}
    for name, group in sets.items():
        extended[name] = {}
        for label, cell in group.items():
            n = cell["n_spans"]
            extended[name][label] = dict(cell)
            if n in cache:
                extended[name][label]["dg_only_remeasured"] = cache[n]["dg_only"]
                extended[name][label]["answer_checked"] = cache[n]["answer_checked"]
                extended[name][label]["delta_ms_median"] = cache[n]["delta_ms_median"]

    report = {
        "extends": str(args.source),
        "dim": DIM, "repeats": args.repeats, "numpy": np.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "storage_dtype": "float16",
        "query": args.query, "query_words": len(args.query.split()),
        "echo_set_size": args.echo_set_size, "echo_min": args.echo_min,
        "eta_a": args.eta_a,
        "note": "serving path only. Before: stored span vectors dotted with the query "
                "embedding the cache already computed for retrieval, then a max. After: "
                "the same dots with an argmax, plus the winning variant's answer_loss "
                "compare, the query's tokenisation and the echo set difference. No model "
                "call on either path.",
        "command": " ".join(sys.argv),
        "serving_ms": timings,
        "query_tokenise_ms": tokenise,
        "sets": extended,
        "sets_note": "ms_median/ms_p95 are the published values copied from --in and "
                     "were measured on another host; compare answer_checked against "
                     "dg_only_remeasured, which this run measured beside it.",
        "bytes_per_entry_formula": {
            "before": "(1 + n_spans) * 384 * 2  # whole + span block, float16",
            "after": "before + n_spans * 2 + echo_tokens_json_bytes",
            "deletion_block_not_charged": "n_deletions * 384 * 2, diagnostic, stored but "
                                          "never read at serving",
            "examples_vectors_only": {str(n): bytes_for(n, 0) for n in wanted},
        },
        "insertion_embedder_calls": {
            "before": {"calls": 1, "texts": "n_variants (spans + deletions), one batch"},
            "after": {"calls": 2, "texts": "n_variants in the same batch, plus the "
                                           "answer in a call of its own"},
            "why_two": "the variant batch must stay byte-identical for the DG numbers to "
                       "be unchanged; a real embedder pads a batch to its longest item "
                       "and an answer is far longer than any variant "
                       "(sentry/cache/defense/deletion.py:build_profile)",
        },
    }
    if args.eval or args.answers:
        if not (args.eval and args.answers):
            raise SystemExit("--eval and --answers go together")
        report["measured"] = measure_entries(args.eval, args.answers,
                                             parse_policy(args.policy), args.limit)
        report["measured"]["set_name"] = args.set_name
        report["measured"]["policy"] = parse_policy(args.policy).fingerprint()
        m = report["measured"]["bytes_per_entry"]["median"]
        print(f"  bytes/entry (median entry): {m['before_bytes']} -> {m['after_bytes']} "
              f"(+{m['overhead_fraction'] * 100:.2f}%)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
