"""Build the fence shipped as ``sentry/cache/fences/e5-small-v2.json``.

    python experiments/build_default_fence.py [--model-path /local/e5-small-v2]

It fits ``calibrate_from_hits`` on the benign hits bundled in ``data/`` (ComQA and
Natural Questions), after checking that this encoder reproduces the stored Deletion
Gain readings. It then reports what the fence blocks on the three attack classes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sentry.cache.defense.calibrate import AnswerColumns, calibrate_from_hits
from sentry.cache.defense.deletion import build_profile, excess
from sentry.cache.defense.spans import deployed_policy
from sentry.cache.quickstart import DEFAULT_FENCE, default_embedder

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "data/runs/supp_appendix_20260923/inputs"
SETS = {"lmp": "CAP", "scp": "SCP", "kca": "KCA"}


def load_rows():
    rows = {}
    for name in SETS:
        path = ROWS / f"rows_{name}.jsonl"
        rows[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_FENCE)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "experiments/paper/results/default_fence/report.json")
    args = parser.parse_args(argv)

    embedder = default_embedder(args.model_path)
    policy = deployed_policy()
    rows = load_rows()

    # Benign hits: ComQA controls (shared by CAP and SCP) and Natural Questions (KCA).
    hits, seen = [], set()
    for name in ("lmp", "kca"):
        for row in rows[name]:
            if row["arm"] == "genuine" and (row["key"], row["query"]) not in seen:
                seen.add((row["key"], row["query"]))
                hits.append({"query": row["query"], "key": row["key"], "answer": row["answer"]})

    # The encoder must reproduce the stored readings before anything is fitted.
    columns = AnswerColumns()
    drift = defaultdict(float)
    readings = {}
    for name, set_rows in rows.items():
        queries = embedder.encode([row["query"] for row in set_rows])
        for row, anchor in zip(set_rows, queries):
            profile = build_profile(row["key"], embedder, policy, answer=row["answer"])
            if not profile.judgeable:
                continue
            reading = excess(profile, anchor)
            readings[(name, row["record_id"])] = reading
            drift["base_cos"] = max(drift["base_cos"], abs(reading.base_cos - row["base_cos"]))
            drift["excess_span"] = max(drift["excess_span"],
                                       abs(reading.excess_span - row["excess_span"]))
            loss, echo = columns(row["query"], reading)
            if row.get("adl_best") is not None and loss is not None:
                drift["adl"] = max(drift["adl"], abs(loss - row["adl_best"]))
            if row.get("echo_best") is not None and echo is not None:
                drift["echo"] = max(drift["echo"], abs(echo - row["echo_best"]))
    print("max drift from stored readings:", dict(drift))
    if drift["excess_span"] > 1e-3 or drift["base_cos"] > 1e-3:
        raise SystemExit("encoder does not reproduce the stored readings; check the model")

    fence, report = calibrate_from_hits(hits, embedder, metadata={
        "source": "benign ComQA and Natural Questions hits bundled in data/",
        "note": "starter fence; recalibrate on your own benign traffic"})

    attacks = {}
    for name, label in SETS.items():
        blocked, total = [], 0
        for row in rows[name]:
            reading = readings.get((name, row["record_id"]))
            if row["arm"] != "attack" or reading is None or reading.base_cos < 0.90:
                continue
            total += 1
            blocked.append(fence.blocks(reading.base_cos, reading.words, reading.excess_span,
                                        *columns(row["query"], reading)))
        attacks[label] = {"retrieved": total,
                          "block_rate": float(np.mean(blocked)) if blocked else None}
    report["attack_block_rate_at_retrieval"] = attacks
    report["max_drift_from_stored_readings"] = dict(drift)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fence.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
