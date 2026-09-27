"""E2 runner — build the SCP records file via ``generate.generate_scp``.

Thin CLI around ``sentry.research.pipeline.generate.generate_scp`` so the
builder stays a library function. Writes a NEW file
(``scp_records.jsonl``); ``validated_records.jsonl`` is never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _root in (_HERE.parents[3], _HERE.parents[0]):
    if (_root / "sentry").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from sentry.research.pipeline.generate import generate_scp  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default="<server-workdir>/final500/datasets/validated_records.jsonl",
    )
    parser.add_argument(
        "--payloads",
        default="<server-workdir>/final500/datasets/scp_payloads.jsonl",
    )
    parser.add_argument(
        "--out",
        default="<server-workdir>/final500/datasets/scp_records.jsonl",
    )
    parser.add_argument("--n-total", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)

    summary = generate_scp(
        args.records, args.payloads, args.out, n_total=args.n_total, seed=args.seed
    )
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
