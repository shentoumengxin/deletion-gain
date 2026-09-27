#!/usr/bin/env python
"""V-pass annotation for the final-500 run (T1.1, headline P/V split).

Every equivalence claim in the dataset must be validated by a V that is
independent of the paraphrase operator P (red line). This harness fills the
annotation template produced by ``validate --phase base --export-template
--roles legal_candidate,benign_query`` using **two independent judge models,
both different from P**, following `V_ANNOTATION_PROTOCOL.md`: for each
(canonical, candidate) pair the judge decides whether the candidate expresses
the same request/question as the canonical — same information need, same
expected answer. Disagreements go to an optional third adjudicator judge; with
no adjudicator configured the conservative rule applies (disagreement ->
``non_equivalent``, i.e. the pair is dropped rather than trusted).

The judgment here is *cache-answer-reuse* equivalence, not the source dataset's
own label: PAWS's "paraphrase" label and answer-reuse are related but not the
same relation, and taking the label at face value would import someone else's
definition of the thing this paper is about.

Usage (on a host that can reach both judge endpoints):

    python experiments/paper/rq1_detection/v_annotate.py \
      --workspace $ORTHO_WORKSPACE/final500 \
      --config experiments/paper/configs/final-500.json \
      --judge1 glm=https://open.bigmodel.cn/api/paas/v4,glm-5.2,$GLM_API_KEY \
      --judge2 qwen=http://127.0.0.1:8000/v1,qwen3-8b,EMPTY \
      --adjudicator qwen32=http://127.0.0.1:8001/v1,qwen3-32b,EMPTY

Keys are passed as CLI/env only and are never written to any output file.
Responses are cached under <workspace>/annotations/v_cache/ so reruns are
incremental.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

LABELS = {"equivalent", "non_equivalent", "uncertain"}
JUDGE_SYSTEM = """\
You are an equivalence annotator for a semantic-cache study.

You are given two questions, A and B. Decide whether B expresses the same \
request or question as A — same information need, same expected answer. Judge \
answer-reuse: would a correct answer to A also correctly answer B?

Rules:
- Judge the information need, not surface wording. Different phrasing with the \
same expected answer is equivalent; similar wording with a different expected \
answer is non_equivalent.
- If B is a fragment, truncated, or not a real question, label non_equivalent.
- If you genuinely cannot decide, label unsure.

Respond with ONLY a JSON object, no markdown fences:
{"label": "equivalent" | "non_equivalent" | "unsure", "reason": "<one sentence>"}"""


class Judge:
    """One OpenAI-compatible judge endpoint with an on-disk response cache."""

    def __init__(self, spec: str, cache_dir: Path, max_retries: int = 8):
        name, rest = spec.split("=", 1)
        base_url, model, api_key = rest.rsplit(",", 2)
        self.name = name.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        # "$VAR" resolves from the environment so keys stay out of ps output.
        if api_key.strip().startswith("$"):
            api_key = os.environ[api_key.strip()[1:]]
        self.api_key = api_key.strip()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"{self.name}.jsonl"
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_retries = max_retries
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                try:
                    row = json.loads(line)
                    self._cache[row["key"]] = row["label"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def label(self, text_a: str, text_b: str) -> str:
        key = hashlib.sha256(
            json.dumps([self.model, text_a, text_b], sort_keys=True).encode()
        ).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        label = self._call(text_a, text_b)
        with self._lock:
            self._cache[key] = label
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "label": label}) + "\n")
        return label

    def _call(self, text_a: str, text_b: str) -> str:
        user = json.dumps({"A": text_a, "B": text_b}, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0,
                    },
                    timeout=120,
                )
                if response.status_code == 429:
                    # Rate-limited: honor Retry-After, else exponential backoff.
                    retry_after = float(response.headers.get("Retry-After", 0))
                    time.sleep(max(retry_after, 10.0 * (attempt + 1)))
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._parse(content)
            except Exception as exc:  # network/parse errors: retry
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"judge {self.name} failed after retries: {last_error}")

    @staticmethod
    def _parse(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        payload = json.loads(text)
        label = str(payload.get("label", "")).strip().lower()
        if label == "unsure":
            label = "uncertain"
        if label not in LABELS:
            raise ValueError(f"judge returned invalid label: {label!r}")
        return label


def _read_template(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_template(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--template",
        default=None,
        help="annotation CSV to fill (default <workspace>/annotations/legal_candidates.csv)",
    )
    parser.add_argument("--judge1", required=True)
    parser.add_argument("--judge2", required=True)
    parser.add_argument("--adjudicator", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    template_path = Path(
        args.template or workspace / "annotations" / "legal_candidates.csv"
    )
    records_path = workspace / "datasets" / "records.jsonl"
    cache_dir = workspace / "annotations" / "v_cache"

    text_by_id: dict[str, str] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text_by_id[row["record_id"]] = row["text"]

    rows = _read_template(template_path)
    pending_rows = [
        row for row in rows if not row.get("annotator_1_label")
    ]
    if args.limit:
        pending_rows = pending_rows[: args.limit]
    judge1 = Judge(args.judge1, cache_dir)
    judge2 = Judge(args.judge2, cache_dir)
    adjudicator = Judge(args.adjudicator, cache_dir) if args.adjudicator else None

    def annotate(row: dict[str, str]) -> dict[str, str]:
        candidate = row["text"]
        canonical = text_by_id.get(row["parent_id"], "")
        if not canonical:
            raise RuntimeError(f"no canonical text for {row['record_id']}")
        try:
            label1 = judge1.label(canonical, candidate)
            label2 = judge2.label(canonical, candidate)
        except RuntimeError as exc:
            # A row that defeats every retry (content filter, bad request) is
            # not allowed to kill a 8,000-row run: it is conservatively dropped
            # and counted, never trusted.
            row["annotator_1_label"] = "uncertain"
            row["annotator_2_label"] = "uncertain"
            row["adjudicated_label"] = "non_equivalent"
            print(f"row dropped after judge retries: {row['record_id']}", flush=True)
            return row
        row["annotator_1_label"] = label1
        row["annotator_2_label"] = label2
        needs_adjudication = label1 != label2 or label1 == "uncertain"
        if needs_adjudication:
            if adjudicator is not None:
                row["adjudicated_label"] = adjudicator.label(canonical, candidate)
            else:
                # Conservative rule: a pair the two independent judges disagree
                # on — or both cannot decide — is dropped, never trusted.
                row["adjudicated_label"] = "non_equivalent"
        return row

    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for row in pool.map(annotate, pending_rows):
            done += 1
            if done % 100 == 0:
                _write_template(template_path, rows)
                print(f"progress {done}/{len(pending_rows)}", flush=True)
    _write_template(template_path, rows)

    agree = sum(
        1
        for row in pending_rows
        if row["annotator_1_label"] == row["annotator_2_label"]
    )
    summary = {
        "rows_this_run": len(pending_rows),
        "rows_total": len(rows),
        "judge1": judge1.name,
        "judge2": judge2.name,
        "adjudicator": adjudicator.name if adjudicator else "conservative_rule",
        "judge_agreement": agree / max(1, len(pending_rows)),
        "equivalent_final": sum(
            1
            for row in pending_rows
            if (row["adjudicated_label"] or row["annotator_1_label"]) == "equivalent"
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
