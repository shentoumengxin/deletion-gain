"""Prepare and score SC-IPI ablation inputs with matched answers and legal queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def prepare(root, work):
    from experiments.paper.rq1_detection.v3_detect import load_rows
    from sentry.research.pipeline.instruction_benign import load_answer_files

    work.mkdir(parents=True, exist_ok=True)
    corpus = root / "datasets/final500/eval/kca_eval.jsonl"
    cell = root / "runs/answer_check_20260909/cells/e5-small-v2__kca.json"
    report = json.loads(cell.read_text())
    answer_files = []
    for f in report["answers"]["files"]:
        path = root / "responses/answer_check_20260909/answers" / Path(f["path"]).name
        assert sha(path) == f["sha256"]
        answer_files.append(path)
    answers, _ = load_answer_files(answer_files)
    response_file = root / "runs/answer_check_20260909/asr/asr_flat.jsonl"
    judged = {r["record_id"].replace("kca-f2-", "gcg-f2-"): r
              for r in read_rows(response_file) if r["set"] == "kca"}
    assert len(judged) == 800
    rows, composition = load_rows(corpus, ["gcg"], ["cacheattack_cleaned_qa"])
    by_id = {r["record_id"]: r for r in read_rows(corpus)}
    exact_answers, flags, identities = {}, [], []
    for r in rows:
        digest = hashlib.sha256(r["text"].encode()).hexdigest()
        if r["arm"] == "attack":
            j = judged[r["record_id"]]
            assert j["prompt"] == r["text"] and j["payload"] == by_id[r["record_id"]]["payload"]
            assert not j.get("judge_error") and type(j["judge_success"]) is bool
            answer = j["response"]
            flags.append({"record_id": r["record_id"], "poisoned": j["judge_success"],
                          "poisoned_strict": None})
        else:
            answer = answers[digest]
        assert answer.strip()
        exact_answers[r["text"]] = {"prompt": r["text"], "prompt_sha": digest, "response": answer}
        identities.append({"record_id": r["record_id"], "intent_id": r["intent_id"],
                           "entry_sha256": digest,
                           "query_sha256": hashlib.sha256(r["anchor"].encode()).hexdigest(),
                           "answer_sha256": hashlib.sha256(answer.encode()).hexdigest()})
    assert len(flags) == 798 and len(rows) == 1297
    (work / "eval.jsonl").write_bytes(corpus.read_bytes())
    write_rows(work / "answers.jsonl", exact_answers.values())
    write_rows(work / "flags.jsonl", flags)
    write_rows(work / "identities.jsonl", identities)
    manifest = {"n_benign": 499, "n_attack": 798, "composition": composition,
                "incoming_query": "same independently validated NQ legal paraphrase as main table",
                "attack_answer": "frozen Qwen response that received the saved injection verdict",
                "benign_answer": "same frozen answer used for main-table benign calibration",
                "victim_calls": 0, "judge_calls": 0,
                "sources": {str(p.relative_to(root)): sha(p) for p in [corpus, cell, response_file]},
                "inputs": {p.name: sha(p) for p in work.glob("*.jsonl")}}
    (work / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ["sources", "inputs"]}))


def score(work, allow_cpu):
    from sentry.research.pipeline.cli import _require_server
    _require_server("SC-IPI embedding replay", False, allow_cpu)
    from experiments.paper.rq1_detection.v3_detect import main

    manifest = json.loads((work / "inputs.json").read_text())
    assert all(sha(work / name) == digest for name, digest in manifest["inputs"].items())
    main(["--eval", str(work / "eval.jsonl"), "--encoder", "intfloat/e5-small-v2",
          "--pooling", "cls", "--text-prefix", "", "--attack-role", "gcg",
          "--benign-generator", "cacheattack_cleaned_qa", "--flags", str(work / "flags.jsonl"),
          "--policy", "multi[count:4+width:2:cap16]/runs", "--storage-dtype", "float16",
          "--answers", str(work / "answers.jsonl"), "--echo-min", "1", "--seed", "0",
          "--out", str(work / "cell.json"), "--dump-rows", str(work / "scores.jsonl")])
    receipt = {"inputs_sha256": sha(work / "inputs.json"),
               "cell_sha256": sha(work / "cell.json"), "scores_sha256": sha(work / "scores.jsonl"),
               "cpu_override": allow_cpu,
               "authorization": "User explicitly selected cpu-server CPU replay in this thread."}
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score"])
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--allow-local-heavy", action="store_true",
                        help="Explicit server CPU override; use only when authorized.")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.data_root:
            parser.error("prepare requires --data-root")
        prepare(args.data_root, args.work_root)
    else:
        score(args.work_root, args.allow_local_heavy)
