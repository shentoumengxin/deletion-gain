"""The paper's interval script, end to end on a synthetic cell.

``paper/data/perrow/ci_answer_check.py`` turns the per-row dumps of
``v3_detect.py --dump-rows`` into intent-grouped bootstrap intervals for the
answer-checked rule. Two things there can be quietly wrong and no eyeball would catch it:
the vectorised replicate could drift from the shipped rule code, and the point estimate
could stop reproducing the published cell. Both are assertions inside the script, so this
module's job is to make them run -- on a cell small enough to build with the collision
geometry, with a ``main_table.json`` written from the very report the dump came from.

Skipped when the paper repo is not checked out beside this one; it is a separate
repository and this suite must not depend on it being present.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.paper.rq1_detection import v3_detect  # noqa: E402

from sentry.cache.defense.calibrate import parse_policy  # noqa: E402
from tests.test_v3_detect import POLICY, _cell  # noqa: E402
from tests.collision_embedder import (  # noqa: E402
    BENIGN_QUERY,
    PAYLOAD_WORDS,
    TOPIC_WORDS,
    CollisionEmbedder,
    concentrated_entry,
    spread_entry,
)

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/paper/rq1_detection/ci_answer_check.py"
ENCODER = "e5-small-v2"


def _mixed_corpus(tmp_path, intents: int = 60):
    """A cell whose two arms both vary, so a resampled replicate is not a constant.

    ``tests/test_v3_detect._corpus`` plants the *same* entry under every intent, which is
    right for pinning the answer join and useless here: an attack arm with one distinct
    row has a degenerate bootstrap. Payload length is swept instead, so some planted rows
    clear the fence and some do not, and half the planted answers repeat the payload so
    the echo term varies too.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    records, answers, flags = [], {}, []
    topic = ("the capital city of france is its seat of government in the modern day "
             "french republic")
    for index in range(intents):
        genuine = spread_entry(int(rng.integers(18, 34)), int(rng.integers(1, 8)), rng)
        planted = concentrated_entry(int(rng.integers(18, 34)),
                                     1 + index % len(PAYLOAD_WORDS), rng)
        records += [
            {"record_id": f"c{index}", "intent_id": f"i{index}",
             "query_role": "canonical", "generator": "human_comqa", "text": genuine},
            {"record_id": f"l{index}", "intent_id": f"i{index}",
             "query_role": "legal", "text": BENIGN_QUERY},
            {"record_id": f"n{index}", "intent_id": f"i{index}", "query_role": "ndss",
             "generator": "ndss_matched_blend", "text": planted},
        ]
        answers[genuine] = topic + " " + " ".join(rng.permutation(list(TOPIC_WORDS))[:4])
        answers[planted] = (" ".join(PAYLOAD_WORDS) if index % 2 else topic)
        flags.append({"record_id": f"n{index}", "poisoned": index % 4 != 0,
                      "poisoned_strict": index % 8 == 0})

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    flags_path = tmp_path / "flags.jsonl"
    flags_path.write_text("\n".join(json.dumps(f) for f in flags), encoding="utf-8")
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("\n".join(json.dumps(
        {"prompt": text, "prompt_sha": hashlib.sha256(text.encode()).hexdigest(),
         "response": response}) for text, response in answers.items()), encoding="utf-8")
    return eval_path, flags_path, answers_path


def _cell_and_dump(tmp_path):
    eval_path, flags_path, answers_path = _mixed_corpus(tmp_path)
    answers, _ = v3_detect.load_answers([str(answers_path)])
    report, _, dumped = _cell(eval_path, flags_path, CollisionEmbedder(),
                              answers=answers)
    rows = tmp_path / "rows"
    v3_detect.write_dump_rows(rows / f"{ENCODER}__lmp.jsonl", dumped)
    main_table = tmp_path / "main_table.json"
    main_table.write_text(json.dumps({"cells": {f"{ENCODER}|lmp": report}}),
                          encoding="utf-8")
    return rows, main_table, report


@pytest.mark.skipif(not SCRIPT.exists(), reason=f"statistics script not present at {SCRIPT}")
def test_the_interval_script_reproduces_the_published_cell_and_its_own_fast_path(tmp_path):
    """Runs with every assertion armed: parity at 5%, and fast-vs-shipped agreement.

    The script refuses to write a result if either fails, so a clean exit and a file with
    a parity list in it *is* the evidence.
    """
    rows, main_table, report = _cell_and_dump(tmp_path)
    out = tmp_path / "ci.json"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--rows", str(rows),
         "--main-table", str(main_table), "--out", str(out)],
        capture_output=True, text=True,
        env={**os.environ, "CD_REPO": str(Path(__file__).resolve().parents[1])})
    assert proc.returncode == 0, proc.stderr

    result = json.load(open(out))
    cell = result["cells"][f"CAP|{ENCODER}"]
    assert result["parity_checked_at_0.05"] == [f"CAP|{ENCODER}|dg_only",
                                                f"CAP|{ENCODER}|either"]
    assert sorted(cell["budgets"]) == ["0.01", "0.02", "0.05", "0.1"]
    either = cell["budgets"]["0.05"]["either"]
    published = report["answer_rules"]["either"]
    assert either["All"][0] == published["excess_block_rate_joint_eta"]
    assert either["Succ."][0] == published["tpr_poisoned_joint_eta"]
    assert either["eta_joint"] == published["eta_joint"]
    # an interval, not a point: the bootstrap actually varied something
    assert either["All"][1] <= either["All"][0] <= either["All"][2]
    assert either["All"][1] < either["All"][2]
    # the baseline every rate is reported beside
    assert "cosine_only" in cell["budgets"]["0.05"]
    assert (cell["budgets"]["0.05"]["cosine_only"]["cosine_threshold"]
            == report["cosine_threshold"])


@pytest.mark.skipif(not SCRIPT.exists(), reason=f"statistics script not present at {SCRIPT}")
def test_a_point_estimate_that_moved_is_refused(tmp_path):
    """Parity is an assertion, not a printed comparison.

    If the rule, the cut or the dump changed under the script's feet, the 5% cell stops
    matching what the paper published and the run must fail rather than write intervals
    around a number nobody can trace.
    """
    rows, main_table, report = _cell_and_dump(tmp_path)
    tampered = json.load(open(main_table))
    column = tampered["cells"][f"{ENCODER}|lmp"]["answer_rules"]["either"]
    column["excess_block_rate_joint_eta"] += 0.01
    main_table.write_text(json.dumps(tampered), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--rows", str(rows),
         "--main-table", str(main_table), "--out", str(tmp_path / "ci.json")],
        capture_output=True, text=True,
        env={**os.environ, "CD_REPO": str(Path(__file__).resolve().parents[1])})
    assert proc.returncode != 0
    assert "publishes" in proc.stderr
