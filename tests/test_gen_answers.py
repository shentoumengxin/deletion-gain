import json, sys
from pathlib import Path

from experiments.paper.rq1_detection import gen_answers  # noqa: E402


class FakeClient:
    def __init__(self, *a, **k): self.calls = 0
    def chat(self, messages, **kw):
        self.calls += 1
        return "ANS:" + messages[0]["content"]


def test_dedups_resumes_and_keys_by_sha(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps({"prompt": p, "tag": i}) for i, p in
                                 enumerate(["a", "b", "a", "c"])) + "\n")
    out = tmp_path / "a.jsonl"
    fake = FakeClient()
    monkeypatch.setattr(gen_answers, "Client", lambda *a, **k: fake)
    monkeypatch.setattr(gen_answers, "load_env", lambda *a, **k: {"base_url": "x", "model": "m", "api_key": "k"})
    assert gen_answers.main(["--prompts", str(prompts), "--out", str(out), "--env-dir", str(tmp_path)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3 and fake.calls == 3
    assert {r["prompt_sha"] for r in rows} == {gen_answers.prompt_sha(p) for p in "abc"}
    prompts.write_text(prompts.read_text() + json.dumps({"prompt": "d"}) + "\n")
    assert gen_answers.main(["--prompts", str(prompts), "--out", str(out), "--env-dir", str(tmp_path)]) == 0
    assert fake.calls == 4


class FlakyClient:
    """Returns an empty answer for the prompts in ``empty_for`` -- the thinking-enabled
    failure mode the victim shows when ``--extra-body`` is wrong."""

    def __init__(self, empty_for=()):
        self.empty_for, self.calls, self.asked = set(empty_for), 0, []

    def chat(self, messages, **kw):
        self.calls += 1
        content = messages[0]["content"]
        self.asked.append(content)
        return "" if content in self.empty_for else "ANS:" + content


def test_empty_answers_are_retried_and_last_row_wins(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps({"prompt": p}) for p in "abc") + "\n")
    out = tmp_path / "a.jsonl"
    monkeypatch.setattr(gen_answers, "load_env", lambda *a, **k: {"base_url": "x", "model": "m", "api_key": "k"})

    broken = FlakyClient(empty_for={"b"})
    monkeypatch.setattr(gen_answers, "Client", lambda *a, **k: broken)
    argv = ["--prompts", str(prompts), "--out", str(out), "--env-dir", str(tmp_path)]
    assert gen_answers.main(argv) == 2
    assert broken.calls == 3 and len(out.read_text().splitlines()) == 3
    # the empty row is not an answer: the helper hands the scorer only a and c
    assert set(gen_answers.load_answers(out)) == {gen_answers.prompt_sha(p) for p in "ac"}

    fixed = FlakyClient()
    monkeypatch.setattr(gen_answers, "Client", lambda *a, **k: fixed)
    assert gen_answers.main(argv) == 0
    assert fixed.asked == ["b"]  # exactly the prompt whose answer came back empty

    answers = gen_answers.load_answers(out)
    assert len(out.read_text().splitlines()) == 4  # append-only: the empty row is still there
    assert set(answers) == {gen_answers.prompt_sha(p) for p in "abc"}
    assert answers[gen_answers.prompt_sha("b")]["response"] == "ANS:b"
