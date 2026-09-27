"""Protect labels, score direction, and the statistical unit of the stress test."""
import importlib

import numpy as np
import pytest


def experiment():
    name = "sentry.research.pipeline.instruction_benign"
    assert importlib.util.find_spec(name) is not None, "instruction experiment is missing"
    return importlib.import_module(name)


def source():
    return {"record_id": "k1", "intent_id": "i1", "corpus": "comqa",
            "text": "What is the capital of France?",
            "anchor": "Which city is France's capital?", "anchor_id": "q1"}


def test_wrappers_preserve_both_cores_and_keep_label_ambiguity_explicit():
    m = experiment()
    rows = m.construct([source()], seed=17)
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert {r["intent_id"] for r in rows} == {"i1"}
    for r in rows:
        assert source()["text"] in r["text"]
        if r["condition"] == "exact_core":
            assert r["anchor"] == source()["text"]
        else:
            assert source()["anchor"] in r["anchor"]
        assert r["malicious"] is False
        if r["kind"] == "constraint" and r["condition"] in ("entry_only", "query_only", "exact_core"):
            assert r["reuse_label"] == "answer_dependent"
        elif r["kind"] == "constraint":
            assert r["reuse_label"] == "compatible_by_construction"
    assert {r["condition"] for r in rows} == {
        "bare", "entry_only", "query_only", "both_same", "both_paraphrase", "exact_core"}


def test_splits_do_not_depend_on_template_or_input_order():
    m = experiment()
    sources = [{**source(), "intent_id": f"i{i}", "record_id": f"k{i}"} for i in range(30)]
    a, b = m.construct(sources, seed=3), m.construct(list(reversed(sources)), seed=3)
    assert {r["sample_id"]: r["split"] for r in a} == {r["sample_id"]: r["split"] for r in b}
    for intent in {r["intent_id"] for r in a}:
        assert len({r["split"] for r in a if r["intent_id"] == intent}) == 1
    assert {r["split"] for r in a} == {"calibration", "test"}


def test_scoring_uses_entry_variants_and_matches_serving_float16():
    m = experiment()
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.deletion import build_profile, excess
    from sentry.embeddings import HashEmbedder
    policy = parse_policy("multi[count:4+width:2:cap16]/runs")
    embedder = HashEmbedder(64)
    row = {**m.construct([source()], seed=1)[1]}
    pool = m.text_pool([row], [policy])
    vectors = dict(zip(pool, embedder.encode(pool)))
    result = m.score_row(row, policy, vectors, "float16")
    expected = excess(build_profile(row["text"], embedder, policy, storage_dtype="float16"),
                      embedder.encode([row["anchor"]])[0])
    assert result["dg"] == pytest.approx(expected.excess_span, abs=1e-12)
    assert result["base_cos"] == pytest.approx(expected.base_cos, abs=1e-12)
    assert result["best_name"] == expected.best_name
    assert result["best_text"] in " ".join(row["text"].split())


def test_short_entries_are_counted_as_unjudgeable_not_zero_gain():
    m = experiment()
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.embeddings import HashEmbedder
    row = {**source(), "text": "France?", "condition": "bare"}
    policy = parse_policy("count:4/runs")
    pool = m.text_pool([row], [policy])
    result = m.score_row(row, policy, dict(zip(pool, HashEmbedder(32).encode(pool))), "float16")
    assert result["judgeable"] is False
    assert result["dg"] is None


def test_threshold_uses_only_bare_calibration_hits():
    m = experiment()
    rows = [dict(split="calibration", condition="bare", dg=x, base_cos=.95, judgeable=True)
            for x in (0., .01, .02)]
    rows += [dict(split="test", condition="bare", dg=100., base_cos=.99, judgeable=True),
             dict(split="calibration", condition="entry_only", dg=50., base_cos=.99, judgeable=True),
             dict(split="calibration", condition="bare", dg=30., base_cos=.1, judgeable=True)]
    thresholds = m.fit_thresholds(rows, budget=.5, retrieval_floor=.9)
    assert thresholds == {"dg": .01, "neg_cos": -.95, "n": 3}


def test_grouped_interval_does_not_treat_templates_as_independent_intents():
    m = experiment()
    groups, values = ["a", "b", "c", "d"], [0, 1, 0, 1]
    a = m.cluster_rate(values, groups, iterations=500, seed=7)
    b = m.cluster_rate(np.repeat(values, 20), np.repeat(groups, 20), iterations=500, seed=7)
    assert a["rate"] == b["rate"] == .5
    assert a["ci95"] == b["ci95"]
    assert b["n_intents"] == 4


def test_retrieval_misses_not_counted_as_successful_dg_passes():
    m = experiment()
    rows = [dict(intent_id="a", base_cos=.95, dg=.2, judgeable=True),
            dict(intent_id="b", base_cos=.8, dg=0., judgeable=True),
            dict(intent_id="c", base_cos=.95, dg=None, judgeable=False)]
    out = m.operating_rates(rows, {"dg": .1, "neg_cos": -.92}, .9, 200, 0)
    assert out["n_hits"] == 2
    assert out["dg_veto_given_hit"]["rate"] == 1.
    assert out["cosine_veto_given_hit"]["rate"] == 0.
    assert out["n_unjudgeable_hits"] == 1


def test_matched_support_requires_distinct_intents_not_template_copies():
    name = "sentry.research.pipeline.instruction_benign_analysis"
    assert importlib.util.find_spec(name) is not None, "analysis module is missing"
    m = importlib.import_module(name)
    a = [dict(intent_id="a", base_cos=.953, dg=.03, words=12) for _ in range(50)]
    b = [dict(intent_id="b", base_cos=.952, dg=.01, words=12) for _ in range(50)]
    result = m.matched_discrimination(a, b, "base_cos", .01, 2)
    assert result["dg_auroc"] is None
    assert result["attack_rows_supported"] == 0


def test_auc_sign_and_ties():
    name = "sentry.research.pipeline.instruction_benign_analysis"
    assert importlib.util.find_spec(name) is not None, "analysis module is missing"
    m = importlib.import_module(name)
    assert m.auc([1., 1.], [1., 1.]) == .5
    assert m.auc([2., 3.], [0., 1.]) == 1.
    assert m.auc([0., 1.], [2., 3.]) == 0.


# --- the answer side ------------------------------------------------------------
# The stress test now attaches the victim's own cached answer to every entry it scores.
# The literal below sits at both ends of the entry, so *every* shortened version drops a
# copy of it and the echo assertion does not depend on which variant happens to win under
# the hash embedder.

LITERAL = "1971-04-19"
BARE_ENTRY = f"{LITERAL} what is the capital of france {LITERAL}"
BARE_ANCHOR = "which city is the capital of france"
LITERAL_ANSWER = f"Paris. The record is dated {LITERAL}."
WRAPPER_ANSWER = "Sure, I will tell you briefly: Paris."


def answer_sources():
    return [{"record_id": "k1", "intent_id": "i1", "corpus": "comqa", "anchor_id": "q1",
             "text": BARE_ENTRY, "anchor": BARE_ANCHOR},
            {"record_id": "k2", "intent_id": "i2", "corpus": "nq", "anchor_id": "q2",
             "text": "who wrote the play hamlet in english",
             "anchor": "which author wrote the play hamlet"},
            # Two words is below the segment floor: answered, but nothing to read.
            {"record_id": "k3", "intent_id": "i3", "corpus": "nq", "anchor_id": "q3",
             "text": "who won?", "anchor": "which side won?"}]


def _smoke_workspace(tmp_path):
    import json
    m = experiment()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rows = m.construct(answer_sources(), seed=20260909)
    m.write_jsonl(workspace / "samples.jsonl", rows)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(
        {"name": "answers-smoke", "seed": 20260909, "n_intents": 4,
         "embedding_model": "deterministic-feature-hash-smoke", "cache_threshold": 0.1,
         "bootstrap_iterations": 64, "min_bin_intents": 2,
         "policies": ["multi[count:4+width:2:cap16]/runs"]}))
    return workspace, config, rows


def _answers_dir(tmp_path, rows):
    """One victim answer per unique i1 entry; i2 gets an empty one, which is no answer."""
    import hashlib
    import json
    directory = tmp_path / "answers"
    directory.mkdir()
    written = []
    for text in sorted({r["text"] for r in rows if r["intent_id"] in ("i1", "i3")}):
        written.append({"prompt": text, "prompt_sha": hashlib.sha256(text.encode()).hexdigest(),
                        "response": LITERAL_ANSWER if text == BARE_ENTRY else WRAPPER_ANSWER,
                        "victim_model": "fake"})
    blank = sorted({r["text"] for r in rows if r["intent_id"] == "i2"})[0]
    written.append({"prompt": blank, "prompt_sha": hashlib.sha256(blank.encode()).hexdigest(),
                    "response": "   ", "victim_model": "fake"})
    (directory / "answers_fake.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in written))
    return directory


def gen_answers():
    """The generator that writes the answers files. It owns the key and the last-row rule.

    Not importable as a package — it lives under ``experiments/`` — so the path insert is
    the import. ``instruction_benign`` deliberately does not do this at runtime (it would
    drag ``requests`` into a scoring run), which is exactly why the agreement is pinned
    here instead.
    """
    import importlib
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1] / "experiments" / "paper" / "analysis")
    sys.path.insert(0, root)
    try:
        return importlib.import_module("experiments.paper.rq1_detection.gen_answers")
    finally:
        sys.path.remove(root)


def test_answer_join_key_is_the_prompt_sha_gen_answers_writes():
    import hashlib
    m = experiment()
    assert m.digest(BARE_ENTRY) == gen_answers().prompt_sha(BARE_ENTRY)
    assert m.digest(BARE_ENTRY) == hashlib.sha256(BARE_ENTRY.encode("utf-8")).hexdigest()


def test_answer_loading_agrees_with_gen_answers(tmp_path):
    """A repeated hash takes its last non-empty row; an all-empty hash has no answer."""
    import json
    m = experiment()
    generator = gen_answers()
    rows = [{"prompt": "a?", "response": "first"},      # superseded two rows below
            {"prompt": "b?", "response": ""},           # all-empty hash: never answered
            {"prompt": "a?", "response": "second"},
            {"prompt": "b?", "response": "   "},
            {"prompt": "c?", "response": "only"}]
    path = tmp_path / "answers_fixture.jsonl"
    path.write_text("".join(
        json.dumps({**row, "prompt_sha": generator.prompt_sha(row["prompt"])}) + "\n"
        for row in rows))

    theirs = generator.load_answers(path)
    ours, files = m.load_answer_files([str(path)])
    assert ours == {sha: record["response"] for sha, record in theirs.items()}
    assert ours[generator.prompt_sha("a?")] == "second"
    assert generator.prompt_sha("b?") not in ours
    assert files == [{"path": str(path), "n_rows": 5, "n_non_empty": 3,
                      "sha256": files[0]["sha256"]}]
    assert m.load_answer_files([str(tmp_path)])[0] == ours  # a directory reads the same


def test_answer_truncations_cut_at_twenty_words_and_the_first_sentence():
    m = experiment()
    variants = m.answer_variants("First one. " + " ".join(f"w{i}" for i in range(40)))
    assert variants["first_sentence"] == "First one."
    assert len(variants["first20"].split()) == 20
    assert variants["full"].startswith("First one. w0")
    assert m.answer_variants("   ") == {}


def test_answer_fields_match_the_profile_the_defense_stores():
    m = experiment()
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.deletion import build_profile, excess
    from sentry.cache.defense.spans import shortened
    from sentry.cache.defense.textnorm import content_tokens
    from sentry.embeddings import HashEmbedder
    policy = parse_policy("multi[count:4+width:2:cap16]/runs")
    embedder = HashEmbedder(64)
    row = {"text": BARE_ENTRY, "anchor": BARE_ANCHOR, "intent_id": "i1", "condition": "bare"}
    variants = shortened(policy, row["text"])
    pool = m.text_pool([row], [policy],
                       extra=list(m.answer_variants(LITERAL_ANSWER).values())
                       + list(variants.removed_texts))
    vectors = dict(zip(pool, embedder.encode(pool)))
    scored = m.score_row(row, policy, vectors, "float16", answer=LITERAL_ANSWER)
    reading = excess(build_profile(row["text"], embedder, policy, storage_dtype="float16",
                                   answer=LITERAL_ANSWER),
                     embedder.encode([row["anchor"]])[0])
    assert scored["best_index"] == reading.best_index
    assert scored["adl_best"] == pytest.approx(reading.answer_loss, abs=1e-9)
    assert scored["echo_best"] == len(reading.echo_tokens - content_tokens(row["anchor"]))
    assert scored["echo_tokens_best"] == sorted(
        reading.echo_tokens - content_tokens(row["anchor"]))
    plain = m.score_row(row, policy, vectors, "float16")
    assert (plain["dg"], plain["best_name"], plain["best_text"], plain["base_cos"]) == (
        scored["dg"], scored["best_name"], scored["best_text"], scored["base_cos"])
    assert plain["has_answer"] is False and plain["adl_best"] is None


def _ablation_fixture(answer):
    """Score ``BARE_ENTRY`` against its anchor with ``answer`` attached, hash embedder."""
    m = experiment()
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.spans import shortened
    from sentry.embeddings import HashEmbedder
    policy = parse_policy("multi[count:4+width:2:cap16]/runs")
    row = {"text": BARE_ENTRY, "anchor": BARE_ANCHOR}
    variants = shortened(policy, row["text"])
    pool = m.text_pool([row], [policy],
                       extra=list(m.answer_variants(answer).values())
                       + list(variants.removed_texts) + [row["text"]])
    vectors = dict(zip(pool, HashEmbedder(64).encode(pool)))
    return m.score_row(row, policy, vectors, "float16", answer=answer), variants, vectors


def test_ablation_columns_read_the_vector_pair_they_name():
    """Each column against its own hand-written dot products: a swap cannot hide."""
    m = experiment()
    from sentry.cache.defense.deletion import store_rows, unit
    scored, variants, vectors = _ablation_fixture(LITERAL_ANSWER)
    best = scored["best_index"]
    whole = unit(vectors[BARE_ENTRY])
    kept = unit(vectors[variants.span_texts[best]])
    dropped = unit(vectors[variants.removed_texts[best]])
    answers = m.answer_variants(LITERAL_ANSWER)

    def against(vector, key):
        return float(vector @ unit(vectors[answers[key]]))

    def rounded(value):
        return float(store_rows(value, "float16"))

    assert scored["adl_best"] == rounded(against(whole, "full") - against(kept, "full"))
    assert scored["adl_best_first20"] == rounded(
        against(whole, "first20") - against(kept, "first20"))
    assert scored["adl_best_first_sentence"] == rounded(
        against(whole, "first_sentence") - against(kept, "first_sentence"))
    assert scored["adl_delta_best"] == rounded(
        against(dropped, "full") - against(kept, "full"))
    # Two sentences, so the truncation really is reading a shorter answer than `full`.
    assert answers["first_sentence"] != answers["full"]
    assert scored["adl_best_first_sentence"] != scored["adl_best"]


def test_ablation_signs_follow_the_removed_content():
    """Deletion Gain does not depend on the answer, so the winner can be read first and
    the answer then set to a text whose sign is known by construction."""
    m = experiment()
    _, variants, _ = _ablation_fixture(LITERAL_ANSWER)
    best = _ablation_fixture(LITERAL_ANSWER)[0]["best_index"]
    kept_text = variants.span_texts[best]
    gone_text = variants.removed_texts[best]

    # Answer is the whole entry: nothing of it survives the cut, so the loss is positive.
    whole_as_answer = _ablation_fixture(BARE_ENTRY)[0]
    assert whole_as_answer["best_index"] == best
    assert whole_as_answer["adl_best"] > 0

    # Answer is exactly what the winning variant kept: the cut loses nothing, so negative.
    kept_as_answer = _ablation_fixture(kept_text)[0]
    assert kept_as_answer["best_index"] == best
    assert kept_as_answer["adl_best"] < 0
    assert kept_as_answer["adl_delta_best"] < 0

    # Answer is exactly what the cut removed: the removed text is the closer of the two.
    gone_as_answer = _ablation_fixture(gone_text)[0]
    assert gone_as_answer["best_index"] == best
    assert gone_as_answer["adl_delta_best"] > 0


def test_score_attaches_answers_without_moving_deletion_gain(tmp_path):
    import json
    m = experiment()
    workspace, config, rows = _smoke_workspace(tmp_path)
    argv = ["score", "--config", str(config), "--workspace", str(workspace), "--smoke"]

    assert m.main(argv) == 0
    before = {(r["sample_id"], r["policy"]): r
              for r in m.read_jsonl(workspace / "scores_smoke.jsonl")}
    assert before
    # Without --answers the columns are still written, so the analysis can read them
    # unconditionally and count what had no answer.
    assert all(r["has_answer"] is False and all(r[f] is None for f in m.ANSWER_FIELDS)
               for r in before.values())

    sets = tmp_path / "sets"
    sets.mkdir()
    m.write_jsonl(sets / "unseen_wrappers.jsonl", [
        {"prompt": BARE_ENTRY, "anchor": BARE_ANCHOR, "intent_id": "i1", "corpus": "comqa",
         "set": "unseen_wrappers", "template": "u0", "position": "prefix",
         "condition": "entry_only", "split": "test"}])
    assert m.main(argv + ["--answers", str(_answers_dir(tmp_path, rows)),
                          "--extra-sets", str(sets),
                          "--extra-sets", str(tmp_path / "absent")]) == 0
    after = m.read_jsonl(workspace / "scores_smoke.jsonl")

    keyed = {(r["sample_id"], r["policy"]): r for r in after if "sample_id" in r}
    assert set(keyed) == set(before)
    for key, row in keyed.items():
        for field in ("dg", "base_cos", "retrieval_cos", "best_name", "best_text",
                      "judgeable", "words", "n_variants"):
            assert row[field] == before[key][field], f"{field} moved for {key}"

    bare = [r for r in after if r.get("condition") == "bare" and r["intent_id"] == "i1"]
    assert bare
    for row in bare:
        assert row["has_answer"] is True
        assert row["echo_best"] >= 1 and LITERAL in row["echo_tokens_best"]
        assert len(row["echo_tokens_best"]) == row["echo_best"]
        assert row["echo_tokens_best"] == sorted(row["echo_tokens_best"])
        assert row["answer_words"] == len(LITERAL_ANSWER.split())
        assert row["answer_sha"] == m.digest(LITERAL_ANSWER)
        for field in ("adl_best", "adl_best_first20", "adl_best_first_sentence",
                      "adl_delta_best"):
            assert isinstance(row[field], float)

    absent = [r for r in after if r["intent_id"] == "i2"]
    assert absent
    assert all(r["has_answer"] is False and r["adl_best"] is None and
               r["echo_best"] is None for r in absent)

    # Answered, but too short to cut: no winning variant, so no readable answer fields.
    short = [r for r in after if r["intent_id"] == "i3" and r.get("condition") == "bare"]
    assert short
    assert all(r["judgeable"] is False and r["has_answer"] is False for r in short)

    tagged = [r for r in after if r.get("set") == "unseen_wrappers"]
    assert len(tagged) == 1 and tagged[0]["has_answer"] is True
    assert tagged[0]["dg"] is not None
    assert all(r.get("set") for r in after)

    provenance = json.loads((workspace / "embedding_provenance.json").read_text())
    assert provenance["model"] == "deterministic-feature-hash-smoke"
    counts = provenance["answers"]
    assert counts["answer_texts_embedded"] and counts["complement_texts_embedded"]
    assert counts["readings_answered_but_unjudgeable"] >= 1
    assert counts["entry_texts_without_answer"] >= 1
    assert counts["rows_with_answer"] >= 1 and counts["rows_without_answer"] >= 1
    assert counts["entry_texts_with_answer"] + counts["entry_texts_without_answer"] == (
        counts["entry_texts"])


# --- the answer rules: per-rule tables ------------------------------------------
# Synthetic scored rows, one policy, two corpora's worth of structure in one corpus.
# Every rule is exercised by a condition group built so that only that rule moves:
# `polite/entry_only` is a wrapped benign entry DG fires on and the answer clears
# (`either` must rescue it), `constraint/entry_only` loses answer similarity to the cut
# (`adl` fires, `echo` does not), `polite/query_only` echoes one content word (`echo`
# fires, `adl` does not). Attacks do both. Two rows are the fail-closed cases.

RULE_POLICY = "multi[count:4+width:2:cap16]/runs"


def analysis():
    name = "sentry.research.pipeline.instruction_benign_analysis"
    assert importlib.util.find_spec(name) is not None, "analysis module is missing"
    return importlib.import_module(name)


def _scored(**kw):
    row = {"corpus": "comqa", "policy": RULE_POLICY, "split": "test", "condition": "bare",
           "kind": "bare", "template": "bare", "position": "none", "intent_id": "i00",
           "record_id": "r00", "sample_id": "s00", "malicious": False,
           "set": "wrapped_benign", "text": "core question here", "anchor": "anchor here",
           "base_text": "core question here", "base_anchor": "anchor here",
           "words": 3, "n_variants": 8, "judgeable": True,
           "dg": 0.0, "base_cos": .95, "retrieval_cos": .95,
           "best_index": 0, "best_name": "pre1", "best_text": "core question here",
           "has_answer": True, "answer_sha": "a" * 8, "answer_words": 12,
           "adl_best": .0, "echo_best": 0, "echo_tokens_best": [],
           "adl_best_first20": .0, "adl_best_first_sentence": .0, "adl_delta_best": .0}
    return {**row, **kw}


def _blank_answer():
    return {"has_answer": False, "answer_sha": None, "answer_words": None,
            "adl_best": None, "echo_best": None, "echo_tokens_best": None,
            "adl_best_first20": None, "adl_best_first_sentence": None,
            "adl_delta_best": None}


def _rule_rows():
    rows = []
    for k in range(40):
        split = "calibration" if k < 20 else "test"
        common = {"intent_id": f"i{k:02d}", "record_id": f"r{k:02d}", "split": split}
        rows.append(_scored(**common, sample_id=f"bare-{k}",
                            dg=k/1000 if split == "calibration" else .0,
                            adl_best=k/1000 if split == "calibration" else .0))
        if split == "calibration":
            continue
        rows.append(_scored(**common, sample_id=f"pe-core-{k}", kind="polite",
                            condition="entry_only", template="polite_tell",
                            position="prefix", dg=.5, adl_best=.005))
        rows.append(_scored(**common, sample_id=f"pe-frag-{k}", kind="polite",
                            condition="entry_only", template="polite_tell",
                            position="suffix", dg=.5, adl_best=.005,
                            best_text="core question"))
        # A retrieval miss: never a defense success, never in a rate's denominator.
        rows.append(_scored(**common, sample_id=f"pe-miss-{k}", kind="polite",
                            condition="entry_only", template="polite_help",
                            position="prefix", dg=.9, adl_best=.9, echo_best=5,
                            base_cos=.5, retrieval_cos=.5))
        rows.append(_scored(**common, sample_id=f"ce-{k}", kind="constraint",
                            condition="entry_only", template="one_sentence",
                            position="prefix", dg=.5, adl_best=.9))
        rows.append(_scored(**common, sample_id=f"pq-{k}", kind="polite",
                            condition="query_only", template="polite_tell",
                            position="prefix", dg=.5, echo_best=1,
                            echo_tokens_best=["paris"]))
        rows.append(_scored(**common, sample_id=f"bs-{k}", kind="polite",
                            condition="both_same", template="polite_tell",
                            position="prefix", dg=None, judgeable=False,
                            best_index=None, best_name=None, best_text=None,
                            **_blank_answer()))
        rows.append(_scored(**common, sample_id=f"bp-{k}", kind="polite",
                            condition="both_paraphrase", template="polite_tell",
                            position="prefix", dg=.5, **_blank_answer()))
        rows.append(_scored(**common, sample_id=f"atk-{k}", kind="attack",
                            condition="attack", template="CAP", position="native",
                            malicious=True, attack_class="CAP", set="instruction_benign",
                            base_text=None, base_anchor=None, best_text="core question",
                            dg=.6, base_cos=.91, retrieval_cos=.91,
                            adl_best=.9, echo_best=1, echo_tokens_best=["1971-04-19"]))
    return rows


def _rule_config(**kw):
    m = experiment()
    return m.InstructionExperimentConfig(
        name="answer-rules", seed=11, n_intents=40,
        embedding_model="deterministic-feature-hash-smoke", cache_threshold=.9,
        bootstrap_iterations=64, min_bin_intents=2, policies=[RULE_POLICY], **kw)


def _run_rules(tmp_path, **kw):
    import json
    m, a = experiment(), analysis()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rows = _rule_rows()
    m.write_jsonl(workspace / "scores.jsonl", rows)
    a.analyze(workspace, _rule_config(), **kw)
    summary = json.loads((workspace / "summary.json").read_text())
    return workspace, rows, summary["answer_rules"][f"comqa|{RULE_POLICY}"]


def test_rule_names_match_the_deployed_answer_rules():
    from sentry.cache.defense.deletion import ANSWER_RULES
    a = analysis()
    assert set(a.RULES) - {"dg_only", "cosine_only"} == set(ANSWER_RULES) - {"none"}


def test_veto_is_the_briefed_rule_and_fails_closed():
    a = analysis()
    clean = {"dg": .0, "judgeable": True, "has_answer": True, "adl_best": .0, "echo_best": 0}
    fired = {**clean, "dg": .5}
    for rule in ("dg_only", "adl", "echo", "either"):
        assert a.veto(clean, .1, .1, rule) is False
        assert a.veto({"dg": None, "judgeable": False, "has_answer": True,
                       "adl_best": .0, "echo_best": 0}, .1, .1, rule) is True
        assert a.veto({**fired, "has_answer": False}, .1, .1, rule) is True
    assert a.veto(fired, .1, .1, "dg_only") is True
    assert a.veto(fired, .1, .1, "adl") is False
    assert a.veto({**fired, "adl_best": .2}, .1, .1, "adl") is True
    assert a.veto({**fired, "echo_best": 1}, .1, .1, "echo") is True
    assert a.veto({**fired, "echo_best": 1}, .1, .1, "echo", 2) is False
    assert a.veto({**fired, "echo_best": 2}, .1, .1, "echo", 2) is True
    assert a.veto({**fired, "echo_best": 1}, .1, .1, "either") is True
    assert a.veto({**fired, "adl_best": .2}, .1, .1, "either") is True
    # An ablation column the row does not carry is a missing reading, so it vetoes.
    assert a.veto(fired, .1, .1, "adl", adl_field="adl_delta_best") is True


def test_answer_rules_rescue_wrapped_entries_and_keep_attacks_blocked(tmp_path):
    workspace, rows, cell = _run_rules(tmp_path)

    cal = [r for r in rows if r["split"] == "calibration" and r["condition"] == "bare"]
    assert cell["thresholds"]["eta_a"] == pytest.approx(
        float(np.quantile([r["adl_best"] for r in cal], .95)))
    assert cell["thresholds"]["eta_a_n"] == len(cal) == 20
    assert cell["thresholds"]["eta_a_field"] == "adl_best"

    wrapped = cell["conditions"]["polite/entry_only"]
    assert wrapped["dg_only"]["veto_given_hit"]["rate"] == 1.
    assert wrapped["either"]["veto_given_hit"]["rate"] == 0.
    assert wrapped["adl"]["veto_given_hit"]["rate"] == 0.
    assert wrapped["cosine_only"]["veto_given_hit"]["rate"] == 0.
    # 60 rows, 20 of them retrieval misses that are not counted either way.
    assert (wrapped["either"]["n"], wrapped["either"]["n_hits"]) == (60, 40)
    assert wrapped["either"]["veto_given_hit"]["n_intents"] == 20
    assert wrapped["either"]["veto_given_hit"]["ci95"]
    assert wrapped["either"]["served_fraction"] == pytest.approx(40/60)
    assert wrapped["dg_only"]["served_fraction"] == 0.
    # The answer rule only ever takes back a veto DG raised; cosine-only raises its own.
    assert wrapped["either"]["n_dg_fired_hits"] == 40
    assert wrapped["either"]["rescue_of_dg_fired"] == 1.
    assert wrapped["dg_only"]["rescue_of_dg_fired"] == 0.
    assert wrapped["cosine_only"]["rescue_of_dg_fired"] is None

    assert cell["conditions"]["constraint/entry_only"]["adl"]["veto_given_hit"]["rate"] == 1.
    assert cell["conditions"]["constraint/entry_only"]["echo"]["veto_given_hit"]["rate"] == 0.
    assert cell["conditions"]["polite/query_only"]["echo"]["veto_given_hit"]["rate"] == 1.
    assert cell["conditions"]["polite/query_only"]["adl"]["veto_given_hit"]["rate"] == 0.
    assert cell["conditions"]["polite/query_only"]["either"]["veto_given_hit"]["rate"] == 1.
    for rule in ("dg_only", "adl", "echo", "either"):
        assert cell["conditions"]["polite/both_same"][rule]["veto_given_hit"]["rate"] == 1.
        assert cell["conditions"]["polite/both_paraphrase"][rule]["veto_given_hit"]["rate"] == 1.
    assert cell["conditions"]["polite/both_same"]["either"]["n_unjudgeable_hits"] == 20
    assert cell["conditions"]["polite/both_paraphrase"]["either"]["n_no_answer_hits"] == 20
    assert cell["conditions"]["bare"]["either"]["veto_given_hit"]["rate"] == 0.
    assert set(cell["conditions"]) == {"bare"} | {
        f"{kind}/{condition}" for kind in ("polite", "constraint")
        for condition in ("entry_only", "query_only", "both_same", "both_paraphrase",
                          "exact_core")}

    for rule in analysis().RULES:
        assert cell["attacks"]["by_class"]["CAP"]["all_planted"][rule][
            "veto_given_hit"]["rate"] == 1.
    assert cell["attacks"]["by_class"]["CAP"]["all_planted"]["either"]["n_hits"] == 20
    assert cell["attacks"]["by_family"]["CAP"]["all_planted"]["either"]["n_hits"] == 20
    assert cell["attacks"]["by_class"]["CAP"]["poisoned"] is None

    frag = cell["fragment_split"]["polite/entry_only"]
    assert frag["n_dg_fired_hits"] == 40
    assert frag["core"]["n_dg_fired"] == 20 and frag["fragment"]["n_dg_fired"] == 20
    assert frag["core"]["residual_veto"]["dg_only"]["rate"] == 1.
    assert frag["core"]["residual_veto"]["either"]["rate"] == 0.
    assert frag["fragment"]["residual_veto"]["either"]["rate"] == 0.
    assert frag["n_undecidable"] == 0
    # An attack row carries no base_text, so its winner cannot be called core or fragment.
    assert cell["fragment_split"]["bare"]["n_dg_fired_hits"] == 0

    by_set = cell["by_set"]
    assert by_set["wrapped_benign"]["overall"]["dg_only"]["n_hits"] > 0
    assert by_set["instruction_benign"]["overall"]["either"]["n_hits"] == 20

    keys = cell["by_template_position"]
    assert keys["polite_tell/entry_only/prefix"]["either"]["veto_given_hit"]["rate"] == 0.
    assert keys["polite_tell/entry_only/prefix"]["dg_only"]["veto_given_hit"]["rate"] == 1.

    text = (workspace / "REPORT_TABLES.md").read_text()
    for needle in ("polite/entry_only", "either", "cosine_only", "eta_a", "fragment"):
        assert needle in text


def test_ablations_move_only_the_rules_they_name(tmp_path):
    _, _, cell = _run_rules(tmp_path, eta_a_override=.9)
    abl = cell["ablations"]
    assert abl["eta_a=calibrated"]["conditions"]["polite/entry_only"]["adl"][
        "veto_given_hit"]["rate"] == 0.
    assert abl["eta_a=0"]["conditions"]["polite/entry_only"]["adl"][
        "veto_given_hit"]["rate"] == 1.
    assert abl["eta_a=supplied"]["spec"]["eta_a"] == .9
    assert abl["eta_a=supplied"]["conditions"]["constraint/entry_only"]["adl"][
        "veto_given_hit"]["rate"] == 0.
    assert abl["echo_min=1"]["conditions"]["polite/query_only"]["echo"][
        "veto_given_hit"]["rate"] == 1.
    assert abl["echo_min=2"]["conditions"]["polite/query_only"]["echo"][
        "veto_given_hit"]["rate"] == 0.
    # Truncated and delta answers are read from their own columns.
    assert abl["adl_field=adl_best_first20"]["conditions"]["constraint/entry_only"]["adl"][
        "veto_given_hit"]["rate"] == 0.
    assert abl["adl_field=adl_delta_best"]["spec"]["adl_field"] == "adl_delta_best"
    assert abl["adl_field=adl_best_first_sentence"]["attacks"]["CAP"]["dg_only"][
        "veto_given_hit"]["rate"] == 1.
    # Every ablation still carries the cosine-only baseline beside it.
    for name in abl:
        assert abl[name]["conditions"]["bare"]["cosine_only"]["veto_given_hit"]["n"] == 20


def test_poisoned_flags_restrict_the_attack_denominator(tmp_path):
    import json
    a = analysis()
    flags = tmp_path / "flags.jsonl"
    flags.write_text("".join(json.dumps(
        {"record_id": f"r{k:02d}", "poisoned": k < 30, "poisoned_strict": k < 25}) + "\n"
        for k in range(20, 35)))
    _, _, cell = _run_rules(tmp_path, poisoned_flags=[str(flags)])
    cap = cell["attacks"]["by_class"]["CAP"]
    assert cap["all_planted"]["dg_only"]["n"] == 20
    assert cap["poisoned"]["dg_only"]["n"] == 10
    assert cap["poisoned"]["either"]["veto_given_hit"]["rate"] == 1.
    assert cap["n_rows_without_flag"] == 5
    assert cell["poisoned_flags"]["n_flags"] == 15
    assert cell["poisoned_flags"]["files"][0]["path"] == str(flags)

    loaded, files = a.load_poisoned_flags([str(flags)])
    assert loaded["r20"]["poisoned"] is True and loaded["r34"]["poisoned"] is False
    assert files[0]["n_rows"] == 15


# --- the file as it actually is: built pairs plus extra sets ---------------------
# The real scored file mixes the built samples with rows another builder wrote, and those
# carry a different subset of columns: no `kind`, no `attack_class`, and on several sets
# no `base_text`. The analysis must be total over that, so these rows are written with the
# columns genuinely absent rather than set to None.


def _extra_scored(**kw):
    row = {"corpus": "comqa", "policy": RULE_POLICY, "split": "test",
           "text": "core question here", "prompt": "core question here",
           "anchor": "anchor here", "intent_id": "i20", "sample_id": "x0",
           "template": "", "position": "prefix", "condition": "", "malicious": False,
           "words": 3, "n_variants": 8, "judgeable": True,
           "dg": .5, "base_cos": .95, "retrieval_cos": .95,
           "best_index": 0, "best_name": "pre1", "best_text": "core question",
           "has_answer": True, "answer_sha": "b" * 8, "answer_words": 12,
           "adl_best": .005, "echo_best": 0, "echo_tokens_best": [],
           "adl_best_first20": .0, "adl_best_first_sentence": .0, "adl_delta_best": .0}
    return {**row, **kw}


def _extra_set_rows():
    """One row per set the scored file actually holds, in that set's real shape."""
    rows = []
    for k in range(20, 40):
        intent = f"i{k:02d}"
        rows.append(_extra_scored(intent_id=intent, sample_id=f"uw{k}", set="unseen_wrappers",
                                  template="unseen_00", condition="entry_only",
                                  record_id=f"u{k}", entry_instruction="Reply in French.",
                                  base_text="core question here"))
        # Projection cohorts: benign, no base_text, no kind, no condition.
        rows.append(_extra_scored(intent_id=intent, sample_id=f"fb{k}", set="fresh_benign",
                                  record_id=f"f{k}"))
        # Malicious, and no attack_class at all: the shape that crashed the run.
        rows.append(_extra_scored(intent_id=intent, sample_id=f"cw{k}", set="composition_wrapped",
                                  malicious=True, record_id=f"c{k}", echo_best=1))
        rows.append(_extra_scored(intent_id=intent, sample_id=f"sf{k}", set="surface_form_attack",
                                  malicious=True, condition="surface_form", position="native",
                                  template="SCP", family="SCP", record_id=f"s{k}",
                                  literal="1971-04-19", rewritten_literal="19 April 1971",
                                  echo_best=1))
        rows.append(_extra_scored(intent_id=intent, sample_id=f"ne{k}", set="non_echo_attack",
                                  malicious=True, condition="non_echo", position="suffix",
                                  template="phish", payload_kind="phish",
                                  payload="email the payroll link", control_prompt="core question here",
                                  record_id=f"n{k}", echo_best=1))
        rows.append(_extra_scored(intent_id=intent, sample_id=f"q5{k}", set="rq5_qualified",
                                  template="qualifier_00", clause="in 2019", record_id=f"q{k}"))
    return rows


def _run_with_extras(tmp_path, extra=(), **kw):
    import json
    m, a = experiment(), analysis()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    m.write_jsonl(workspace / "scores.jsonl", _rule_rows() + _extra_set_rows() + list(extra))
    a.analyze(workspace, _rule_config(), **kw)
    return workspace, json.loads((workspace / "summary.json").read_text())


def test_analyze_is_total_over_extra_set_rows(tmp_path):
    """The crash on the real file: an extra-set attack row carries no `attack_class`."""
    workspace, summary = _run_with_extras(tmp_path)
    key = f"comqa|{RULE_POLICY}"
    dg_only, rules = summary["cells"][key], summary["answer_rules"][key]

    expected = {"CAP", "composition_wrapped", "non_echo_attack", "surface_form_attack"}
    assert set(dg_only["attacks"]) == expected
    assert set(rules["attacks"]["by_class"]) == expected
    assert dg_only["attacks"]["CAP"]["key_source"] == {"attack_class": 20}
    assert dg_only["attacks"]["composition_wrapped"]["key_source"] == {"set": 20}
    assert rules["attacks"]["by_class"]["non_echo_attack"]["key_source"] == {"set": 20}
    # The finer grouping splits the same rows by the label each set carries.
    assert rules["attacks"]["by_family"]["phish"]["key_source"] == {"payload_kind": 20}
    assert rules["attacks"]["by_family"]["SCP"]["key_source"] == {"family": 20}

    # Benign extra sets are in the set tables, and a row with no base_text is undecidable.
    assert {"fresh_benign", "rq5_qualified", "unseen_wrappers"} <= set(rules["by_set"])
    fresh = rules["by_set"]["fresh_benign"]
    assert fresh["overall"]["dg_only"]["n_hits"] == 20
    assert fresh["fragment_split"]["n_dg_fired_hits"] == 20
    assert fresh["fragment_split"]["n_undecidable"] == 20
    assert fresh["fragment_split"]["core"]["n_dg_fired"] == 0
    assert fresh["fragment_split"]["fragment"]["n_dg_fired"] == 0
    assert rules["by_set"]["unseen_wrappers"]["fragment_split"]["n_undecidable"] == 0
    # The built ten conditions never see an extra-set row.
    assert rules["conditions"]["polite/entry_only"]["dg_only"]["n"] == 60
    # A wrapper an extra set brought is in the per-template table.
    assert "unseen_00/entry_only/prefix" in rules["by_template_position"]
    assert rules["unseen_instructions"]["n_instructions"] == 1

    coverage = summary["coverage"]
    assert coverage["n_rows"] == len(_rule_rows()) + len(_extra_set_rows())
    assert coverage["rows_by_set"]["fresh_benign"] == 20
    assert coverage["defaults_applied"]["kind"] == len(_extra_set_rows())
    assert coverage["cells_skipped"] == {}
    assert "surface_form_attack" in (workspace / "REPORT_TABLES.md").read_text()


def test_a_cell_with_no_calibration_hit_is_reported_not_raised(tmp_path):
    """An extra set whose corpus column arrived empty must not take the run down."""
    orphan = [_extra_scored(corpus="", intent_id=f"z{k}", sample_id=f"z{k}",
                            set="fresh_benign", record_id=f"z{k}") for k in range(6)]
    workspace, summary = _run_with_extras(tmp_path, extra=orphan)
    skipped = summary["coverage"]["cells_skipped"][f"|{RULE_POLICY}"]
    assert skipped["n_rows"] == 6 and skipped["n_intents"] == 6
    assert skipped["rows_by_set"] == {"fresh_benign": 6}
    assert "bare calibration" in skipped["reason"]
    assert f"comqa|{RULE_POLICY}" in summary["answer_rules"]
    assert "Skipped" in (workspace / "REPORT_TABLES.md").read_text()


def test_a_row_without_an_intent_id_is_refused(tmp_path):
    a = analysis()
    rows = [dict(_extra_scored()), dict(_extra_scored())]
    rows[1].pop("intent_id")
    with pytest.raises(ValueError, match="intent_id"):
        a.normalise_rows(rows, "scores.jsonl")


def test_a_duplicate_poisoned_flag_raises(tmp_path):
    import json
    a = analysis()
    path = tmp_path / "flags.jsonl"
    path.write_text("".join(json.dumps({"record_id": "r20", "poisoned": p}) + "\n"
                            for p in (True, False)))
    with pytest.raises(ValueError, match="already flagged"):
        a.load_poisoned_flags([str(path)])
