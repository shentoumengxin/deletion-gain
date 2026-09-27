"""Frozen cohort sources preserve intent isolation during reorganization."""

def test_fresh_clusters_exclude_any_old_question_and_do_not_split_intents():
    from sentry.research.pipeline import cohort_sources as m
    cfg = m.CohortConfig(name="test", seed=1, n_intents=100, fresh_intents=10)
    old = [dict(text="old", anchor="Who is Alice?", base_text="old", base_anchor="Who is Alice?")]
    raw = [dict(cluster_id="overlap", questions=["Different phrasing?", "WHO IS ALICE"]),
           dict(cluster_id="new", questions=["Who is Bob?", "What is Bob's identity?"]),
           dict(cluster_id="singleton", questions=["Single question?"])]
    rows, info = m.fresh_benign(raw, old, cfg)
    assert info["selected_clusters"] == ["new"]
    assert {r["intent_id"] for r in rows} == {"comqa-fresh:new"}
    assert len(rows) == 31
    for r in rows:
        assert "Who is Bob?" in r["text"]
        assert r["stage"] == "fresh_test"
        assert r["malicious"] is False

