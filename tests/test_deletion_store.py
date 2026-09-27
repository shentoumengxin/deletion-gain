"""Profile storage: the join between the write path and the serving path.

GPTCache assigns an entry id inside ``import_data`` and ``save`` does not return it, so
the two paths cannot be joined on the id. They can be joined on the entry *text*: the
writer is handed it as ``save``'s ``question`` argument and the evaluator is handed it
back as ``eval_cache_data["question"]``.

The subtlety worth a test is that GPTCache wraps both halves of an entry, and names
their payloads differently. A question arrives as a ``Question`` dataclass whose
``__str__`` is its *repr*, so a key built with ``str()`` would be
``"Question(content='...', deps=None)"`` on one path and the bare text on the other, and
nothing would ever match. An answer arrives as an ``Answer`` dataclass whose payload is
``.answer`` rather than ``.content``, so unwrapping only the question would hand the
answer check a dataclass repr to embed.

Only the three wrapper tests need ``gptcache``, so the skip is scoped to them via the
``question_cls`` and ``answer_cls`` fixtures rather than declared at module scope.
GPTCache is an optional dependency by design; a module-level ``importorskip`` would
silently delete the rest -- eviction, reconciliation, mismatch refusal, disk round-trip,
the answer fields -- on any machine without it, and the suite would still report green
over a store that is the join between the write path and the serving path.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentry.cache.defense.entry_store import (
    InMemoryProfileStore,
    entry_text,
    text_key,
)
from sentry.cache.defense.deletion import DeletionProfile


@pytest.fixture
def answer_cls():
    """GPTCache's ``Answer`` wrapper, skipping only the test that needs it."""
    module = pytest.importorskip(
        "gptcache.manager.scalar_data.base",
        reason="gptcache is optional; only the Answer-wrapper test needs it")
    return module.Answer


@pytest.fixture
def question_cls():
    """GPTCache's ``Question`` wrapper, skipping only the tests that need it."""
    module = pytest.importorskip(
        "gptcache.manager.scalar_data.base",
        reason="gptcache is optional; only the Question-wrapper tests need it")
    return module.Question


def profile(embedder: str = "e5", policy: str = "count:6",
            dimension: int = 8) -> DeletionProfile:
    rng = np.random.default_rng(0)
    return DeletionProfile(
        whole=rng.normal(size=dimension),
        spans=rng.normal(size=(10, dimension)),
        deletions=rng.normal(size=(6, dimension)),
        span_names=tuple(f"s{i}" for i in range(10)),
        deletion_names=tuple(f"d{i}" for i in range(6)),
        words=12, segment_count=6, embedder=embedder, policy=policy, min_segments=3)


def test_a_gptcache_question_keys_the_same_as_its_text(question_cls):
    assert text_key(question_cls("what is the capital of france")) == \
        text_key("what is the capital of france")


def test_entry_text_reads_content_not_repr(question_cls):
    assert entry_text(question_cls("hello world")) == "hello world"


def test_key_is_stable_under_incidental_whitespace():
    assert text_key("  what  is   the capital of france ") == \
        text_key("what is the capital of france")


def test_key_distinguishes_different_texts():
    assert text_key("capital of france") != text_key("capital of germany")


def test_put_then_get():
    store = InMemoryProfileStore("e5", "count:6")
    store.put("k", profile())
    assert store.get("k") is not None
    assert len(store) == 1


def test_get_is_none_for_an_unknown_key():
    assert InMemoryProfileStore("e5", "count:6").get("nope") is None


def test_get_forgets_a_profile_built_in_another_space():
    """A stale profile is worse than none: it is a confidently wrong answer."""
    store = InMemoryProfileStore("e5", "count:6")
    store._profiles["k"] = profile(embedder="bge")     # simulate a config change
    assert store.get("k") is None
    assert len(store) == 0


def test_get_forgets_a_profile_cut_under_another_policy():
    store = InMemoryProfileStore("e5", "count:6")
    store._profiles["k"] = profile(policy="width:3")
    assert store.get("k") is None


def test_put_refuses_a_mismatched_profile():
    store = InMemoryProfileStore("e5", "count:6")
    with pytest.raises(ValueError):
        store.put("k", profile(embedder="bge"))
    with pytest.raises(ValueError):
        store.put("k", profile(policy="width:3"))


def test_capacity_evicts_least_recently_used():
    store = InMemoryProfileStore("e5", "count:6", capacity=2)
    store.put("a", profile())
    store.put("b", profile())
    store.get("a")                 # touch a, so b is oldest
    store.put("c", profile())

    assert store.get("b") is None
    assert store.get("a") is not None
    assert store.get("c") is not None


def test_reconcile_drops_keys_the_host_no_longer_holds():
    """Guards against an evicted entry's profile being served for a new entry."""
    store = InMemoryProfileStore("e5", "count:6")
    store.put("a", profile())
    store.put("b", profile())

    assert store.reconcile(["a"]) == 1
    assert store.get("b") is None
    assert store.get("a") is not None


def test_roundtrips_through_disk(tmp_path):
    store = InMemoryProfileStore("e5", "count:6")
    store.put("a", profile())
    store.save(tmp_path)

    loaded = InMemoryProfileStore.load(tmp_path)
    got = loaded.get("a")

    assert got is not None
    assert got.policy == "count:6"
    assert got.embedder == "e5"
    assert got.words == 12
    assert got.span_names == tuple(f"s{i}" for i in range(10))
    np.testing.assert_allclose(got.whole, store.get("a").whole)
    np.testing.assert_allclose(got.spans, store.get("a").spans)
    np.testing.assert_allclose(got.deletions, store.get("a").deletions)


def test_entry_text_reads_a_gptcache_answer(answer_cls):
    """The write path's answer arrives wrapped too, and in a *differently* named field.

    ``Question`` calls its payload ``content``; ``Answer`` calls it ``answer``. Unwrapping
    only the first one would hand the answer check ``"Answer(answer='john adams',
    answer_type=0)"`` to embed and to tokenise -- an answer digest over a dataclass repr,
    and echo tokens that include the field names.
    """
    assert entry_text(answer_cls("john adams", 0)) == "john adams"


def test_store_round_trips_answer_fields(tmp_path):
    """The three answer fields survive disk, and an entry written without one stays without."""
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.deletion import build_profile
    from tests.test_deletion_statistic import ToyEmbedder

    emb, policy = ToyEmbedder(), parse_policy("multi[count:4+width:2:cap16]/runs")
    store = InMemoryProfileStore(emb.model_name, policy.fingerprint())
    with_answer = build_profile("what high school did eminem attend reply 1971-04-19",
                                emb, policy, answer="The answer is 1971-04-19.")
    without = build_profile("what city hosted the 1998 winter olympics", emb, policy)
    store.put("a", with_answer)
    store.put("b", without)
    store.save(tmp_path)

    again = InMemoryProfileStore.load(tmp_path)
    a, b = again.get("a"), again.get("b")

    assert a.has_answer and a.answer_digest == with_answer.answer_digest
    assert a.echo_tokens == with_answer.echo_tokens
    assert (a.answer_loss == with_answer.answer_loss).all()
    assert not b.has_answer
    assert b.answer_loss is None and b.echo_tokens is None and b.answer_digest is None


def test_a_store_written_before_the_answer_fields_still_loads(tmp_path):
    """Existing on-disk stores predate the three fields and must not become unreadable.

    The record they wrote has no ``has_answer``/``echo_tokens``/``answer_digest`` keys and
    the bundle has no ``aloss_`` array. Such an entry loads with the fields absent, which
    the serving path already handles: the veto is decided by the deletion gain alone and
    the fallback is counted in ``no_answer_fields``.
    """
    import json

    store = InMemoryProfileStore("e5", "count:6")
    store.put("a", profile())
    store.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    for record in meta["profiles"]:
        for field in ("has_answer", "echo_tokens", "answer_digest"):
            record.pop(field, None)
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    got = InMemoryProfileStore.load(tmp_path).get("a")

    assert got is not None and not got.has_answer
    assert got.answer_loss is None and got.echo_tokens is None and got.answer_digest is None


def test_the_answer_digest_does_not_round_trip_on_its_own(tmp_path):
    """The three answer fields are one fact, so none of them survives without the others.

    A profile carrying a digest but no per-variant fields is incoherent -- ``has_answer``
    already reads False for it -- and letting the digest through the round trip would
    leave a stored record that names the answer it was built from while carrying nothing
    the check can read against it.
    """
    from dataclasses import replace

    store = InMemoryProfileStore("e5", "count:6")
    store.put("a", replace(profile(), answer_digest="deadbeef"))
    store.save(tmp_path)

    got = InMemoryProfileStore.load(tmp_path).get("a")

    assert not got.has_answer
    assert got.answer_digest is None
