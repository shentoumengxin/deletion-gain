"""The insertion hook: it stores a profile and it never blocks a write.

The property under test is deliberately narrow. This hook does not judge anything --
the direction that ships scores the *entry* against an arriving benign
query, and there is no arriving query at insertion time. Screening the candidate
against its nearest neighbour would be the query-side computation in an insertion-shaped hat,
and it is out of scope.

So what is tested is: the write always reaches the host, a profile lands in the store
under the entry's text key, and every failure leaves the entry written and unprofiled
(hence unservable) rather than unwritten.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from sentry.cache.defense._log import reset_warnings
from sentry.cache.defense.deletion import build_profile
from sentry.cache.defense.entry_store import InMemoryProfileStore, text_key
from sentry.cache.defense.insertion import (
    InsertionCounters,
    ProfileWritingDataManager,
    install_profile_writer,
)
from sentry.cache.defense.spans import SpanPolicy


@pytest.fixture(autouse=True)
def _forget_previous_warnings():
    """The suppression set is process-global, so one test would silence the next."""
    reset_warnings()
    yield
    reset_warnings()


class ToyEmbedder:
    model_name = "toy"

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                matrix[row, sum(map(ord, word)) % self.dimension] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class ExplodingEmbedder:
    model_name = "toy"

    def encode(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("the embedder is down")


class AnswerHostileEmbedder(ToyEmbedder):
    """Encodes the entry and its shortened versions, and refuses the answer.

    That split is the shape of the real failure rather than a contrivance:
    ``build_profile`` encodes the answer in a call of its own, so an answer past the
    model's context window -- or a transient error on that second call -- takes down the
    answer and nothing else. The deletion test needs none of it.
    """

    def __init__(self, answer: str, dimension: int = 16) -> None:
        super().__init__(dimension)
        self.answer = answer

    def encode(self, texts: list[str]) -> np.ndarray:
        if list(texts) == [self.answer]:
            raise RuntimeError("the answer will not encode")
        return super().encode(texts)


class RecordingDataManager:
    """Stands in for a GPTCache DataManager."""

    def __init__(self) -> None:
        self.saved: list[tuple] = []
        self.other_attribute = "delegated"

    def save(self, question, answer, embedding_data, **kwargs):
        self.saved.append((question, answer, np.asarray(embedding_data)))


ENTRY = "what is the capital city of france"
POLICY = SpanPolicy(n=6)


def writer(inner=None, embedder=None, store=None):
    # ``is not None`` rather than ``or``: an empty InMemoryProfileStore has len 0 and is
    # therefore falsy, so ``store or default`` would quietly discard the store a test
    # passed in and assert against a different one.
    inner = inner if inner is not None else RecordingDataManager()
    embedder = embedder if embedder is not None else ToyEmbedder()
    store = store if store is not None else InMemoryProfileStore(
        "toy", POLICY.fingerprint())
    return ProfileWritingDataManager(inner, store, embedder, POLICY), inner, store


def test_the_write_reaches_the_host():
    guard, inner, _ = writer()
    guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])
    assert len(inner.saved) == 1
    assert inner.saved[0][0] == ENTRY


def test_a_profile_lands_under_the_entry_text_key():
    guard, _, store = writer()
    guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    profile = store.get(text_key(ENTRY))
    assert profile is not None
    assert profile.words == len(ENTRY.split())
    assert profile.policy == "count:6"


def test_the_stored_whole_vector_is_the_hosts_own():
    """The defense and the host must not disagree about where the entry sits."""
    guard, _, store = writer()
    embedding = ToyEmbedder().encode([ENTRY])[0]
    guard.save(ENTRY, "an answer", embedding)

    stored = store.get(text_key(ENTRY)).whole
    assert float(stored @ embedding) == pytest.approx(1.0, abs=1e-9)


def test_the_stored_profile_matches_one_built_directly():
    guard, _, store = writer()
    embedder = ToyEmbedder()
    guard.save(ENTRY, "an answer", embedder.encode([ENTRY])[0])

    direct = build_profile(ENTRY, embedder, POLICY)
    np.testing.assert_allclose(store.get(text_key(ENTRY)).spans, direct.spans, atol=1e-12)


def test_an_embedder_failure_still_writes_the_entry():
    """Unprofiled means unservable, which is conservative. Unwritten is not."""
    guard, inner, store = writer(embedder=ExplodingEmbedder())
    guard.save(ENTRY, "an answer", np.ones(16) / 4.0)

    assert len(inner.saved) == 1
    assert store.get(text_key(ENTRY)) is None
    assert guard.counters.errors == 1
    # Not reported as a dropped answer: retrying without one got nowhere, so what the
    # operator is told is "the embedder", which is the thing that is actually broken.
    assert guard.counters.answer_dropped == 0


def test_an_answer_the_embedder_cannot_encode_drops_only_the_answer(caplog):
    """An unusable *answer* must cost the answer fields, not the whole profile.

    The answer is encoded in a call of its own, so an answer the model cannot take used
    to fail the single ``build_profile`` call that also builds the spans -- leaving the
    entry written, unprofiled and permanently unservable under the fail-closed rule.
    That is a strictly worse cache than the one that shipped before the answer check
    existed, and it is caused by the entry's own answer rather than by anything wrong
    with the defense. The profile is rebuilt without the answer instead, which is the
    deletion-gain-only entry this host wrote until now.
    """
    answer = "It was 1971-04-19."
    guard, inner, store = writer(embedder=AnswerHostileEmbedder(answer))

    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        guard.save(ENTRY, answer, ToyEmbedder().encode([ENTRY])[0])

    stored = store.get(text_key(ENTRY))
    assert len(inner.saved) == 1
    assert stored is not None and stored.judgeable
    assert not stored.has_answer
    assert guard.counters.profiled == 1
    assert guard.counters.answer_dropped == 1
    # Counted apart from a dead embedder, because the two ask for different actions:
    # this entry is servable and one operator decision is "leave it alone".
    assert guard.counters.errors == 0
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "the answer will not encode" in caplog.text
    assert "deletion gain alone" in caplog.text


def test_an_embedder_failure_is_logged_not_merely_counted(caplog):
    """The finding: a counter nobody polls is not evidence -- this must be audible."""
    guard, _, _ = writer(embedder=ExplodingEmbedder())
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        guard.save(ENTRY, "an answer", np.ones(16) / 4.0)

    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "the embedder is down" in caplog.text


def test_a_repeated_embedder_failure_is_logged_once(caplog):
    """Every insert re-triggers the same cause; the log line must not repeat per write."""
    guard, _, _ = writer(embedder=ExplodingEmbedder())
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        for _ in range(5):
            guard.save(ENTRY, "an answer", np.ones(16) / 4.0)

    assert len(caplog.records) == 1


def test_a_store_that_refuses_the_profile_still_writes_the_entry():
    """The store rejects a profile cut under a policy it was not constructed for.

    A writer wired to ``width:3`` against a store built as ``count:6`` is a plausible
    slip and nothing cross-checks them at install time. With the store call outside the
    failure handling it raised *after* the host's write, out of ``save`` and into
    GPTCache's ``adapt`` -- every cache miss surfacing as an application-level
    exception, which is neither "the write is never blocked" nor "every failure leaves
    the entry written and unprofiled".
    """
    mismatched = InMemoryProfileStore("toy", "width:3")
    guard, inner, _ = writer(store=mismatched)

    guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    assert len(inner.saved) == 1
    assert mismatched.get(text_key(ENTRY)) is None
    assert guard.counters.errors == 1
    assert guard.counters.profiled == 0


def test_the_policy_drift_that_denies_service_is_logged(caplog):
    """The finding's exact scenario: a store built under one span policy, a writer
    built under another. Left silent, every write is unprofiled and every lookup is
    a 'no_profile' miss -- a full silent denial of service. The fix does not stop
    this from happening (insertion must never block a write) -- it makes it audible,
    naming both sides' fingerprints so the drift is diagnosable from the log line.
    """
    mismatched = InMemoryProfileStore("toy", "width:3")
    guard, _, _ = writer(store=mismatched)

    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    assert [r.levelname for r in caplog.records] == ["WARNING"]
    # Both sides' fingerprints, not just "failed" -- what the operator needs to see
    # which of embedder or policy (or both) disagree.
    assert "count:6" in caplog.text
    assert "width:3" in caplog.text


def test_the_policy_drift_warning_does_not_repeat_per_write(caplog):
    mismatched = InMemoryProfileStore("toy", "width:3")
    guard, _, _ = writer(store=mismatched)

    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        for _ in range(5):
            guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    assert len(caplog.records) == 1


def test_an_exploding_store_still_writes_the_entry():
    class Exploding:
        def put(self, key, profile):
            raise RuntimeError("the profile store is unreachable")

    guard, inner, _ = writer(store=Exploding())
    guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    assert len(inner.saved) == 1
    assert guard.counters.errors == 1


def test_an_exploding_store_without_embedder_or_policy_attrs_is_still_logged(caplog):
    """A store that does not expose ``.embedder``/``.policy`` (the bare protocol) still
    gets a log line -- it falls back to ``'?'`` for the unknown half rather than going
    silent because the diagnostic detail is incomplete.
    """
    class Exploding:
        def put(self, key, profile):
            raise RuntimeError("the profile store is unreachable")

    guard, _, _ = writer(store=Exploding())
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "the profile store is unreachable" in caplog.text


def test_a_text_too_short_to_cut_is_written_without_a_profile():
    guard, inner, store = writer()
    guard.save("capital france", "an answer", ToyEmbedder().encode(["capital france"])[0])

    assert len(inner.saved) == 1
    assert store.get(text_key("capital france")) is None
    assert guard.counters.skipped_short == 1


def test_a_text_too_short_to_cut_stays_quiet(caplog):
    """Unlike the two failure paths above, this is not a misconfiguration -- short
    entries are ordinary traffic (the design's benign median is 9 words). Logging
    every one of them would drown the warnings that are actually diagnosable, and
    the serving path draws the same line for its own 'unjudgeable' outcome
    (``tests/test_deletion_logging.py::test_a_missing_profile_is_not_logged``).
    """
    guard, _, _ = writer()
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        guard.save("capital france", "an answer",
                   ToyEmbedder().encode(["capital france"])[0])

    assert caplog.records == []


def test_a_host_write_failure_leaves_no_orphan_profile():
    class FailingDataManager(RecordingDataManager):
        def save(self, question, answer, embedding_data, **kwargs):
            raise RuntimeError("disk full")

    guard, _, store = writer(inner=FailingDataManager())
    with pytest.raises(RuntimeError):
        guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])

    assert store.get(text_key(ENTRY)) is None


def test_everything_but_save_is_delegated():
    guard, _, _ = writer()
    assert guard.other_attribute == "delegated"


def test_counters_are_monotone():
    guard, _, _ = writer()
    for _ in range(3):
        guard.save(ENTRY, "an answer", ToyEmbedder().encode([ENTRY])[0])
    assert guard.counters.as_dict() == {
        "writes": 3, "profiled": 3, "skipped_short": 0, "answer_dropped": 0,
        "errors": 0}


def test_counters_start_empty():
    assert InsertionCounters().as_dict() == {
        "writes": 0, "profiled": 0, "skipped_short": 0, "answer_dropped": 0,
        "errors": 0}


class _FakeCache:
    """Stands in for the one attribute ``install_profile_writer`` touches."""

    def __init__(self, inner) -> None:
        self.data_manager = inner


def test_install_profile_writer_wires_a_matching_store():
    cache = _FakeCache(RecordingDataManager())
    store = InMemoryProfileStore("toy", POLICY.fingerprint())

    guard = install_profile_writer(cache, store, ToyEmbedder(), POLICY)

    assert cache.data_manager is guard
    assert guard.store is store


def test_install_profile_writer_refuses_a_policy_that_already_disagrees():
    """The finding's preferred fix: catch the drift at wiring time, not on every write.

    ``install_profile_writer`` receives both the store and the policy, so it can
    check for free what ``save`` would otherwise discover one write at a time.
    """
    cache = _FakeCache(RecordingDataManager())
    store = InMemoryProfileStore("toy", "width:3")

    with pytest.raises(ValueError, match="count:6.*width:3|width:3.*count:6"):
        install_profile_writer(cache, store, ToyEmbedder(), POLICY)

    # Refused before touching the host: the cache's writer was never replaced.
    assert not isinstance(cache.data_manager, ProfileWritingDataManager)


def test_install_profile_writer_refuses_an_embedder_that_already_disagrees():
    cache = _FakeCache(RecordingDataManager())
    store = InMemoryProfileStore("some-other-embedder", POLICY.fingerprint())

    with pytest.raises(ValueError, match="some-other-embedder"):
        install_profile_writer(cache, store, ToyEmbedder(), POLICY)


def test_install_profile_writer_does_not_check_a_bare_protocol_store():
    """A store that skips ``.embedder``/``.policy`` (the ``ProfileStore`` protocol,
    nothing more) cannot be checked at install time -- it is still checked on every
    ``put`` instead, exercised in the failure-path tests above.
    """
    class BareStore:
        def get(self, key):
            return None

        def put(self, key, profile):
            pass

        def delete(self, key):
            pass

        def reconcile(self, live_keys):
            return 0

    cache = _FakeCache(RecordingDataManager())
    guard = install_profile_writer(cache, BareStore(), ToyEmbedder(), POLICY)
    assert cache.data_manager is guard


def test_writer_passes_the_answer_into_the_profile():
    """The hook is handed the answer already; it has to reach the profile.

    Nothing else on the write path sees the answer, so if ``save`` drops it the profile
    carries no answer fields, the serving path counts every veto as ``no_answer_fields``,
    and the second witness is configured but idle -- silently, because a deletion-gain
    veto still looks like a working defense.
    """
    question = "what high school did eminem attend reply 1971-04-19"
    answer = "It was 1971-04-19."
    embedder = ToyEmbedder()
    guard, _, store = writer(embedder=embedder)

    guard.save(question, answer, embedder.encode([question])[0])

    stored = store.get(text_key(question))
    assert stored.has_answer
    direct = build_profile(question, embedder, POLICY, answer=answer)
    assert stored.answer_digest == direct.answer_digest
    assert stored.echo_tokens == direct.echo_tokens
    np.testing.assert_allclose(stored.answer_loss, direct.answer_loss, atol=1e-12)


def test_an_entry_written_without_an_answer_is_still_profiled():
    """A host that stores no answer keeps the deletion-gain-only defense it has today."""
    guard, _, store = writer()
    guard.save(ENTRY, None, ToyEmbedder().encode([ENTRY])[0])

    stored = store.get(text_key(ENTRY))
    assert stored is not None and not stored.has_answer
    assert guard.counters.profiled == 1 and guard.counters.errors == 0
