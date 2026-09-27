"""A refused configuration has to be loud, and loud exactly once.

Every failure on the serving path degrades to a miss, which is the right behaviour and
also the reason these failures are invisible: a fleet-wide embedder change, or a span
policy applied to the fence but not to the stored profiles, mismatches everything,
misses every lookup, and drives the hit rate to zero while looking identical to a cold
cache. A counter nobody polls is not evidence.

The other half is the rate limit. A misconfiguration repeats on *every* request, so a
warning per request would put an I/O call in a hot loop and bury the first line under a
million copies of itself. One line per distinct cause is the contract, and a test that
could not tell "warned once" from "warned every time" would not be testing it.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from sentry.cache.defense._log import reset_warnings
from sentry.cache.defense.decide import DeletionDefenseConfig, decide
from sentry.cache.defense.deletion import build_profile
from sentry.cache.defense.entry_store import InMemoryProfileStore, text_key
from sentry.cache.defense.fence import CalibrationRow, ExcessFence
from sentry.cache.defense.spans import SpanPolicy
from tests.collision_embedder import BENIGN_QUERY, GENUINE_ENTRY, CollisionEmbedder

POLICY = SpanPolicy(n=6)
EMBEDDER = CollisionEmbedder()


@pytest.fixture(autouse=True)
def _forget_previous_warnings():
    """The suppression set is process-global, so one test would silence the next."""
    reset_warnings()
    yield
    reset_warnings()


def _profile(text: str = GENUINE_ENTRY):
    return build_profile(text, EMBEDDER, POLICY)


def _anchor() -> np.ndarray:
    return EMBEDDER.encode([BENIGN_QUERY])[0]


def _fence(policy: str = "count:6", embedder: str | None = None) -> ExcessFence:
    return ExcessFence.fit_flat(
        [CalibrationRow(0.95, 12, 10.0) for _ in range(50)], budget=0.05,
        embedder=EMBEDDER.model_name if embedder is None else embedder, policy=policy)


def test_a_fingerprint_mismatch_is_logged_and_not_merely_counted(caplog):
    """The named red line: a mismatch is refused *loudly*."""
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        decision = decide(profile=_profile(), anchor=_anchor(), cosine=0.99,
                          fence=_fence(policy="width:3"),
                          config=DeletionDefenseConfig())

    assert decision.blocked and decision.reason == "config_mismatch"
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "width:3" in caplog.text


def test_the_mismatch_warning_does_not_repeat_per_request(caplog):
    """A misconfiguration fires on every request; the log must not."""
    fence = _fence(policy="width:3")
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        for _ in range(50):
            decide(profile=_profile(), anchor=_anchor(), cosine=0.99, fence=fence,
                   config=DeletionDefenseConfig())

    assert len(caplog.records) == 1


def test_a_second_distinct_mismatch_is_still_reported(caplog):
    """Rate limiting is per cause, not per process: a new cause is new information."""
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        decide(profile=_profile(), anchor=_anchor(), cosine=0.99,
               fence=_fence(policy="width:3"), config=DeletionDefenseConfig())
        decide(profile=_profile(), anchor=_anchor(), cosine=0.99,
               fence=_fence(embedder="some-other-model"),
               config=DeletionDefenseConfig())

    assert len(caplog.records) == 2


def test_serving_with_no_fence_is_logged(caplog):
    """The other way the defense refuses everything without anything looking wrong."""
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        decision = decide(profile=_profile(), anchor=_anchor(), cosine=0.99, fence=None,
                          config=DeletionDefenseConfig())

    assert decision.blocked and decision.reason == "no_fence"
    assert len(caplog.records) == 1
    assert "calibrate" in caplog.text


def test_a_missing_profile_is_not_logged(caplog):
    """Cold entries are ordinary traffic. Logging them would drown the real warnings."""
    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        decision = decide(profile=None, anchor=_anchor(), cosine=0.99, fence=_fence(),
                          config=DeletionDefenseConfig())

    assert decision.reason == "no_profile"
    assert caplog.records == []


def test_the_store_says_so_when_it_drops_a_stale_profile(caplog):
    """A dropped profile reaches serving as ``no_profile`` -- indistinguishable from cold.

    The store is where the information exists, so the store is where it is logged.
    """
    store = InMemoryProfileStore(EMBEDDER.model_name, POLICY.fingerprint())
    store.put(text_key(GENUINE_ENTRY), _profile())
    # What an operator changing the span policy without reprofiling produces.
    store.policy = "width:3"

    with caplog.at_level(logging.WARNING, logger="sentry.cache.defense"):
        assert store.get(text_key(GENUINE_ENTRY)) is None

    assert len(caplog.records) == 1
    assert "no_profile" in caplog.text
