"""GPTCache integration: the deletion test as a veto on an existing evaluator.

``DeletionVetoEvaluation`` implements GPTCache's ``SimilarityEvaluation`` interface and
wraps whatever evaluator the deployment already uses. It can only *remove* hits -- when
it accepts, it forwards the inner evaluator's score unchanged. Adopting the defense is
therefore a configuration change rather than a fork, and disabling it restores the
host's original behaviour exactly.

**This is the entry side.** The text under inspection is the *cached entry*; the
anchor is the *arriving query*. That is the direction the threat model needs: the
attacker owns the entry, plants a payload near a popular question, and ordinary users
collide into it afterwards. The victim's query is clean, so scoring it finds nothing --
only scoring the entry against the user's question finds the payload.

It is also the direction that costs nothing here. The entry's shortened versions were
embedded once at insertion (``insertion.py``); serving scores the stored candidate vectors against the
arriving query, with no model call and no text generation. The answer check the ladder
may run adds one tokenisation of the arriving query and one set difference, which is why
the query's *text* is handed to :func:`~sentry.cache.defense.decide.decide` alongside its
vector.

GPTCache calls the evaluator as::

    rank = evaluation_func(
        {"question": ..., "embedding": ...},
        {"question": ..., "answer": ..., "search_result": (distance, vector_id),
         "embedding": ...},
    )

``cache_dict["question"]`` is the cached entry's text, which is how the stored profile
is found -- the host's ``vector_id`` is never learned by the write path, so it cannot be
the join key. See ``entry_store.py``.

Two behaviours are load-bearing:

- **The excess is logged before it is acted on.** Refitting the fence on served traffic
  alone censors the upper tail and ratchets it downward every epoch until it blocks
  everything. Blocked candidates must reach the calibration log too.
- **Every failure degrades toward a stricter cache, never an undefended one.** A missing
  profile, a mismatched fingerprint, an unreadable store, a missing fence, an
  unjudgeable entry, or an unexpected exception all reduce to "serve only
  near-duplicates", never to "serve anything".

The verdict itself is **not** written here. It is :func:`sentry.cache.defense.decide.decide`,
used by the real GPTCache runtime; that module documents the order and the reason for each rung. What lives in this file is the
GPTCache plumbing around it: where the query vector and the cosine come from, and how a
verdict is expressed as a score the host reads as a hit or a miss.

Sharing the ladder is a correctness property. While it was written twice the two copies
ordered their checks differently, only one enforced the ``cache_threshold`` floor, and a
fail-open hole fixed here was reintroduced on the other host, because nothing connected
them.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol
from dataclasses import replace
import hashlib
from .textnorm import normalise_answer

import numpy as np

from .decide import (
    DefenseCounters,
    DeletionDecision,
    DeletionDefenseConfig,
    InMemoryCalibrationLog,
    cosine_from_l2_distance,
    decide,
    l2_distance_from_cosine,
)
from .deletion import DeletionProfile, unit
from .entry_store import ProfileStore, entry_text, text_key
from .fence import ExcessFence

__all__ = [
    "DefenseCounters",
    "DeletionDecision",
    "DeletionDefenseConfig",
    "DeletionVetoEvaluation",
    "InMemoryCalibrationLog",
    "InnerEvaluation",
    "cosine_from_l2_distance",
    "l2_distance_from_cosine",
]

try:  # GPTCache is optional: the library and its tests must not require it.
    from gptcache.similarity_evaluation import SimilarityEvaluation as _SimilarityEvaluation
except Exception:  # pragma: no cover - exercised only where gptcache is absent
    class _SimilarityEvaluation:  # type: ignore[no-redef]
        """Stand-in with the same surface GPTCache expects."""

        def __init__(self, **kwargs: Any) -> None:
            del kwargs


class InnerEvaluation(Protocol):
    def range(self) -> tuple[float, float]: ...
    def evaluation(self, src_dict: dict, cache_dict: dict, **kwargs: Any) -> float: ...


class DeletionVetoEvaluation(_SimilarityEvaluation):
    """The entry-side deletion test in front of an existing GPTCache similarity evaluator."""

    def __init__(
        self,
        inner: InnerEvaluation,
        store: ProfileStore,
        fence: Optional[ExcessFence] = None,
        config: Optional[DeletionDefenseConfig] = None,
        log: Optional[InMemoryCalibrationLog] = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.store = store
        self.config = config or DeletionDefenseConfig()
        self._fence: Optional[ExcessFence] = None
        self.fence = fence
        self.log = log if log is not None else InMemoryCalibrationLog()
        self.counters = DefenseCounters()
        self.last_decision: Optional[DeletionDecision] = None

    @property
    def fence(self) -> Optional[ExcessFence]:
        return self._fence

    @fence.setter
    def fence(self, value: Optional[ExcessFence]) -> None:
        """Attach a boundary, refusing one fitted for the other direction.

        A query-side fence applied to the entry side is not a weaker defense, it is an
        arbitrary one: The query side's benign population is legitimate *queries* and the entry side's is
        genuine *entries*, so the boundary would neither block attacks nor hold the
        false-block budget, and nothing in the served output would look wrong. This is
        a property rather than a constructor check because callers attach the boundary
        after construction -- calibration happens once traffic exists.
        """
        if value is not None and value.direction != "entry":
            raise ValueError(
                f"fence was fitted for direction {value.direction!r} but this evaluator "
                f"serves 'entry' (the cached entry against the arriving query); refit it "
                f"on entry-side rows rather than reusing one calibrated for the query "
                f"side")
        if value is not None and value.statistic != self.config.statistic:
            raise ValueError(
                f"fence was fitted on {value.statistic!r} but the evaluator serves "
                f"{self.config.statistic!r}")
        self._fence = value

    def refit_fence(self, max_drift: float = 0.05, min_rows: int = 200) -> Optional[str]:
        """Refit the attached fence on the log's anchor tier, under the drift bound.

        This is the *only* supported refit path, and it exists because the obvious one
        is wrong. The calibration log deliberately holds every evaluated candidate,
        attacks included -- refitting on served traffic alone censors the upper tail --
        so feeding ``log.rows()`` to :meth:`ExcessFence.fit` fits a benign quantile over
        rows that are not benign, and the boundary rises exactly where the attacks are.
        The anchor tier is the subset an attacker cannot cheaply pollute: to place a row
        in it they must reach ``anchor_cosine`` against the arriving query, which is the
        near-duplication they were trying to avoid.

        Returns ``None`` when the new boundary was adopted, or a reason when it was not.
        The evaluator keeps serving its current fence in every rejected case, so a
        caller that ignores the return value degrades to "no refit", never to "no
        boundary".

        ``min_rows`` guards the other end: a quantile fitted on a handful of rows is
        noise, and the drift bound would happily adopt it.
        """
        if self._fence is None:
            return "no fence attached"
        rows = self.log.rows(anchor_only=True)
        if len(rows) < min_rows:
            return f"only {len(rows)} anchor rows, need {min_rows}"
        candidate, rejection = self._fence.bounded_refit(rows, max_drift=max_drift)
        if rejection is not None:
            return rejection
        self.fence = candidate
        return None

    # ---- GPTCache surface -------------------------------------------------
    def range(self) -> tuple[float, float]:
        """Delegate, so the host's hit threshold keeps its meaning."""
        return self.inner.range()

    def evaluation(self, src_dict: dict, cache_dict: dict, **kwargs: Any) -> float:
        if not self.config.enabled:
            return self.inner.evaluation(src_dict, cache_dict, **kwargs)
        try:
            return self._evaluate(src_dict, cache_dict, **kwargs)
        except Exception:
            # Never raise into the host, and never fail open.
            self.counters.errors += 1
            self.last_decision = DeletionDecision(
                blocked=True, reason="internal_error", cosine=float("nan"))
            return self._reject()

    # ---- decision path ----------------------------------------------------
    def _evaluate(self, src_dict: dict, cache_dict: dict, **kwargs: Any) -> float:
        """Gather what the ladder needs from GPTCache's two dicts, then run it.

        Everything host-specific happens here and the verdict happens in
        :func:`~sentry.cache.defense.decide.decide`: the query vector arrives as
        ``src_dict["embedding"]``, the entry text as ``cache_dict["question"]`` (the
        join key, because the host's ``vector_id`` is never learned by the write path),
        and the cosine from whichever source is available.
        """
        query_vector = _as_vector(src_dict.get("embedding"))
        profile = self.store.get(text_key(cache_dict.get("question")))
        if profile is not None and profile.has_answer:
            answer = cache_dict.get("answer")
            digest = (hashlib.sha256(normalise_answer(entry_text(answer)).encode("utf-8")).hexdigest()
                      if answer is not None else None)
            if digest is None or profile.answer_digest != digest:
                profile = replace(profile, answer_loss=None, echo_tokens=None, answer_digest=None)
        decision = decide(
            profile=profile,
            anchor=query_vector,
            cosine=self._cosine(cache_dict, profile, query_vector),
            fence=self.fence,
            config=self.config,
            counters=self.counters,
            log=self.log,
            query_text=entry_text(src_dict.get("question")),
        )
        self.last_decision = decision
        if decision.blocked:
            return self._reject()
        return self.inner.evaluation(src_dict, cache_dict, **kwargs)

    # ---- helpers ----------------------------------------------------------
    def _cosine(self, cache_dict: dict, profile: Optional[DeletionProfile],
                query_vector: Optional[np.ndarray]) -> float:
        """Cosine to the entry, from the best source available.

        The profile's whole vector is exact and is what the excess is measured against.
        The fallbacks exist only so the ``cache_threshold`` floor stays enforceable for
        an entry that has no profile -- that floor has to hold regardless of how the
        surrounding cache is tuned.
        """
        if profile is not None and query_vector is not None:
            return float(profile.whole @ unit(query_vector))
        entry_vector = _as_vector(cache_dict.get("embedding"))
        if entry_vector is not None and query_vector is not None:
            return float(unit(entry_vector) @ unit(query_vector))
        distance, _ = _search_result(cache_dict)
        return (self.config.cosine_from_distance(distance) if distance is not None
                else float("nan"))

    def _reject(self) -> float:
        """The host's "not similar enough" score, i.e. a miss."""
        return float(self.range()[0])


def _search_result(cache_dict: dict) -> tuple[Optional[float], Optional[str]]:
    """Unpack ``(distance, vector_id)``, tolerating absence and extra fields."""
    result = cache_dict.get("search_result")
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        return None, None
    distance = result[0]
    vector_id = result[1]
    distance = float(distance) if isinstance(distance, (int, float, np.floating)) else None
    return distance, None if vector_id is None else str(vector_id)


def _as_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    return array if array.size else None
