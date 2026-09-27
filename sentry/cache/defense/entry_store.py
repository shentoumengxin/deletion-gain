"""Side-car storage for per-entry deletion profiles.

An entry's shortened versions never change, so they are embedded once at insertion and
read at serving with nothing but dot products. That is what makes the entry side free
on the serving path, and this is where the vectors live.

**Keyed by the entry text, not by the host's vector id.** GPTCache assigns an id inside
``import_data`` and ``save`` does not return it, so the write path never learns it. Both
paths do see the text: the writer as ``save``'s ``question`` argument, the evaluator as
``eval_cache_data["question"]`` read back from the scalar store. The text is the join.

Two hazards get explicit handling, both inherited from the store this replaces. A
profile is only meaningful in the space that produced it *and* under the cut that
produced it, so every record carries an embedder name and a span-policy fingerprint and
a mismatch invalidates the record rather than mixing them -- a stale profile is a
confidently wrong answer, which is worse than a missed detection. And an entry the host
has evicted must not keep a profile that a later entry could be served against, so the
store supports reconciliation against the host's live key set.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np

from ._log import warn_once
from .deletion import DeletionProfile


def entry_text(value: Any) -> str:
    """The text of a cached entry, whatever wrapper the host handed over.

    GPTCache wraps a question in a ``Question`` dataclass whose ``__str__`` is its
    *repr*, so ``str(value)`` would produce ``"Question(content='...', deps=None)"`` on
    the write path and the bare text on the serving path. Nothing would ever match, and
    the only symptom would be a cache that never serves anything.

    The answer is wrapped too, and its payload has a *different* name: every GPTCache
    adapter that is not ``put`` hands ``data_manager.save`` an ``Answer`` dataclass whose
    text is ``.answer``, not ``.content``. Unwrapping only the question would leave the
    answer check reading ``"Answer(answer='...', answer_type=0)"`` -- an answer digest
    over a dataclass repr, and echo tokens containing the field names -- which nothing
    downstream could tell apart from a genuine answer.
    """
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    answer = getattr(value, "answer", None)
    if isinstance(answer, str):
        return answer
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def text_key(value: Any) -> str:
    """Stable identity for one entry's text.

    Whitespace is collapsed because the two paths can differ in incidental spacing, and
    nothing stronger is applied: both paths see the same string, so the key has to be
    stable, not canonical. Normalising harder (casefolding, punctuation stripping) would
    merge entries the host considers distinct.
    """
    return hashlib.sha256(
        " ".join(entry_text(value).split()).encode("utf-8")).hexdigest()


class ProfileStore(Protocol):
    """What the serving path and the insertion hook need from storage."""

    def get(self, key: str) -> Optional[DeletionProfile]: ...
    def put(self, key: str, profile: DeletionProfile) -> None: ...
    def delete(self, key: str) -> None: ...
    def reconcile(self, live_keys: Iterable[str]) -> int: ...


class InMemoryProfileStore:
    """Thread-safe in-memory store with LRU capacity and on-disk persistence.

    The serving path reads while the insertion path writes, so every mutation takes the
    lock.
    """

    def __init__(self, embedder: str, policy: str, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.embedder = str(embedder)
        self.policy = str(policy)
        self.capacity = int(capacity)
        self._lock = threading.RLock()
        self._profiles: dict[str, DeletionProfile] = {}
        self._recency: "OrderedDict[str, None]" = OrderedDict()

    # ---- reads -----------------------------------------------------------
    def get(self, key: str) -> Optional[DeletionProfile]:
        """The entry's profile, or None when absent or built under another config.

        A profile built under another embedder or another cut is dropped rather than
        returned, because a stale profile is a confidently wrong answer. But the caller
        cannot tell that apart from a cold entry -- both arrive as ``None`` and serving
        reports ``no_profile`` -- so the drop is logged. A configuration change
        invalidates *every* stored profile at once, and without a line here the only
        symptom is a cache that suddenly serves nothing and looks freshly started.
        """
        with self._lock:
            profile = self._profiles.get(key)
            if profile is None:
                return None
            if profile.embedder != self.embedder or profile.policy != self.policy:
                self._forget(key)
                warn_once(
                    f"stale_profile:{profile.embedder}|{profile.policy}"
                    f"->{self.embedder}|{self.policy}",
                    "deletion-test defense: dropping stored profiles built with "
                    "embedder=%r policy=%r; this store now holds embedder=%r policy=%r. "
                    "Every affected entry reports 'no_profile' and is unservable until "
                    "it is profiled again.",
                    profile.embedder, profile.policy, self.embedder, self.policy)
                return None
            self._touch(key)
            return profile

    def __len__(self) -> int:
        with self._lock:
            return len(self._profiles)

    def tracked_keys(self) -> set[str]:
        with self._lock:
            return set(self._recency)

    # ---- writes ----------------------------------------------------------
    def put(self, key: str, profile: DeletionProfile) -> None:
        with self._lock:
            if profile.embedder != self.embedder:
                raise ValueError(
                    f"profile embedder {profile.embedder!r} does not match store "
                    f"{self.embedder!r}")
            if profile.policy != self.policy:
                raise ValueError(
                    f"profile span policy {profile.policy!r} does not match store "
                    f"{self.policy!r}")
            self._profiles[key] = profile
            self._touch(key)
            self._enforce_capacity()

    def delete(self, key: str) -> None:
        with self._lock:
            self._forget(key)

    def reconcile(self, live_keys: Iterable[str]) -> int:
        """Drop records the host no longer holds; returns how many were dropped."""
        live = set(live_keys)
        with self._lock:
            stale = [key for key in self._recency if key not in live]
            for key in stale:
                self._forget(key)
            return len(stale)

    # ---- persistence -----------------------------------------------------
    def save(self, directory: Path | str) -> None:
        """Write to ``directory`` as one metadata file plus one array bundle."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            keys = list(self._profiles)
            arrays: dict[str, np.ndarray] = {}
            records = []
            for index, key in enumerate(keys):
                profile = self._profiles[key]
                arrays[f"whole_{index}"] = profile.whole
                arrays[f"spans_{index}"] = profile.spans
                arrays[f"dels_{index}"] = profile.deletions
                # The answer fields are optional per entry, so the array is written only
                # when there is one; the record's ``has_answer`` is what says whether to
                # look for it. All three are gated on that one predicate, on both sides
                # of the round trip: they are one fact about one answer, and a record
                # that kept the digest alone would name the answer it was built from
                # while carrying nothing the check can read against it. The token sets
                # go into the metadata as sorted lists
                # because ``allow_pickle=False`` -- which stays, a profile bundle is read
                # from disk by the serving process -- rules out an object array, and
                # sorting makes the file byte-stable across runs.
                if profile.has_answer:
                    arrays[f"aloss_{index}"] = profile.answer_loss
                records.append({
                    "key": key,
                    "span_names": list(profile.span_names),
                    "deletion_names": list(profile.deletion_names),
                    "words": profile.words,
                    "segment_count": profile.segment_count,
                    "embedder": profile.embedder,
                    "policy": profile.policy,
                    "min_segments": profile.min_segments,
                    "storage_dtype": profile.storage_dtype,
                    "has_answer": profile.has_answer,
                    "echo_tokens": ([sorted(tokens) for tokens in profile.echo_tokens]
                                    if profile.has_answer else None),
                    "answer_digest": (profile.answer_digest if profile.has_answer
                                      else None),
                })
            meta = {
                "embedder": self.embedder,
                "policy": self.policy,
                "capacity": self.capacity,
                "profiles": records,
                "recency": list(self._recency),
            }
        (directory / "profiles.npz").unlink(missing_ok=True)
        np.savez_compressed(directory / "profiles.npz", **arrays)
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> "InMemoryProfileStore":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store = cls(meta["embedder"], meta["policy"],
                    capacity=int(meta.get("capacity", 100_000)))
        with np.load(directory / "profiles.npz", allow_pickle=False) as arrays:
            for index, record in enumerate(meta.get("profiles", [])):
                # A store written before the answer check existed carries none of the
                # three keys, and ``has_answer`` is then absent rather than False. It
                # loads with the fields unset, which the serving path already handles:
                # the veto is decided on the deletion gain alone and counted in
                # ``no_answer_fields``.
                has_answer = bool(record.get("has_answer"))
                store._profiles[record["key"]] = DeletionProfile(
                    whole=arrays[f"whole_{index}"],
                    spans=arrays[f"spans_{index}"],
                    deletions=arrays[f"dels_{index}"],
                    span_names=tuple(record["span_names"]),
                    deletion_names=tuple(record["deletion_names"]),
                    words=int(record["words"]),
                    segment_count=int(record["segment_count"]),
                    embedder=record["embedder"],
                    policy=record["policy"],
                    min_segments=int(record["min_segments"]),
                    storage_dtype=record.get("storage_dtype", str(arrays[f"spans_{index}"].dtype)),
                    answer_loss=arrays[f"aloss_{index}"] if has_answer else None,
                    echo_tokens=(tuple(frozenset(tokens)
                                       for tokens in record["echo_tokens"])
                                 if has_answer else None),
                    answer_digest=(record.get("answer_digest") if has_answer
                                   else None),
                )
        for key in meta.get("recency", []):
            store._recency[key] = None
        for key in store._profiles:
            store._recency.setdefault(key, None)
        return store

    # ---- internals -------------------------------------------------------
    def _touch(self, key: str) -> None:
        self._recency.pop(key, None)
        self._recency[key] = None

    def _forget(self, key: str) -> None:
        self._profiles.pop(key, None)
        self._recency.pop(key, None)

    def _enforce_capacity(self) -> None:
        while len(self._recency) > self.capacity:
            oldest, _ = self._recency.popitem(last=False)
            self._profiles.pop(oldest, None)
