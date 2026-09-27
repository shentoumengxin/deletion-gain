"""The insertion hook: embed an entry's shortened versions once, and keep them.

This hook renders **no verdict**. It is not an admission controller and it never
refuses a write.

That is a deliberate narrowing. The direction this defense ships scores the
cached *entry* against the arriving benign query -- because the attacker owns the entry,
not the query: the payload is planted, ordinary users collide into it afterwards, and
the victim's text is clean, so scoring the arriving query finds nothing. At insertion
there is no arriving query to anchor against. Screening the candidate against its
nearest neighbour instead would be the query-side computation wearing an insertion-shaped hat,
and it would need its own benign population and its own fitted fence. See
``docs/ARCHITECTURE.md`` §3.3 and §11.

What this hook buys is the whole reason the entry side costs nothing at serving: an entry's
shortened versions never change, so the policy's shortened texts are embedded here, once,
and serving scores the stored candidate vectors.

The entry's own cached answer is read here too, and for the same reason. Given one, the
profile carries what the answer check reads at serving -- a scalar per variant and a
small token set per variant, both derived from the answer -- so the second witness costs
one more embedder call at insertion and no model call at all when a query arrives.

**Every failure leaves the entry written and unprofiled.** An unprofiled entry is
unservable under the serving path's fail-closed rule, which is the conservative
outcome; refusing the write instead would be a question the cache can never learn, and
this hook has no evidence on which to make that call. Silent is not the same as free,
though: a failure that repeats on every write drives the whole cache to `no_profile`
misses while looking like nothing worse than a cold start, so each distinct cause is
also reported once through ``_log.warn_once`` -- the counters below are for dashboards,
the log line is for the operator who has to find out why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ._log import warn_once
from .deletion import DEFAULT_STORAGE_DTYPE, Embedder, build_profile
from .entry_store import ProfileStore, entry_text, text_key
from .spans import SpanPolicy


@dataclass
class InsertionCounters:
    """Cheap observability; all monotone."""

    writes: int = 0
    profiled: int = 0
    skipped_short: int = 0
    #: Entries profiled without their answer because the answer alone would not encode.
    #: Counted apart from ``errors`` because the two ask for different things: this entry
    #: is servable and its veto falls back to deletion gain, while an error means nothing
    #: is being profiled at all.
    answer_dropped: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class ProfileWritingDataManager:
    """Wraps a GPTCache ``DataManager`` so every new entry gets a deletion profile.

    Delegates everything except ``save``. Like the serving veto it only ever *adds* a
    side effect and never originates or suppresses a write, so removing the wrapper
    restores the host exactly.
    """

    def __init__(self, inner: Any, store: ProfileStore, embedder: Embedder,
                 policy: SpanPolicy,
                 counters: Optional[InsertionCounters] = None,
                 storage_dtype: str = DEFAULT_STORAGE_DTYPE) -> None:
        self.inner = inner
        self.store = store
        self.embedder = embedder
        self.policy = policy
        self.counters = counters or InsertionCounters()
        # What the stored vectors are rounded to. A deployment carrying a large cache
        # wants "float16" -- it halves the per-entry cost and moves `excess` by ~1e-4
        # against a fence near 7e-3 -- but the default stays exact because parity with
        # the reference analysis code is a red line. See `deletion.STORAGE_DTYPES`.
        self.storage_dtype = storage_dtype
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def close(self) -> None:
        """Close the owned host once, including GPTCache's atexit callback."""
        if not self._closed:
            self._closed = True
            close = getattr(self.inner, "close", None)
            if close is not None:
                close()

    def _build(self, text: str, whole: Any, answer: Optional[str]):
        """Profile the entry, dropping the *answer* rather than the profile if it fails.

        The two failures reachable here are not the same failure. ``build_profile``
        encodes the answer in a call of its own, so an answer past the model's context
        window -- or a transient error on that one call -- takes down the answer and
        nothing else; the deletion test needs none of it. Letting that take the profile
        with it would leave the entry written, unprofiled and permanently unservable
        under the fail-closed rule: a strictly worse cache than the one that shipped
        before the answer check existed, caused by the entry's own answer rather than by
        anything wrong with the defense.

        So the answer is dropped and the profile rebuilt without it, which is exactly the
        deletion-gain-only entry this hook wrote until now. The degradation is counted and
        logged under its own key, because "this entry's answer would not encode" and "the
        embedder is down and nothing is being profiled" want different actions from the
        operator and would otherwise arrive as the same line. The rebuild re-encodes the
        variants, which is waste -- on a path that has already failed once.

        The degradation is reported only when it *worked*. If the retry fails too, the
        answer was never the problem and what gets reported is the original exception,
        unchanged from what a dead embedder produced before this branch existed.
        """
        try:
            return build_profile(text, self.embedder, self.policy, whole=whole,
                                 storage_dtype=self.storage_dtype, answer=answer)
        except Exception as exc:
            first = exc

        if answer is not None:
            try:
                profile = build_profile(text, self.embedder, self.policy, whole=whole,
                                        storage_dtype=self.storage_dtype, answer=None)
            except Exception:
                pass
            else:
                self.counters.answer_dropped += 1
                warn_once(
                    f"insertion_answer_encode_error:{self.embedder.model_name}:"
                    f"{type(first).__name__}",
                    "deletion-test defense: insertion could not encode an entry's "
                    "stored answer (embedder=%r): %s. The entry is profiled and "
                    "servable, but without answer fields, so its vetoes are decided on "
                    "deletion gain alone and counted in 'no_answer_fields'. If this "
                    "repeats, the answer check is configured and idle on this cache.",
                    self.embedder.model_name, first)
                return profile

        # An unusable embedder must not cost the host a write. The entry lands without a
        # profile and is unservable until one is built for it. If this is not a one-off
        # (a truncation, a transient timeout) but the embedder itself -- swapped, downed,
        # reconfigured -- every subsequent write goes unprofiled the same way and the
        # cache quietly stops serving. That is the counter-only gap this warns about:
        # distinct per embedder+policy so a fleet-wide misconfiguration says so once
        # instead of never.
        self.counters.errors += 1
        warn_once(
            f"insertion_build_profile_error:{self.embedder.model_name}:"
            f"{self.policy.fingerprint()}:{type(first).__name__}",
            "deletion-test defense: insertion could not build a deletion "
            "profile (embedder=%r policy=%r): %s. The entry is written "
            "unprofiled and unservable; if this repeats on every write, the "
            "embedder itself is unusable and no entry is being profiled.",
            self.embedder.model_name, self.policy.fingerprint(), first)
        return None

    def save(self, question: Any, answer: Any, embedding_data: Any, **kwargs: Any) -> None:
        text = entry_text(question)
        # The answer is what the second witness reads, and this hook is the only place on
        # the write path that sees it. It is unwrapped the same way the question is --
        # GPTCache hands most adapters an ``Answer`` dataclass.
        profile = self._build(text, embedding_data,
                              entry_text(answer) if answer is not None else None)

        # A failed host write must leave its old answer and profile usable. Once
        # the host has committed, invalidate the old profile before installing the
        # replacement. The serving hook also binds rescue fields to the actual answer
        # digest, so a failing external store cannot rescue with a stale answer.
        self.inner.save(question, answer, embedding_data, **kwargs)
        self.counters.writes += 1
        delete = getattr(self.store, "delete", None)
        if delete is not None:
            try:
                delete(text_key(question))
            except Exception as exc:
                self.counters.errors += 1
                warn_once(
                    f"insertion_delete_error:{type(exc).__name__}",
                    "deletion-test defense: old profile invalidation failed (%s); "
                    "the host write succeeded, and answer-digest validation disables "
                    "stale rescue fields.", type(exc).__name__)

        if profile is None:
            return
        if not profile.judgeable:
            # Too short to cut. Written, unprofiled, and therefore unservable -- see
            # the spec's §6 for why that trade is small at a benign median of 9 words.
            self.counters.skipped_short += 1
            return

        try:
            self.store.put(text_key(question), profile)
        except Exception as exc:
            # The store refuses a profile whose embedder or span policy does not match
            # its own, and a writer wired to a policy the store was not constructed for
            # would otherwise raise *after* the host's write -- turning every cache miss
            # into an application-level exception out of GPTCache's ``adapt``. This hook
            # renders no verdict and must cost the host nothing: an unstorable profile
            # leaves the entry written and unprofiled, which is the same conservative
            # outcome as an unusable embedder.
            #
            # This is the drift scenario the review flagged: a store built under one
            # policy and a writer installed under another, with nothing cross-checking
            # them (``install_profile_writer`` now does, see below -- but a store can
            # still be wired up by hand). Once it happens it happens on *every* write,
            # so counting it is not evidence anyone will see; naming both sides' own
            # idea of embedder and policy is what makes the mismatch diagnosable from
            # the log line alone.
            store_embedder = getattr(self.store, "embedder", "?")
            store_policy = getattr(self.store, "policy", "?")
            self.counters.errors += 1
            warn_once(
                f"insertion_store_put_error:{self.embedder.model_name}:"
                f"{self.policy.fingerprint()}:{store_embedder}:{store_policy}",
                "deletion-test defense: insertion could not store a profile -- "
                "writer is embedder=%r policy=%r, store is embedder=%r policy=%r "
                "(%s). Every write from here on lands unprofiled and every lookup "
                "against it reports 'no_profile': the store and this writer's "
                "policy have drifted apart and need to be rebuilt from the same "
                "embedder and span policy.",
                self.embedder.model_name, self.policy.fingerprint(),
                store_embedder, store_policy, exc)
            return
        self.counters.profiled += 1


def install_profile_writer(cache: Any, store: ProfileStore, embedder: Embedder,
                           policy: SpanPolicy,
                           storage_dtype: str = DEFAULT_STORAGE_DTYPE) -> ProfileWritingDataManager:
    """Put profile construction in front of ``cache``'s writer.

    The store handed here must be the same object the serving evaluator reads, or the
    profiles this writes are never found.

    Raises immediately if ``store`` already carries its own idea of embedder and
    policy (:class:`~sentry.cache.defense.entry_store.InMemoryProfileStore` and
    anything shaped like it) and it disagrees with what is handed here. That
    combination is exactly the "plausible slip" the review flagged: nothing else
    cross-checks the store's construction-time config against the writer's, so
    without this every write would instead fail one at a time, deep in ``save``,
    with only a log line (see ``ProfileWritingDataManager.save``) to notice by. A
    store built generically enough to skip ``embedder``/``policy`` attributes
    (matching the bare :class:`ProfileStore` protocol) is not checked here -- it is
    checked on every ``put`` instead, which is the best this function can do without
    assuming more than the protocol promises.
    """
    store_embedder = getattr(store, "embedder", None)
    store_policy = getattr(store, "policy", None)
    if store_embedder is not None and store_embedder != str(embedder.model_name):
        raise ValueError(
            f"install_profile_writer: store embedder {store_embedder!r} does not "
            f"match writer embedder {embedder.model_name!r}; every write would "
            f"fail store.put() from here on and land unprofiled -- rebuild the "
            f"store and the writer from the same embedder")
    if store_policy is not None and store_policy != policy.fingerprint():
        raise ValueError(
            f"install_profile_writer: store policy {store_policy!r} does not "
            f"match writer policy {policy.fingerprint()!r}; every write would "
            f"fail store.put() from here on and land unprofiled -- rebuild the "
            f"store and the writer from the same span policy")
    guard = ProfileWritingDataManager(cache.data_manager, store, embedder, policy,
                                      storage_dtype=storage_dtype)
    cache.data_manager = guard
    return guard
