"""The deletion test: does some shortened version of a text match the anchor better?

    excess = max over contiguous sub-spans S of  cos(e_S, e_k)  -  cos(e_t, e_k)

Every part of a genuine question is about the same thing, so removing any part can
only make the match worse and ``excess`` is at or below zero. A collision is a
question part plus a payload; removing the payload leaves the question part, which
matches the anchor better than the two together, and ``excess`` is positive.

Nothing is generated. The only model involved is the embedder, and it only ever
*reads* the text -- it is never asked to follow it, which matters because the text
under inspection may be an instruction.

**One function, two sides.** ``excess(profile, anchor)`` does not know which side it is
serving. The *query side* profiles the arriving query and anchors on the entry it nearly
matched; the *entry side* profiles the cached entry and anchors on the arriving query.
The entry side is what the threat model needs -- the payload is already cached and the
victim's query is clean, so scoring the query finds nothing -- and it is the cheap one,
because an entry's spans never change and are computed once at insertion.

**The served statistic is ``excess_span``**, the maximum over prefixes and suffixes.
``excess_del`` is computed and stored but decides nothing: dropping an interior
segment leaves two disjoint pieces, which is not the contiguous run the method is
defined over. See ``docs/METHOD.md`` §1 and ``tests/reference/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .spans import SpanPolicy, shortened
from .textnorm import content_tokens, normalise_answer

#: How the answer check may rule on a veto DG raised.
#:
#: ``excess`` says some shortened version of the entry matches the arriving query better
#: than the whole entry does -- the content that version dropped is not what the query
#: asked for. That alone does not say the content *did* anything: benign padding is
#: removable too, and vetoing on it is a false block. The entry's own cached answer is
#: the second witness. If the removed content shaped that answer -- the entry's match to
#: it falls when the content goes, or the answer repeats the content's own words -- then
#: what is cached answers something this query never asked and the veto stands. If the
#: removed content cost the answer nothing, it was inert and the veto is rescued.
#:
#: - ``"none"``  -- no second witness; DG alone decides, which is what shipped before.
#: - ``"adl"``   -- veto only when the answer loss clears ``eta_a``.
#: - ``"echo"``  -- veto only when the answer repeats content words the query never used.
#: - ``"either"`` -- veto when either witness says the removed content shaped the answer.
ANSWER_RULES = ("none", "adl", "echo", "either")

#: Precisions a stored profile may be kept at. Storage is ``n_variants x D`` numbers per
#: entry and the widened variant sets of ``form="runs"`` / ``mode="multi"`` make that the
#: dominant cost, so it is worth halving -- but only if the rounding is invisible against
#: the quantity being thresholded, which is a fitted fence around ``+0.007`` in cosine.
#:
#: ``float16`` moves ``excess`` by at most ``1.4e-4`` on the comqa corpus, 52x below the
#: fence, and the realised block rate does not move.
#:
#: ``int8`` is deliberately **not** offered. For a 384-dimensional unit vector the largest
#: component is ~0.15, so a symmetric per-vector step is ~0.15/127 and the accumulated
#: cosine error lands near ``4.5e-3`` -- the fence's own order. Measured, it costs 1.8
#: points of block rate. That is not a cheaper defense, it is a broken one.
STORAGE_DTYPES = {"float64": np.float64, "float32": np.float32,
                  "float16": np.float16}

#: What a profile is stored at unless the caller says otherwise.
#:
#: **float64, and the reason is a red line rather than caution.**
#: ``tests/reference/deletion_reference.py`` is the analysis code that produced the
#: paper's numbers and parity with it is asserted to 1e-12; a lossy storage precision
#: cannot satisfy that by construction, since its entire justification is that the error
#: is small rather than absent. So the shipped default reproduces the reference exactly
#: and ``float16`` is opt-in, with its error *bounded* by a test instead of being assumed
#: (``tests/test_deletion_statistic.py``).
DEFAULT_STORAGE_DTYPE = "float64"


def store_rows(rows: np.ndarray, storage_dtype: str) -> np.ndarray:
    """Round ``rows`` to the precision a profile is kept at.

    Reads promote back to float64 automatically, so nothing downstream has to know: the
    only visible effect is the rounding, which :data:`STORAGE_DTYPES` bounds.
    """
    if storage_dtype not in STORAGE_DTYPES:
        raise ValueError(
            f"unknown storage dtype {storage_dtype!r}; expected one of "
            f"{sorted(STORAGE_DTYPES)} (int8 is excluded on purpose -- its cosine error "
            f"is the same order as the fence)")
    return np.asarray(rows, dtype=STORAGE_DTYPES[storage_dtype])


class Embedder(Protocol):
    """What building a profile needs. Matches ``sentry.research.pipeline.embed``."""

    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class CallableEmbedder:
    """Adapts a single-text embedding callable to the batched protocol.

    GPTCache hands deployments an ``embedding_func`` that takes one string. Wrapping it
    means the defense can be installed without also being handed the underlying model,
    at the cost of losing the batching that ``docs/METHOD.md`` §1's cost claim
    assumes -- 16 sequential forward passes rather than one batch of 16. A deployment
    that cares about insertion latency should pass the model itself.
    """

    def __init__(self, fn: Callable[[str], Any], model_name: str) -> None:
        self.fn = fn
        self.model_name = str(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([
            np.asarray(self.fn(text), dtype=float).reshape(-1) for text in texts])


def unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` scaled to unit length; unchanged if it is degenerate.

    Cosines are read as dot products everywhere in this module, so normalisation is
    the caller's contract made explicit rather than an assumption about the embedder.
    """
    array = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array if norm < 1e-12 else array / norm


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


@dataclass(frozen=True)
class DeletionProfile:
    """A text's shortened versions, already embedded.

    Built once, at insertion, for the *cached entry* -- that is what the entry side
    scores, and an
    entry's shortened versions never change -- and then read at serving with nothing but
    dot products against whatever query arrives. Nothing profiles the arriving query:
    that would be the query side, which this project measures but does not deploy.

    ``embedder`` and ``policy`` travel with the vectors because a profile is only
    meaningful against a fence fitted on the same pair, and the failure when they
    disagree is silent rather than loud.
    """

    whole: np.ndarray
    #: ``(2n - 2, D)`` prefix and suffix vectors -- the block the served statistic reads.
    spans: np.ndarray
    #: ``(n, D)`` one-segment-deletion vectors -- diagnostic only.
    deletions: np.ndarray
    span_names: tuple[str, ...]
    deletion_names: tuple[str, ...]
    words: int
    segment_count: int
    embedder: str
    policy: str
    min_segments: int
    #: Precision the vectors above are kept at; see :data:`STORAGE_DTYPES`. It is *not*
    #: part of :meth:`SpanPolicy.fingerprint` and deliberately does not invalidate a
    #: fence: the rounding is two decades below what a fence thresholds, so a boundary
    #: fitted at one precision is valid at the other. Recorded for provenance.
    storage_dtype: str = DEFAULT_STORAGE_DTYPE
    #: Per span variant: how much the entry's cosine to its own stored answer falls when
    #: that variant's complement is removed, ``cos(k, y) - cos(s, y)``. None when the
    #: profile was built without an answer. Computed once here, so serving reads one
    #: number rather than embedding anything. Kept at ``storage_dtype``.
    answer_loss: Optional[np.ndarray] = None
    #: Per span variant, aligned with :attr:`spans`: content words of the complement --
    #: what that variant dropped -- that also occur in the answer.
    echo_tokens: Optional[tuple[frozenset[str], ...]] = None
    #: sha256 of the normalised answer the two fields above were computed from, so a
    #: profile can be matched against the answer the host actually holds.
    answer_digest: Optional[str] = None

    @property
    def has_answer(self) -> bool:
        """Whether the answer check has anything to read on this profile."""
        return self.answer_loss is not None and self.echo_tokens is not None

    @property
    def dimension(self) -> int:
        return int(self.whole.shape[0])

    @property
    def judgeable(self) -> bool:
        """Whether this text has enough segments for a deletion to mean anything."""
        return (self.segment_count >= self.min_segments
                and self.spans.size > 0 and self.deletions.size > 0)


def build_profile(text: str, embedder: Embedder, policy: SpanPolicy,
                  whole: Optional[np.ndarray] = None,
                  min_segments: Optional[int] = None,
                  storage_dtype: str = DEFAULT_STORAGE_DTYPE,
                  answer: Optional[str] = None) -> DeletionProfile:
    """Cut ``text``, embed every shortened version, and keep the vectors.

    ``whole`` lets the caller supply a vector it already has -- the insertion path is
    handed the host's own embedding of the entry, and re-encoding it would be waste and
    a chance to disagree with the host about what the entry's vector is. It is
    re-normalised regardless of what the host did.

    ``answer`` is the entry's own cached answer. Given one, the profile also carries what
    the answer check reads: per variant, how much the entry's match to that answer falls
    when the variant's complement is removed, and which of the removed content words the
    answer repeats.

    **The statistic itself does not see it.** The text-and-variants batch is exactly the
    texts, in exactly the order, it would be without an answer, so the span vectors and
    every DG quantity read off them are bit-identical with and without one. The answer is
    encoded in a call of its own, which is what the design record budgets. Putting it in
    the variant batch would not be free: a real embedder pads a batch to its longest item,
    and a cached LLM answer is far longer than any variant, so every variant row would be
    computed at a different padded width -- bit-identity gone, and a wide batch costing
    more than the second call it saved.
    """
    # One helper decides what "shortened version" means, so insertion and serving can
    # never disagree about it. It also honours the policy's `form` and `multi` components,
    # which is where the two narrowings of §1's definition were fixed.
    spans = shortened(policy, text)
    floor = policy.min_segments if min_segments is None else int(min_segments)

    # An answer that normalises to nothing -- empty, or markdown scaffolding only -- is
    # no answer. Storing a loss against a degenerate vector would be a number the check
    # could read and believe.
    cleaned = normalise_answer(answer) if answer is not None else ""

    to_encode = list(spans.all_texts)
    if whole is None:
        to_encode = [text] + to_encode

    encoded = (embedder.encode(to_encode) if to_encode
               else np.zeros((0, 0), dtype=float))
    if whole is None:
        whole_vector = unit(encoded[0])
        variants = encoded[1:]
    else:
        whole_vector = unit(whole)
        variants = encoded

    span_count = len(spans.span_names)
    variants = _unit_rows(variants) if variants.size else np.zeros(
        (0, whole_vector.shape[0]), dtype=float)

    answer_loss = echo_tokens = answer_digest = None
    if cleaned:
        answer_vector = unit(embedder.encode([cleaned])[0])
        base = float(whole_vector @ answer_vector)
        answer_loss = store_rows(base - variants[:span_count] @ answer_vector,
                                 storage_dtype)
        answer_words = content_tokens(cleaned)
        echo_tokens = tuple(content_tokens(gone) & answer_words
                            for gone in spans.removed_texts)
        answer_digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    return DeletionProfile(
        whole=store_rows(whole_vector, storage_dtype),
        spans=store_rows(variants[:span_count], storage_dtype),
        deletions=store_rows(variants[span_count:], storage_dtype),
        span_names=spans.span_names,
        deletion_names=spans.deletion_names,
        words=len(text.split()),
        segment_count=spans.segment_count,
        embedder=str(embedder.model_name),
        policy=policy.fingerprint(),
        min_segments=floor,
        storage_dtype=storage_dtype,
        answer_loss=answer_loss,
        echo_tokens=echo_tokens,
        answer_digest=answer_digest,
    )


@dataclass(frozen=True)
class ExcessReading:
    """What the deletion test concluded about one (text, anchor) pair."""

    base_cos: float
    #: The served statistic: best prefix or suffix, minus the whole text.
    excess_span: float
    #: Diagnostic: best one-segment deletion, minus the whole text. Decides nothing.
    excess_del: float
    best_name: str
    #: ``"prefix"`` or ``"suffix"`` -- which end the winning span kept.
    best_end: str
    #: Fraction of the text's segments the winning span kept.
    best_kept: float
    words: int
    #: Row of :attr:`DeletionProfile.spans` the winning variant is, so the answer fields
    #: precomputed for it can be read back without re-deriving anything from the name.
    best_index: int = 0
    #: ``answer_loss`` of the winning variant; None when the profile carries no answer.
    answer_loss: Optional[float] = None
    #: Content words the winning variant dropped that the stored answer contains; None
    #: when the profile carries no answer.
    echo_tokens: Optional[frozenset[str]] = None


def excess(profile: DeletionProfile, anchor: np.ndarray) -> ExcessReading:
    """Read ``profile`` against ``anchor``. Pure dot products; no model call.

    On the deployed entry side ``profile`` describes the cached entry and ``anchor``
    is the arriving benign query's vector.

    Raises on an unjudgeable profile rather than returning a sentinel. A text with too
    few segments has no meaningful deletion, and any number returned for it would read
    as "nothing found" -- which is the *serve* verdict, the opposite of what the
    fail-closed rule requires. The serving hook has to route such an entry to its own
    ``unjudgeable`` branch (a miss by default), and the insertion hook has to notice it
    before it stores anything, so a caller that forgot to route one is told rather than
    handed a plausible-looking zero.
    """
    if not profile.judgeable:
        raise ValueError(
            f"unjudgeable profile: {profile.segment_count} segments is below the "
            f"floor of {profile.min_segments}; route it to the hook's policy for "
            f"texts too short to cut")

    direction = unit(anchor)
    base = float(profile.whole @ direction)

    span_scores = profile.spans @ direction
    best_index = int(np.argmax(span_scores))
    best_name = profile.span_names[best_index]
    end, kept = _describe(best_name, profile.segment_count)

    return ExcessReading(
        base_cos=base,
        excess_span=float(span_scores[best_index]) - base,
        excess_del=float((profile.deletions @ direction).max()) - base,
        best_name=best_name,
        best_end=end,
        best_kept=kept,
        words=profile.words,
        best_index=best_index,
        answer_loss=(float(profile.answer_loss[best_index])
                     if profile.has_answer else None),
        echo_tokens=(profile.echo_tokens[best_index] if profile.has_answer else None),
    )


def _describe(name: str, segments: int) -> tuple[str, float]:
    """Which end the winning variant kept, and what share of the segments it is.

    ``docs/METHOD.md`` §3.4's mechanism claim rests on this: for the append
    families the winning variant is a *prefix* 93-98% of the time, which is what says the
    statistic is finding the question rather than scoring noise.

    Three name shapes exist because a policy can widen the variant set. ``pre<k>`` and
    ``suf<k>`` are the original prefixes and suffixes; ``run<i>_<j>`` is a contiguous run
    under ``form="runs"``; and a ``multi`` policy prefixes either with its component's
    fingerprint, which itself contains colons. Parsing this by slicing a fixed offset --
    which is what the code did -- reads ``count:6:run3_6`` as an integer and raises. The
    interior runs of the general form are neither end, so they report ``"interior"``
    rather than being forced into one.
    """
    head, _, bare = name.rpartition(":")
    bare = bare or name
    if "#" in head:
        # A multi policy stamps the component's own segment count into the name, because
        # a fine cut has more segments than the primary one and the share must be against
        # the cut the variant actually came from.
        segments = int(head.rsplit("#", 1)[-1])
    if bare.startswith("pre"):
        cut = int(bare[3:])
        return "prefix", cut / segments
    if bare.startswith("suf"):
        cut = int(bare[3:])
        return "suffix", (segments - cut) / segments
    if bare.startswith("run"):
        start, end = (int(v) for v in bare[3:].split("_"))
        kept = (end - start) / segments
        if start == 0:
            return "prefix", kept
        if end == segments:
            return "suffix", kept
        return "interior", kept
    raise ValueError(f"unrecognised variant name {name!r}")


@dataclass(frozen=True)
class AnswerVerdict:
    """Whether the entry's own cached answer backs the veto DG raised."""

    #: Whether the veto stands. None means the profile carried no answer to ask.
    fires: Optional[bool]
    #: Echoed content words left after the query's own vocabulary is subtracted.
    echo: Optional[int]
    answer_loss: Optional[float]
    reason: str


def answer_check(reading: Optional[ExcessReading], query_text: str, rule: str,
                 eta_a: float, echo_min: int) -> AnswerVerdict:
    """Does the content DG found removable also matter to the stored answer?

    ``rule="none"`` always fires (DG-only, what shipped before). Otherwise the query's
    own content words are subtracted from the winning variant's echo set first -- a word
    the query itself asked about is not something the query "did not need" -- and the
    configured rule is applied. A reading without answer fields returns ``fires=None``
    so the caller can fall back to the DG-only rule and count that it did.

    Pure set arithmetic on numbers computed at insertion: no model call, and no second
    pass over the entry's spans.
    """
    if rule not in ANSWER_RULES:
        raise ValueError(f"unknown answer rule {rule!r}; expected one of {ANSWER_RULES}")
    if echo_min < 1:
        # Zero would make every reading "echo" -- an answer-checked rule collapsing to
        # DG-only while still reporting itself as answer-checked, which is the quiet
        # wrong number this project refuses to produce.
        raise ValueError(f"echo_min must be at least 1; got {echo_min}")
    if rule == "none":
        return AnswerVerdict(True, None, None, "rule_none")
    if reading is None or reading.answer_loss is None or reading.echo_tokens is None:
        return AnswerVerdict(None, None, None, "no_answer_fields")
    echo = len(reading.echo_tokens - content_tokens(query_text))
    by_loss = reading.answer_loss > eta_a
    by_echo = echo >= echo_min
    fires = {"adl": by_loss, "echo": by_echo, "either": by_loss or by_echo}[rule]
    return AnswerVerdict(bool(fires), echo, reading.answer_loss,
                         f"rule_{rule}:{'fires' if fires else 'clean'}")
