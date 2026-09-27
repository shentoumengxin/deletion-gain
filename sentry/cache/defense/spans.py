"""Cutting a text into contiguous spans, and the shortened versions built from them.

The deletion test asks whether some shortened version of a text matches an anchor
better than the whole text does. This module decides what "shortened version" means,
and nothing else — no vectors, no embedder, no policy about what to do with the answer.

Two ways to cut, and the choice is not cosmetic:

- **A fixed count** (``n``, default 6) makes every deletion remove the same *fraction*
  of the text, so a 41-word plant and a 9-word question are asked the same question.
  This is the configuration behind the per-family tables in ``docs/METHOD.md``
  §3.1 and §3.3.
- **A fixed width** (3 words) makes every segment carry the same amount of text.
  This is the configuration behind the pooled numbers in §3.1 and all of §3.5. Its
  cost grows with length, which ``max_segments`` bounds.

Which is right is a measured question and the owner intends to sweep it, so the policy
is a value that travels with the data it produced: :meth:`SpanPolicy.fingerprint` is
recorded on every stored profile and every fitted fence, and a mismatch is refused.
A fence fitted under ``count:6`` and applied to ``width:3`` profiles raises no error,
blocks no attacks, and holds no false-block budget.

Parity with ``tests/reference/deletion_reference.py`` is a red line: that file is the
analysis code that produced the paper's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Segments per text in ``count`` mode. ``deletion_test.py``'s default.
DEFAULT_SEGMENTS = 6
#: Words per segment in ``width`` mode. ``deletion_profile.py``'s three-word setting.
DEFAULT_WIDTH = 3
#: Below this many segments a text has no meaningful deletion. The offline scripts
#: dropped such rows; a deployment cannot, so it marks them unjudgeable instead.
MIN_SEGMENTS = 3

_MODES = ("count", "width", "multi")

#: Which shortened versions the *served* statistic reads.
#:
#: ``excess`` is the max over **every contiguous sub-span**. A payload at both ends leaves
#: no clean prefix and no clean suffix but a clean *interior* run, which a prefix/suffix-only
#: form never inspects. ``runs`` implements the contiguous-window definition; ``span`` is
#: the prefix/suffix form.
_FORMS = ("span", "runs")

#: Component cuts of a ``multi`` policy. A single granularity cannot be right for both a
#: 20-word payload and a 2-word one: a fixed count makes every deletion drop the same
#: *fraction*, so a payload shorter than one span never gets isolated and `excess` cannot
#: see it (`RQ6`). Since ``excess`` is a max over sub-spans, taking it over the union of
#: several cuts is the same operator on a larger set -- monotone, so it can only find
#: more, at the price of a higher benign fence that has to be re-measured.
DEFAULT_MULTI = ("count:6", "width:2:cap16")


@dataclass(frozen=True)
class SpanPolicy:
    """How a text is cut. Frozen, and identified by :meth:`fingerprint`."""

    mode: str = "count"
    n: int = DEFAULT_SEGMENTS
    width: int = DEFAULT_WIDTH
    #: ``width`` mode only: fall back to equal division once a text would exceed this
    #: many segments, so cost and storage stay bounded on long texts.
    max_segments: Optional[int] = None
    #: Fewer segments than this and the text is unjudgeable.
    min_segments: int = MIN_SEGMENTS
    #: Which shortened versions the served statistic reads; see :data:`_FORMS`.
    form: str = "span"
    #: ``multi`` mode only: component cut fingerprints, unioned. See :data:`DEFAULT_MULTI`.
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {_MODES}")
        if self.form not in _FORMS:
            raise ValueError(f"unknown form {self.form!r}; expected one of {_FORMS}")
        if self.mode == "multi" and not self.components:
            # Empty means "the default pair", so a caller can ask for multi-resolution
            # without restating what it is. Resolved here rather than at the field default
            # because the field default has to stay empty: it is also what every
            # single-cut policy carries, and a non-empty default made those raise.
            object.__setattr__(self, "components", DEFAULT_MULTI)
        if self.mode == "multi" and len(self.components) < 2:
            raise ValueError("multi mode needs at least two component cuts")
        if self.mode != "multi" and self.components:
            raise ValueError("components are only meaningful in multi mode")
        if self.n < 2:
            raise ValueError("n must be at least 2; one segment has no deletion")
        if self.width < 1:
            raise ValueError("width must be at least 1 word")
        if self.min_segments < 2:
            raise ValueError("min_segments must be at least 2")
        if self.max_segments is not None and self.max_segments < self.min_segments:
            raise ValueError("max_segments must not be below min_segments")

    def fingerprint(self) -> str:
        """Stable identity of the *cut*, for matching profiles against fences.

        ``min_segments`` is deliberately excluded: it decides which texts are judgeable,
        not where the cuts fall, so two policies differing only in it produce identical
        geometry and must share a fence rather than silently invalidating one.
        """
        if self.mode == "multi":
            cut = "multi[" + "+".join(self.components) + "]"
        elif self.mode == "count":
            cut = f"count:{self.n}"
        else:
            cap = f":cap{self.max_segments}" if self.max_segments is not None else ""
            cut = f"width:{self.width}{cap}"
        # The default form is left off so that a policy identical to the one every fitted
        # fence on disk was built under keeps fingerprinting as `count:6`. Widening the
        # variant set is a different geometry and must not silently share a fence.
        return cut if self.form == "span" else f"{cut}/{self.form}"

    def segments(self, text: str) -> list[str]:
        """Contiguous word spans of ``text`` under this policy.

        ``multi`` has no single cut, so it reports its first component's. Callers that
        need every variant must use :meth:`shortened`, not this.
        """
        if self.mode == "multi":
            return _parse_component(self.components[0]).segments(text)
        words = text.split()
        if self.mode == "count":
            return _equal_segments(words, self.n)
        parts = [" ".join(words[i:i + self.width])
                 for i in range(0, len(words), self.width)]
        if self.max_segments is not None and len(parts) > self.max_segments:
            return _equal_segments(words, self.max_segments)
        return parts


def _equal_segments(words: list[str], n: int) -> list[str]:
    """``n`` spans of near-equal length. Parity with ``ref_segments``.

    Fewer words than segments returns one word per segment rather than padding, which
    is what the reference does and is why a two-word text ends up below
    ``min_segments`` and unjudgeable.
    """
    if len(words) < n:
        return list(words)
    edges = [round(i * len(words) / n) for i in range(n + 1)]
    return [" ".join(words[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]


@dataclass(frozen=True)
class SpanSet:
    """The shortened versions of one text, split by which statistic reads them.

    Prefixes and suffixes are kept apart from one-segment deletions because only the
    former feed the served statistic: dropping an *interior* segment leaves two
    disjoint pieces, which is not the contiguous run ``docs/METHOD.md`` §1
    defines. Keeping them in separate blocks means serving reads a whole array rather
    than gathering rows out of a mixed one.
    """

    span_names: tuple[str, ...]
    span_texts: tuple[str, ...]
    deletion_names: tuple[str, ...]
    deletion_texts: tuple[str, ...]
    segment_count: int
    #: Aligned with ``span_texts``: the words of the text that variant dropped, joined by
    #: single spaces. Two disjoint pieces (an interior run drops both ends) are joined
    #: the same way -- the answer check reads this as a bag of words, never as prose.
    removed_texts: tuple[str, ...] = ()

    @property
    def all_texts(self) -> tuple[str, ...]:
        """Every variant, spans first — the order :func:`build_profile` encodes in."""
        return self.span_texts + self.deletion_texts


def build_spans(parts: list[str]) -> SpanSet:
    """Every prefix, every suffix, and every one-segment deletion of ``parts``.

    ``3n - 2`` variants, 16 at ``n = 6``. Prefixes and suffixes are interleaved
    (``pre1, suf1, pre2, suf2, ...``) to match the order the reference iterates its
    variant dict in, so that ``argmax`` breaks a tie the way the reference's
    ``max(..., key=...)`` breaks it. Ties are measure-zero on floats, but the
    tie-break decides ``best_end``, which is the mechanism claim of §3.4.
    """
    n = len(parts)
    span_names: list[str] = []
    span_texts: list[str] = []
    removed: list[str] = []
    for cut in range(1, n):
        span_names.append(f"pre{cut}")
        span_texts.append(" ".join(parts[:cut]))
        removed.append(" ".join(parts[cut:]))
        span_names.append(f"suf{cut}")
        span_texts.append(" ".join(parts[cut:]))
        removed.append(" ".join(parts[:cut]))

    deletion_names: list[str] = []
    deletion_texts: list[str] = []
    for index in range(n):
        deletion_names.append(f"del{index}")
        deletion_texts.append(" ".join(parts[:index] + parts[index + 1:]))

    spans = [(name, text, gone) for name, text, gone in zip(span_names, span_texts, removed)
             if text.strip()]
    dels = [(name, text) for name, text in zip(deletion_names, deletion_texts)
            if text.strip()]
    return SpanSet(
        span_names=tuple(name for name, _, _ in spans),
        span_texts=tuple(text for _, text, _ in spans),
        deletion_names=tuple(name for name, _ in dels),
        deletion_texts=tuple(text for _, text in dels),
        segment_count=n,
        removed_texts=tuple(gone for _, _, gone in spans),
    )


def _parse_component(spec: str) -> "SpanPolicy":
    """``"count:6"`` / ``"width:2"`` / ``"width:2:cap16"`` as a single-cut policy.

    Deliberately narrow: a component is a *cut*, never a form and never another
    ``multi``, so a policy cannot nest and a fingerprint stays finite.
    """
    bits = spec.split(":")
    if bits[0] == "count" and len(bits) == 2:
        return SpanPolicy(mode="count", n=int(bits[1]))
    if bits[0] == "width" and len(bits) in (2, 3):
        cap = int(bits[2][3:]) if len(bits) == 3 and bits[2].startswith("cap") else None
        return SpanPolicy(mode="width", width=int(bits[1]), max_segments=cap)
    raise ValueError(f"unrecognised component cut {spec!r}")


def build_runs(parts: list[str]) -> SpanSet:
    """Every contiguous run of ``parts`` bar the whole text — §1's definition as written.

    ``n(n+1)/2 - 1`` variants against :func:`build_spans`'s ``3n - 2``: 20 versus 16 at
    ``n = 6``. The extra ones are the interior runs, which is exactly what a payload
    placed at both ends survives on.

    Order is by start then end, which is stable but *not* the reference's interleaved
    prefix/suffix order. That matters only for tie-breaks, and ties are measure-zero on
    floats; the reference form is still what :func:`build_spans` produces, so parity with
    ``tests/reference/deletion_reference.py`` is untouched.
    """
    n = len(parts)
    names, texts, removed = [], [], []
    for start in range(n):
        for end in range(start + 1, n + 1):
            if start == 0 and end == n:
                continue
            text = " ".join(parts[start:end])
            if not text.strip():
                continue
            names.append(f"run{start}_{end}")
            texts.append(text)
            removed.append(" ".join(parts[:start] + parts[end:]))
    deletion_names, deletion_texts = [], []
    for index in range(n):
        text = " ".join(parts[:index] + parts[index + 1:])
        if text.strip():
            deletion_names.append(f"del{index}")
            deletion_texts.append(text)
    return SpanSet(span_names=tuple(names), span_texts=tuple(texts),
                   deletion_names=tuple(deletion_names),
                   deletion_texts=tuple(deletion_texts), segment_count=n,
                   removed_texts=tuple(removed))


def shortened(policy: SpanPolicy, text: str) -> SpanSet:
    """The variants ``policy`` asks for, deduplicated, in a stable order.

    One place decides what "shortened version" means, so insertion and serving cannot
    drift apart. For ``multi`` the component variant sets are unioned: a text can be cut
    coarsely *and* finely, and ``excess`` -- a max over sub-spans -- reads the union.
    Duplicates are dropped because the same run often survives two cuts, and paying for
    its embedding twice buys nothing.
    """
    builder = build_runs if policy.form == "runs" else build_spans
    if policy.mode != "multi":
        return builder(policy.segments(text))

    seen: set = set()
    names, texts, removed = [], [], []
    for spec in policy.components:
        component = _parse_component(spec)
        built = builder(component.segments(text))
        # The component's own segment count travels in the name. Without it `_describe`
        # would divide a fine cut's `pre3` by the *primary* cut's segment count and
        # report a share that is simply wrong -- a diagnostic that looks right, which is
        # worse than one that raises.
        n_component = len(component.segments(text))
        for name, variant, gone in zip(built.span_names, built.span_texts,
                                       built.removed_texts):
            if variant in seen:
                continue
            seen.add(variant)
            names.append(f"{spec}#{n_component}:{name}")
            texts.append(variant)
            removed.append(gone)
    primary = builder(_parse_component(policy.components[0]).segments(text))
    return SpanSet(span_names=tuple(names), span_texts=tuple(texts),
                   deletion_names=primary.deletion_names,
                   deletion_texts=primary.deletion_texts,
                   segment_count=primary.segment_count,
                   removed_texts=tuple(removed))


def deployed_policy() -> SpanPolicy:
    """Current paper policy; historical controls must select their policy explicitly.

    The value type retains its count:6 default. Switching the deployment to 4+2
    changes the fingerprint and therefore requires a matching calibration artifact.
    A legacy 6+2 fence is never silently reused for the current runtime.
    """
    return SpanPolicy(mode="multi", components=("count:4", "width:2:cap16"), form="runs")
