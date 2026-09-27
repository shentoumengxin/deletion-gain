"""Golden copy of the deletion-test analysis code. DO NOT EDIT TO MATCH AN IMPLEMENTATION.

Verbatim extract of ``deletion_test.py`` (``segments``, ``variants``) and
``deletion_profile.py`` (the ``spans``/``excess_raw`` block) from
``<server-workdir>/`` on ``cpu-server``, which is the code that produced every
number in ``docs/DELETION_TEST.md``.

The project's rule is that the offline statistic and the online statistic must be the
same statistic. Under the residual that rule was enforced by ``OrbitDefense.evaluate``
mirroring ``compute_pair_metrics``; here it is enforced by
``tests/test_deletion_spans.py`` and ``tests/test_deletion_statistic.py`` diffing
``sentry.cache.defense`` against this file.

If a parity test fails, the implementation is wrong. This file is the specification.
"""
from __future__ import annotations

import numpy as np


def ref_segments(text: str, n: int) -> list[str]:
    """deletion_test.py:segments -- n contiguous spans of near-equal length."""
    words = text.split()
    if len(words) < n:
        return words
    edges = [round(i * len(words) / n) for i in range(n + 1)]
    return [" ".join(words[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def ref_cut_width(text: str, width: int) -> list[str]:
    """deletion_profile.py:cut -- fixed-width segments."""
    words = text.split()
    return [" ".join(words[i:i + width]) for i in range(0, len(words), width)]


def ref_variants(parts: list[str]) -> dict[str, str]:
    """deletion_test.py:variants -- deletions first, then prefix/suffix pairs."""
    out: dict[str, str] = {}
    n = len(parts)
    for i in range(n):
        out[f"del{i}"] = " ".join(parts[:i] + parts[i + 1:])
    for p in range(1, n):
        out[f"pre{p}"] = " ".join(parts[:p])
        out[f"suf{p}"] = " ".join(parts[p:])
    return {name: text for name, text in out.items() if text.strip()}


def ref_excess(variant_vectors: dict[str, np.ndarray], whole: np.ndarray,
               anchor: np.ndarray) -> tuple[float, float]:
    """(excess_span, excess_del) -- deletion_profile.py:209 and deletion_test.py:251.

    ``excess_span`` is the served statistic: the max over prefixes and suffixes only.
    A one-segment deletion of an *interior* segment leaves two disjoint pieces, which
    is not a contiguous run, so the two maxima are separate statistics rather than one
    pooled maximum.
    """
    base = float(whole @ anchor)
    spans = [name for name in variant_vectors if name.startswith(("pre", "suf"))]
    dels = [name for name in variant_vectors if name.startswith("del")]
    excess_span = max(float(variant_vectors[n] @ anchor) for n in spans) - base
    excess_del = max(float(variant_vectors[n] @ anchor) for n in dels) - base
    return excess_span, excess_del
