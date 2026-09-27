"""Base defense interface for cache poisoning detection"""

from typing import Optional

from sentry.research.tracing import LookupTrace


class DefenseBase:
    """
    Base class for cache defense signals.

    Subclasses implement score() to compute a suspicion score.
    should_reject() uses a threshold on that score by default.
    """

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    def score(
        self,
        query: str,
        cached_query: str,
        similarity: float,
        trace: LookupTrace,
        *,
        response: Optional[str] = None,
    ) -> float:
        """
        Compute a defense signal score (higher = more suspicious).

        Args:
            query: The incoming query
            cached_query: The matched cached query
            similarity: Cosine similarity score
            trace: Full lookup trace
            response: Cached response text (keyword-only)

        Returns:
            Suspicion score (0.0 = benign, higher = more suspicious)
        """
        raise NotImplementedError

    def should_reject(
        self,
        query: str,
        cached_query: str,
        similarity: float,
        trace: LookupTrace,
        *,
        response: Optional[str] = None,
    ) -> bool:
        """
        Determine if a cache hit should be rejected.

        Default: reject if score() exceeds threshold.
        """
        return self.score(
            query, cached_query, similarity, trace, response=response
        ) > self.threshold
