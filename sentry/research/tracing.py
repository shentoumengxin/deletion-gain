"""Trace capture for cache operations"""

from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import numpy as np


@dataclass
class LookupTrace:
    """Complete trace of a cache lookup operation"""

    # Input
    query: str
    timestamp: datetime

    # Embedding
    query_embedding: np.ndarray
    embedding_time_ms: float

    # Vector search
    top_k_indices: list[int]
    top_k_similarities: list[float]
    search_time_ms: float

    # Similarity evaluation
    hit: bool
    max_similarity: Optional[float]
    threshold: float
    matched_entry_id: Optional[str]
    matched_query: Optional[str]
    matched_response: Optional[str]

    # Cache state
    cache_size: int
    total_time_ms: float

    # Base-stage candidate before the optional defense
    base_hit: bool = False
    base_candidate_entry_id: Optional[str] = None

    # Online deletion-test defense (the cached entry scored against this query)
    defense_enabled: bool = False
    defense_checked: bool = False
    defense_blocked: bool = False
    defense_reason: Optional[str] = None
    #: max over the entry's prefixes/suffixes of cos(span, query), minus cos(entry, query)
    defense_excess: Optional[float] = None
    #: the fence height this excess was compared against
    defense_fence: Optional[float] = None
    #: "prefix" or "suffix" -- which end the winning span kept
    defense_best_end: Optional[str] = None
    #: fraction of the entry's segments the winning span kept
    defense_best_kept: Optional[float] = None
    #: word count of the entry, the fence's second feature
    defense_words: int = 0
    defense_scoring_time_ms: float = 0.0
    defense_total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary (for JSON serialization)"""
        data = asdict(self)
        data["query_embedding"] = self.query_embedding.tolist()
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class InsertTrace:
    """Trace of a cache insertion operation."""

    query: str
    timestamp: datetime
    entry_id: str
    embedding_time_ms: float
    #: Whether the deletion-test defense is on. Insertion builds the entry's profile
    #: either way; it renders no verdict.
    defense_enabled: bool
    #: Whether this entry got a deletion profile. False means the text had too few
    #: segments to cut, which the serving path reads as a miss rather than a free pass.
    defense_profiled: bool
    #: Cost of cutting the entry and embedding its shortened versions. This is where the
    #: deletion test spends its model calls; serving spends none.
    defense_profile_time_ms: float
    total_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary (for JSON serialization)."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class CacheInsertResult:
    """Result of an insertion with observability details."""

    entry_id: str
    trace: InsertTrace


@dataclass
class CacheLookupResult:
    """Result of a cache lookup"""

    hit: bool
    response: Optional[str]
    matched_entry_id: Optional[str]
    trace: LookupTrace
