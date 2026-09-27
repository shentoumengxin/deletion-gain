"""Entry-side deletion gain and answer checking; the core requires only NumPy."""

from .deletion import (
    CallableEmbedder,
    DeletionProfile,
    Embedder,
    ExcessReading,
    build_profile,
    excess,
    unit,
)
from .decide import (
    DefenseCounters,
    DeletionDecision,
    DeletionDefenseConfig,
    InMemoryCalibrationLog,
    cosine_from_l2_distance,
    decide,
    l2_distance_from_cosine,
)
from .entry_store import InMemoryProfileStore, ProfileStore, entry_text, text_key
from .fence import CalibrationRow, ExcessFence, achieved_block_rate
from .insertion import (
    InsertionCounters,
    ProfileWritingDataManager,
    install_profile_writer,
)
from .spans import SpanPolicy, SpanSet, build_spans


def __getattr__(name):
    # Importing a statistic must not initialize the optional host or its clients.
    if name == "DeletionVetoEvaluation":
        from .gptcache_plugin import DeletionVetoEvaluation
        return DeletionVetoEvaluation
    raise AttributeError(name)

__all__ = [
    # the statistic
    "SpanPolicy",
    "SpanSet",
    "build_spans",
    "DeletionProfile",
    "ExcessReading",
    "Embedder",
    "CallableEmbedder",
    "build_profile",
    "excess",
    "unit",
    # storage and the boundary
    "ProfileStore",
    "InMemoryProfileStore",
    "text_key",
    "entry_text",
    "CalibrationRow",
    "ExcessFence",
    "achieved_block_rate",
    # the two hooks
    "InsertionCounters",
    "ProfileWritingDataManager",
    "install_profile_writer",
    "DeletionDefenseConfig",
    "DeletionDecision",
    "DeletionVetoEvaluation",
    "DefenseCounters",
    "InMemoryCalibrationLog",
    "decide",
    "cosine_from_l2_distance",
    "l2_distance_from_cosine",
]
