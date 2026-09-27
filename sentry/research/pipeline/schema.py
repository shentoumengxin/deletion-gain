from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

QUERY_ROLES = {
    "canonical",
    "legal_candidate",
    "legal",
    "wu",
    "gcg",
    "ndss",
    # SCP — Semantic Cache Poisoning (Wu et al., NDSS 2026) black-box Qadv.
    "scp",
    "orbit",
    "hard_negative",
    # Incoming side of an independent benign pair (QQP/PAWS-QQP). Never receives
    # an orbit: only the entry side of a benign pair may touch the operator P.
    "benign_query",
    "external",
}
SEMANTIC_LABELS = {"equivalent", "non_equivalent", "pending"}
SPLITS = {"train", "val", "test", "external"}


@dataclass
class QueryRecord:
    record_id: str
    intent_id: str
    source: str
    split: str
    query_role: str
    text: str
    generator: str
    seed: int
    payload: str | None = None
    attempt: int = 0
    semantic_label: str = "pending"
    cache_hit: bool | None = None
    response_success: bool | None = None
    parent_id: str | None = None
    canonical_answer: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.record_id:
            raise ValueError("record_id is required")
        if not self.intent_id:
            raise ValueError("intent_id is required")
        if self.split not in SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        if self.query_role not in QUERY_ROLES:
            raise ValueError(f"invalid query_role: {self.query_role}")
        if self.semantic_label not in SEMANTIC_LABELS:
            raise ValueError(f"invalid semantic_label: {self.semantic_label}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be non-empty")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryRecord":
        record = cls(**payload)
        record.validate()
        return record
