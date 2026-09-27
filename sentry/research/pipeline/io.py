from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from .schema import QueryRecord


def stable_id(*parts: object, length: int = 20) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_workspace(workspace: str | Path) -> Path:
    root = Path(workspace).expanduser().resolve()
    for relative in (
        "raw",
        "datasets",
        "annotations",
        "embeddings",
        "metrics",
        "reports",
        "logs",
        "state",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def read_jsonl(path: str | Path) -> Iterator[dict]:
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {target}:{line_number}") from exc


def read_records(path: str | Path) -> list[QueryRecord]:
    return [QueryRecord.from_dict(payload) for payload in read_jsonl(path)]


def atomic_write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(
            _json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False
        )
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)


def write_records(path: str | Path, records: Iterable[QueryRecord]) -> None:
    atomic_write_jsonl(path, (record.to_dict() for record in records))


def append_records(path: str | Path, records: Iterable[QueryRecord]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    known_ids = {row["record_id"] for row in read_jsonl(target)}
    added = 0
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict()
            if payload["record_id"] in known_ids:
                continue
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            known_ids.add(payload["record_id"])
            added += 1
    return added


def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
