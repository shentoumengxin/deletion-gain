"""Encode validated corpus records without generating legacy orbit statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sentry.embeddings import HashEmbedder, TransformerCLSEmbedder
from .config import ExperimentConfig
from .io import atomic_write_json, read_records, write_records


def embed_records(
    records_path: str | Path,
    output_npz: str | Path,
    embedded_records_path: str | Path,
    config: ExperimentConfig,
    smoke: bool = False,
) -> dict:
    """Save vectors in record order, enriched records, and their manifest.

    The NPZ retains the record_ids/embeddings/model_name keys consumed by the
    corpus merge tools. Experiment-specific scores belong to the RQ analyses.
    """
    records = read_records(records_path)
    if not records:
        raise ValueError("no records to embed")
    embedder = HashEmbedder() if smoke else TransformerCLSEmbedder(config.embedding_model)
    embeddings = embedder.encode([record.text for record in records])
    ids = np.asarray([record.record_id for record in records], dtype=str)
    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        record_ids=ids,
        embeddings=embeddings,
        model_name=np.asarray([embedder.model_name], dtype=str),
    )
    embedding_by_id = {
        record_id: embeddings[index] for index, record_id in enumerate(ids.tolist())
    }
    canonical_by_intent = {
        record.intent_id: record
        for record in records
        if record.query_role == "canonical"
    }
    for record in records:
        if record.query_role not in {"wu", "gcg", "ndss", "benign_query", "hard_negative"}:
            continue
        canonical = canonical_by_intent[record.intent_id]
        similarity = float(np.dot(
            embedding_by_id[record.record_id], embedding_by_id[canonical.record_id]
        ))
        record.cache_hit = similarity >= config.cache_threshold
        record.metadata = {
            **record.metadata,
            "embedded_cosine_to_canonical": similarity,
            "embedding_model": embedder.model_name,
        }
    write_records(embedded_records_path, records)
    manifest = {
        "model_name": embedder.model_name,
        "record_count": len(records),
        "embedding_dimension": int(embeddings.shape[1]),
        "cache_threshold": config.cache_threshold,
        "records": str(Path(records_path)),
        "embeddings": str(output),
        "embedded_records": str(Path(embedded_records_path)),
    }
    atomic_write_json(output.with_name("embedding_manifest.json"), manifest)
    return manifest
