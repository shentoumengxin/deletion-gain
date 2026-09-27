"""Shared normalized encoders. Model packages are imported only on construction."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    model_name = "deterministic-feature-hash-smoke"

    def __init__(self, dimension: int = 256):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    @property
    def signature(self) -> dict:
        return {"model_name": self.model_name, "pooling": "feature-hash",
                "text_prefix": "", "normalization": "l2",
                "dimension": self.dimension, "revision": "builtin-v1"}

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            normalized = re.sub(r"\s+", " ", text.casefold()).strip()
            tokens = re.findall(r"[a-z0-9]+", normalized)
            features = tokens + [
                normalized[index : index + 3]
                for index in range(max(0, len(normalized) - 2))
            ]
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row_index, column] += sign
        return _normalize(matrix)


class TransformerCLSEmbedder:
    def __init__(self, model_name: str, batch_size: int = 128, pooling: str = "cls",
                 text_prefix: str = "", revision: str | None = None,
                 source: str | None = None):
        """``source`` loads the weights from a local copy of ``model_name``.

        ``model_name`` and ``revision`` stay the encoder's identity, so a fence fitted on
        the Hub model also matches a local snapshot of the same revision.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.pooling = pooling          # "cls": last_hidden_state[:, 0]; "mean": mask-weighted mean
        self.text_prefix = text_prefix  # e.g. "query: " for the e5 model-card recipe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_from, load_revision = (source, None) if source else (model_name, revision)
        self.tokenizer = AutoTokenizer.from_pretrained(load_from, revision=load_revision)
        self.model = AutoModel.from_pretrained(load_from, revision=load_revision).to(self.device).eval()
        self.dimension = int(self.model.config.hidden_size)

    @property
    def signature(self) -> dict:
        return {"model_name": self.model_name, "pooling": self.pooling,
                "text_prefix": self.text_prefix, "normalization": "l2",
                "dimension": self.dimension, "revision": self.revision or "unknown"}

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self.text_prefix:
                batch = [self.text_prefix + t for t in batch]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
                if self.pooling == "cls":
                    output = hidden[:, 0, :]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    output = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                output = torch.nn.functional.normalize(output, dim=1)
            batches.append(output.cpu().numpy().astype(np.float32))
        return np.vstack(batches) if batches else np.empty((0, self.dimension), dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class CLSEmbeddingModel:
    """Online embedder using CLS-token pooling (``last_hidden_state[:, 0, :]`` + L2).

    Wraps the research pipeline's ``TransformerCLSEmbedder`` so online numbers reproduce
    the offline validation exactly (the scale-up used CLS pooling, NOT
    SentenceTransformer mean pooling). It is also the pooling CacheAttack's own hit-rate
    measurement uses on e5/bge, so the same embedder matches both the validated defense
    and the attack's HR. Presents the ``EmbeddingModel`` interface
    (``encode``/``encode_batch``/``model_name``/``dimension``) that ``SemanticCache``
    depends on, and which its ``_BatchedEmbedder`` adapts for ``build_profile``.
    """

    def __init__(self, model_name: str, batch_size: int = 128):
        # Lazy import keeps `import sentry.embeddings` light (torch/transformers
        # and the research package are only pulled when a CLS model is instantiated).

        self._embedder = TransformerCLSEmbedder(model_name, batch_size=batch_size)
        self.model_name = model_name
        self.dimension = int(self._embedder.model.config.hidden_size)

    def encode(self, text: str) -> np.ndarray:
        return self._embedder.encode([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._embedder.encode(texts)


# Short aliases used by the e2e driver / config; values are HF model ids.
EMBEDDER_ALIASES = {
    "minilm-cls": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "bge-small": "BAAI/bge-small-en-v1.5",
}


def build_embedder(name: str, batch_size: int = 128) -> CLSEmbeddingModel:
    """Build a CLS-pooled online embedder by short alias or raw HF id.

    Used for the 3-embedder generalization sweep: ``minilm-cls`` (the validated
    embedder), ``e5-small-v2`` (CacheAttack target), ``bge-small`` (CacheAttack
    surrogate). Unknown names are passed through as raw HF model ids.
    """
    model_name = EMBEDDER_ALIASES.get(name, name)
    return CLSEmbeddingModel(model_name, batch_size=batch_size)
