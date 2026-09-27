"""Signal 3: Multi-view Embedding Disagreement"""

from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import DefenseBase
from sentry.research.tracing import LookupTrace


class MultiViewDisagreement(DefenseBase):
    """
    Detects key-collision attacks via primary-vs-auxiliary similarity inflation.

    A key-collision attack optimizes the query against the *primary* embedding
    space the cache uses, inflating its cosine to the cached key beyond what
    independent auxiliary encoders corroborate. So the attack signature is
    one-sided: primary similarity high, auxiliary similarities lower.

    Score = max(0, similarity_primary - mean(auxiliary_cosines)).

    Note: an earlier version returned the raw spread max(cos) - min(cos). That
    is symmetric and label-inverted on verbatim-injection attacks (where the
    victim query is a near-duplicate of the cached query, so every encoder
    agrees and the spread is tiny) while genuine loose paraphrases disagree
    more across encoders. The directional form keys on the actual attack
    mechanism instead of raw disagreement, so true paraphrases and verbatim
    near-duplicates both score ~0; only primary-space inflation scores high.
    """

    AUXILIARY_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    ]

    def __init__(
        self,
        threshold: float = 0.15,
        model_names: Optional[list[str]] = None,
    ):
        super().__init__(name="multiview_disagreement", threshold=threshold)
        self._model_names = model_names or self.AUXILIARY_MODELS
        self._models: Optional[list[SentenceTransformer]] = None

    def _load_models(self):
        if self._models is None:
            self._models = [
                SentenceTransformer(name) for name in self._model_names
            ]

    def _cosine(self, model: SentenceTransformer, text_a: str, text_b: str) -> float:
        embeddings = model.encode(
            [text_a, text_b], normalize_embeddings=True, convert_to_numpy=True
        )
        return float(np.dot(embeddings[0], embeddings[1]))

    def score(
        self,
        query: str,
        cached_query: str,
        similarity: float,
        trace: LookupTrace,
        *,
        response: Optional[str] = None,
    ) -> float:
        self._load_models()

        aux = [self._cosine(model, query, cached_query) for model in self._models]
        mean_aux = sum(aux) / len(aux)

        # One-sided: only primary-space inflation (the key-collision signature)
        # is suspicious. Auxiliaries agreeing or exceeding the primary is not.
        return max(0.0, similarity - mean_aux)
