"""Signal 2: Bidirectional NLI Entailment"""

from typing import Optional
from sentence_transformers import CrossEncoder

from .base import DefenseBase
from sentry.research.tracing import LookupTrace


class BiNLI(DefenseBase):
    """
    Detects attacks via bidirectional NLI entailment.

    Paraphrases exhibit mutual entailment (A entails B AND B entails A).
    Attacks show asymmetric entailment: Q_cached entails Q_in (contains it)
    but Q_in does NOT entail Q_cached (lacks injected content).

    Score = 1 - min(P_entail(forward), P_entail(backward)).
    Higher score = more suspicious.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
    ):
        super().__init__(name="binli", threshold=threshold)
        self._model_name = model_name
        self._model: Optional[CrossEncoder] = None
        self._entail_idx: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
            # Resolve the entailment class index from the model config,
            # since NLI checkpoints disagree on label ordering.
            id2label = getattr(self._model.model.config, "id2label", None)
            if not id2label:
                raise RuntimeError(
                    f"NLI model {self._model_name} has no id2label; cannot identify entailment class."
                )
            matches = [
                int(idx) for idx, label in id2label.items()
                if str(label).lower() == "entailment"
            ]
            if not matches:
                raise RuntimeError(
                    f"NLI model {self._model_name} has no 'entailment' label in {id2label!r}."
                )
            self._entail_idx = matches[0]

    def _entailment_prob(self, premise: str, hypothesis: str) -> float:
        """P(hypothesis is entailed by premise)."""
        self._load_model()
        scores = self._model.predict(
            [(premise, hypothesis)], apply_softmax=True
        )
        return float(scores[0][self._entail_idx])

    def score(
        self,
        query: str,
        cached_query: str,
        similarity: float,
        trace: LookupTrace,
        *,
        response: Optional[str] = None,
    ) -> float:
        self._load_model()
        # Score both entailment directions in a single batched forward pass
        # instead of two sequential predict() calls — the cross-encoder runs
        # the two (premise, hypothesis) pairs as one batch, roughly halving
        # latency on the cache hot path.
        scores = self._model.predict(
            [(cached_query, query), (query, cached_query)],
            apply_softmax=True,
        )
        p_forward = float(scores[0][self._entail_idx])   # cached_query |= query
        p_backward = float(scores[1][self._entail_idx])  # query |= cached_query
        return 1.0 - min(p_forward, p_backward)
