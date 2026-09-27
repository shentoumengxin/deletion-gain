"""Signal 1: Conditional Perplexity Asymmetry"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import DefenseBase
from sentry.research.tracing import LookupTrace


class PerplexityAsymmetry(DefenseBase):
    """
    Detects attacks via conditional perplexity asymmetry.

    Measures information asymmetry between query pairs using a causal LM.
    Legitimate paraphrases have symmetric conditional surprisal.
    Attacks inject extra content, creating asymmetric information flow.

    Score = |S(Q_cached|Q_in) - S(Q_in|Q_cached)|
    where S(A|B) = mean NLL of A tokens conditioned on B as prefix.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        model_name: str = "distilgpt2",
        max_length: int = 512,
    ):
        super().__init__(name="perplexity_asymmetry", threshold=threshold)
        self._model_name = model_name
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_model(self):
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

    def _conditional_surprisal(self, target: str, context: str) -> float:
        """Compute mean NLL of target tokens conditioned on context."""
        self._load_model()

        separator = " | "
        context_ids = self._tokenizer.encode(
            context + separator, add_special_tokens=False
        )
        target_ids = self._tokenizer.encode(target, add_special_tokens=False)

        if not target_ids:
            return 0.0

        # Cap target so it always fits with at least one context token.
        if len(target_ids) >= self._max_length:
            target_ids = target_ids[: self._max_length - 1]

        # Reserve room for the target by left-truncating the context.
        # Right-truncating the concatenation (the previous approach) silently
        # dropped target tokens and let an attacker pad `context` past
        # max_length to force this function to return 0.0.
        max_context_len = max(1, self._max_length - len(target_ids))
        if len(context_ids) > max_context_len:
            context_ids = context_ids[-max_context_len:]

        full_ids = context_ids + target_ids
        target_start = len(context_ids)

        input_ids = torch.tensor([full_ids], device=self._device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits

        # Compute NLL for target tokens only
        total_nll = 0.0
        count = 0
        for i in range(target_start, len(full_ids)):
            token_logits = logits[0, i - 1]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            token_id = full_ids[i]
            total_nll -= log_probs[token_id].item()
            count += 1

        return total_nll / count if count > 0 else 0.0

    def score(
        self,
        query: str,
        cached_query: str,
        similarity: float,
        trace: LookupTrace,
        *,
        response: Optional[str] = None,
    ) -> float:
        s_cached_given_query = self._conditional_surprisal(cached_query, query)
        s_query_given_cached = self._conditional_surprisal(query, cached_query)
        return abs(s_cached_given_query - s_query_given_cached)
