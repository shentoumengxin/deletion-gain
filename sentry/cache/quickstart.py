"""A defended GPTCache in a few lines, with the paper's encoder and a shipped fence.

    from sentry.cache import open_cache

    with open_cache(llm=my_llm) as cache:
        print(cache.ask("what city hosted the 1998 winter olympics?").answer)

The shipped fence was calibrated on public benign hits (ComQA and Natural Questions)
with ``intfloat/e5-small-v2``. It is a starting point. Recalibrate on your own benign
traffic with ``sentry-cache calibrate`` before relying on its false-rejection rate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .runtime import CacheRuntime, RuntimeConfig, create_runtime

DEFAULT_MODEL = "intfloat/e5-small-v2"
DEFAULT_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
DEFAULT_FENCE = Path(__file__).with_name("fences") / "e5-small-v2.json"


def default_embedder(model_path: Optional[str] = None, batch_size: int = 64):
    """The paper's encoder: e5-small-v2 at a pinned revision, CLS pooling, no prefix.

    ``model_path`` loads a local copy of that revision instead of the Hugging Face Hub.
    """
    from sentry.embeddings import TransformerCLSEmbedder
    return TransformerCLSEmbedder(DEFAULT_MODEL, batch_size=batch_size, pooling="cls",
                                  revision=DEFAULT_REVISION, source=model_path)


def open_cache(llm: Callable[[str], str], *, data_dir: Optional[str | Path] = None,
               fence: Optional[str | Path] = None, embedder=None,
               model_path: Optional[str] = None) -> CacheRuntime:
    """Open a GPTCache with the Deletion Gain hit filter in front of it.

    ``llm`` maps a query to an answer and is called on every miss and every rejected
    hit. ``data_dir`` persists the cache across restarts, and None keeps it in memory.
    ``fence`` is a fence file from ``sentry-cache calibrate``, and None uses the
    shipped one. ``embedder`` must match the fence's encoder signature.
    """
    embedder = embedder or default_embedder(model_path)
    config = RuntimeConfig(fence_path=Path(fence) if fence else DEFAULT_FENCE,
                           encoder_signature=embedder.signature, data_dir=data_dir)
    return create_runtime(config, embedder, llm)
