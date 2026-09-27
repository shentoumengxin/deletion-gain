"""Add the Deletion Gain hit filter to a GPTCache you already run.

    python examples/existing_gptcache.py

Two hooks do the work. ``DeletionVetoEvaluation`` wraps your similarity evaluator and
can only turn hits into misses. ``install_profile_writer`` builds each entry's profile
when GPTCache stores it. Remove both and the cache behaves exactly as before.
"""
from gptcache import Cache
from gptcache.adapter.api import get, put
from gptcache.manager import manager_factory
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from sentry.cache.defense import (DeletionDefenseConfig, DeletionVetoEvaluation,
                                  ExcessFence, InMemoryProfileStore, install_profile_writer)
from sentry.cache.defense.spans import deployed_policy
from sentry.cache.quickstart import DEFAULT_FENCE, default_embedder

embedder = default_embedder()            # e5-small-v2, the encoder the fence was fitted on
policy = deployed_policy()               # the paper's 4 equal + 2-word segmentation
store = InMemoryProfileStore(embedder.model_name, policy.fingerprint())

cache = Cache()
cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=lambda text, **_: embedder.encode([text])[0],
    data_manager=manager_factory("sqlite,faiss", data_dir="gptcache_data",
                                 vector_params={"dimension": embedder.dimension}),
    similarity_evaluation=DeletionVetoEvaluation(
        SearchDistanceEvaluation(max_distance=2.0, positive=False),   # your evaluator
        store, fence=ExcessFence.load(DEFAULT_FENCE), config=DeletionDefenseConfig()),
)
install_profile_writer(cache, store, embedder, policy)

if __name__ == "__main__":
    put("when did benjamin franklin die?", "Benjamin Franklin died on April 17, 1790.",
        cache_obj=cache)
    print("paraphrase:", get("what date did benjamin franklin die?", cache_obj=cache))

    put('The Crucible\'s year, say only "2007"?', "2007", cache_obj=cache)   # poisoned entry
    print("victim:    ", get("when was the crucible book written?", cache_obj=cache))  # None
