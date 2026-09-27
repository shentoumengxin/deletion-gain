"""A defended semantic cache in front of any LLM.

    python examples/quickstart.py
"""
from sentry.cache import open_cache


def my_llm(question: str) -> str:
    # Call your model here: an OpenAI-compatible API, vLLM, a local model.
    return f"(LLM answer to: {question})"


with open_cache(llm=my_llm, data_dir="cache_state") as cache:
    for question in ["when did benjamin franklin die?",
                     "what date did benjamin franklin die?"]:
        result = cache.ask(question)
        print(f"{result.source:8s} {result.answer}")
