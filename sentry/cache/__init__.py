"""SENTRY's entry-side defense; ``open_cache`` gives a defended GPTCache in a few lines."""
from . import defense

__all__ = [*defense.__all__, "open_cache"]

def __getattr__(name):
    if name == "open_cache":
        from .quickstart import open_cache
        return open_cache
    if name in defense.__all__:
        return getattr(defense, name)
    raise AttributeError(name)
