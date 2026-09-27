"""The content-word tokeniser the answer check reads.

The last test is the one with teeth: the check must not know what a request wrapper
looks like. A function-word list containing "please" or "tell" would make the
false-veto number a whitelist in disguise.
"""
from __future__ import annotations

from sentry.cache.defense.textnorm import FUNCTION_WORDS, content_tokens, normalise_answer


def test_normalise_strips_markdown_and_collapses_whitespace():
    raw = "**Eminem**, whose real name is Marshall Mathers,\n\n### attended  *Highland Park*"
    assert normalise_answer(raw) == "Eminem, whose real name is Marshall Mathers, attended Highland Park"


def test_content_tokens_drop_function_words_keep_dates_and_names():
    toks = content_tokens("The answer is: Mariah Carey and Nick Cannon were married on 2008-12-22.")
    assert "2008-12-22" in toks
    assert "mariah" in toks and "cannon" in toks
    assert "the" not in toks and "is" not in toks and "and" not in toks and "on" not in toks


def test_request_vocabulary_is_not_a_function_word():
    # The check must not know what a wrapper looks like.
    for word in ("please", "tell", "answer", "question", "briefly", "sentence", "help"):
        assert word not in FUNCTION_WORDS
        assert word in content_tokens(f"x {word} y")


def test_content_tokens_is_a_frozenset_and_lowercases():
    toks = content_tokens("Spanish SPANISH spanish")
    assert toks == frozenset({"spanish"})
