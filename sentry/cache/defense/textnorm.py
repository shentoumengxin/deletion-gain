"""Answer text normalisation and the content-word tokeniser the answer check reads.

Deliberately small and deliberately ignorant. The function-word list is general English
only -- articles, pronouns, auxiliaries, prepositions, conjunctions, wh-words -- and must
never contain request vocabulary ("please", "tell", "answer"). The answer check may not
know what a wrapper looks like; if it did, the wrapper templates would leak into the
detector and the false-veto number would be a whitelist in disguise.
"""

from __future__ import annotations

import re

FUNCTION_WORDS: frozenset[str] = frozenset("""
a an the this that these those there here
i me my mine we us our ours you your yours he him his she her hers it its they them
their theirs who whom whose which what when where why how
am is are was were be been being do does did doing have has had having
will would shall should can could may might must
and or but nor so yet if then than as of at by for from in into on onto to with
without about above below over under between among through during before after
until while up down out off again further once
not no only own same such too very just also both each few more most other some any
all
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_MARKDOWN = re.compile(r"[*_`#>]+")
_SPACE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Strip markdown emphasis and headings, collapse whitespace."""
    cleaned = _MARKDOWN.sub("", str(text))
    return _SPACE.sub(" ", cleaned).strip()


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase word tokens minus function words. Digits, dates and hyphenated forms
    survive as single tokens (``2008-12-22``), which is what a planted literal looks like."""
    tokens = _TOKEN.findall(normalise_answer(text).lower())
    return frozenset(t.strip("'-") for t in tokens
                     if t.strip("'-") and t.strip("'-") not in FUNCTION_WORDS)
