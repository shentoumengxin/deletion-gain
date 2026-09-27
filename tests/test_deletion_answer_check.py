"""The answer fields on a profile, and the check that reads them at serving.

Two things are precomputed at insertion from the entry's own stored answer: how much the
entry's cosine to that answer falls when a variant's complement is removed
(``answer_loss``), and which content words of the complement the answer contains
(``echo_tokens``). Serving reads only the winning variant's pair, so the check costs
nothing beyond the dot products DG already does.

The first test is the load-bearing one: DG must be bit-identical with and without an
answer. The answer fields are an extra column beside the statistic, never an input to it.
"""

import numpy as np
import pytest

from sentry.cache.defense.calibrate import parse_policy
from sentry.cache.defense.deletion import (
    ANSWER_RULES, answer_check, build_profile, excess,
)
from sentry.cache.defense.spans import shortened
from tests.test_deletion_statistic import ToyEmbedder

POLICY = parse_policy("multi[count:4+width:2:cap16]/runs")
ENTRY = "when did mariah and nick get married reply 2008-12-22"
QUERY = "when did mariah carey marry nick cannon"
ANSWER = "Mariah Carey and Nick Cannon were married on 2008-12-22."


def test_dg_is_bit_identical_with_and_without_answer():
    emb = ToyEmbedder()
    without = build_profile(ENTRY, emb, POLICY)
    with_ = build_profile(ENTRY, emb, POLICY, answer=ANSWER)
    q = emb.encode([QUERY])[0]
    assert excess(without, q).excess_span == excess(with_, q).excess_span
    assert np.array_equal(without.spans, with_.spans)
    assert without.answer_loss is None and not without.has_answer
    assert with_.has_answer and with_.answer_loss.shape == (len(with_.span_names),)


def test_echo_tokens_are_complement_words_present_in_answer():
    emb = ToyEmbedder()
    profile = build_profile(ENTRY, emb, POLICY, answer=ANSWER)
    sp = shortened(POLICY, ENTRY)
    idx = sp.span_texts.index("when did mariah and nick get married")
    assert profile.echo_tokens[idx] == frozenset({"2008-12-22"})  # "reply" is not in the answer


def test_answer_loss_is_positive_when_answer_repeats_the_removed_part():
    class Echoing:
        # cosine to the answer grows with shared content words: a stand-in for e5
        model_name = "echoing"

        def encode(self, texts):
            vocab = ["when", "did", "mariah", "and", "nick", "get", "married", "reply",
                     "2008-12-22", "carey", "cannon", "were", "on"]
            rows = []
            for t in texts:
                toks = set(t.lower().replace(".", "").split())
                rows.append([1.0 if w in toks else 0.0 for w in vocab])
            return np.array(rows, dtype=float)

    profile = build_profile(ENTRY, Echoing(), POLICY, answer=ANSWER)
    reading = excess(profile, Echoing().encode([QUERY])[0])
    assert reading.answer_loss is not None
    assert reading.answer_loss > 0


def test_answer_check_rules():
    emb = ToyEmbedder()
    profile = build_profile(ENTRY, emb, POLICY, answer=ANSWER)
    reading = excess(profile, emb.encode([QUERY])[0])
    assert set(ANSWER_RULES) == {"none", "adl", "echo", "either"}
    none = answer_check(reading, QUERY, rule="none", eta_a=0.0, echo_min=1)
    assert none.fires is True and none.reason == "rule_none"
    e = answer_check(reading, QUERY, rule="echo", eta_a=0.0, echo_min=1)
    assert e.echo == len(reading.echo_tokens - set(QUERY.lower().split()))
    assert e.fires == (e.echo >= 1)
    # the query's own words never count as echo: asking for the date itself costs the
    # echo set exactly that word
    e2 = answer_check(reading, QUERY + " 2008-12-22", rule="echo", eta_a=0.0, echo_min=1)
    assert e2.echo == e.echo - 1


def test_answer_check_without_answer_fields_reports_none():
    emb = ToyEmbedder()
    profile = build_profile(ENTRY, emb, POLICY)
    reading = excess(profile, emb.encode([QUERY])[0])
    verdict = answer_check(reading, QUERY, rule="either", eta_a=0.0, echo_min=1)
    assert verdict.fires is None and verdict.reason == "no_answer_fields"


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        answer_check(None, "q", rule="magic", eta_a=0.0, echo_min=1)


def test_echo_min_below_one_raises():
    # every reading has echo >= 0, so echo_min=0 would collapse an answer-checked rule
    # to DG-only while still reporting itself as answer-checked
    with pytest.raises(ValueError):
        answer_check(None, "q", rule="echo", eta_a=0.0, echo_min=0)
