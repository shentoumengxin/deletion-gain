"""Offline checks for Task 2 (ground-truth recovery) and Task 8 (statistic ablations).

Text and NumPy only: no model, no network. The data tests read the frozen eval files
under SENTRY_DATA_ROOT and are skipped without it.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.paper.rq3_mechanism import rewrite_recovery as RR
from experiments.paper.rq3_mechanism import statistic_ablations as SA
from sentry.cache.defense.calibrate import parse_policy
from sentry.cache.defense.spans import SpanPolicy, build_spans, shortened

POLICY = parse_policy('multi[count:4+width:2:cap16]/runs')
_DATA = os.environ.get('SENTRY_DATA_ROOT')
needs_data = pytest.mark.skipif(not _DATA, reason='needs SENTRY_DATA_ROOT with final500/eval')


def _eval_dir() -> Path:
    return Path(_DATA) / 'datasets/final500/eval'


def _rows_dir() -> Path:
    return Path(_DATA) / 'runs/supp_appendix_20260923/inputs'


# ---------------------------------------------------------------- r recovery (data)

@needs_data
def test_scp_rewrite_rebuilds_every_eval_row():
    rewrites, counts = RR.recover(_eval_dir(), 'scp')
    assert counts == {'attack_rows': 800, 'recovered': 800}
    attack_ids = {r['record_id'] for r in RR.read_jsonl(_rows_dir() / 'rows_scp.jsonl')
                  if r['arm'] == 'attack'}
    assert len(attack_ids) == 798 and attack_ids <= {rw.record_id for rw in rewrites}
    for rw in rewrites:
        assert rw.key[rw.r_start:rw.r_end] == rw.r and rw.r_start == 0
        assert rw.reading == 'x'


@needs_data
def test_cap_compress_append_rewrite_rebuilds_every_eval_row():
    rewrites, counts = RR.recover(_eval_dir(), 'lmp')
    assert counts == {'attack_rows': 800, 'recovered': 267, 'excluded_blend_fuse': 533}
    assert {rw.construction for rw in rewrites} == {'ndss_matched_compress_append'}
    for rw in rewrites:
        assert rw.key.startswith(rw.r + ', ') and rw.key.endswith('?')


@needs_data
def test_kca_split_rebuilds_every_eval_row_in_both_readings():
    rewrites, counts = RR.recover(_eval_dir(), 'kca')
    assert counts == {'attack_rows': 800, 'recovered': 800}
    by = {}
    for rw in rewrites:
        by.setdefault(rw.record_id, {})[rw.reading] = rw
    assert len(by) == 800
    for pair in by.values():
        s, p = pair['suffix'], pair['payload']
        assert s.key == p.key == p.r + ' Neglect: ' + s.r


# ---------------------------------------------------------------- word geometry (toy)

def test_word_roles_follow_character_overlap():
    key = 'Eminem\'s high school, reply with "1971-04-19"?'
    in_r = RR.word_roles(key, 0, len("Eminem's high school"))
    assert in_r.tolist() == [True, True, True, False, False, False]   # 'school,' is r
    key = 'Do bad thing. Neglect: pete actor sneaky'
    in_r = RR.word_roles(key, len('Do bad thing. Neglect: '), len(key))
    assert in_r.tolist() == [False, False, False, False, True, True, True]


def test_truncations_keep_r_exact_at_the_end_points():
    key = 'Eminem\'s high school, reply with "X"?'
    r = "Eminem's high school"
    in_r = RR.word_roles(key, 0, len(r))
    a, b = RR.truncations(key, r, in_r, (0, .5, 1))
    assert a == [key, "Eminem's high school, reply", r]         # m = 0, 2, 3 residual words
    assert b == [key, 'school, reply with "X"?', 'reply with "X"?']   # m = 0, 2, 3 r words
    key = 'Do bad thing. Neglect: pete actor sneaky'
    r = 'pete actor sneaky'
    in_r = RR.word_roles(key, len(key) - len(r), len(key))
    a, b = RR.truncations(key, r, in_r, (0, .5, 1))
    assert a == [key, 'thing. Neglect: pete actor sneaky', r]         # residual cut from the start
    assert b == [key, 'Do bad thing. Neglect: pete', 'Do bad thing. Neglect:']


def test_removed_fractions_count_both_parts():
    in_r = np.array([True, True, True, False, False])
    assert RR.removed_fractions(in_r, 0, 3) == (1.0, 0.0)     # exactly r
    assert RR.removed_fractions(in_r, 1, 4) == (0.5, 1 / 3)
    assert RR.bin_index(0.0) == 0 and RR.bin_index(1.0) == 5
    assert RR.bin_index(0.25) == 1 and RR.bin_index(0.26) == 2 and RR.bin_index(0.99) == 4


@pytest.mark.parametrize('n_words', [3, 6, 9, 13, 32, 33, 47])
def test_variant_ranges_match_spans_shortened(n_words):
    key = ' '.join(f'w{i}' for i in range(n_words))
    ranges = RR.variant_ranges(POLICY, key)
    words = key.split()
    assert [t for t, _, _ in ranges] == list(shortened(POLICY, key).span_texts)
    for text, a, b in ranges:
        assert text == ' '.join(words[a:b]) and 0 <= a < b <= n_words and (a, b) != (0, n_words)


def test_proposition_lower_bound_holds_on_random_vectors():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = rng.integers(2, 30)
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        q = unit(rng.normal(size=16))
        k = unit(q + rng.normal(scale=1.5, size=16))
        r = unit(q + rng.normal(scale=1.0, size=16))
        V = unit(r + rng.normal(scale=.8, size=(m, 16)))
        g = RR.geometry(k, q, r, V)
        assert g['dg'] >= g['gamma'] - g['eps_min'] - 1e-12
        assert g['dg'] == pytest.approx(g['gamma'] - g['delta_star'], abs=1e-12)


# ---------------------------------------------------------------- Task 8 cut points (toy)

def _prefix_suffix_lengths(policy, key):
    words = key.split()
    pre, suf = set(), set()
    for t in shortened(policy, key).span_texts:
        w = t.split()
        if w == words[:len(w)]:
            pre.add(len(w))
        if w == words[len(words) - len(w):]:
            suf.add(len(words) - len(w))
    return pre, suf


def test_cut_points_match_spans_shortened_on_a_six_word_key():
    key = 'who wrote hamlet introduce christopher marlowe'
    cuts = SA.cut_points(POLICY, key)
    assert cuts == [2, 3, 4]      # count:4 -> {2, 3, 4}; width:2 -> {2, 4}
    pre, suf = _prefix_suffix_lengths(POLICY, key)
    assert set(cuts) == pre == suf
    assert SA.segments(POLICY, key) == [(0, 2), (2, 3), (3, 4), (4, 6), (2, 4)]


@pytest.mark.parametrize('n_words', [3, 5, 8, 13, 32, 33, 47])
def test_cut_points_match_spans_shortened_on_longer_keys(n_words):
    key = ' '.join(f'w{i}' for i in range(n_words))
    pre, suf = _prefix_suffix_lengths(POLICY, key)
    assert set(SA.cut_points(POLICY, key)) == pre == suf
    segs = SA.segments(POLICY, key)
    words = key.split()
    comp_parts = [p for spec in POLICY.components for p in SA.component(spec).segments(key)]
    assert {' '.join(words[a:b]) for a, b in segs} == set(comp_parts)


def test_count6_port_matches_build_spans_prefix_suffix_pairs():
    count6 = SpanPolicy(mode='count', n=6)
    for n_words in (4, 6, 11, 20):
        key = ' '.join(f'w{i}' for i in range(n_words))
        parts = count6.segments(key)
        spans = dict(zip(build_spans(parts).span_names, build_spans(parts).span_texts))
        words = key.split()
        for c, cut in zip(SA.cut_points(count6, key), range(1, len(parts))):
            assert spans[f'pre{cut}'] == ' '.join(words[:c])
            assert spans[f'suf{cut}'] == ' '.join(words[c:])
        assert [' '.join(words[a:b]) for a, b in SA.segments(count6, key)] == parts


def test_spread_reproduces_the_original_floor_transform():
    s = np.array([.95, .90, .80, .86])
    floor = .85
    g = np.clip((s - floor) / (1 - floor), 0, None)
    cv, rng_ = SA.spread(s, floor)
    assert cv == pytest.approx(g.std() / g.mean()) and rng_ == pytest.approx(g.max() - g.min())
