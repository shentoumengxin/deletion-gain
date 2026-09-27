"""Offline checks for the pair classifier splits and the response-perplexity scorer.

No model weights, no network: the splits run on synthetic rows, and the perplexity
scorer runs on a fake model with fixed logits.
"""
import numpy as np
import pytest

from experiments.paper.rq1_detection import pair_classifier as P
from experiments.paper.baselines import response_perplexity as RP


# ---------------------------------------------------------------- pair classifier splits

def synthetic_union(seed=0):
    """Rows shaped like the real union: ComQA intents carry benign + CAP/SCP attacks,
    NQ intents carry benign + KCA attacks; attacks share their intent's benign query."""
    rng = np.random.default_rng(seed)
    intent, cls, attack, corpus = [], [], [], []
    for i in range(60):
        intent.append(f'comqa-{i}'); cls.append(''); attack.append(False); corpus.append('comqa')
        for c in ('CAP', 'SCP'):
            for _ in range(rng.integers(0, 4)):
                intent.append(f'comqa-{i}'); cls.append(c); attack.append(True); corpus.append('comqa')
    for i in range(50):
        intent.append(f'nq-{i}'); cls.append(''); attack.append(False); corpus.append('nq')
        for _ in range(rng.integers(1, 3)):
            intent.append(f'nq-{i}'); cls.append('KCA'); attack.append(True); corpus.append('nq')
    return {'intent_id': np.array(intent, dtype=object), 'cls': np.array(cls, dtype=object),
            'attack': np.array(attack), 'corpus': np.array(corpus, dtype=object)}


def test_intent_folds_keep_an_intent_together_and_depend_on_seed():
    u = synthetic_union()
    f0, f1 = P.intent_folds(u['intent_id'], 5, 0), P.intent_folds(u['intent_id'], 5, 1)
    for k in np.unique(u['intent_id']):
        assert len(set(f0[u['intent_id'] == k])) == 1
    assert set(f0) == set(range(5)) and not np.array_equal(f0, f1)


@pytest.mark.parametrize('held_out', ['CAP', 'SCP', 'KCA'])
@pytest.mark.parametrize('drop', [False, True])
def test_loco_never_trains_on_the_held_out_class(held_out, drop):
    u = synthetic_union()
    target = (u['attack'] & (u['cls'] == held_out)) | (
        ~u['attack'] & (u['corpus'] == P.CLASSES[held_out][1]))
    for seed in range(3):
        folds = P.intent_folds(u['intent_id'], 5, seed)
        covered = np.zeros(len(folds), int)
        for f in range(5):
            train, test = P.split_masks(u, folds, f, held_out=held_out,
                                        drop_heldout_benign=drop)
            assert not (train & test).any()
            assert not (train & u['attack'] & (u['cls'] == held_out)).any()
            assert not set(u['intent_id'][train]) & set(u['intent_id'][test])
            assert not (test & ~target).any()          # only the class and its corpus benign
            if drop:                                    # no benign row of that corpus trains
                corpus = u['corpus'] == P.CLASSES[held_out][1]
                assert not (train & ~u['attack'] & corpus).any()
                if held_out == 'KCA':                   # NQ carries only KCA: no NQ row at all
                    assert not (train & corpus).any()
            assert (train & u['attack']).any() and (train & ~u['attack']).any()
            covered += test
        assert np.array_equal(covered, target.astype(int))   # every target row scored once


def test_in_distribution_split_is_intent_disjoint_and_covers_every_row():
    u = synthetic_union()
    folds = P.intent_folds(u['intent_id'], 5, 3)
    covered = np.zeros(len(folds), int)
    for f in range(5):
        train, test = P.split_masks(u, folds, f)
        assert not set(u['intent_id'][train]) & set(u['intent_id'][test])
        assert (train | test).all()
        covered += test
    assert (covered == 1).all()


def test_pair_features_layout():
    ek, eq = np.arange(6.).reshape(2, 3), np.ones((2, 3))
    x = P.pair_features(ek, eq)
    assert x.shape == (2, 12)
    assert np.array_equal(x[:, 6:9], ek - eq) and np.array_equal(x[:, 9:], ek * eq)


def test_fit_and_score_rejects_intent_overlap():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 4)), np.array([0, 1] * 20)
    groups = np.array([f'g{i // 2}' for i in range(40)], dtype=object)
    train = np.arange(40) < 21                         # row 20 and 21 share intent g10
    with pytest.raises(AssertionError):
        P.fit_and_score('logreg', x, y, groups, train, ~train, seed=0, inner_folds=3)


# ---------------------------------------------------------------- response perplexity

def fake_logprob_fn(table):
    """log p(ids[i] | ids[:i]) read from a fixed next-token table indexed by ids[i-1]."""
    lp = table - np.log(np.exp(table).sum(1, keepdims=True))
    return lambda ids: np.array([lp[ids[i - 1], ids[i]] for i in range(1, len(ids))])


def test_nll_is_averaged_over_answer_tokens_only():
    # vocabulary of 4; query token 3 is very unlikely after anything, answer tokens are not
    table = np.zeros((4, 4))
    table[:, 3] = -20.0
    fn = fake_logprob_fn(table)
    lp = table - np.log(np.exp(table).sum(1, keepdims=True))
    context, target = [3, 3, 3, 2], [0, 1, 0]
    nll, info = RP.mean_target_nll(fn, context, target, max_length=512)
    expected = -(lp[2, 0] + lp[0, 1] + lp[1, 0]) / 3        # y's three tokens, nothing else
    assert nll == pytest.approx(expected)
    assert info == {'n_target': 3, 'context_truncated': False, 'windows': 1}
    assert nll < 2.0                                        # a q-token term would add ~20


def test_windows_left_truncate_context_and_score_every_target_token_once():
    ctx, tgt = list(range(100, 110)), list(range(20))
    (ids, start), = RP.windows(ctx, tgt, max_length=24)
    assert ids == ctx[-4:] + tgt and start == 4              # context cut from the left
    wins = RP.windows(ctx, list(range(50)), max_length=16)
    scored = [t for ids, s in wins for t in ids[s:]]
    assert scored == list(range(50)) and all(len(ids) <= 16 for ids, _ in wins)
    assert all(s >= 1 for _, s in wins)                      # every scored token has a prefix


def test_long_answer_is_scored_in_full():
    table = np.random.default_rng(0).normal(size=(6, 6))
    fn = fake_logprob_fn(table)
    lp = table - np.log(np.exp(table).sum(1, keepdims=True))
    ctx, tgt = [5, 4], [i % 5 for i in range(37)]
    nll, info = RP.mean_target_nll(fn, ctx, tgt, max_length=16)
    stream = ctx + tgt
    # windows of 16 lose long-range context but a bigram table has none to lose
    expected = -np.mean([lp[stream[i - 1], stream[i]] for i in range(len(ctx), len(stream))])
    assert nll == pytest.approx(expected) and info['windows'] > 1 and info['n_target'] == 37


def test_torch_path_matches_existing_perplexity_helper():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from experiments.paper.baselines.defense.perplexity import PerplexityAsymmetry

    logits_table = torch.randn(7, 7, generator=torch.Generator().manual_seed(0))

    class FakeModel:                                  # next-token logits depend on ids[i]
        def __call__(self, input_ids):
            return SimpleNamespace(logits=logits_table[input_ids])

    class FakeTokenizer:
        eos_token_id = 6
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 6 for c in text]

    helper = PerplexityAsymmetry(model_name='fake', max_length=512)
    helper._model, helper._tokenizer = FakeModel(), FakeTokenizer()
    helper._device = torch.device('cpu')
    scorer = RP.ResponseScorer(helper)
    q, y = 'who wrote hamlet', 'marlowe did'
    nll, info = scorer.conditional(y, q)
    assert nll == pytest.approx(helper._conditional_surprisal(y, q), abs=1e-5)
    assert info['n_target'] == len(y)
    ids = [FakeTokenizer.eos_token_id] + [ord(c) % 6 for c in y]
    lp = torch.log_softmax(logits_table[torch.tensor(ids[:-1])], -1)
    manual = -float(lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].mean())
    assert scorer.unconditional(y)[0] == pytest.approx(manual, abs=1e-5)


# ---------------------------------------------------------------- benign rows never trained on

def test_ood_rows_are_read_by_the_fold_model_that_never_saw_their_intent():
    u = synthetic_union()
    folds = P.intent_folds(u['intent_id'], 5, 0)
    fold_of = dict(zip(u['intent_id'], folds))
    seen_intents = ['comqa-3', 'nq-7', 'comqa-11']
    new_intents = ['fresh-0', 'qqp-1']
    rows = seen_intents + new_intents
    # fold model f "leaks": it scores 100 on every intent it trained on (fold != f), else 0
    scores = np.array([[0.0 if fold_of.get(k) == f else 100.0 for k in rows] for f in range(5)])
    scores[:, 3:] = [[100.0], [0.0], [0.0], [0.0], [0.0]]   # unseen intents: one model of five
    frac = P.ood_block_fraction(scores, np.array(rows, dtype=object), fold_of, t=50.0)
    assert np.array_equal(frac[:3], [0.0, 0.0, 0.0])        # never read by a model that saw it
    assert np.allclose(frac[3:], [0.2, 0.2])                # averaged over the five models


def test_replay_blocks_follow_remote_replay_block():
    v = {'dg': np.array([.5, .5, .5, .001, np.nan, .5]),
         'judgeable': np.array([True, True, True, True, False, True]),
         'has_answer': np.array([True, True, False, True, True, True]),
         'adl_best': np.array([.1, 0., 0., .1, 0., 0.]),
         'echo_best': np.array([0, 0, 0, 3, 0, 1])}
    assert P.replay_blocks(v, .01, .05).tolist() == [True, False, True, False, True, True]
    assert P.replay_blocks(v, .01).tolist() == [True, True, True, False, True, True]


def test_group_matching_uses_set_split_kind_and_condition():
    base = {'set': 'instruction_benign', 'split': 'test', 'kind': 'polite', 'condition': 'entry_only'}
    assert P.match_group(base) == 'Template'
    assert P.match_group({**base, 'split': 'calibration'}) is None
    assert P.match_group({**base, 'kind': 'constraint'}) == 'Constraint entry_only'
    assert P.match_group({**base, 'condition': 'query_only'}) is None
    assert P.match_group({**base, 'set': 'unseen_wrappers', 'kind': 'unseen_wrapper'}) == \
        'LLM-written entry_only'
    assert P.match_group({'set': 'fresh_benign', 'split': 'fresh_test', 'kind': 'bare',
                          'condition': 'bare'}) == 'Fresh bare'
