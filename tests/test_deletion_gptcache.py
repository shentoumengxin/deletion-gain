"""The entry-side deletion test driven by a real GPTCache process.

``tests/test_deletion_serving.py`` stands the plugin in front of a fake inner evaluator.
This puts it inside GPTCache itself: GPTCache embeds the prompt, searches its own faiss
index, calls the registered ``SimilarityEvaluation``, and decides hit or miss from the
score it gets back. Three assumptions are only checkable here -- that ``search_result``
carries ``(squared L2 distance, vector id)``, that returning the range minimum is read
as "not similar enough", and that the entry text handed to ``data_manager.save`` is the
entry text handed back as ``cache_dict["question"]``.

**The attack is in the entry, never in the query.** Both scenarios below issue the same
clean benign query; what differs is what was planted. A test that put the payload in the
query would be exercising the query side and would pass for the wrong reason.

Geometry is constructed rather than learned. Question words all point at one topic
direction; payload words tilt away from it by a fixed angle. Mean-pooled and normalised,
a planted entry lands at a cosine the cache will still retrieve, while the prefix that
holds only the question words sits closer to the topic than the whole entry does -- which
is exactly the structure ``excess`` is built to find, in miniature.

The embedder that produces that geometry is the one the mechanism tests use; it lives in
``tests/collision_embedder.py`` so that every test measuring this defense measures it
with the same instrument.

**Two fences, two sections.** The first section runs the flat fence over the narrow
benign corpus -- the calibration the rest of this repo's serving tests share, and the
form ``docs/DELETION_TEST.md`` §4's fence-transfer result is reported under. The second
runs the *shipped* form: the conditional surface over ``(cos, log words)`` that
``ExcessFence.fit`` produces, fitted on a benign corpus that actually spans a cosine
range, and applied at both ends of it. Nothing exercised that form end to end before,
which meant the boundary a deployment installs had never blocked a planted entry in a
real cache. See ``tests/test_deletion_fence_conditional.py`` for why the shipped form is
the conditional one; this file is where it has to work.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("gptcache", reason="gptcache is optional; integration tests need it")

from gptcache import Cache  # noqa: E402
from gptcache.adapter.api import get as cache_get, put as cache_put  # noqa: E402
from gptcache.manager import manager_factory  # noqa: E402
from gptcache.processor.pre import get_prompt  # noqa: E402
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation  # noqa: E402

from sentry.cache.defense.decide import decide  # noqa: E402
from sentry.cache.defense.entry_store import InMemoryProfileStore, text_key  # noqa: E402
from sentry.cache.defense.fence import ExcessFence  # noqa: E402
from sentry.cache.defense.gptcache_plugin import (  # noqa: E402
    DeletionDefenseConfig,
    DeletionVetoEvaluation,
    cosine_from_l2_distance,
)
from sentry.cache.defense.insertion import install_profile_writer  # noqa: E402
from sentry.cache.defense.spans import SpanPolicy  # noqa: E402
from tests.collision_embedder import (  # noqa: E402
    BENIGN_QUERY,
    DIM,
    FAR_GENUINE_ENTRY,
    FAR_PLANTED_ENTRY,
    GENUINE_ENTRY,
    NEAR_GENUINE_ENTRY,
    NEAR_PLANTED_ENTRY,
    PARAPHRASE_ENTRY,
    PLANTED_ENTRY,
    CollisionEmbedder,
    a2_reading,
    benign_fence as _benign_fence,
    wide_benign_fence,
)

POLICY = SpanPolicy(n=6)
EMBEDDER = CollisionEmbedder()


def benign_fence() -> ExcessFence:
    """The shared entry-side calibration for this geometry; see ``tests/collision_embedder.py``.

    It lives beside the embedder rather than here so that every test measuring this
    defense measures it against the same boundary. A fence with no benign headroom used
    to pass on whichever host happened to assert on the one fixture below it.
    """
    return _benign_fence(POLICY, embedder=EMBEDDER)


class RecordingEvaluation:
    """Delegates to the host's own evaluator and keeps the last ``cache_dict``.

    The point of the recording is ``search_result``: nothing in the defense's own
    bookkeeping shows what the *host* handed it, and one of the three assumptions under
    test here is about the shape and units of that tuple.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last_cache_dict: dict | None = None

    def range(self) -> tuple[float, float]:
        return self.inner.range()

    def evaluation(self, src_dict: dict, cache_dict: dict, **kwargs) -> float:
        self.last_cache_dict = cache_dict
        return self.inner.evaluation(src_dict, cache_dict, **kwargs)


def make_cache(tmp_path, defended: bool = True, fence: ExcessFence | None = None):
    store = InMemoryProfileStore(EMBEDDER.model_name, POLICY.fingerprint())
    manager = manager_factory("sqlite,faiss", data_dir=str(tmp_path),
                              vector_params={"dimension": DIM})
    inner = RecordingEvaluation(SearchDistanceEvaluation(max_distance=2.0, positive=False))
    defense = DeletionVetoEvaluation(
        inner, store, fence=benign_fence() if fence is None else fence,
        config=DeletionDefenseConfig(enabled=defended))

    cache = Cache()
    cache.init(pre_embedding_func=get_prompt,
               embedding_func=EMBEDDER,
               data_manager=manager,
               similarity_evaluation=defense)
    install_profile_writer(cache, store, EMBEDDER, POLICY)
    return cache, defense, store


def test_a_planted_entry_is_not_served_to_a_benign_query(tmp_path):
    cache, defense, _ = make_cache(tmp_path)
    cache_put(PLANTED_ENTRY, "the poisoned answer", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) is None
    assert defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_above_fence"
    assert defense.last_decision.best_end == "prefix"


@pytest.mark.parametrize("entry", [GENUINE_ENTRY, PARAPHRASE_ENTRY])
def test_a_genuine_entry_is_served_to_the_same_benign_query(tmp_path, entry):
    """Both benign fixtures, not just the one that happens to sit below the boundary.

    ``PARAPHRASE_ENTRY`` is a thirteen-word legitimate paraphrase at cosine 0.99991 and
    it used to be blocked: the calibration corpus held nothing but the cleanest possible
    text, so the 5% the budget spends was spent on ordinary traffic. Asserting it here
    makes a fence with no benign headroom a test failure rather than a production
    surprise.
    """
    cache, defense, _ = make_cache(tmp_path)
    cache_put(entry, "paris", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) == "paris"
    assert not defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_within_fence"


def test_the_planted_entry_is_retrievable_without_the_defense(tmp_path):
    """Otherwise the block above proves nothing -- the host might have missed anyway."""
    cache, _, _ = make_cache(tmp_path, defended=False)
    cache_put(PLANTED_ENTRY, "the poisoned answer", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) == "the poisoned answer"


def test_the_writer_and_the_evaluator_agree_on_the_entry_text(tmp_path):
    """The join between the two paths, which is the entry text and not a vector id."""
    cache, defense, store = make_cache(tmp_path)
    cache_put(GENUINE_ENTRY, "paris", cache_obj=cache)
    assert len(store) == 1

    cache_get(BENIGN_QUERY, cache_obj=cache)
    assert defense.last_decision.reason != "no_profile"


def test_search_result_carries_a_squared_l2_distance(tmp_path):
    """The cosine the defense computes must agree with the host's own distance."""
    cache, defense, _ = make_cache(tmp_path)
    cache_put(GENUINE_ENTRY, "paris", cache_obj=cache)
    cache_get(BENIGN_QUERY, cache_obj=cache)

    direct = float(EMBEDDER.encode([GENUINE_ENTRY])[0] @ EMBEDDER.encode([BENIGN_QUERY])[0])
    assert defense.last_decision.cosine == pytest.approx(direct, abs=1e-5)

    # And the host's own tuple, read the way the plugin's fallback would read it: the
    # first field is a *squared* L2 distance on unit vectors, the second a vector id.
    # Nothing outside a real host can check that this is the convention faiss reports in.
    distance, vector_id = defense.inner.last_cache_dict["search_result"]
    assert cosine_from_l2_distance(distance) == pytest.approx(direct, abs=1e-5)
    assert vector_id is not None


def test_the_range_minimum_is_read_as_a_miss(tmp_path):
    """The veto's only lever on the host is its return value, so that has to land.

    ``_reject`` returns ``range()[0]``. If GPTCache read that as a hit -- or if the
    delegated range no longer matched the host's threshold arithmetic -- the defense
    would compute the right verdict and serve the entry anyway.
    """
    cache, defense, _ = make_cache(tmp_path)
    cache_put(PLANTED_ENTRY, "the poisoned answer", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) is None
    assert defense.range() == defense.inner.range()
    assert defense._reject() == defense.range()[0]


# --------------------------------------------------------------------------------
# The shipped form: a conditional fence, applied at both ends of the cosine range
# --------------------------------------------------------------------------------
#: Fitted once. ``ExcessFence.fit`` over a benign corpus spanning cosine ~0.914 to
#: ~0.9999 -- entries whose off-topic mass is *spread*, which is what makes them benign
#: -- and its flat comparator from the same rows at the same 5% budget.
CONDITIONAL_FENCE = wide_benign_fence("conditional", embedder=EMBEDDER, policy=POLICY)
FLAT_OVER_WIDE_RANGE = wide_benign_fence("flat", embedder=EMBEDDER, policy=POLICY)


@pytest.mark.parametrize("entry", [NEAR_PLANTED_ENTRY, FAR_PLANTED_ENTRY])
def test_the_conditional_fence_blocks_a_plant_at_either_end_of_the_range(tmp_path, entry):
    """The coverage that was missing: the shipped boundary, in a real cache, at both ends.

    Every end-to-end test before this one used ``fit_flat`` over a corpus spanning
    0.9995 to 0.9999 and applied it to a single plant at 0.9413. That works only because
    a flat boundary ignores cosine; the form that ships does not, and had never run
    against a planted entry.

    The two plants carry the same construction and differ only in how much payload the
    distance allows: six off-axis words at cosine ~0.9919, fourteen at ~0.9293. The
    near one is the harder case and the one a flat fence cannot reach -- an entry that
    close has only ~0.008 of room to improve by deletion, far under any height
    calibrated over the far end.
    """
    cache, defense, _ = make_cache(tmp_path, fence=CONDITIONAL_FENCE)
    cache_put(entry, "the poisoned answer", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) is None
    assert defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_above_fence"
    # The prefix that stops before the payload block is what recovered the question.
    assert defense.last_decision.best_end == "prefix"


@pytest.mark.parametrize("entry", [NEAR_GENUINE_ENTRY, FAR_GENUINE_ENTRY])
def test_the_conditional_fence_serves_a_genuine_entry_at_either_end(tmp_path, entry):
    """The other half of a boundary: it has to let ordinary traffic through, everywhere.

    ``FAR_GENUINE_ENTRY`` sits at cosine ~0.9189 -- *further* from the arriving query
    than the far plant above, which is at ~0.9293 -- and carries as much off-topic mass
    as that plant does. It differs in one thing only: the mass is spread across the text
    instead of gathered into one contiguous block, so no prefix or suffix recovers the
    question and ``excess`` stays at ~0.029 against a ceiling of ~0.052. Serving this
    while blocking the plant is therefore a decision about arrangement and not about
    distance, which is the deletion test's entire claim.
    """
    cache, defense, _ = make_cache(tmp_path, fence=CONDITIONAL_FENCE)
    cache_put(entry, "paris", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) == "paris"
    assert not defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_within_fence"


@pytest.mark.parametrize("entry", [NEAR_PLANTED_ENTRY, FAR_PLANTED_ENTRY])
def test_both_plants_are_retrievable_without_the_defense(tmp_path, entry):
    """Otherwise the blocks above prove nothing -- the host might have missed anyway.

    The near plant matters most here: at cosine 0.9919 it is exactly the entry a cache
    is most confident about serving.
    """
    cache, _, _ = make_cache(tmp_path, defended=False)
    cache_put(entry, "the poisoned answer", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) == "the poisoned answer"


def test_a_flat_fence_over_the_same_range_serves_the_near_end_plant(tmp_path):
    """What conditioning on cosine buys, priced in served attacks.

    Same corpus, same 5% budget, same cache, same planted entry -- only the *form* of the
    boundary differs. One height calibrated across a wide cosine range lands at ~0.029,
    which is above anything an entry at cosine 0.9919 can produce, so the near-end plant
    is served with the poisoned answer. The conditional fence allows ~0.0063 there and
    refuses it.

    This is the concrete version of the flat form's under-blocking end
    (``tests/test_deletion_fence_conditional.py``): a band where the boundary blocks
    nothing is a band where attacks are not being tested.
    """
    cache, defense, _ = make_cache(tmp_path, fence=FLAT_OVER_WIDE_RANGE)
    cache_put(NEAR_PLANTED_ENTRY, "the poisoned answer", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) == "the poisoned answer"
    assert not defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_within_fence"


def test_a_flat_fence_over_the_same_range_refuses_the_far_end_genuine_entry(tmp_path):
    """And what it costs on the other side: legitimate far traffic, refused.

    The same flat boundary that is too high at 0.9919 is too low at 0.9189, where
    ordinary entries have real room to improve by deletion and use it. The entry served
    by the conditional fence above comes back a miss here. Over-blocking at one end and
    under-blocking at the other are the same defect -- one height, two very different
    benign populations -- which is why the shipped fence conditions on cosine.
    """
    cache, defense, _ = make_cache(tmp_path, fence=FLAT_OVER_WIDE_RANGE)
    cache_put(FAR_GENUINE_ENTRY, "paris", cache_obj=cache)

    assert cache_get(BENIGN_QUERY, cache_obj=cache) is None
    assert defense.last_decision.blocked
    assert defense.last_decision.reason == "excess_above_fence"


def test_the_four_fixtures_sit_where_the_tests_above_claim_they_do(tmp_path):
    """Pins the geometry, so a fixture change is a loud failure rather than a quiet one.

    Every assertion in this section is an argument about *where* these entries sit. If a
    change to the embedder or the corpus moved the near plant to cosine 0.96, the tests
    would still pass and would no longer be testing the near end. Reading the entry-side statistic
    directly -- the cached entry against the arriving benign query, never the reverse --
    is what makes that a failure.
    """
    near_plant = a2_reading(NEAR_PLANTED_ENTRY, EMBEDDER, POLICY)
    far_plant = a2_reading(FAR_PLANTED_ENTRY, EMBEDDER, POLICY)
    near_genuine = a2_reading(NEAR_GENUINE_ENTRY, EMBEDDER, POLICY)
    far_genuine = a2_reading(FAR_GENUINE_ENTRY, EMBEDDER, POLICY)

    assert near_plant.base_cos > 0.99 and near_genuine.base_cos > 0.99
    assert far_plant.base_cos < 0.94 and far_genuine.base_cos < 0.94
    # The far genuine entry is the *further* of the two far fixtures, so nothing below
    # can be a decision about distance.
    assert far_genuine.base_cos < far_plant.base_cos
    # And the near plant's excess is smaller than the far genuine entry's, so nothing
    # can be a decision about the raw statistic either.
    assert near_plant.excess_span < far_genuine.excess_span
    # Both plants clear the conditional boundary; neither genuine entry does.
    for reading in (near_plant, far_plant):
        assert CONDITIONAL_FENCE.blocks(reading.base_cos, reading.words,
                                        reading.excess_span)
    for reading in (near_genuine, far_genuine):
        assert not CONDITIONAL_FENCE.blocks(reading.base_cos, reading.words,
                                            reading.excess_span)


# --------------------------------------------------------------------------------
# The answer check, in a real cache: the entry's own answer as the second witness
# --------------------------------------------------------------------------------
#: A benign entry carrying a harmless request wrapper. Deletion gain vetoes it -- dropping
#: "please tell me" leaves the bare question, which matches the arriving query better --
#: and the cached answer is perfectly reusable, which is the false veto the answer check
#: exists to buy back (design record §1).
WRAPPED_ENTRY = "please tell me who was the first president born in massachusetts"
WRAPPED_QUERY = "who was the first president born in massachusetts"
WRAPPED_ANSWER = "John Adams was the first president born in Massachusetts."


def test_plugin_passes_query_text_and_can_rescue(tmp_path):
    """The wrapper is removable, the answer never depended on it, so the entry is served.

    Both halves of the plumbing are load-bearing here and neither is visible from inside
    the defense's own bookkeeping. The insertion hook has to pass the *answer* into the
    profile, or the veto falls back to deletion gain alone and this entry is refused. The
    evaluator has to pass the arriving *query text* into the ladder, or the echo set still
    holds "first" -- a word the removed span shares with the answer and with the question
    the user actually asked -- and the veto stands for a word nobody planted.
    """
    fence = replace(benign_fence(), answer_rule="echo")
    cache, defense, store = make_cache(tmp_path, fence=fence)
    cache_put(WRAPPED_ENTRY, WRAPPED_ANSWER, cache_obj=cache)

    assert cache_get(WRAPPED_QUERY, cache_obj=cache) == WRAPPED_ANSWER
    assert defense.last_decision.reason == "rescued_by_answer"
    assert defense.last_decision.echo == 0

    # Deletion gain did veto it -- the rescue is what served it -- and the query text is
    # what emptied the echo set. Re-running the same profile against the same fence
    # without the query is the counterfactual, and it blocks.
    blind = decide(profile=store.get(text_key(WRAPPED_ENTRY)),
                   anchor=EMBEDDER.encode([WRAPPED_QUERY])[0],
                   cosine=defense.last_decision.cosine, fence=fence,
                   config=DeletionDefenseConfig())
    assert blind.reason == "excess_above_fence"
    assert blind.echo == 1
