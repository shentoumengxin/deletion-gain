"""The out-of-the-box path: benign-only calibration, the shipped fence, open_cache."""
import json

import numpy as np
import pytest

from sentry.cache.defense.calibrate import calibrate_from_hits
from sentry.cache.defense.fence import ExcessFence
from sentry.cache.defense.spans import deployed_policy
from sentry.cache.quickstart import DEFAULT_FENCE, DEFAULT_MODEL, DEFAULT_REVISION
from sentry.embeddings import HashEmbedder

TOPICS = ["capital of france", "tallest mountain in africa", "author of hamlet",
          "boiling point of water", "largest ocean on earth", "speed of light in vacuum",
          "first man on the moon", "currency of japan", "longest river in asia",
          "inventor of the telephone", "smallest planet in the solar system",
          "chemical symbol for gold", "year the berlin wall fell", "language spoken in brazil",
          "painter of the mona lisa", "number of bones in the human body",
          "freezing point of mercury", "founder of microsoft", "capital of australia",
          "main gas in the atmosphere", "distance from earth to the sun",
          "largest desert in the world", "composer of the four seasons",
          "height of mount everest"]


def benign_hits():
    hits = []
    for topic in TOPICS:
        key = f"what is the {topic} and why is it well known"
        hits.append({"key": key, "query": f"what is the {topic} and why is it famous",
                     "answer": f"The {topic} is a well known fact."})
        hits.append({"key": key, "query": key, "answer": f"The {topic} is a well known fact."})
    return hits


def test_shipped_fence_matches_the_default_runtime():
    fence = ExcessFence.load(DEFAULT_FENCE)
    assert fence.is_flat and fence.direction == "entry"
    assert fence.answer_rule == "either" and fence.echo_min == 1 and fence.eta_a is not None
    assert fence.policy == deployed_policy().fingerprint()
    assert fence.embedder == DEFAULT_MODEL
    assert fence.metadata["joint_calibrated"] is True
    signature = fence.metadata["encoder_signature"]
    assert signature == {"model_name": DEFAULT_MODEL, "pooling": "cls", "text_prefix": "",
                         "normalization": "l2", "dimension": 384,
                         "revision": DEFAULT_REVISION}


def test_calibrate_from_hits_fits_a_runtime_fence():
    embedder = HashEmbedder()
    fence, report = calibrate_from_hits(benign_hits(), embedder, min_cosine=0.0)
    assert fence.metadata["joint_calibrated"] is True
    assert fence.metadata["encoder_signature"] == embedder.signature
    assert fence.policy == deployed_policy().fingerprint()
    assert fence.is_flat and fence.answer_rule == "either" and fence.eta_a is not None
    assert report["n_used"] == len(benign_hits())
    # A quantile over n hits can overshoot the budget by at most one hit.
    assert report["in_sample_false_rejection"] <= 0.05 + 1 / report["n_used"]
    assert report["heldout_false_rejection"] is not None


def test_calibrate_from_hits_refuses_bad_input():
    with pytest.raises(ValueError, match="nonempty string 'answer'"):
        calibrate_from_hits([{"key": "a b c d", "query": "a b c d"}], HashEmbedder())
    with pytest.raises(ValueError, match="at least 20 usable hits"):
        calibrate_from_hits(benign_hits()[:6], HashEmbedder(), min_cosine=0.0)
    with pytest.raises(ValueError, match="at least 20 usable hits"):
        calibrate_from_hits(benign_hits(), HashEmbedder(), min_cosine=1.01)


def test_open_cache_serves_through_a_calibrated_fence(tmp_path):
    pytest.importorskip("gptcache")
    from sentry.cache import open_cache
    embedder = HashEmbedder()
    fence, _ = calibrate_from_hits(benign_hits(), embedder, min_cosine=0.0)
    fence.save(tmp_path / "fence.json")
    calls = []
    llm = lambda query: calls.append(query) or "Paris."
    with open_cache(llm, fence=tmp_path / "fence.json", embedder=embedder) as cache:
        first = cache.ask("what is the capital of france and why is it well known")
        again = cache.ask("what is the capital of france and why is it well known")
    assert (first.source, again.source) == ("backend", "cache")
    assert again.decision is not None and not again.decision.blocked
    assert calls == ["what is the capital of france and why is it well known"]


def test_calibrate_cli_writes_a_fence(tmp_path, monkeypatch):
    from sentry.cache import runtime_cli
    hits = tmp_path / "hits.jsonl"
    hits.write_text("\n".join(json.dumps(hit) for hit in benign_hits()) + "\n")
    monkeypatch.setattr(runtime_cli, "_load_embedder", lambda args: HashEmbedder())
    runtime_cli.main(["calibrate", "--hits", str(hits), "--output", str(tmp_path / "f.json"),
                      "--min-cosine", "0"])
    fence = ExcessFence.load(tmp_path / "f.json")
    assert fence.metadata["encoder_signature"] == HashEmbedder().signature
    assert np.isfinite(fence.coefficients[0])
