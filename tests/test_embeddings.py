"""Shared encoders must be usable without loading the research pipeline."""
import numpy as np


def test_hash_encoder_has_a_complete_stable_identity_and_normalized_batches():
    from sentry.embeddings import HashEmbedder

    encoder = HashEmbedder(32)
    assert encoder.signature == {
        "model_name": encoder.model_name,
        "pooling": "feature-hash",
        "text_prefix": "",
        "normalization": "l2",
        "dimension": 32,
        "revision": "builtin-v1",
    }
    texts = ["Who wrote this book?", "Where does the river start?"]
    np.testing.assert_array_equal(encoder.encode(texts), encoder.encode(texts))
    np.testing.assert_allclose(np.linalg.norm(encoder.encode(texts), axis=1), 1)
    assert encoder.encode([]).shape == (0, 32)


def test_core_import_does_not_load_research_or_model_clients():
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-c",
        "import sys; import sentry.cache.defense.deletion; "
        "assert 'requests' not in sys.modules; "
        "assert 'pandas' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'gptcache' not in sys.modules",
    ], check=True)
