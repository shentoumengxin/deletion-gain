"""Research clients resolve explicit environment configuration without secret files."""
from pathlib import Path


def test_credentials_are_read_only_from_explicit_environment(monkeypatch):
    from sentry.research.operators import load_env

    monkeypatch.setenv("PARAPHRASE_API_BASE_URL", "https://model.invalid/v1/")
    monkeypatch.setenv("PARAPHRASE_API_KEY", "test-placeholder")
    monkeypatch.setenv("PARAPHRASE_API_MODEL", "fixture-model")
    monkeypatch.setattr(Path, "exists", lambda self: True)

    def deny_read(self, *args, **kwargs):
        raise AssertionError("A credential file must not be read")

    monkeypatch.setattr(Path, "read_text", deny_read)
    assert load_env() == {"base_url": "https://model.invalid/v1",
                          "api_key": "test-placeholder", "model": "fixture-model"}


def test_default_response_cache_uses_the_external_data_root(monkeypatch, tmp_path):
    from sentry.research.operators import default_cache_dir

    monkeypatch.delenv("ORBIT_CACHE_DIR", raising=False)
    monkeypatch.setenv("SENTRY_DATA_ROOT", str(tmp_path))
    assert default_cache_dir() == tmp_path / "responses" / "local"
