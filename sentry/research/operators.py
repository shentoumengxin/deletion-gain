"""Paraphrase operators and cached model responses used to construct research corpora.

P constructs candidates; the independent validator V decides equivalence. No serving
module imports this client. Responses are stored in the external data root, and
credentials are supplied through explicit environment variables.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

# Resolve only when constructing a client, so importing offline analysis needs no data root.
_DEFAULT_CACHE_DIR = None


def default_cache_dir() -> Path:
    explicit = os.environ.get("ORBIT_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    from sentry.artifacts import data_path
    return data_path("responses", "local")


def load_env() -> dict:
    """Read operator credentials from the explicit process environment only."""
    base_url = os.environ.get("PARAPHRASE_API_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("PARAPHRASE_API_KEY", "")
    model = os.environ.get("PARAPHRASE_API_MODEL", "")
    if base_url and api_key and model:
        return {"base_url": base_url, "api_key": api_key, "model": model}
    raise RuntimeError("Set PARAPHRASE_API_BASE_URL, PARAPHRASE_API_KEY and PARAPHRASE_API_MODEL")


class Client:
    """OpenAI-compatible chat client with on-disk response caching.

    Identical model requests reuse their saved responses. This client is used
    by corpus construction and offline baselines, never the serving defense.
    """

    def __init__(self, creds: dict, cache_name: str, cache_dir: Path | None = _DEFAULT_CACHE_DIR):
        self.creds = creds
        self.url = creds["base_url"]
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"{cache_name}.jsonl"
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:  # tolerate a partial line left by a killed process
                    row = json.loads(line)
                    self._cache[row["key"]] = row["content"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def _key(self, messages, temperature, seed, max_tokens, extra_body=None) -> str:
        parts = [messages, temperature, seed, max_tokens, self.creds["model"]]
        # Appended only when non-empty, so every key written by a caller that does not
        # use extra_body keeps
        # hashing to what it already hashed to.
        if extra_body:
            parts.append(extra_body)
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def chat(self, messages, temperature=0.9, seed=0, max_tokens=400,
             extra_body: dict | None = None) -> str:
        """One completion, served from disk when the same draw was made before.

        ``extra_body`` carries provider parameters the OpenAI-compatible shape has no
        field for. The one that matters here is DeepSeek's ``thinking``: left at its
        default, ``deepseek-v4-flash`` spends the entire token budget on reasoning and
        returns **empty content** — measured at 2,000 reasoning tokens for nothing,
        against 19 completion tokens with thinking disabled. A caller that forgets it
        pays for every call and receives no paraphrase.
        """
        key = self._key(messages, temperature, seed, max_tokens, extra_body)
        if key in self._cache:
            with self._lock:
                self.cache_hits += 1
            return self._cache[key]
        last = ""
        for attempt in range(4):
            try:
                response = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.creds['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.creds["model"],
                        "messages": messages,
                        "temperature": temperature,
                        "seed": seed + attempt,
                        "max_tokens": max_tokens,
                        **(extra_body or {}),
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    with self._lock:
                        self.calls += 1
                        self._cache[key] = content
                        with self.cache_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({"key": key, "content": content}) + "\n")
                    return content
                last = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"operator chat failed after retries: {last}")
