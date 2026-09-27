"""Owned GPTCache lifecycle with entry-side defense and atomic snapshot publication.

One runtime serializes calls within its process. Persistent directories have one owner;
this is not a distributed cache. Only explicitly flushed snapshots survive a restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import uuid
from typing import Callable

import numpy as np

from .defense.decide import DeletionDecision, DeletionDefenseConfig
from .defense.deletion import unit
from .defense.entry_store import InMemoryProfileStore
from .defense.fence import ExcessFence
from .defense.gptcache_plugin import DeletionVetoEvaluation
from .defense.insertion import install_profile_writer
from .defense.spans import SpanPolicy

@dataclass(frozen=True)
class RuntimeConfig:
    fence_path: Path | str
    encoder_signature: dict
    data_dir: Path | str | None = None
    policy: SpanPolicy = field(default_factory=lambda: SpanPolicy(mode='multi', components=('count:4','width:2:cap16'), form='runs'))
    storage_dtype: str = 'float16'
    cache_threshold: float = .90
    capacity: int = 10000

@dataclass(frozen=True)
class AskResult:
    answer: str
    source: str
    decision: DeletionDecision | None
    cache_write: str
    write_error: str | None = None

class _ClosedDataManager:
    def close(self):
        pass


class _CosineEvaluation:
    def range(self): return (0., 1.)
    def evaluation(self, src_dict, cache_dict, **kwargs):
        return float(unit(src_dict['embedding']) @ unit(cache_dict['embedding']))

class _OfflineEvaluation(DeletionVetoEvaluation):
    def refit_fence(self, *args, **kwargs):
        raise RuntimeError('online fence refitting is disabled; calibrate offline')

class CacheRuntime:
    def __init__(self, config: RuntimeConfig, embedder, backend: Callable[[str], str]):
        from gptcache import Cache
        from gptcache.config import Config
        from gptcache.manager import manager_factory
        self.config, self.embedder, self.backend = config, embedder, backend
        self._lock = threading.RLock()
        self._closed = False
        self._owner = None
        signature = getattr(embedder, 'signature', None)
        if (not isinstance(signature, dict)
                or not {'model_name','revision','pooling','text_prefix','normalization','dimension'} <= signature.keys()
                or signature.get('dimension') != embedder.dimension
                or signature.get('model_name') != embedder.model_name
                or signature != config.encoder_signature):
            raise ValueError('embedder signature does not match runtime encoder_signature')
        if config.storage_dtype != 'float16' or config.cache_threshold != .90:
            raise ValueError('runtime requires calibrated float16 storage and cosine threshold 0.90')
        fence = ExcessFence.load(config.fence_path)
        if (fence.embedder != embedder.model_name or fence.policy != config.policy.fingerprint()
                or fence.direction != 'entry' or not fence.is_flat
                or fence.answer_rule != 'either' or fence.echo_min != 1
                or not fence.metadata.get('joint_calibrated')
                or fence.metadata.get('encoder_signature') != signature):
            raise ValueError('fence identity/signature must match the entry-side flat joint either policy')
        if not np.all(np.isfinite(fence.coefficients)) or not np.isfinite(fence.eta_a):
            raise ValueError('fence thresholds must be finite')
        self._identity = {'schema':1, 'encoder_signature':signature,
                          'policy':config.policy.fingerprint(), 'min_segments':config.policy.min_segments,
                          'storage_dtype':config.storage_dtype, 'cache_threshold':config.cache_threshold,
                          'capacity':config.capacity,
                          'fence':fence.to_dict()}
        self._root = Path(config.data_dir).resolve() if config.data_dir is not None else None
        self._temporary = tempfile.TemporaryDirectory(prefix='sentry-runtime-')
        self._working = Path(self._temporary.name)
        try:
            if self._root is not None:
                self._root.mkdir(parents=True, exist_ok=True)
                # OS advisory ownership lock is released even if the process dies.
                import fcntl
                self._owner = (self._root / 'LOCK').open('a+b')
                try: fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: raise RuntimeError('cache directory already has a runtime owner') from None
                current = self._root / 'CURRENT'
                if current.exists():
                    name = current.read_text().strip()
                    if Path(name).name != name or not name.startswith('snapshot-'):
                        raise ValueError('invalid cache snapshot pointer')
                    snapshot = self._root / name
                    saved = json.loads((snapshot / 'identity.json').read_text())
                    if saved != self._identity:
                        raise ValueError('persisted cache encoder/fence/policy identity mismatch')
                    manifest = json.loads((snapshot / 'manifest.json').read_text())
                    required = {'identity.json','host/sqlite.db','host/faiss.index',
                                'profiles/meta.json','profiles/profiles.npz'}
                    if not required <= manifest.keys():
                        raise ValueError('cache snapshot manifest is incomplete')
                    for relative, digest in manifest.items():
                        path = snapshot / relative
                        if path.resolve().parent not in (snapshot.resolve(), (snapshot/'host').resolve(), (snapshot/'profiles').resolve()):
                            raise ValueError('invalid snapshot member')
                        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                            raise ValueError('cache snapshot checksum mismatch')
                    shutil.copytree(snapshot / 'host', self._working / 'host')
                    self.store = InMemoryProfileStore.load(snapshot / 'profiles')
                else:
                    self.store = InMemoryProfileStore(embedder.model_name, config.policy.fingerprint(), config.capacity)
            else:
                self.store = InMemoryProfileStore(embedder.model_name, config.policy.fingerprint(), config.capacity)
            if self.store.embedder != embedder.model_name or self.store.policy != config.policy.fingerprint():
                raise ValueError('persisted profile store identity mismatch')
            manager = manager_factory('sqlite,faiss', data_dir=str(self._working/'host'),
                                      max_size=config.capacity, eviction_params={'max_size':config.capacity},
                                      vector_params={'dimension':embedder.dimension, 'top_k':1})
            self.evaluator = _OfflineEvaluation(_CosineEvaluation(), self.store, fence,
                                                DeletionDefenseConfig(cache_threshold=.90))
            self.cache = Cache()
            self.cache.init(pre_embedding_func=lambda data, **kwargs: data['prompt'],
                            embedding_func=self._encode_query, data_manager=manager,
                            similarity_evaluation=self.evaluator,
                            config=Config(similarity_threshold=.90, auto_flush=1000000000, disable_report=True),
                            next_cache=None)
            self.writer = install_profile_writer(self.cache, self.store, embedder, config.policy,
                                                  storage_dtype=config.storage_dtype)
        except BaseException:
            if self._owner is not None: self._owner.close()
            self._temporary.cleanup()
            raise

    def _encode_query(self, query, **kwargs):
        vector = unit(self.embedder.encode([query])[0]).astype(np.float32)
        if vector.shape != (self.embedder.dimension,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) < 1e-12:
            raise ValueError('embedder returned an invalid query vector')
        return vector

    def ask(self, query: str) -> AskResult:
        from gptcache.adapter.adapter import adapt
        with self._lock:
            if self._closed: raise RuntimeError('runtime is closed')
            if not isinstance(query, str) or not query.strip(): raise ValueError('query must be nonempty text')
            self.evaluator.last_decision = None
            state = {'source':'cache', 'write':'not_attempted', 'error':None, 'answer':None}
            def call_backend(*args, **kwargs):
                state['source'] = 'backend'
                answer = self.backend(query)
                if not isinstance(answer, str): raise TypeError('backend must return str')
                state['answer'] = answer
                return answer
            def update(answer, update_cache_func, *args, **kwargs):
                try:
                    errors = self.writer.counters.errors
                    profiled = self.writer.counters.profiled
                    update_cache_func(answer)
                    state['write'] = ('written' if self.writer.counters.profiled > profiled else 'unprofiled')
                    if self.writer.counters.errors > errors: state['error'] = 'profile construction or storage failed'
                except Exception as exc:
                    state['write'], state['error'] = 'failed', type(exc).__name__
                return answer
            answer = adapt(call_backend, lambda answer: answer, update,
                           prompt=query, cache_obj=self.cache, top_k=1)
            # GPTCache treats an empty backend string as no result; preserve the API result.
            if state['source']=='backend' and state['answer']=='': answer=''
            return AskResult(answer, state['source'], self.evaluator.last_decision, state['write'], state['error'])

    def flush(self):
        with self._lock:
            if self._closed: raise RuntimeError('runtime is closed')
            self.cache.flush()
            if self._root is None: return
            current = self._root / 'CURRENT'
            previous = current.read_text().strip() if current.exists() else None
            snapshot = self._root / ('snapshot-' + uuid.uuid4().hex)
            pointer = self._root / ('CURRENT-' + uuid.uuid4().hex)
            published = False
            try:
                snapshot.mkdir()
                shutil.copytree(self._working/'host', snapshot/'host')
                # Include committed SQLite WAL pages while the host stays open.
                with sqlite3.connect(self._working/'host'/'sqlite.db') as src:
                    with sqlite3.connect(snapshot/'host'/'sqlite.db') as dst: src.backup(dst)
                self.store.save(snapshot/'profiles')
                (snapshot/'identity.json').write_text(json.dumps(self._identity, sort_keys=True))
                manifest = {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in snapshot.rglob('*') if p.is_file()}
                (snapshot/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
                for path in snapshot.rglob('*'):
                    if path.is_file():
                        with path.open('rb') as stream: os.fsync(stream.fileno())
                for directory in (snapshot/'host', snapshot/'profiles', snapshot):
                    self._fsync_directory(directory)
                with pointer.open('w') as stream:
                    stream.write(snapshot.name + '\n'); stream.flush(); os.fsync(stream.fileno())
                os.replace(pointer, current)
                published = True
                self._fsync_directory(self._root)
                # Publication succeeds before any older recovery point is removed.
                for old in self._root.iterdir():
                    if (old.name not in {snapshot.name, previous}
                            and re.fullmatch(r'snapshot-[0-9a-f]{32}', old.name)
                            and old.is_dir() and not old.is_symlink()):
                        shutil.rmtree(old)
                self._fsync_directory(self._root)
            except BaseException:
                if not published:
                    shutil.rmtree(snapshot, ignore_errors=True)
                pointer.unlink(missing_ok=True)
                raise

    @staticmethod
    def _fsync_directory(directory):
        fd = os.open(directory, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def close(self):
        with self._lock:
            if self._closed: return
            try:
                self.flush()
            finally:
                try:
                    self.writer.close()
                finally:
                    try:
                        # GPTCache's SQLStorage.close() is a no-op in 0.1.44.
                        self.writer.inner.s._engine.dispose()
                    finally:
                        # Cache.init registers an atexit closure holding the Cache.
                        # Leave that closure a tiny no-op owner, not a route back to
                        # this runtime, the encoder, profiles, or a deleted workdir.
                        self.cache.data_manager = _ClosedDataManager()
                        self.cache.embedding_func = None
                        self.cache.similarity_evaluation = None
                        self.cache.has_init = False
                        self.embedder = self.backend = None
                        self.writer = self.evaluator = self.store = None
                        self._closed = True
                        if self._owner is not None: self._owner.close()
                        self._temporary.cleanup()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

def create_runtime(config: RuntimeConfig, embedder, backend: Callable[[str], str]) -> CacheRuntime:
    return CacheRuntime(config, embedder, backend)
