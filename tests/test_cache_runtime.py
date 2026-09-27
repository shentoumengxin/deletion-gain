"""Offline lifecycle tests against a real GPTCache host; no model downloads."""
from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('gptcache')
from sentry.cache.runtime import RuntimeConfig, create_runtime
from sentry.cache.defense.fence import ExcessFence
from sentry.cache.defense.entry_store import text_key, InMemoryProfileStore

class Embedder:
    model_name = 'runtime-fixture'
    dimension = 3
    signature = {'model_name': model_name, 'revision':'fixture-v1', 'pooling':'synthetic', 'normalization':'l2', 'text_prefix':'', 'dimension':3}
    def __init__(self): self.calls=[]
    def encode(self, texts):
        self.calls.append(list(texts))
        return np.tile([1.,0.,0.], (len(texts),1))

def config(tmp_path):
    cfg=RuntimeConfig(fence_path=tmp_path/'fence.json', encoder_signature=Embedder.signature, data_dir=tmp_path/'state')
    fence=ExcessFence(np.array([.01,0,0]), .05, embedder=Embedder.model_name, policy=cfg.policy.fingerprint(), answer_rule='either', eta_a=.01, metadata={'joint_calibrated':True,'encoder_signature':Embedder.signature})
    fence.save(cfg.fence_path)
    return cfg

def test_miss_hit_restart_and_close(tmp_path):
    cfg=config(tmp_path); emb=Embedder(); calls=[]
    with create_runtime(cfg,emb,lambda q: calls.append(q) or 'blue') as rt:
        first=rt.ask('what color is the clear sky today')
        assert (first.answer, first.source, first.cache_write)==('blue','backend','written')
        emb.calls.clear()
        hit=rt.ask('what color is the clear sky today')
        assert hit.source=='cache' and len(emb.calls)==1
        assert len(calls)==1
    rt.close()
    with pytest.raises(RuntimeError): rt.ask('closed')
    with create_runtime(cfg,Embedder(),lambda q: pytest.fail('unexpected backend')) as restored:
        assert restored.ask('what color is the clear sky today').source=='cache'
        profile=restored.store.get(text_key('what color is the clear sky today'))
        assert profile.storage_dtype=='float16' and profile.spans.dtype==np.float16

def test_backend_failure_does_not_write(tmp_path):
    with create_runtime(config(tmp_path),Embedder(),lambda q: (_ for _ in ()).throw(ValueError('backend'))) as rt:
        with pytest.raises(ValueError,match='backend'): rt.ask('what color is the clear sky today')
        assert len(rt.store)==0

def test_write_failure_preserves_answer(tmp_path,monkeypatch):
    with create_runtime(config(tmp_path),Embedder(),lambda q:'blue') as rt:
        monkeypatch.setattr(rt.writer.inner,'save',lambda *a,**k: (_ for _ in ()).throw(OSError('disk')))
        result=rt.ask('what color is the clear sky today')
        assert result.answer=='blue' and result.cache_write=='failed'

def test_signature_mismatch_refused(tmp_path):
    cfg=config(tmp_path)
    with pytest.raises(ValueError,match='signature'):
        create_runtime(replace(cfg,encoder_signature={'wrong':True}),Embedder(),lambda q:'blue')

@pytest.mark.parametrize('answer', ['other answer', None])
def test_stale_answer_fields_cannot_rescue(tmp_path,answer):
    cfg=config(tmp_path)
    with create_runtime(cfg,Embedder(),lambda q:'blue') as rt:
        text='what color is the clear sky today'
        rt.ask(text)
        profile=rt.store.get(text_key(text))
        # Harmless geometric fixture: shortened versions align with the query.
        profile=replace(profile, whole=np.array([.95,np.sqrt(1-.95**2),0.]), answer_loss=np.zeros(len(profile.spans)))
        rt.store.put(text_key(text),profile)
        rt.evaluator.evaluation({'question':text,'embedding':np.array([1.,0,0])}, {'question':text,'answer':'blue','embedding':profile.whole})
        assert not rt.evaluator.last_decision.blocked
        rt.evaluator.evaluation({'question':text,'embedding':np.array([1.,0,0])}, {'question':text,'answer':answer,'embedding':profile.whole})
        assert rt.evaluator.last_decision.blocked
        assert rt.evaluator.counters.no_answer_fields==1

def test_failed_profile_replacement_removes_old_profile(tmp_path,monkeypatch):
    with create_runtime(config(tmp_path),Embedder(),lambda q:'blue') as rt:
        text='what color is the clear sky today'; rt.ask(text)
        monkeypatch.setattr(rt.writer,'_build',lambda *a,**k:None)
        rt.writer.save(text,'new answer',np.array([1.,0,0]))
        assert rt.store.get(text_key(text)) is None
        result=rt.ask(text)
        assert result.source=='backend' and result.decision.reason=='no_profile'

def test_failed_snapshot_keeps_previous_current(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    rt=create_runtime(cfg,Embedder(),lambda q:'blue'); rt.ask('what color is the clear sky today'); rt.flush()
    pointer=(cfg.data_dir/'CURRENT').read_text()
    with monkeypatch.context() as patch:
        patch.setattr(rt.store,'save',lambda *a: (_ for _ in ()).throw(OSError('snapshot')))
        with pytest.raises(OSError): rt.flush()
        assert (cfg.data_dir/'CURRENT').read_text()==pointer
    rt.close()

def test_veto_backend_once_and_online_refit_disabled(tmp_path):
    calls=[]
    with create_runtime(config(tmp_path),Embedder(),lambda q:calls.append(q) or 'blue') as rt:
        text='what color is the clear sky today'; rt.ask(text)
        p=rt.store.get(text_key(text))
        rt.store.put(text_key(text),replace(p,whole=np.array([.95,np.sqrt(1-.95**2),0]),answer_loss=np.ones(len(p.spans))))
        result=rt.ask(text)
        assert result.source=='backend' and result.decision.blocked and len(calls)==2
        with pytest.raises(RuntimeError,match='disabled'): rt.evaluator.refit_fence()

def test_persistent_directory_has_one_owner(tmp_path):
    cfg=config(tmp_path)
    with create_runtime(cfg,Embedder(),lambda q:'blue'):
        with pytest.raises(RuntimeError,match='owner'): create_runtime(cfg,Embedder(),lambda q:'blue')

def test_export_selects_joint_fit_partition(tmp_path):
    from sentry.cache.defense.calibrate import export_runtime_fence
    cfg=config(tmp_path)
    base=ExcessFence.load(cfg.fence_path)
    report={'fence_form':'flat','storage_dtype':'float16','embedder':Embedder.model_name,
            'fence':base.to_dict(),'n_fit_intents':20,
            'answer_rules':{'either':{'fitted':True,'echo_min':1,'eta_joint_holdout':.012,
              'eta_a_holdout':.02,'joint_budget_reachable_holdout':True,
              'achieved_benign_block_rate_joint_eta':.05}}}
    exported=export_runtime_fence(report,tmp_path/'export.json',Embedder.signature)
    assert exported.coefficients[0]==.012 and exported.eta_a==.02
    assert exported.metadata['calibration_partition']=='fit_intents'
    create_runtime(replace(cfg,fence_path=tmp_path/'export.json',data_dir=None),Embedder(),lambda q:'blue').close()

def test_rescue_is_a_real_cache_hit_without_backend(tmp_path):
    calls=[]
    with create_runtime(config(tmp_path),Embedder(),lambda q:calls.append(q) or 'blue') as rt:
        text='what color is the clear sky today'; rt.ask(text)
        p=rt.store.get(text_key(text))
        rt.store.put(text_key(text),replace(p,whole=np.array([.95,np.sqrt(1-.95**2),0]),answer_loss=np.zeros(len(p.spans))))
        rt.embedder.calls.clear()
        result=rt.ask(text)
        assert result.source=='cache' and not result.decision.blocked
        assert result.decision.answer_reason and len(calls)==1 and len(rt.embedder.calls)==1

def test_corrupted_snapshot_rejected(tmp_path):
    cfg=config(tmp_path)
    with create_runtime(cfg,Embedder(),lambda q:'blue') as rt: rt.ask('what color is the clear sky today')
    snapshot=cfg.data_dir/(cfg.data_dir/'CURRENT').read_text().strip()
    with (snapshot/'profiles'/'profiles.npz').open('ab') as out: out.write(b'broken')
    with pytest.raises(ValueError,match='checksum'): create_runtime(cfg,Embedder(),lambda q:'blue')

def test_host_write_failure_preserves_existing_profile(tmp_path,monkeypatch):
    with create_runtime(config(tmp_path),Embedder(),lambda q:'blue') as rt:
        text='what color is the clear sky today'; rt.ask(text)
        original=rt.store.get(text_key(text))
        monkeypatch.setattr(rt.writer.inner,'save',lambda *a,**k: (_ for _ in ()).throw(OSError('write')))
        with pytest.raises(OSError): rt.writer.save(text,'new answer',np.array([1.,0,0]))
        assert rt.store.get(text_key(text)) is original

def test_delete_failure_does_not_prevent_host_write(tmp_path,monkeypatch):
    with create_runtime(config(tmp_path),Embedder(),lambda q:'blue') as rt:
        monkeypatch.setattr(rt.store,'delete',lambda *a: (_ for _ in ()).throw(OSError('delete')))
        result=rt.ask('what color is the clear sky today')
        assert result.answer=='blue' and result.cache_write=='written'
        assert rt.ask('what color is the clear sky today').source=='cache'

def test_snapshots_retain_only_current_and_previous(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    with create_runtime(cfg,Embedder(),lambda q:'blue') as rt:
        rt.ask('what color is the clear sky today')
        untouched=cfg.data_dir/'snapshot-user-owned'; untouched.mkdir()
        names=[]
        for _ in range(4):
            rt.flush(); names.append((cfg.data_dir/'CURRENT').read_text().strip())
        assert {p.name for p in cfg.data_dir.glob('snapshot-*')}==set(names[-2:])|{untouched.name}
        before=set(cfg.data_dir.iterdir())
        with monkeypatch.context() as patch:
            patch.setattr(rt.store,'save',lambda *a: (_ for _ in ()).throw(OSError('snapshot')))
            with pytest.raises(OSError): rt.flush()
        assert set(cfg.data_dir.iterdir())==before

def test_closed_runtime_and_encoder_are_collectable(tmp_path):
    import gc
    import weakref
    encoder=Embedder(); rt=create_runtime(config(tmp_path),encoder,lambda q:'blue')
    runtime_ref=weakref.ref(rt); encoder_ref=weakref.ref(encoder)
    rt.close(); del rt,encoder; gc.collect()
    assert runtime_ref() is None and encoder_ref() is None
