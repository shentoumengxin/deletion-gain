"""Intent-grouped bootstrap 95% CIs for every BR@5% cell of Table 1 and the ISR/ASR cells.

Resamples benign intents and attack intents with replacement (cluster bootstrap),
refits the nominal threshold (benign 1-budget quantile) on the resampled benign arm,
reads BR on the resampled attack arm. B=2000, percentile interval; every cell draws from its
own generator seeded by (set, method, budget, arm) (boot.py), so adding a method leaves the
other cells' intervals unchanged.
Parity: with no resampling every cell must reproduce the published Table 1 value.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import json, os, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ.get('CD_REPO', str(REPO_ROOT)))
from sentry.cache.defense.calibrate import auroc
sys.path.insert(0, os.path.join(os.environ.get('CD_REPO', str(REPO_ROOT)), 'experiments/paper/rq1_detection'))
from experiments.paper.rq1_detection.rq1_operating_points import operating_point  # realized FPR over 200 intent-grouped half-splits

V3 = os.environ.get('CD_V3', str(RESULTS_ROOT / "v3"))
SC = (os.environ.get('CD_SC') or str(data_root() / "runs/paper/granularity32/scores"))
FLAGS = (os.environ.get('CD_FLAGS') or str(data_root() / "datasets/final500/eval/poisoned_flags.jsonl"))
SPEC = 'multi[count:4+width:2:cap16]/runs'

flags = {}
for l in open(FLAGS):
    r = json.loads(l)
    flags[r['record_id']] = r
    # KCA f2 remap gcg-f2-NNNN -> kca-f2-NNNN (README)
    if r['record_id'].startswith('gcg'):
        flags['kca' + r['record_id'][3:]] = r


def sets_map(name):
    s = json.load(open(f'{data_root()}/datasets/final500/sets/{name}_800.json'))
    return {rec['id']: rec['intent_id'] for rec in s['records']}


from experiments.paper.rq1_detection.boot import boot_br as _boot_br, boot_rate as _boot_rate  # per-cell seeded (boot.py)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


results = {}
for setname, scname, setfile in [('CAP', 'cap', 'lmp'), ('SCP', 'scp', 'scp'), ('KCA', 'kca', 'kca')]:
    d = json.load(open(f'{SC}/{scname}.json'))
    k = d['specs'].index(SPEC)
    rows = d['rows']
    att_rows = [r for r in rows if r['arm'] == 'attack']
    ben_rows = [r for r in rows if r['arm'] == 'genuine']
    dg_att = [r['exc'][k] for r in att_rows]; dg_att_int = [r['intent_id'] for r in att_rows]
    dg_ben = [r['exc'][k] for r in ben_rows]; dg_ben_int = [r['intent_id'] for r in ben_rows]
    cos_att = [-r['cos'] for r in att_rows]; cos_ben = [-r['cos'] for r in ben_rows]

    salt = json.load(open(f'{data_root()}/runs/paper/operating/{setfile}_salting_scores.json'))
    s128 = salt['salting, compressive k=128']
    rec_ids = s128['attack_record_ids']
    id2int = sets_map(setfile)
    salt_att_int = [id2int[i] for i in rec_ids]
    # alignment check: DG attack intent sequence == salting attack intent sequence
    assert len(salt_att_int) == len(dg_att_int), (len(salt_att_int), len(dg_att_int))
    assert salt_att_int == dg_att_int, 'attack row order differs between score dumps'
    assert s128['benign_intents'] == dg_ben_int, 'benign row order differs'
    def flag(i):
        for cand in (i, i.replace('gcg-f2', 'kca-f2'), 'kca' + i[3:]):
            if cand in flags: return bool(flags[cand]['poisoned'])
        raise KeyError(i)
    pois = np.array([flag(i) for i in rec_ids])
    n_pois = int(pois.sum())

    base = json.load(open(f'{data_root()}/runs/paper/operating/{setfile}_baseline_scores.json'))
    methods = {
        'cosine threshold': (cos_att, cos_ben),
        'cond. perplexity': (base['cond. perplexity']['attack'], base['cond. perplexity']['benign']),
        'bidirectional NLI': (base['bidirectional NLI']['attack'], base['bidirectional NLI']['benign']),
        # compressive random projection k=128 (rq2_salting.py); superseded in Table 1 by the
        # text-level key salting of Zhang et al. Sec. 7.1 below, kept for the record.
        'salted vectors (projection)': (s128['attack'], s128['benign']),
        # DG's erasure set read against an absolute cosine threshold (max_i cos(span_i, q) =
        # cos + exc). Not erase-and-check as released and not reported; kept for the record.
        'erasure, cosine read (not reported)': ([r['cos'] + r['exc'][k] for r in att_rows],
                                                [r['cos'] + r['exc'][k] for r in ben_rows]),
        'deletion gain': (dg_att, dg_ben),
    }
    # multi-encoder agreement (baselines/defense/multiview.py) from multiview.py output
    mv_path = perrow_root().joinpath(f'multiview_{scname}_e5.json')
    if mv_path.exists():
        mv = json.load(open(mv_path))
        mv_att = [r for r in mv['rows'] if r['arm'] == 'attack']; mv_ben = [r for r in mv['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in mv_att] == dg_att_int and [r['intent_id'] for r in mv_ben] == dg_ben_int
        assert max(abs(r['cos_primary'] - x['cos']) for r, x in zip(mv_att, att_rows)) < 2e-3
        dis = lambda r: max(0.0, r['cos_primary'] - (r['cos_aux0'] + r['cos_aux1']) / 2)
        mn = lambda r: -min(r['cos_primary'], r['cos_aux0'], r['cos_aux1'])
        methods['multi-encoder'] = ([dis(r) for r in mv_att], [dis(r) for r in mv_ben])
        methods['multi-encoder min-cos'] = ([mn(r) for r in mv_att], [mn(r) for r in mv_ben])
    def nominal(a, bn, budget=0.05, mask=None):
        a = np.asarray(a, float); bn = np.asarray(bn, float)
        if mask is not None: a = a[np.asarray(mask, bool)]
        return float((a > np.quantile(bn, 1 - budget)).mean())
    # key salting (Zhang et al. 2026, Sec. 7.1) from salting_text.py: a secret 5-token salt
    # written into the text before encoding; score = -salted cosine. Table 1 row: prefix, seed 0;
    # every (augmentation, seed) is summarized without bootstrap in res['salting_variants'].
    sl_path = perrow_root().joinpath(f'salting_{scname}_e5.json')
    salting_variants = {}
    if sl_path.exists():
        sl = json.load(open(sl_path))
        sl_att = [r for r in sl['rows'] if r['arm'] == 'attack']; sl_ben = [r for r in sl['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in sl_att] == dg_att_int and [r['intent_id'] for r in sl_ben] == dg_ben_int
        assert max(abs(r['plain'] - x['cos']) for r, x in zip(sl_att, att_rows)) < 2e-3
        methods['key salting'] = ([-r['prefix:0'] for r in sl_att], [-r['prefix:0'] for r in sl_ben])
        for key in [f'{aug}:{seed}' for aug in sl['augmentations'] for seed in sl['salts']]:
            a_ = [-r[key] for r in sl_att]; b_ = [-r[key] for r in sl_ben]
            salting_variants[key] = {'auc': float(auroc(np.asarray(a_), np.asarray(b_))),
                                     'att': nominal(a_, b_), 'succ': nominal(a_, b_, mask=pois)}
    # erase-and-check as released (eac_original.py: suffix mode, d=20, the authors' DistilBERT
    # safety classifier on the cached entry). Binary any-erasure flag -> rates with Wilson CIs
    # (res['eac_flag']); the continuous read max_i P(harmful) is placed at the budget.
    eac_path = perrow_root().joinpath(f'eac_{scname}.json')
    eac_flag = {}
    if eac_path.exists():
        ea = json.load(open(eac_path))
        ea_att = [r for r in ea['rows'] if r['arm'] == 'attack']; ea_ben = [r for r in ea['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in ea_att] == dg_att_int and [r['intent_id'] for r in ea_ben] == dg_ben_int
        methods['erase-and-check'] = ([r['p_max'] for r in ea_att], [r['p_max'] for r in ea_ben])
        fa = [r['flag'] for r in ea_att]; fb = [r['flag'] for r in ea_ben]
        fs = [f for f, m in zip(fa, pois) if m]
        eac_flag = {'attack': wilson(sum(fa), len(fa)), 'benign': wilson(sum(fb), len(fb)),
                    'succ': wilson(sum(fs), len(fs)), 'ms_per_hit_median': ea['ms_per_hit_median'],
                    'verify': ea.get('verify')}
    # LLM judge, continuous by voting (judge_vote.py): fraction of UNSAFE votes over N samples
    # of the reported prompt at temperature 1; placed at the budget like the other rows.
    jv_path = perrow_root().joinpath(f'judge_vote_{scname}.json')
    if jv_path.exists():
        jv = json.load(open(jv_path))
        jv_att = [r for r in jv['rows'] if r['arm'] == 'attack']; jv_ben = [r for r in jv['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in jv_att] == dg_att_int and [r['intent_id'] for r in jv_ben] == dg_ben_int
        methods['LLM judge (vote)'] = ([r['score'] for r in jv_att], [r['score'] for r in jv_ben])
    for m in ('cond. perplexity', 'bidirectional NLI'):
        assert base[m]['benign_intents'] == dg_ben_int
        assert len(base[m]['attack']) == len(dg_att)
    res = {'n_attack': len(dg_att), 'n_poisoned': n_pois, 'methods': {}, 'salting_variants': salting_variants, 'eac_flag': eac_flag}
    for m, (a, bn) in methods.items():
        att_ci = _boot_br(f'{setname}|{m}|0.05|att', a, dg_att_int, bn, dg_ben_int)
        suc_ci = _boot_br(f'{setname}|{m}|0.05|succ', a, dg_att_int, bn, dg_ben_int, mask=pois)
        att1 = _boot_br(f'{setname}|{m}|0.01|att', a, dg_att_int, bn, dg_ben_int, budget=0.01)
        auc = float(auroc(np.asarray(a, float), np.asarray(bn, float)))
        rf = {str(b): operating_point(np.asarray(a, float), np.asarray(bn, float), np.asarray(dg_ben_int), b) for b in (0.05, 0.01)}
        rf = {b: (v['realized_fpr_mean'], v['realized_fpr_sd']) for b, v in rf.items()}
        res['methods'][m] = {'att': att_ci, 'succ': suc_ci, 'att_1pct': att1, 'auc': auc, 'realized_fpr': rf}
        print(f'{setname:4s} {m:24s} AUC {auc:.3f} Att {att_ci[0]:.3f} [{att_ci[1]:.3f},{att_ci[2]:.3f}]  '
              f'Succ {suc_ci[0]:.3f} [{suc_ci[1]:.3f},{suc_ci[2]:.3f}]  '
              f'Att@1% {att1[0]:.3f} [{att1[1]:.3f},{att1[2]:.3f}]', flush=True)
    # ISR / ASR (defended = poisoned and not blocked at nominal threshold)
    thr = np.quantile(np.asarray(dg_ben), 0.95)
    blocked = np.asarray(dg_att) > thr
    isr = _boot_rate(f'{setname}|ISR', pois.astype(float), dg_att_int)
    asr = _boot_rate(f'{setname}|ASR+DG', (pois & ~blocked).astype(float), dg_att_int)
    res['isr'] = isr; res['asr_defended'] = asr
    print(f'{setname:4s} ISR {isr[0]:.3f} [{isr[1]:.3f},{isr[2]:.3f}]  ASR+DG {asr[0]:.4f} [{asr[1]:.4f},{asr[2]:.4f}]  n_pois={n_pois}', flush=True)
    results[setname] = res

# verdict-only LaCache: Wilson intervals on the block counts of the k=20 runs (CAP: detection/
# lacache_k20.json on 800 rows; SCP and KCA: the 798-row rescoring, scp_lacache_k20_798.json /
# kca_lacache_k20_full.json next to this script when present).
lac = {'CAP': (0.3925, 800), 'SCP': (0.36875, 800), 'KCA': (1.0, 798)}
for s_, f in (('SCP', 'scp_lacache_k20_798.json'), ('KCA', 'kca_lacache_k20_full.json')):
    fp = perrow_root().joinpath(f)
    if fp.exists():
        d = json.load(open(fp))['all_planted']['lacache_main']['POOLED']
        lac[s_] = (d['block'], d['n_attack'])
for s_, (p_, n) in lac.items():
    k_ = round(p_ * n); w = wilson(k_, n)
    results[s_]['lacache_att_wilson'] = w
    print(f'{s_} LaCache Att {w[0]:.3f} [{w[1]:.3f},{w[2]:.3f}] (Wilson, n={n})')

json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else str(perrow_root() / 'ci_table1.json'), 'w'), indent=1)
