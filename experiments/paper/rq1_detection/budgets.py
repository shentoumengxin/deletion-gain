"""BR at nominal FPR budgets 10/5/2/1% (tab/budgets.tex): threshold at the benign 1-b
quantile of the whole benign arm, same rows/scores as Table 1. Prints the LaTeX rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import json, numpy as np
import os
V3 = os.environ.get('CD_V3', str(RESULTS_ROOT / "v3"))
SC = (os.environ.get('CD_SC') or str(data_root() / "runs/paper/granularity32/scores"))
SPEC = 'multi[count:4+width:2:cap16]/runs'
budgets = [0.10, 0.05, 0.02, 0.01]
for setname, sc, sf in [('CAP', 'cap', 'lmp'), ('SCP', 'scp', 'scp'), ('KCA', 'kca', 'kca')]:
    d = json.load(open(f'{SC}/{sc}.json')); k = d['specs'].index(SPEC)
    att = [r for r in d['rows'] if r['arm'] == 'attack']; ben = [r for r in d['rows'] if r['arm'] == 'genuine']
    base = json.load(open(f'{data_root()}/runs/paper/operating/{sf}_baseline_scores.json'))
    s128 = json.load(open(f'{data_root()}/runs/paper/operating/{sf}_salting_scores.json'))['salting, compressive k=128']
    methods = {'cosine': ([-r['cos'] for r in att], [-r['cos'] for r in ben]),
               'perplexity': (base['cond. perplexity']['attack'], base['cond. perplexity']['benign']),
               'NLI': (base['bidirectional NLI']['attack'], base['bidirectional NLI']['benign']),
               'salted k=128 (projection, not reported)': (s128['attack'], s128['benign']),
               'erasure cosine read (not reported)': ([r['cos'] + r['exc'][k] for r in att], [r['cos'] + r['exc'][k] for r in ben]),
               'deletion gain': ([r['exc'][k] for r in att], [r['exc'][k] for r in ben])}
    import os
    mv_path = os.path.join(str(perrow_root()), f'multiview_{sc}_e5.json')
    if os.path.exists(mv_path):
        mv = json.load(open(mv_path)); ma = [r for r in mv['rows'] if r['arm'] == 'attack']; mb = [r for r in mv['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in ma] == [r['intent_id'] for r in att]
        dis = lambda r: max(0.0, r['cos_primary'] - (r['cos_aux0'] + r['cos_aux1']) / 2)
        mn = lambda r: -min(r['cos_primary'], r['cos_aux0'], r['cos_aux1'])
        methods['multi-encoder'] = ([dis(r) for r in ma], [dis(r) for r in mb])
        methods['multi-encoder min-cos'] = ([mn(r) for r in ma], [mn(r) for r in mb])
    here = str(perrow_root())
    sl_path = os.path.join(here, f'salting_{sc}_e5.json')   # key salting (Zhang et al. Sec. 7.1), prefix seed 0
    if os.path.exists(sl_path):
        sl = json.load(open(sl_path)); sa = [r for r in sl['rows'] if r['arm'] == 'attack']; sb = [r for r in sl['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in sa] == [r['intent_id'] for r in att]
        methods['key salting'] = ([-r['prefix:0'] for r in sa], [-r['prefix:0'] for r in sb])
    ea_path = os.path.join(here, f'eac_{sc}.json')   # erase-and-check as released, continuous read max_i P(harmful)
    if os.path.exists(ea_path):
        ea = json.load(open(ea_path)); xa = [r for r in ea['rows'] if r['arm'] == 'attack']; xb = [r for r in ea['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in xa] == [r['intent_id'] for r in att]
        methods['erase-and-check'] = ([r['p_max'] for r in xa], [r['p_max'] for r in xb])
    jv_path = os.path.join(here, f'judge_vote_{sc}.json')   # LLM judge, UNSAFE vote fraction
    if os.path.exists(jv_path):
        jv = json.load(open(jv_path)); ja = [r for r in jv['rows'] if r['arm'] == 'attack']; jb = [r for r in jv['rows'] if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in ja] == [r['intent_id'] for r in att]
        methods['LLM judge (vote)'] = ([r['score'] for r in ja], [r['score'] for r in jb])
    for m, (a, b) in methods.items():
        a = np.array(a, float); b = np.array(b, float)
        print(setname, m, [round(float((a > np.quantile(b, 1 - bud)).mean()), 3) for bud in budgets])
