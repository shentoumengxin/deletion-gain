"""Table 4 (cross-encoder) recomputation from per-row dumps at the deployed cut.

Per encoder: AUC, cosine-matched AUC (+support), BR Att./Succ. at the nominal 5% budget,
realized FPR mean+-sd over 200 intent-grouped half-splits (repo protocol), and
intent-grouped bootstrap 95% CIs for Att., Succ., and matched AUC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import sys, json, collections
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments/paper/rq1_detection"))
import numpy as np
from sentry.cache.defense.calibrate import matched_auroc, auroc
from experiments.paper.rq1_detection.rq1_operating_points import operating_point

S = str(perrow_root())
V3 = str(RESULTS_ROOT / "v3")
FLAGS = str(data_root() / "datasets/final500/eval/poisoned_flags.jsonl")
files = {
    'e5': str(perrow_root() / "deletion_v3_cap_e5_42.jsonl"),
    'bge': f'{S}/deletion_v3_cap_bge_42.jsonl',
    'gte': f'{S}/deletion_v3_cap_gte_42.jsonl',
    'minilm': f'{S}/deletion_v3_cap_minilm_42.jsonl',
}
sys.path.insert(0, S)
from experiments.paper.rq1_detection.boot import boot_br, boot_stat  # per-cell seeded; the e5 row shares its keys with ci_table1.py's CAP deletion gain

flags = {json.loads(l)['record_id']: json.loads(l)['poisoned'] for l in open(FLAGS)}
rec_ids = json.load(open(f'{data_root()}/runs/paper/operating/lmp_salting_scores.json'))['salting, compressive k=128']['attack_record_ids']
id2int = {r['id']: r['intent_id'] for r in json.load(open(f'{data_root()}/datasets/final500/sets/lmp_800.json'))['records']}
pois = np.array([bool(flags[i]) for i in rec_ids])
ref_int = [id2int[i] for i in rec_ids]


out = {}
for enc, path in files.items():
    rows = [json.loads(l) for l in open(path)]
    a = [r for r in rows if r['arm'] == 'attack']; b = [r for r in rows if r['arm'] == 'genuine']
    assert [r['intent_id'] for r in a] == ref_int, enc
    ax = np.array([r['excess_span'] for r in a]); bx = np.array([r['excess_span'] for r in b])
    ac = np.array([r['base_cos'] for r in a]); bc = np.array([r['base_cos'] for r in b])
    a_int = np.array([r['intent_id'] for r in a]); b_int = np.array([r['intent_id'] for r in b])
    thr = np.quantile(bx, 0.95)
    att = float((ax > thr).mean()); succ = float((ax[pois] > thr).mean())
    m, sup = matched_auroc(ax, bx, ac, bc, width=0.01)
    cell = operating_point(ax, bx, b_int, 0.05)
    name = 'deletion gain' if enc == 'e5' else f'deletion gain {enc}'
    att_ci = boot_br(f'CAP|{name}|0.05|att', ax, a_int, bx, b_int)
    suc_ci = boot_br(f'CAP|{name}|0.05|succ', ax, a_int, bx, b_int, mask=pois)
    assert abs(att_ci[0] - att) < 1e-12 and abs(suc_ci[0] - succ) < 1e-12
    ci = {'att': list(att_ci[1:]), 'succ': list(suc_ci[1:]),
          'matched': boot_stat(f'CAP|{name}|matched', lambda as_, bs: matched_auroc(ax[as_], bx[bs], ac[as_], bc[bs], width=0.01)[0], a_int, b_int),
          'auc': boot_stat(f'CAP|{name}|auc', lambda as_, bs: auroc(ax[as_], bx[bs]), a_int, b_int)}
    n_matched_nan = 0
    out[enc] = dict(auc=float(auroc(ax, bx)), matched=float(m), support=int(sup), att=att, succ=succ,
                    realized_fpr_mean=cell['realized_fpr_mean'], realized_fpr_sd=cell['realized_fpr_sd'],
                    realized_br_mean=cell['realized_br_mean'], realized_br_sd=cell['realized_br_sd'],
                    nominal_thr=float(thr), ci=ci, n_matched_nan=n_matched_nan)
    print(f"{enc:7s} AUC {out[enc]['auc']:.3f} [{ci['auc'][0]:.3f},{ci['auc'][1]:.3f}] matched {m:.3f} ({sup}) "
          f"[{ci['matched'][0]:.3f},{ci['matched'][1]:.3f}] Att {att:.3f} [{ci['att'][0]:.3f},{ci['att'][1]:.3f}] "
          f"Succ {succ:.3f} [{ci['succ'][0]:.3f},{ci['succ'][1]:.3f}] realizedFPR {cell['realized_fpr_mean']:.3f}+-{cell['realized_fpr_sd']:.3f} "
          f"nanmatched={out[enc]['n_matched_nan']}", flush=True)
json.dump(out, open(f'{S}/../table4.json', 'w'), indent=1)
# reference: published cand42 cells
for enc, f in [('bge', 'bge'), ('gte', 'gte'), ('minilm', 'minilm')]:
    c = json.load(open(f'{V3}/granularity32/cand42_crossenc/{f}.json'))
    print('published', enc, round(c['excess_auroc_cos_matched'], 3), c['matched_support'], c['excess_block_rate'], round(c['tpr_poisoned'], 3), round(c['achieved_benign_block_rate'], 3))
