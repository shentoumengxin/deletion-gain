"""Recompute the appendix NLI comparison from frozen, aligned scores.

No model calls. The original evaluation loader establishes score-array order;
record IDs and intent IDs are checked against the current response-label cohort.
Only aggregates, grouped bootstrap intervals, and source hashes are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from recompute_e2e_asr import arrays, decisions, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--paper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root))
    from experiments.paper.rq1_detection.v3_detect import load_rows

    sources = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        assert not raw.startswith(b'version https://git-lfs')
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines() if s.strip()] if lines else json.loads(raw)

    def auc(a, b):
        return float((a[:, None] > b).mean() + .5 * (a[:, None] == b).mean())

    root = args.data_root / 'runs/paper'
    previous = read(args.paper_root / 'data/baseline_asr_revision.json')
    current = read(args.paper_root / 'data/e2e_asr_revision.json')
    ci = read(root / 'perrow/ci_table1.json')
    result = {
        'model': 'cross-encoder/nli-MiniLM2-L6-H768',
        'score': '1 - min(P_entail(key, query), P_entail(query, key))',
        'nominal_fpr': .05,
        'asr': 'mean(retrieved AND defense_accepts AND poisoned_backend_response)',
        'retrieval_floor': .90,
        'bootstrap': {'replicates': args.replicates, 'unit': 'intent_id', 'refit_benign_threshold': True},
        'sources_sha256': sources,
        'cells': {},
    }
    specs = [('CAP', 'lmp', 'ndss', 'human_comqa'), ('SCP', 'scp', 'scp', 'human_comqa'),
             ('KCA', 'kca', 'gcg', 'cacheattack_cleaned_qa')]
    for name, stem, role, benign_generator in specs:
        ep = args.data_root / f'datasets/final500/eval/{stem}_eval.jsonl'
        read(ep, lines=True)
        loaded, _ = load_rows(ep, [role], [benign_generator])
        cp = root / f'perrow/answer_check_rows/e5-small-v2__{stem}.jsonl'
        rows = read(cp, lines=True)
        ours_cell = current['cells'][name]
        assert sources[str(cp)] == ours_cell.get('original_cohort_source_sha256', ours_cell['source_sha256'])
        if sources[str(cp)] != ours_cell['source_sha256']:
            aligned_path = args.data_root / ours_cell['source']
            aligned_rows = read(aligned_path, lines=True)
            assert sources[str(aligned_path)] == ours_cell['source_sha256']
            keyed = {r['record_id']: r for r in aligned_rows}
            assert len(keyed) == len(aligned_rows) == len(rows)
            aligned_rows = [keyed[r['record_id']] for r in rows]
            for old, new in zip(rows, aligned_rows):
                assert old.keys() == new.keys()
                assert all(old[k] == new[k] for k in old if k not in ('adl_best', 'echo_best'))
                if old['arm'] == 'genuine':
                    assert old == new
            rows = aligned_rows
        identity = lambda rr: [(r['record_id'], r['intent_id'], r['arm']) for r in rr]
        assert identity(loaded) == identity(rows)
        assert len({r['record_id'] for r in rows}) == len(rows)
        assert all(r['has_answer'] for r in rows)
        ar = [r for r in rows if r['arm'] == 'attack']
        br = [r for r in rows if r['arm'] == 'genuine']
        assert all(type(r['poisoned']) is bool for r in ar)
        a, b = arrays(ar), arrays(br)
        stored = read(root / f'operating/{stem}_baseline_scores.json')['bidirectional NLI']
        sidecar = read(root / f'operating/{stem}_salting_scores.json')['salting, compressive k=128']
        assert sidecar['attack_record_ids'] == [r['record_id'] for r in ar]
        assert stored['benign_intents'] == sidecar['benign_intents'] == [r['intent_id'] for r in br]
        x, y = np.asarray(stored['attack']), np.asarray(stored['benign'])
        assert len(x) == len(ar) and len(y) == len(br)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        threshold = float(np.quantile(y, .95))
        blocked = x > threshold
        poison = a['poisoned'].astype(bool)
        joint = (a['base_cos'] >= .90) & poison
        success = joint & ~blocked
        nli = {'auc': auc(x, y), 'br': float(blocked.mean()),
               'asr': float(success.mean()), 'successes': int(success.sum()),
               'threshold': threshold, 'poisoned_br': float(blocked[poison].mean())}
        historical = read(root / f'figures/v3/detection/baselines/{stem}_pplnli.json')['baselines_ppl_nli']['binli']
        assert abs(nli['auc'] - historical['auroc']) < 1e-14
        assert nli['br'] == historical['block_at_budget']
        assert nli['br'] == ci[name]['methods']['bidirectional NLI']['att'][0]
        assert nli['poisoned_br'] == ci[name]['methods']['bidirectional NLI']['succ'][0]

        ag, bg = groups(a), groups(b)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(('NLI|' + name).encode()).digest()[:8], 'little'))
        draws = []
        for _ in range(args.replicates):
            ai = np.concatenate([ag[j] for j in rng.integers(len(ag), size=len(ag))])
            bi = np.concatenate([bg[j] for j in rng.integers(len(bg), size=len(bg))])
            draws.append(float((joint[ai] & (x[ai] <= np.quantile(y[bi], .95))).mean()))
        nli['asr_ci95'] = np.percentile(draws, [2.5, 97.5]).tolist()
        nli['br_ci95'] = ci[name]['methods']['bidirectional NLI']['att'][1:]
        nli['br_by_fpr'] = {
            str(budget): float((x > np.quantile(y, 1 - budget)).mean())
            for budget in [.10, .05, .02, .01]
        }
        assert nli['br_by_fpr']['0.05'] == nli['br']
        assert nli['br_by_fpr']['0.01'] == ci[name]['methods']['bidirectional NLI']['att_1pct'][0]

        accept, _ = decisions(b, a)
        ours = {'auc': auc(a['excess_span'], b['excess_span']),
                'br': float((~accept['Ours']).mean()),
                'asr': float((joint & accept['Ours']).mean()),
                'successes': int((joint & accept['Ours']).sum())}
        assert ours['successes'] == previous['cells'][name]['methods']['Ours']['successes']
        cosine = previous['cells'][name]['methods']['Cosine']
        assert int((joint & accept['Cosine']).sum()) == cosine['successes']
        result['cells'][name] = {
            'n_attack': len(ar), 'n_benign': len(br), 'benign_generator': benign_generator,
            'n_attack_intents': len(ag), 'record_order_verified': True,
            'original_nli_auc_br_parity': True,
            'methods': {'Cosine': {k: cosine[k] for k in ['auc', 'br', 'asr', 'successes']},
                        'Bidirectional NLI': nli, 'Ours': ours},
        }
        print(name, 'NLI', nli['auc'], nli['br'], f"ASR {nli['successes']}/{len(ar)} = {nli['asr']:.6f}")
    for rel in ['experiments/paper/baselines/defense/binli.py',
                'experiments/paper/rq1_detection/rq1_baseline_scores.py',
                'experiments/paper/rq1_detection/v3_detect.py']:
        path = args.code_root / rel
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
