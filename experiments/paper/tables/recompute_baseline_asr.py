"""Compute all main-table ASRs from aligned, frozen defensive evaluation rows.

No attacks or responses are generated. Per-row inputs remain external; the output
contains aggregate counts, grouped bootstrap intervals and source hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from recompute_e2e_asr import arrays, decisions, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--lacache-scores', type=Path, required=True)
    parser.add_argument('--lacache-source', required=True,
                        help='Persistent location of the external score artifact')
    parser.add_argument('--ours-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    sources = {}
    def read(path, lines=False):
        raw = path.read_bytes()
        assert not raw.startswith(b'version https://git-lfs')
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)
    root = args.data_root / 'runs/paper'
    ref = read(root / 'perrow/ci_table1.json')
    lacache = read(args.lacache_scores)
    sources[args.lacache_source] = sources.pop(str(args.lacache_scores))
    ours = read(args.ours_summary)
    result = {'metric': 'End-to-End Attack Success Rate', 'abbreviation': 'ASR',
              'definition': 'mean(retrieved AND accepted AND poisoned_backend_response)',
              'denominator': 'all attack records in the detection cohort',
              'retrieval_floor': .90, 'nominal_fpr': .05,
              'retrieval': 'Key salting uses salted cosine; all other methods use the original key-query cosine.',
              'bootstrap_replicates': args.replicates,
              'bootstrap': 'resample benign and attack intent groups and refit every threshold',
              'sources_sha256': sources, 'cells': {}}
    for name, stem in [('CAP', 'lmp'), ('SCP', 'scp'), ('KCA', 'kca')]:
        short = name.lower()
        cohort_path = root / f'perrow/answer_check_rows/e5-small-v2__{stem}.jsonl'
        records = read(cohort_path, lines=True)
        br = [r for r in records if r['arm'] == 'genuine']
        ar = [r for r in records if r['arm'] == 'attack']
        ids = {r['record_id'] for r in records}
        assert len(ids) == len(records) and all(r['has_answer'] for r in records)
        assert all(type(r['poisoned']) is bool for r in ar)
        b, a = arrays(br), arrays(ar)
        ours_cell = ours['cells'][name]
        assert sources[str(cohort_path)] == ours_cell.get('original_cohort_source_sha256', ours_cell['source_sha256'])
        ours_records = records
        if ours_cell['source_sha256'] != sources[str(cohort_path)]:
            aligned_path = args.data_root / ours_cell['source']
            aligned_rows = read(aligned_path, lines=True)
            assert sources[str(aligned_path)] == ours_cell['source_sha256']
            keyed = {r['record_id']: r for r in aligned_rows}
            assert len(keyed) == len(aligned_rows) == len(records)
            ours_records = [keyed[r['record_id']] for r in records]
            for old, new in zip(records, ours_records):
                assert old.keys() == new.keys()
                assert all(old[k] == new[k] for k in old if k not in ('adl_best', 'echo_best'))
                if old['arm'] == 'genuine':
                    assert old == new
        assert sources[str(cohort_path)] == lacache['sets'][name]['cohort_sha256']
        def aligned(rows, score):
            mapping = {r['record_id']: r for r in rows}
            assert len(mapping) == len(rows) and set(mapping) == ids
            for r in records:
                assert mapping[r['record_id']]['intent_id'] == r['intent_id']
                assert mapping[r['record_id']]['arm'] == r['arm']
            return (np.array([score(mapping[r['record_id']]) for r in ar]),
                    np.array([score(mapping[r['record_id']]) for r in br]))
        scores = {}
        for method, filename, score in [
            ('Multi-embedding', f'multiview_{short}_e5.json', lambda r: -min(r['cos_primary'], r['cos_aux0'], r['cos_aux1'])),
            ('Key salting', f'salting_{short}_e5.json', lambda r: -r['prefix:0']),
            ('LLM judge', f'judge_vote_{short}.json', lambda r: r['score']),
            ('Erase-and-check', f'eac_{short}.json', lambda r: r['p_max']),
        ]:
            scores[method] = aligned(read(root / 'perrow' / filename)['rows'], score)
        scores['LaCache'] = aligned(lacache['sets'][name]['rows'], lambda r: r['score'])
        # The original baseline exporter stores ordered arrays. Its ID sidecar
        # and the original cosine dump establish the same attack/benign order.
        sidecar = read(root / f'operating/{stem}_salting_scores.json')['salting, compressive k=128']
        assert sidecar['attack_record_ids'] == [r['record_id'] for r in ar]
        assert sidecar['benign_intents'] == [r['intent_id'] for r in br]
        cosine_rows = read(root / f'granularity32/scores/{short}.json')['rows']
        ca = [r for r in cosine_rows if r['arm'] == 'attack']
        cb = [r for r in cosine_rows if r['arm'] == 'genuine']
        assert [r['intent_id'] for r in ca] == [r['intent_id'] for r in ar]
        assert [r['intent_id'] for r in cb] == sidecar['benign_intents']
        scores['Cosine'] = (np.array([-r['cos'] for r in ca]), np.array([-r['cos'] for r in cb]))
        ppl = read(root / f'operating/{stem}_baseline_scores.json')['cond. perplexity']
        assert ppl['benign_intents'] == sidecar['benign_intents']
        scores['Perplexity'] = (np.array(ppl['attack']), np.array(ppl['benign']))
        if name == 'CAP':
            lr = read(root / 'figures/v3/detection/lacache_k20.json')['sets']['LMP']['encoders']['e5-small-v2']
            lac_ref = (lr['all_planted']['POOLED']['block_at_5pct'],
                       lr['poisoned']['POOLED']['block_at_5pct'], lr['all_planted']['POOLED']['auroc'])
        else:
            lr = read(root / f'perrow/{short}_lacache_k20_{"798" if name == "SCP" else "full"}.json')
            lac_ref = (lr['all_planted']['lacache_main']['POOLED']['block'],
                       lr['poisoned']['lacache_main']['POOLED']['block'],
                       lr['all_planted']['lacache_main']['POOLED']['auroc'])
        names = {'Cosine': 'cosine threshold', 'Perplexity': 'cond. perplexity',
                 'Multi-embedding': 'multi-encoder min-cos', 'Key salting': 'key salting',
                 'LLM judge': 'LLM judge (vote)', 'Erase-and-check': 'erase-and-check'}
        poison = a['poisoned'].astype(bool)
        retrieval = {m: ((-x >= .90) if m == 'Key salting' else a['base_cos'] >= .90)
                     for m, (x, y) in scores.items()}
        methods = {}
        for m, (x, y) in scores.items():
            assert len(x) == len(ar) and len(y) == len(br)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            threshold = float(np.quantile(y, .95))
            blocked = x > threshold
            auc = float((x[:, None] > y).mean() + .5 * (x[:, None] == y).mean())
            br_value, poisoned_br = float(blocked.mean()), float(blocked[poison].mean())
            expected = lac_ref if m == 'LaCache' else (ref[name]['methods'][names[m]]['att'][0], ref[name]['methods'][names[m]]['succ'][0], ref[name]['methods'][names[m]]['auc'])
            assert br_value == expected[0], (name, m, 'BR', br_value, expected[0])
            assert poisoned_br == expected[1], (name, m, 'poisoned BR', poisoned_br, expected[1])
            assert abs(auc - expected[2]) < (1e-5 if m == 'LaCache' else 1e-14), (name, m, 'AUC', auc, expected[2])
            successful = poison & retrieval[m] & ~blocked
            methods[m] = {'successes': int(successful.sum()), 'n': len(ar), 'asr': float(successful.mean()),
                          'br': br_value, 'auc': auc, 'threshold': threshold,
                          'reference_auc': expected[2], 'original_br_parity': True,
                          'original_poisoned_br_parity': True}
        bg, ag = groups(b), groups(a)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little'))
        draws = {m: [] for m in scores}
        for _ in range(args.replicates):
            bi = np.concatenate([bg[j] for j in rng.integers(len(bg), size=len(bg))])
            ai = np.concatenate([ag[j] for j in rng.integers(len(ag), size=len(ag))])
            for m, (x, y) in scores.items():
                accepted = x[ai] <= np.quantile(y[bi], .95)
                draws[m].append(float((poison[ai] & retrieval[m][ai] & accepted).mean()))
        for m in scores:
            methods[m]['ci95'] = np.percentile(draws[m], [2.5, 97.5]).tolist()
        ours_b = arrays([r for r in ours_records if r['arm'] == 'genuine'])
        ours_a = arrays([r for r in ours_records if r['arm'] == 'attack'])
        accepted, _ = decisions(ours_b, ours_a)
        success = int((poison & (a['base_cos'] >= .9) & accepted['Ours']).sum())
        assert success == ours['cells'][name]['methods']['Ours']['successes']
        methods['Ours'] = ours['cells'][name]['methods']['Ours']
        assert methods['Cosine']['successes'] == ours['cells'][name]['methods']['Cosine']['successes']
        result['cells'][name] = {'n': len(ar), 'n_benign': len(br), 'methods': methods,
                                'None': ours['cells'][name]['methods']['None'],
                                'lacache_prefix_source': {k: v for k, v in lacache['sets'][name].items() if k.endswith('sha256') or k == 'source'}}
        print(name, {m: f"{v['successes']}/{v['n']} = {v['asr']:.3f}" for m, v in methods.items()}, flush=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
