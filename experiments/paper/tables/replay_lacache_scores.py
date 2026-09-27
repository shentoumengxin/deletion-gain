"""Replay LaCache scores from frozen answer prefixes, without answer generation.

Uses the original e5-small-v2 CLS/L2 recipe and batch sizes. Only scalar scores
and record identifiers are written. Keep the output in the external data root.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).cpu().eval()
    result = {'model': str(args.model), 'pooling': 'cls', 'normalization': 'l2',
              'text_prefix': '', 'dtype': 'float32', 'device': 'cpu', 'threads': 4,
              'k': 20, 'sets': {}}
    sources = [('CAP', 'lmp', 'out/v3/lacache/lmp_lacache_input.jsonl'),
               ('SCP', 'scp', 'rebase/scp_lacache_input_798.jsonl'),
               ('KCA', 'kca', 'out/v3/lacache/kca_lacache_input.jsonl')]
    for name, stem, relative in sources:
        source = args.experiment_root / relative
        raw = source.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert {r['k'] for r in rows} == {20}
        cohort = args.experiment_root / f'answer_check_20260909/dumps/e5-small-v2__{stem}__cls.jsonl'
        cohort_raw = cohort.read_bytes()
        records = [json.loads(line) for line in cohort_raw.splitlines() if line.strip()]
        by_id = {r['record_id']: r for r in records}
        assert len(by_id) == len(records) == len(rows)
        # Prefix exports use descriptive benign IDs and the kca-f2 spelling;
        # the cohort retains gcg-f2. Join attacks by record ID and controls by
        # intent; require a bijection for both populations.
        benign_by_intent = {r['intent_id']: r for r in records if r['arm'] == 'genuine'}
        assert len(benign_by_intent) == sum(r['arm'] == 'genuine' for r in records)
        def resolve(r):
            if r['set'] == 'benign':
                return benign_by_intent[r['intent_id']]
            return by_id[r['record_id'].replace('kca-f2', 'gcg-f2')]
        assert {resolve(r)['record_id'] for r in rows} == set(by_id)
        assert all(resolve(r)['intent_id'] == r['intent_id'] for r in rows)
        uniq = sorted({r[key] or '' for r in rows for key in ('entry_prefix', 'query_prefix')})
        encoded = []
        for start in range(0, len(uniq), 256):
            chunk = [s or 'empty' for s in uniq[start:start + 256]]
            for offset in range(0, len(chunk), 128):
                inputs = tokenizer(chunk[offset:offset + 128], padding=True,
                                   truncation=True, return_tensors='pt')
                with torch.no_grad():
                    hidden = model(**inputs).last_hidden_state[:, 0, :]
                    vectors = torch.nn.functional.normalize(hidden, dim=1)
                encoded.append(vectors.numpy().astype(np.float32))
        matrix = np.concatenate(encoded, axis=0)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)
        index = {s: i for i, s in enumerate(uniq)}
        scored = []
        for r in rows:
            c = resolve(r)
            assert c['arm'] == ('genuine' if r['set'] == 'benign' else 'attack')
            score = 1.0 - float(matrix[index[r['entry_prefix'] or '']] @
                                matrix[index[r['query_prefix'] or '']])
            scored.append({'record_id': c['record_id'], 'arm': c['arm'],
                           'intent_id': c['intent_id'], 'score': score})
        b = np.array([r['score'] for r in scored if r['arm'] == 'genuine'])
        a = np.array([r['score'] for r in scored if r['arm'] == 'attack'])
        threshold = float(np.quantile(b, .95))
        result['sets'][name] = {'source': str(source),
                               'source_sha256': hashlib.sha256(raw).hexdigest(),
                               'cohort_sha256': hashlib.sha256(cohort_raw).hexdigest(),
                               'threshold': threshold, 'br': float((a > threshold).mean()),
                               'auc': float(((a[:, None] > b).mean() + .5 * (a[:, None] == b).mean())),
                               'rows': scored}
        print(name, 'n', len(a), 'threshold', threshold,
              'BR', result['sets'][name]['br'], 'AUC', result['sets'][name]['auc'], flush=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
