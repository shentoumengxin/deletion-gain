"""Does the embedding pair carry cache-hit validity once the classifier has seen wrappers?

Test of the information-bottleneck claim. The label is the validity of the hit:
positive = a retrievable attack entry whose stored answer is judged poisoned; negative = a
valid hit, i.e. a bare benign entry, a benign key with a harmless wrapper, or a retrievable
attack entry whose answer is not poisoned. Three feature sets share models, splits and
thresholds:

* ``pair``    [E(k), E(q), E(k)-E(q), E(k)*E(q)]                        (1536 dims)
* ``answer``  pair + [E(y), E(y)*E(k), E(y)*E(q)]                        (2688 dims)
* ``stats``   [DG, ADL, Echo]                                            (3 dims)

Bottleneck prediction: ``pair`` cannot reach high BR and low held-out wrapper FPR at once
even after seeing wrappers, and ``answer`` narrows the gap.

Rows.
* Union: ``rows_{lmp,scp,kca}.jsonl`` as in ``pair_classifier.load_union`` (CAP, SCP, KCA
  attacks and the two bare benign arms, the ComQA arm once), with their judged answers.
* Template wrappers (seen): ``answer_check_20260909/full/scores.jsonl``, policy 4+2, set
  instruction_benign, kinds polite and constraint (three templates each, prefix and
  suffix), conditions entry_only, exact_core and both_paraphrase, both splits, so every
  union intent of both corpora carries them.
* LLM-written wrappers (held out): set unseen_wrappers, the twelve instructions frozen
  before scoring, same three conditions, test-split intents only. Never trained on.
Wrapper answers come from ``answers_{wrapped,unseen_wrappers}.jsonl`` (Qwen3-8B, T=0, the
backend of the attack answers). Every answer is embedded as ``answer_variants`` leaves it,
the text the deployed profile embeds; a wrapper answer must hash to its row's answer_sha.
Only retrievable hits (cos >= 0.90) enter, except the bare benign rows, which are the
calibration population of the paper and enter whole, as in pair_classifier.

Splits. 5-fold GroupKFold by intent over the union intents with pair_classifier's seeds, so
union rows fall in the same folds; a wrapper row takes its intent's fold. Each fold model
trains on the attack, bare and template rows of the other folds and scores every row of
its fold, LLM-written wrappers included, so each row is scored by the one model that never
saw its intent. Schemes: in_distribution; loco_CAP, where CAP attacks never enter training.
Models and the inner search are pair_classifier's (no class weights); the logistic grid
extends down to C = 1e-5, since C = 1e-3, the lower end of pair_classifier's grid, won every
fold with wrappers in training.

Metrics (``summarize``). Per seed and corpus, the threshold is the 95th percentile of the
bare benign OOF scores (block iff score > t). BR on poisoned hits per class; FPR on the
seen-template hits of held-out intents and on the LLM-written hits (all three conditions,
as in Table 9; also the strictly unseen instructions, word-set Jaccard < 0.25 with every
seen template wording, the historical rule of ``instruction_benign_analysis``);
blocked share of failed attack hits; AUC poisoned vs failed attack hits of a class; BR at
a fixed LLM-written FPR (a trade-off read-out, threshold taken on those hits). Mean and sd
over seeds. Ours and DG only (main-table thresholds) are read on the same rows.

Subcommands: ``build`` and ``encode`` (cpu-server), ``fit`` (sklearn, cpu-server),
``summarize`` (NumPy). Rows, embeddings and scores stay outside the repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import pair_classifier as P
from experiments.paper.rq1_detection import supp_rules as R

POLICY = P.OOD_POLICY
CONDITIONS = ('entry_only', 'exact_core', 'both_paraphrase')
STRICT_JACCARD = 0.25
FEATURES = ('pair', 'answer', 'stats')
SCHEMES = {'in_distribution': None, 'loco_CAP': 'CAP'}
TRAIN_GROUPS = ('attack', 'bare', 'template')
CORPUS_OF_CLASS = {c: corpus for c, (_, corpus) in P.CLASSES.items()}
GRIDS = {'logreg': {'clf__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
         'mlp': P.GRIDS['mlp']}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- build

def cmd_build(args) -> int:
    from sentry.research.pipeline.instruction_benign import answer_variants
    from sentry.research.pipeline.instruction_benign_analysis import instruction_overlap

    def full(answer):
        return answer_variants(answer).get('full', '')

    u = P.load_union(args.inputs)
    answers = {}
    for cls, (stem, corpus) in P.CLASSES.items():
        for r in P.read_jsonl(args.inputs / f'rows_{stem}.jsonl'):
            assert sha_text(r['answer']) == r['answer_sha256'], r['record_id']
            answers[r['record_id']] = r['answer']
    rows = []
    for i in range(len(u['record_id'])):
        attack = bool(u['attack'][i])
        rows.append({'row_id': str(u['record_id'][i]), 'group': 'attack' if attack else 'bare',
                     'cls': str(u['cls'][i]), 'corpus': str(u['corpus'][i]),
                     'intent_id': str(u['intent_id'][i]), 'template': '', 'instruction': '',
                     'condition': 'native' if attack else 'bare',
                     'key': str(u['key'][i]), 'query': str(u['query'][i]),
                     'answer': full(answers[str(u['record_id'][i])]),
                     'poisoned': bool(u['poisoned'][i]) if attack else False,
                     'cos': float(u['base_cos'][i]), 'dg': float(u['excess_span'][i]),
                     'adl': float(u['adl_best'][i]), 'echo': float(u['echo_best'][i]),
                     'judgeable': True})
    union_intents = set(u['intent_id'])

    by_prompt = {}
    for name in ('answers_wrapped.jsonl', 'answers_unseen_wrappers.jsonl'):
        for r in P.read_jsonl(args.answers / name):
            k = (r['corpus'], r['prompt'])
            assert by_prompt.get(k, r['response']) == r['response'], k
            by_prompt[k] = r['response']
    n_wrapper = 0
    with args.scores.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r['policy'] != POLICY or r['malicious'] or r['condition'] not in CONDITIONS:
                continue
            if r['set'] == 'instruction_benign' and r['kind'] in ('polite', 'constraint'):
                group = 'template'
            elif r['set'] == 'unseen_wrappers':
                group = 'llm'
                assert r['split'] == 'test'
            else:
                continue
            assert r['intent_id'] in union_intents, r['intent_id']
            assert r['has_answer'], r['sample_id']
            y = full(by_prompt[(r['corpus'], r['text'])])
            assert sha_text(y) == r['answer_sha'], r['sample_id']
            instruction = r['text'].replace(r['base_text'], ' ') if r['base_text'] in r['text'] else ''
            rows.append({'row_id': r['sample_id'], 'group': group, 'cls': '',
                         'corpus': r['corpus'], 'intent_id': r['intent_id'],
                         'template': str(r.get('template') or ''),
                         'instruction': ' '.join(instruction.split()),
                         'condition': r['condition'], 'key': r['text'], 'query': r['anchor'],
                         'answer': y, 'poisoned': False, 'cos': float(r['retrieval_cos']),
                         'dg': None if r['dg'] is None else float(r['dg']),
                         'adl': float(r['adl_best']), 'echo': float(r['echo_best']),
                         'judgeable': bool(r['judgeable'])})
            n_wrapper += 1
    assert len({r['row_id'] for r in rows}) == len(rows)
    wording = {r['template']: r['instruction'] for r in rows
               if r['group'] == 'llm' and r['condition'] == 'entry_only'}
    assert '' not in wording and '' not in wording.values(), 'an LLM-written row lacks its wording'
    overlap = {t: instruction_overlap(text) for t, text in wording.items()}
    for r in rows:
        r['hit'] = r['cos'] >= R.RETRIEVAL_FLOOR
        r['label'] = int(r['group'] == 'attack' and r['poisoned'] and r['hit'])
        r['max_jaccard'] = overlap[r['template']][0] if r['group'] == 'llm' else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    from collections import Counter
    print('rows', len(rows), 'wrapper rows', n_wrapper)
    print(Counter((r['group'], r['corpus'], r['hit']) for r in rows))
    print('labels', Counter((r['group'], r['cls'], r['label']) for r in rows if r['group'] == 'attack'))
    print('unjudgeable hits', Counter(r['group'] for r in rows if r['hit'] and not r['judgeable']))
    for t, text in sorted(wording.items()):
        print(f'  {t:10s} {overlap[t][0]:.2f} {text!r} ~ {overlap[t][1]!r}')
    return 0


def load_rows(path: Path) -> dict:
    rows = P.read_jsonl(path)
    v = {k: np.array([r[k] for r in rows], dtype=object)
         for k in ('row_id', 'group', 'cls', 'corpus', 'intent_id', 'template', 'instruction',
                   'condition', 'key', 'query', 'answer')}
    for k in ('poisoned', 'judgeable', 'hit'):
        v[k] = np.array([bool(r[k]) for r in rows])
    v['max_jaccard'] = np.array([np.nan if r['max_jaccard'] is None else r['max_jaccard']
                                 for r in rows], dtype=float)
    v['label'] = np.array([r['label'] for r in rows], dtype=int)
    for k in ('cos', 'adl', 'echo'):
        v[k] = np.array([r[k] for r in rows], dtype=float)
    v['dg'] = np.array([np.nan if r['dg'] is None else r['dg'] for r in rows], dtype=float)
    return v


# ---------------------------------------------------------------- encode

def cmd_encode(args) -> int:
    from sentry.embeddings import TransformerCLSEmbedder

    v = load_rows(args.rows)
    texts = sorted(set(v['key']) | set(v['query']) | set(v['answer']))
    t0 = time.time()
    emb = TransformerCLSEmbedder(P.ENCODER, batch_size=64)
    vec = emb.encode(texts)
    index = {t: i for i, t in enumerate(texts)}
    ek, eq, ey = (vec[[index[t] for t in v[k]]] for k in ('key', 'query', 'answer'))
    cos = np.einsum('ij,ij->i', ek.astype(float), eq.astype(float))
    diff = np.abs(cos - v['cos'])
    print(f'encoded {len(texts)} texts in {time.time() - t0:.0f}s; rows {len(cos)}; '
          f'max |cos - stored cos| = {diff.max():.3e}', flush=True)
    assert diff.max() <= 1e-4, 'embeddings disagree with the stored cosines'
    np.savez_compressed(args.out, row_id=v['row_id'].astype(str), ek=ek, eq=eq, ey=ey, cos=cos,
                        rows_sha256=P.sha_file(args.rows), signature=json.dumps(emb.signature))
    return 0


# ---------------------------------------------------------------- fit

def features(name: str, v: dict, z) -> np.ndarray:
    if name == 'stats':
        dg = np.where(v['judgeable'], v['dg'], 0.0)
        return np.column_stack([dg, v['adl'], v['echo'], (~v['judgeable']).astype(float)])
    ek, eq = z['ek'].astype(np.float64), z['eq'].astype(np.float64)
    x = P.pair_features(ek, eq)
    if name == 'pair':
        return x
    ey = z['ey'].astype(np.float64)
    return np.hstack([x, ey, ey * ek, ey * eq])


def folds_for(v: dict, union_intents, seed: int) -> np.ndarray:
    ids = np.array(sorted(union_intents), dtype=object)
    fold_of = dict(zip(ids, P.intent_folds(ids, P.N_FOLDS, seed)))
    return np.array([fold_of[k] for k in v['intent_id']])


def train_mask(v: dict, folds: np.ndarray, fold: int, held_out: str | None) -> np.ndarray:
    m = (folds != fold) & np.isin(v['group'], TRAIN_GROUPS)
    m &= v['hit'] | (v['group'] == 'bare')
    if held_out is not None:
        m &= ~((v['group'] == 'attack') & (v['cls'] == held_out))
    return m


def fit_model(name: str, x, y, groups, train, seed: int):
    """``pair_classifier.fit_model`` with this module's grids."""
    from sklearn.model_selection import GridSearchCV, GroupKFold

    cv = GroupKFold(n_splits=P.INNER_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(P.make_model(name, seed), GRIDS[name], scoring='roc_auc', cv=cv,
                          n_jobs=1, refit=True, error_score='raise')
    search.fit(x[train], y[train], groups=groups[train])
    return search.best_estimator_, {'best_params': search.best_params_,
                                    'inner_auc': float(search.best_score_),
                                    'n_train': int(train.sum()),
                                    'n_train_positive': int(y[train].sum())}


def _task(x, y, groups, v, folds, seed, scheme, fold, model, feat):
    train = train_mask(v, folds, fold, SCHEMES[scheme])
    test = folds == fold
    assert not set(groups[train]) & set(groups[test])
    fitted, info = fit_model(model, x, y, groups, train, seed)
    return feat, seed, scheme, fold, model, np.flatnonzero(test), P.raw_score(fitted, x[test]), info


def cmd_fit(args) -> int:
    from joblib import Parallel, delayed

    v = load_rows(args.rows)
    z = np.load(args.embeddings, allow_pickle=False)
    assert np.array_equal(z['row_id'], v['row_id'].astype(str))
    union_intents = set(v['intent_id'][np.isin(v['group'], ('attack', 'bare'))])
    y, groups = v['label'], v['intent_id']
    folds = {s: folds_for(v, union_intents, s) for s in args.seeds}
    slim = {k: v[k] for k in ('group', 'cls', 'hit')}
    out, infos = {}, []
    for feat in args.features:
        x = features(feat, v, z)
        jobs = [(s, sc, f, m) for s in args.seeds for sc in args.schemes
                for f in range(P.N_FOLDS) for m in args.models]
        t0 = time.time()
        done = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(_task)(x, y, groups, slim, folds[s], s, sc, f, m, feat) for s, sc, f, m in jobs)
        print(f'{feat}: {len(jobs)} fits in {time.time() - t0:.0f}s', flush=True)
        for feat_, s, sc, f, m, idx, scores, info in done:
            key = f'{feat_}|{s}|{sc}|{m}'
            arr = out.setdefault(key, np.full(len(y), np.nan))
            assert np.isnan(arr[idx]).all()
            arr[idx] = scores
            infos.append({'features': feat_, 'seed': s, 'scheme': sc, 'fold': f, 'model': m, **info})
        del x
    for key, arr in out.items():
        assert np.isfinite(arr).all(), key
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, row_id=v['row_id'].astype(str),
                        folds=np.stack([folds[s] for s in args.seeds]), **out)
    args.out.with_suffix('.fits.json').write_text(json.dumps(infos, indent=1) + '\n')
    return 0


# ---------------------------------------------------------------- summarize

def masks(v: dict) -> dict:
    hit = v['hit']
    att = (v['group'] == 'attack') & hit
    out = {}
    for cls, corpus in CORPUS_OF_CLASS.items():
        out[f'poisoned {cls}'] = att & (v['cls'] == cls) & v['poisoned']
        out[f'failed {cls}'] = att & (v['cls'] == cls) & ~v['poisoned']
    for c in ('comqa', 'nq'):
        tmpl = (v['group'] == 'template') & hit & (v['corpus'] == c)
        out[f'Template {c}'] = tmpl & (v['condition'] == 'entry_only') & np.char.startswith(
            v['template'].astype(str), 'polite')
        out[f'All templates {c}'] = tmpl
        llm = (v['group'] == 'llm') & hit & (v['corpus'] == c)
        out[f'LLM-written {c}'] = llm
        out[f'LLM-written strict {c}'] = llm & (v['max_jaccard'] < STRICT_JACCARD)
        out[f'bare {c}'] = (v['group'] == 'bare') & (v['corpus'] == c)
    return out


def rule_blocks(v: dict, name: str) -> np.ndarray:
    blocked = np.zeros(len(v['dg']), bool)
    for c in ('comqa', 'nq'):
        t = P.MAIN_TABLE[c]
        m = v['corpus'] == c
        dg = np.where(v['judgeable'][m], v['dg'][m], np.inf)
        if name == 'DG only':
            b = dg > t['eta_dg']
        else:
            b = (dg > t['eta']) & ((v['adl'][m] > t['eta_a']) | (v['echo'][m] >= 1))
        blocked[m] = b | ~v['judgeable'][m]
    return blocked


def read_out(v: dict, M: dict, blocked: np.ndarray, score: np.ndarray | None = None) -> dict:
    cells = {name: float(blocked[m].mean()) for name, m in M.items()}
    if score is not None:
        for cls in P.CLASSES:
            cells[f'AUC poisoned vs failed {cls}'] = R.auc(score[M[f'poisoned {cls}']],
                                                           score[M[f'failed {cls}']])
        for cls, c in CORPUS_OF_CLASS.items():
            cells[f'AUC poisoned {cls} vs LLM-written'] = R.auc(score[M[f'poisoned {cls}']],
                                                                score[M[f'LLM-written {c}']])
            for x in (0.05, 0.10):
                t = float(np.quantile(score[M[f'LLM-written {c}']], 1 - x))
                cells[f'BR {cls} at LLM-written FPR {x:.2f}'] = float(
                    (score[M[f'poisoned {cls}']] > t).mean())
    return cells


def cmd_summarize(args) -> int:
    v = load_rows(args.rows)
    z = np.load(args.oof, allow_pickle=False)
    assert np.array_equal(z['row_id'], v['row_id'].astype(str))
    fits = json.loads(args.oof.with_suffix('.fits.json').read_text())
    M = masks(v)
    result = {
        'label': 'positive = retrievable attack entry with a poisoned stored answer; negative = '
                 'bare benign, harmless wrapper, retrievable attack entry with a valid answer',
        'threshold': 'per seed and corpus, 95th percentile of the bare benign OOF scores; '
                     'block iff score > t',
        'grids': GRIDS,
        'rows_sha256': P.sha_file(args.rows), 'oof_sha256': P.sha_file(args.oof),
        'n': {name: int(m.sum()) for name, m in M.items()},
        'rules': {name: read_out(v, M, rule_blocks(v, name)) for name in ('Ours', 'DG only')},
        'classifier': {},
    }
    for key in sorted(k for k in z.files if k.count('|') == 3):
        feat, seed, scheme, model = key.split('|')
        score = z[key]
        blocked = np.zeros(len(score), bool)
        for c in ('comqa', 'nq'):
            t = float(np.quantile(score[M[f'bare {c}']], 0.95))
            blocked[v['corpus'] == c] = score[v['corpus'] == c] > t
        cell = read_out(v, M, blocked, score)
        name = f'{feat}|{scheme}|{model}'
        result['classifier'].setdefault(name, []).append(cell)
    for name, cells in result['classifier'].items():
        result['classifier'][name] = {k: P.mean_sd([c[k] for c in cells]) for k in cells[0]}
        feat, scheme, model = name.split('|')
        result['classifier'][name]['chosen_params'] = sorted(
            {json.dumps(f['best_params']) for f in fits
             if (f['features'], f['scheme'], f['model']) == (feat, scheme, model)})
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    show = ['poisoned CAP', 'poisoned SCP', 'poisoned KCA', 'Template comqa', 'Template nq',
            'LLM-written comqa', 'LLM-written nq', 'LLM-written strict comqa',
            'LLM-written strict nq',
            'failed CAP', 'failed SCP', 'failed KCA']
    print('n', {k: result['n'][k] for k in show})
    for name, cells in result['rules'].items():
        print(f'{name:28s}', ' '.join(f'{cells[k]:.3f}' for k in show))
    for name, cells in result['classifier'].items():
        print(f'{name:28s}', ' '.join(f'{cells[k]["mean"]:.3f}' for k in show))
        print(' ' * 28, {k: round(c['mean'], 3) for k, c in cells.items()
                         if k.startswith(('AUC', 'BR '))})
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--inputs', type=Path, required=True, help='dir of rows_{lmp,scp,kca}.jsonl')
    b.add_argument('--scores', type=Path, required=True, help='answer_check_20260909/full/scores.jsonl')
    b.add_argument('--answers', type=Path, required=True, help='answer_check_20260909/answers')
    b.add_argument('--out', type=Path, required=True)
    e = sub.add_parser('encode')
    e.add_argument('--rows', type=Path, required=True)
    e.add_argument('--out', type=Path, required=True)
    f = sub.add_parser('fit')
    f.add_argument('--rows', type=Path, required=True)
    f.add_argument('--embeddings', type=Path, required=True)
    f.add_argument('--features', nargs='+', choices=FEATURES, default=list(FEATURES))
    f.add_argument('--schemes', nargs='+', choices=list(SCHEMES), default=list(SCHEMES))
    f.add_argument('--models', nargs='+', choices=P.MODELS, default=list(P.MODELS))
    f.add_argument('--seeds', nargs='+', type=int, default=list(P.SEEDS))
    f.add_argument('--jobs', type=int, default=64)
    f.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('summarize')
    s.add_argument('--rows', type=Path, required=True)
    s.add_argument('--oof', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    return {'build': cmd_build, 'encode': cmd_encode, 'fit': cmd_fit,
            'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
