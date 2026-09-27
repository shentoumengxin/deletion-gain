"""Can a detector that sees only the embedding pair (E(k), E(q)) find poisoned entries?

Tests Contribution 1 from the other side: if a classifier on the raw pair already
separates poisoned from benign entries, and transfers to an attack class it never saw,
the deletion structure is not what makes the signal. Design:

* Data. The union of ``rows_{lmp,scp,kca}.jsonl``: CAP, SCP and KCA attacks plus the two
  benign arms. CAP and SCP share the 499 ComQA benign rows, which enter once (deduplicated
  by ``record_id``). Attack rows carry their intent's benign query, so E(q) alone never
  separates an attack from the benign row of the same intent.
* Encoder. e5-small-v2, CLS pooling, no prefix, L2 norm (``sentry/embeddings.py``); every
  row's cos(E(k), E(q)) must match the dump's ``base_cos`` to 1e-4.
* Features. x = [E(k), E(q), E(k) - E(q), E(k) * E(q)] (4 x 384), standardised inside
  each training fold (the scaler is part of the fitted pipeline).
* Models. L2 logistic regression, C in {1e-3 ... 100}; an MLP with one hidden layer of
  256, ``early_stopping=True``, alpha in {1e-4, 1e-3, 1e-2}. The grid point is chosen by
  an inner 5-fold GroupKFold (by intent, ROC AUC) on the training fold only.
* Splits. 5-fold GroupKFold by ``intent_id`` over the union (one fold assignment for all
  classes), 5 seeds that reshuffle the intents. In-distribution: train on all three
  attack classes and all benign rows of the training folds. Leave-one-class-out (LOCO):
  the held-out class's attacks never enter training; its test rows are that class's
  attacks plus the benign rows of its own corpus. For KCA a second LOCO variant also
  drops the NQ benign rows from training, which removes the shortcut "NQ => benign".
* Metrics. Out-of-fold scores are pooled over the 5 folds. Per class: AUC, BR at the
  benign 95th percentile of the OOF scores of that class's corpus (block iff score > t),
  realized FPR, ASR (``supp_rules.asr``); mean and sd over seeds, plus a 2,000-draw
  intent bootstrap on seed 0. Cosine, DG only and Ours (full-cohort calibration, the
  definitions of ``dg_necessity``) are reported on the same rows.
* Controls for what the classifier reads. ``fit --features key`` runs the same pipeline on
  E(k) alone; the summary also reports the key's word count as a one-number detector.
* Benign rows never trained on (``ood_benign_fpr``). The held-out wrapper, fresh-benign
  and QQP populations are scored by the same fold models: a row whose intent is in the
  training union by the one model that never saw that intent, any other row by all five
  (blocked fraction averaged). The threshold is the corpus' OOF benign 95th percentile
  (QQP takes ComQA's). Ours, DG only and cosine are read on the same retrievable hits.

Subcommands: ``encode`` and ``ood-encode`` (model, cpu-server), ``fit`` and ``ood-score``
(sklearn, cpu-server), ``summarize`` (NumPy). Embeddings and scores stay outside the repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R

ENCODER = 'intfloat/e5-small-v2'
CLASSES = {'CAP': ('lmp', 'comqa'), 'SCP': ('scp', 'comqa'), 'KCA': ('kca', 'nq')}
SCHEMES = {  # name -> (held-out class or None, drop the held-out corpus' benign rows)
    'in_distribution': (None, False),
    'loco_CAP': ('CAP', False),
    'loco_SCP': ('SCP', False),
    'loco_KCA': ('KCA', False),
    'loco_KCA_drop_nq_benign': ('KCA', True),
}
MODELS = ('logreg', 'mlp')
GRIDS = {'logreg': {'clf__C': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
         'mlp': {'clf__alpha': [1e-4, 1e-3, 1e-2]}}
N_FOLDS, SEEDS, INNER_FOLDS = 5, (0, 1, 2, 3, 4), 5
BUDGETS = (0.01, 0.05)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- data

def load_union(inputs: Path) -> dict:
    """All attack rows of the three tables plus each benign row once, as aligned arrays."""
    rows, seen = [], {}
    for cls, (stem, corpus) in CLASSES.items():
        for r in read_jsonl(inputs / f'rows_{stem}.jsonl'):
            if r['arm'] == 'genuine':
                if r['record_id'] in seen:                 # ComQA benign, shared by CAP/SCP
                    old = seen[r['record_id']]
                    assert all(old[f] == r[f] for f in ('key', 'query', 'answer', 'base_cos',
                                                        'intent_id', 'excess_span')), r['record_id']
                    continue
                seen[r['record_id']] = r
                rows.append({**r, 'cls': '', 'corpus': corpus, 'attack': False})
            else:
                rows.append({**r, 'cls': cls, 'corpus': corpus, 'attack': True})
    u = {k: np.array([r[k] for r in rows], dtype=object)
         for k in ('record_id', 'intent_id', 'cls', 'corpus', 'family', 'key', 'query')}
    u['attack'] = np.array([r['attack'] for r in rows])
    u['poisoned'] = np.array([bool(r['poisoned']) for r in rows])
    for k in ('base_cos', 'excess_span', 'adl_best', 'echo_best'):
        u[k] = np.array([r[k] for r in rows], dtype=float)
    assert len(set(u['record_id'])) == len(rows)
    return u


def intent_folds(intent_ids, n_folds: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold index per row: shuffle the distinct intents with ``seed``, deal them round-robin."""
    intent_ids = np.asarray(intent_ids, dtype=object)
    uniq = np.array(sorted(set(intent_ids)), dtype=object)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {k: i % n_folds for i, k in enumerate(uniq[order])}
    return np.array([fold_of[k] for k in intent_ids])


def split_masks(u: dict, folds: np.ndarray, fold: int, held_out: str | None = None,
                drop_heldout_benign: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) row masks for one outer fold.

    In-distribution (``held_out`` None): train on every other fold, test on this one.
    LOCO: the held-out class's attacks are removed from training; the test rows are that
    class's attacks and its corpus' benign rows in this fold.
    """
    train, test = folds != fold, folds == fold
    if held_out is None:
        return train, test
    corpus = CLASSES[held_out][1]
    heldout_attack = u['attack'] & (u['cls'] == held_out)
    corpus_benign = ~u['attack'] & (u['corpus'] == corpus)
    train = train & ~heldout_attack
    if drop_heldout_benign:
        train = train & ~corpus_benign
    return train, test & (heldout_attack | corpus_benign)


def pair_features(ek: np.ndarray, eq: np.ndarray) -> np.ndarray:
    return np.hstack([ek, eq, ek - eq, ek * eq])


# ---------------------------------------------------------------- models

def make_model(name: str, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == 'logreg':            # l1_ratio defaults to 0: an L2 penalty
        clf = LogisticRegression(max_iter=5000)
    elif name == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True,
                            max_iter=300, random_state=seed)
    else:
        raise KeyError(name)
    return Pipeline([('scale', StandardScaler()), ('clf', clf)])


def raw_score(model, x: np.ndarray) -> np.ndarray:
    """Pre-sigmoid score, so saturated probabilities do not tie at 1.0."""
    clf = model.named_steps['clf']
    z = model.named_steps['scale'].transform(x)
    if hasattr(clf, 'decision_function'):
        return clf.decision_function(z)
    h = np.maximum(z @ clf.coefs_[0] + clf.intercepts_[0], 0.0)      # relu hidden layer
    return (h @ clf.coefs_[1] + clf.intercepts_[1])[:, 0]


def fit_model(name: str, x, y, groups, train, seed: int, inner_folds: int = INNER_FOLDS):
    """Grid-search on the training rows (inner GroupKFold by intent); the refit winner."""
    from sklearn.model_selection import GridSearchCV, GroupKFold

    cv = GroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(make_model(name, seed), GRIDS[name], scoring='roc_auc', cv=cv,
                          n_jobs=1, refit=True, error_score='raise')
    search.fit(x[train], y[train], groups=groups[train])
    info = {'best_params': search.best_params_, 'inner_auc': float(search.best_score_),
            'n_train': int(train.sum()), 'n_train_attack': int(y[train].sum())}
    return search.best_estimator_, info


def fit_and_score(name: str, x, y, groups, train, test, seed: int,
                  inner_folds: int = INNER_FOLDS) -> tuple[np.ndarray, dict]:
    """Fit on the training rows, score the test rows; they must share no intent."""
    assert not set(groups[train]) & set(groups[test]), 'an intent is in train and test'
    model, info = fit_model(name, x, y, groups, train, seed, inner_folds)
    return raw_score(model, x[test]), info


# ---------------------------------------------------------------- metrics

def class_rows(u: dict, cls: str) -> tuple[np.ndarray, np.ndarray]:
    corpus = CLASSES[cls][1]
    return u['attack'] & (u['cls'] == cls), ~u['attack'] & (u['corpus'] == corpus)


def score_metrics(b: dict, a: dict, budget: float = .05) -> dict:
    """b, a: dicts with 'score' (and a: 'base_cos', 'poisoned'). Block iff score > t."""
    t = float(np.quantile(b['score'], 1 - budget))
    blocked = a['score'] > t
    k, n, rate = R.asr(a['base_cos'], a['poisoned'], blocked)
    return {'auc': R.auc(a['score'], b['score']), 'threshold': t,
            'br': float(blocked.mean()), 'realized_fpr': float((b['score'] > t).mean()),
            'br_poisoned': float(blocked[a['poisoned']].mean()),
            'asr': rate, 'asr_successes': k, 'n_attack': n, 'n_benign': int(len(b['score']))}


def arms(u: dict, cls: str, score: np.ndarray) -> tuple[dict, dict]:
    ai, bi = class_rows(u, cls)
    keys = ('intent_id', 'base_cos', 'poisoned', 'excess_span', 'adl_best', 'echo_best', 'family')
    b = {k: u[k][bi] for k in keys}
    a = {k: u[k][ai] for k in keys}
    b['score'], a['score'] = score[bi], score[ai]
    return b, a


def bootstrap_metrics(b: dict, a: dict, replicates: int, seed: str) -> dict:
    def stat(bb, aa):
        m = score_metrics(bb, aa)
        return [m['auc'], m['br'], m['asr']]
    lo, hi = R.intent_bootstrap(b, a, stat, replicates, seed=seed)
    return {'auc_ci95': [lo[0], hi[0]], 'br_ci95': [lo[1], hi[1]], 'asr_ci95': [lo[2], hi[2]]}


def baseline_cells(u: dict, cls: str, replicates: int) -> dict:
    """Cosine, DG only and Ours on the same rows, full-cohort benign calibration."""
    from experiments.paper.rq1_detection import dg_necessity as D

    b, a = arms(u, cls, np.zeros(len(u['attack'])))
    out = {}
    for m in ('Cosine', 'DG only', 'Ours'):
        th = D.fit(m, b, .05)
        blocked = D.blocks(m, a, th)
        k, n, rate = R.asr(a['base_cos'], a['poisoned'], blocked)

        def stat(bb, aa, m=m):
            bl = D.blocks(m, aa, D.fit(m, bb, .05))
            return [float(bl.mean()), R.asr(aa['base_cos'], aa['poisoned'], bl)[2]]
        lo, hi = R.intent_bootstrap(b, a, stat, replicates, seed=f'pair|{cls}|{m}')
        out[m] = {'thresholds': th, 'auc': D.ranking_auc(m, b, a), 'br': float(blocked.mean()),
                  'br_ci95': [lo[0], hi[0]], 'realized_fpr': float(D.blocks(m, b, th).mean()),
                  'br_poisoned': float(blocked[a['poisoned']].mean()),
                  'asr': rate, 'asr_successes': k, 'asr_ci95': [lo[1], hi[1]],
                  'n_attack': n, 'n_benign': int(len(b['base_cos']))}
    return out


def mean_sd(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': v.tolist()}


# ---------------------------------------------------------------- benign rows never trained on

OOD_POLICY = 'multi[count:4+width:2:cap16]/runs'
OOD_GROUPS = (  # label, set, split, kind (None = any), condition; answer_check_20260909 rows
    ('bare', 'instruction_benign', 'test', 'bare', 'bare'),
    ('Template', 'instruction_benign', 'test', 'polite', 'entry_only'),
    ('Same query', 'instruction_benign', 'test', 'polite', 'exact_core'),
    ('Constraint entry_only', 'instruction_benign', 'test', 'constraint', 'entry_only'),
    ('LLM-written entry_only', 'unseen_wrappers', 'test', None, 'entry_only'),
    ('LLM-written exact_core', 'unseen_wrappers', 'test', None, 'exact_core'),
    ('LLM-written both_paraphrase', 'unseen_wrappers', 'test', None, 'both_paraphrase'),
    ('Fresh bare', 'fresh_benign', 'fresh_test', None, 'bare'),
    ('Fresh entry_only', 'fresh_benign', 'fresh_test', None, 'entry_only'),
)
REPORT_GROUPS = {  # reported name -> row labels; Table 9's LLM-written pools all conditions
    **{g[0]: (g[0],) for g in OOD_GROUPS},
    'LLM-written (Table 9: all three conditions)': ('LLM-written entry_only',
                                                    'LLM-written exact_core',
                                                    'LLM-written both_paraphrase'),
    'QQP': ('QQP',),
}
OOD_SCHEMES, OOD_FEATURES = ('in_distribution', 'loco_CAP'), ('pair', 'key')
THRESHOLD_CORPUS = {'comqa': 'comqa', 'nq': 'nq', 'qqp': 'comqa'}   # QQP: frozen ComQA transfer
MAIN_TABLE = {  # joint (eta, eta_A) and DG-only thresholds of the main table
    'comqa': {'eta': 0.002645233293430393, 'eta_a': 0.01965637207031249,
              'eta_dg': 0.006797628467845549},
    'nq': {'eta': 0.00016426084972413917, 'eta_a': 0.017378234863281244,
           'eta_dg': 0.002621655716369286},
}


def match_group(r: dict) -> str | None:
    for label, st, split, kind, cond in OOD_GROUPS:
        if (r['set'] == st and r.get('split') == split and r['condition'] == cond
                and (kind is None or r['kind'] == kind)):
            return label
    return None


def ood_block_fraction(scores: np.ndarray, row_intents, fold_of: dict, t: float) -> np.ndarray:
    """Per row, the blocked fraction under threshold ``t`` (block iff score > t).

    ``scores[f, i]`` is fold model f's score of row i; fold model f was trained on every
    fold but f. A row whose intent is in the training union is read by the one model that
    never saw its intent (fold_of[intent]); any other row by all fold models, averaged.
    """
    blocked = scores > t
    out = blocked.mean(0)
    for i, k in enumerate(row_intents):
        if k in fold_of:
            out[i] = float(blocked[fold_of[k], i])
    return out


def replay_blocks(v: dict, eta: float, eta_a: float | None = None) -> np.ndarray:
    """``supp_controls_20260920/remote_replay.py::block``: an unjudgeable entry is blocked;
    DG <= eta serves; without an answer (or eta_a None) DG alone decides."""
    dg = np.where(v['judgeable'], v['dg'], np.inf)
    out = dg > eta
    if eta_a is not None:
        fires = (v['adl_best'] > eta_a) | (v['echo_best'] >= 1)
        out = out & (~v['has_answer'] | fires)
    return out | ~v['judgeable']


# ---------------------------------------------------------------- subcommands

def cmd_encode(args) -> int:
    from sentry.embeddings import TransformerCLSEmbedder

    u = load_union(args.inputs)
    texts = sorted(set(u['key']) | set(u['query']))
    t0 = time.time()
    emb = TransformerCLSEmbedder(ENCODER, batch_size=64)
    vec = emb.encode(texts)
    index = {t: i for i, t in enumerate(texts)}
    ek = vec[[index[t] for t in u['key']]]
    eq = vec[[index[t] for t in u['query']]]
    cos = np.einsum('ij,ij->i', ek.astype(float), eq.astype(float))
    diff = np.abs(cos - u['base_cos'])
    print(f'encoded {len(texts)} texts in {time.time() - t0:.1f}s; rows {len(cos)}; '
          f'max |cos - base_cos| = {diff.max():.3e} (row {u["record_id"][diff.argmax()]})',
          flush=True)
    assert diff.max() <= 1e-4, 'embeddings disagree with the dump'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, record_id=u['record_id'].astype(str), ek=ek, eq=eq,
                        cos=cos, max_abs_cos_diff=diff.max(),
                        signature=json.dumps(emb.signature))
    return 0


def _task(x, y, groups, u, folds, seed, scheme, fold, model):
    held_out, drop = SCHEMES[scheme]
    train, test = split_masks(u, folds, fold, held_out, drop)
    if held_out is not None:
        assert not (train & u['attack'] & (u['cls'] == held_out)).any()
    scores, info = fit_and_score(model, x, y, groups, train, test, seed)
    return seed, scheme, fold, model, np.flatnonzero(test), scores, info


def cmd_fit(args) -> int:
    from joblib import Parallel, delayed

    u = load_union(args.inputs)
    z = np.load(args.embeddings, allow_pickle=False)
    assert np.array_equal(z['record_id'], u['record_id'].astype(str))
    ek, eq = z['ek'].astype(np.float64), z['eq'].astype(np.float64)
    x = pair_features(ek, eq) if args.features == 'pair' else ek
    y = u['attack'].astype(int)
    groups = u['intent_id']
    folds = {s: intent_folds(groups, N_FOLDS, s) for s in SEEDS}
    jobs = [(s, sc, f, m) for s in SEEDS for sc in SCHEMES for f in range(N_FOLDS) for m in MODELS]
    t0 = time.time()
    done = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_task)(x, y, groups, u, folds[s], s, sc, f, m) for s, sc, f, m in jobs)
    print(f'{len(jobs)} fits in {time.time() - t0:.0f}s', flush=True)
    oof = {f'{s}|{sc}|{m}': np.full(len(y), np.nan) for s in SEEDS for sc in SCHEMES for m in MODELS}
    infos = []
    for s, sc, f, m, idx, scores, info in done:
        key = f'{s}|{sc}|{m}'
        assert np.isnan(oof[key][idx]).all()
        oof[key][idx] = scores
        infos.append({'seed': s, 'scheme': sc, 'fold': f, 'model': m, 'n_test': len(idx), **info})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, record_id=u['record_id'].astype(str), features=args.features,
                        folds=np.stack([folds[s] for s in SEEDS]), **oof)
    args.out.with_suffix('.fits.json').write_text(json.dumps(infos, indent=1) + '\n')
    return 0


def load_ood_rows(scores_path: Path, qqp_path: Path) -> list[dict]:
    """Benign rows of the groups in OOD_GROUPS (4+2 policy) plus the QQP duplicate pairs."""
    keep = ('set', 'corpus', 'intent_id', 'text', 'anchor', 'base_cos', 'retrieval_cos', 'dg',
            'adl_best', 'echo_best', 'judgeable', 'has_answer')
    rows = []
    with scores_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r['policy'] != OOD_POLICY or r['malicious']:
                continue
            label = match_group(r)
            if label:
                rows.append({'label': label, 'record_id': r.get('record_id', ''),
                             **{k: r.get(k) for k in keep}})
    recs = read_jsonl(qqp_path)
    by_id = {r['record_id']: r for r in recs}
    for q in recs:
        if q['query_role'] != 'benign_query':
            continue
        k = by_id[q['parent_id']]
        assert k['query_role'] == 'canonical' and k['intent_id'] == q['intent_id']
        rows.append({'label': 'QQP', 'record_id': q['record_id'], 'set': 'qqp', 'corpus': 'qqp',
                     'intent_id': q['intent_id'], 'text': k['text'], 'anchor': q['text'],
                     'base_cos': None, 'retrieval_cos': None, 'dg': None, 'adl_best': None,
                     'echo_best': None, 'judgeable': None, 'has_answer': False})
    return rows


def cmd_ood_encode(args) -> int:
    """E(k), E(q) for every benign row never trained on; 4+2 DG for QQP, which has none."""
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.deletion import build_profile, excess
    from sentry.embeddings import TransformerCLSEmbedder

    rows = load_ood_rows(args.scores, args.qqp)
    texts = sorted({r['text'] for r in rows} | {r['anchor'] for r in rows})
    t0 = time.time()
    emb = TransformerCLSEmbedder(ENCODER, batch_size=64)
    vec = emb.encode(texts)
    index = {t: i for i, t in enumerate(texts)}
    ek = vec[[index[r['text']] for r in rows]]
    eq = vec[[index[r['anchor']] for r in rows]]
    cos = np.einsum('ij,ij->i', ek.astype(float), eq.astype(float))
    print(f'encoded {len(texts)} texts for {len(rows)} rows in {time.time() - t0:.0f}s', flush=True)
    policy = parse_policy(OOD_POLICY)
    for i, r in enumerate(rows):
        if r['label'] != 'QQP':
            continue
        profile = build_profile(r['text'], emb, policy, storage_dtype='float16')
        r['retrieval_cos'] = float(cos[i])
        r['judgeable'] = bool(profile.judgeable)
        if profile.judgeable:
            x = excess(profile, eq[i])
            r['base_cos'], r['dg'] = x.base_cos, x.excess_span
        else:
            r['base_cos'] = float(cos[i])
    num = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)
    arr = lambda k: np.array([r[k] for r in rows]).astype(str)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, ek=ek, eq=eq, cos=cos, label=arr('label'), set=arr('set'), corpus=arr('corpus'),
        intent_id=arr('intent_id'), record_id=arr('record_id'), key=arr('text'), query=arr('anchor'),
        **{k: num(k) for k in ('base_cos', 'retrieval_cos', 'dg', 'adl_best', 'echo_best')},
        judgeable=np.array([bool(r['judgeable']) for r in rows]),
        has_answer=np.array([bool(r['has_answer']) for r in rows]),
        scores_sha256=sha_file(args.scores), qqp_sha256=sha_file(args.qqp),
        signature=json.dumps(emb.signature))
    d = np.abs(cos - num('retrieval_cos'))
    print(f'max |cos - retrieval_cos| over dump rows = {np.nanmax(d[arr("label") != "QQP"]):.3e}',
          flush=True)
    return 0


def _ood_task(x, x_ood, y, groups, u, folds, seed, scheme, fold, model, feat, chunk=4096):
    held_out, drop = SCHEMES[scheme]
    train = split_masks(u, folds, fold, held_out, drop)[0]
    test = folds == fold
    assert not set(groups[train]) & set(groups[test])
    fitted, info = fit_model(model, x, y, groups, train, seed)
    ood = np.concatenate([raw_score(fitted, x_ood[i:i + chunk])
                          for i in range(0, len(x_ood), chunk)])
    return feat, seed, scheme, fold, model, np.flatnonzero(test), raw_score(fitted, x[test]), ood, info


def cmd_ood_score(args) -> int:
    """Refit the in-distribution and LOCO-CAP fold models (same data, seeds and search as
    ``fit``; checked against its OOF scores) and score every row of ``ood-encode``."""
    from joblib import Parallel, delayed

    u = load_union(args.inputs)
    z = np.load(args.embeddings, allow_pickle=False)
    assert np.array_equal(z['record_id'], u['record_id'].astype(str))
    o = np.load(args.ood_rows, allow_pickle=False)
    feats = {'pair': (pair_features(z['ek'].astype(np.float64), z['eq'].astype(np.float64)),
                      pair_features(o['ek'].astype(np.float64), o['eq'].astype(np.float64))),
             'key': (z['ek'].astype(np.float64), o['ek'].astype(np.float64))}
    y, groups = u['attack'].astype(int), u['intent_id']
    folds = {s: intent_folds(groups, N_FOLDS, s) for s in SEEDS}
    jobs = [(ft, s, sc, f, m) for ft in OOD_FEATURES for s in SEEDS for sc in OOD_SCHEMES
            for f in range(N_FOLDS) for m in MODELS]
    t0 = time.time()
    done = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_ood_task)(*feats[ft], y, groups, u, folds[s], s, sc, f, m, ft)
        for ft, s, sc, f, m in jobs)
    print(f'{len(jobs)} fits in {time.time() - t0:.0f}s', flush=True)
    out, infos = {}, []
    for ft, s, sc, f, m, idx, union_scores, ood, info in done:
        key = f'{ft}|{s}|{sc}|{m}'
        out.setdefault(key + '|union', np.full(len(y), np.nan))[idx] = union_scores
        out.setdefault(key + '|ood', np.full((N_FOLDS, len(ood)), np.nan))[f] = ood
        infos.append({'features': ft, 'seed': s, 'scheme': sc, 'fold': f, 'model': m, **info})
    saved = {'pair': np.load(args.oof), 'key': np.load(args.oof_key_only)}
    check = {}
    for key, v in out.items():
        assert np.isfinite(v).all(), key
        ft, s, sc, m, part = key.split('|')
        if part == 'union':                    # the refit must be the model that made the OOF
            ref = saved[ft][f'{s}|{sc}|{m}']
            mask = np.isfinite(ref)
            check[key] = float(np.abs(v[mask] - ref[mask]).max())
    print('max |refit - saved OOF| =', max(check.values()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, record_id=u['record_id'].astype(str), **out)
    args.out.with_suffix('.fits.json').write_text(
        json.dumps({'refit_vs_saved_oof_max_abs': check, 'fits': infos}, indent=1) + '\n')
    return 0


def ood_section(u: dict, rows_path: Path, scores_path: Path, frozen_path: Path) -> dict:
    """FPR of the classifier, Ours, DG only and cosine on benign rows the classifier never
    trained on, retrievable hits only (retrieval cos >= 0.90)."""
    o = np.load(rows_path, allow_pickle=False)
    sc = np.load(scores_path, allow_pickle=False)
    refit = json.loads(scores_path.with_suffix('.fits.json').read_text())
    frozen = json.loads(frozen_path.read_text())
    label, corpus, intents = o['label'], o['corpus'], o['intent_id'].astype(object)
    hit = o['retrieval_cos'] >= R.RETRIEVAL_FLOOR
    union_intents = set(u['intent_id'])
    # the bare test rows are union benign rows: they must carry the same pair of texts
    ukey = {r: (k, q) for r, k, q, a in zip(u['record_id'], u['key'], u['query'], u['attack']) if not a}
    bare = label == 'bare'
    same = [ukey.get(r) == (k, q) for r, k, q in
            zip(o['record_id'][bare], o['key'][bare], o['query'][bare])]
    emb_diff = np.abs(o['cos'] - o['retrieval_cos'])[label != 'QQP']
    floors = {c: float(np.quantile(u['base_cos'][~u['attack'] & (u['corpus'] == c)], .05))
              for c in ('comqa', 'nq')}
    tab9 = {c: frozen['calibrations'][c] for c in ('comqa', 'nq')}
    v = {k: o[k] for k in ('dg', 'adl_best', 'echo_best', 'judgeable', 'has_answer')}
    result = {
        'protocol': {
            'rows': 'answer_check_20260909/full/scores.jsonl, policy ' + OOD_POLICY +
                    ', malicious False; QQP = 2,636 human duplicate pairs (k = canonical '
                    'parent, q = benign_query)',
            'hit_filter': 'retrieval_cos >= 0.90 (remote_replay.py::hit); rates are over hits',
            'classifier': 'fold models refit from the same data, seeds and search as the '
                          'OOF run (max |refit - saved OOF| reported); never trained on these rows',
            'scoring': 'intent in the training union: the one fold model that never saw the '
                       'intent (per seed); intent outside the union (fresh_benign, QQP): every '
                       'fold model, blocked fraction averaged over the 5 models',
            'threshold': 'the benign 95th percentile of the pooled OOF scores of the row\'s '
                         'corpus (ComQA for comqa and QQP, NQ for nq), per seed; block iff '
                         'score > t. QQP uses the frozen ComQA threshold.',
            'fpr': 'mean and sd over the 5 seeds',
            'ours_main_table': MAIN_TABLE,
            'ours_table9': {c: {'eta': t['eta_joint'], 'eta_a': t['eta_a'],
                                'eta_dg': t['eta_dg'], 'cosine_floor': t['cosine_floor']}
                            for c, t in tab9.items()},
            'cosine_floor_row_tables': floors,
            'block_rule': 'remote_replay.py::block (unjudgeable blocks; no answer -> DG alone)'},
        'checks': {'max_abs_cos_vs_retrieval_cos': float(emb_diff.max()),
                   'n_bare_rows': int(bare.sum()),
                   'bare_rows_same_texts_as_union_row': int(sum(same)),
                   'refit_vs_saved_oof_max_abs': max(refit['refit_vs_saved_oof_max_abs'].values()),
                   'qqp_unjudgeable': int((~o['judgeable'] & (label == 'QQP')).sum())},
        'groups': {},
    }
    fold_of = {s: dict(zip(u['intent_id'], intent_folds(u['intent_id'], N_FOLDS, s)))
               for s in SEEDS}
    blocked = {}      # (feat, scheme, model) -> [per-seed blocked fraction per row]
    for ft in OOD_FEATURES:
        for scheme in OOD_SCHEMES:
            for model in MODELS:
                per_seed = []
                for s in SEEDS:
                    oof = sc[f'{ft}|{s}|{scheme}|{model}|union']
                    t = {c: float(np.quantile(oof[~u['attack'] & (u['corpus'] == c)], .95))
                         for c in ('comqa', 'nq')}
                    ood = sc[f'{ft}|{s}|{scheme}|{model}|ood']
                    frac = np.empty(len(label))
                    for c in ('comqa', 'nq', 'qqp'):
                        m = corpus == c
                        frac[m] = ood_block_fraction(ood[:, m], intents[m], fold_of[s],
                                                     t[THRESHOLD_CORPUS[c]])
                    per_seed.append(frac)
                blocked[(ft, scheme, model)] = per_seed
    for name, labels in REPORT_GROUPS.items():
        for c in ('comqa', 'nq', 'qqp'):
            rows = np.isin(label, labels) & (corpus == c)
            if not rows.any():
                continue
            h = rows & hit
            cell = {'n_rows': int(rows.sum()), 'n_hits': int(h.sum()),
                    'n_intents': int(len(set(intents[rows]))),
                    'intents_in_training_union': int(len(set(intents[rows]) & union_intents)),
                    'threshold_corpus': THRESHOLD_CORPUS[c], 'classifier': {}, 'rules': {}}
            for (ft, scheme, model), per_seed in blocked.items():
                cell['classifier'].setdefault(ft, {}).setdefault(scheme, {})[model] = mean_sd(
                    [float(p[h].mean()) for p in per_seed])
            vh = {k: x[h] for k, x in v.items()}
            tc = THRESHOLD_CORPUS[c]
            rules = {'DG only (main table)': replay_blocks(vh, MAIN_TABLE[tc]['eta_dg']),
                     'Cosine (row-table 5% floor)': o['base_cos'][h] < floors[tc]}
            if c != 'qqp':
                rules['Ours (main table)'] = replay_blocks(vh, MAIN_TABLE[c]['eta'], MAIN_TABLE[c]['eta_a'])
                rules['Ours (Table 9 calibration)'] = replay_blocks(vh, tab9[c]['eta_joint'], tab9[c]['eta_a'])
                rules['DG only (Table 9 calibration)'] = replay_blocks(vh, tab9[c]['eta_dg'])
                rules['Cosine (Table 9 floor)'] = o['base_cos'][h] < tab9[c]['cosine_floor']
            cell['rules'] = {k: {'fpr': float(b.mean()), 'blocked': int(b.sum())} for k, b in rules.items()}
            paper = {'bare': 'bare', 'Template': 'Template', 'Same query': 'Same query',
                     'LLM-written (Table 9: all three conditions)': 'LLM-written'}.get(name)
            if paper and c != 'qqp':
                fr = frozen['wrappers'][c][paper]
                cell['frozen_replay'] = {'n_hits': fr['n_hits'],
                                         'Ours': fr['fpr_or_br_given_hit']['joint_answer_check'],
                                         'DG only': fr['fpr_or_br_given_hit']['dg_only'],
                                         'Cosine': fr['fpr_or_br_given_hit']['cosine_only']}
            result['groups'].setdefault(name, {})[c] = cell
            print(name, c, cell['n_hits'], {k: round(r['fpr'], 3) for k, r in cell['rules'].items()},
                  {f'{ft}/{sch}/{m}': round(x['mean'], 3) for ft, a in cell['classifier'].items()
                   for sch, b in a.items() for m, x in b.items()}, flush=True)
    return result


def scheme_cells(u: dict, oof: Path, replicates: int, label: str) -> dict:
    """Per scheme, model and class: seed mean/sd of the OOF metrics, seed-0 bootstrap."""
    z = np.load(oof, allow_pickle=False)
    assert np.array_equal(z['record_id'], u['record_id'].astype(str))
    fits = json.loads(oof.with_suffix('.fits.json').read_text())
    out = {}
    for scheme, (held_out, _) in SCHEMES.items():
        cells = {}
        for model in MODELS:
            chosen = [f['best_params'] for f in fits if f['scheme'] == scheme and f['model'] == model]
            for cls in ([held_out] if held_out else CLASSES):
                per_seed = []
                for s in SEEDS:
                    b, a = arms(u, cls, z[f'{s}|{scheme}|{model}'])
                    assert np.isfinite(b['score']).all() and np.isfinite(a['score']).all()
                    m = score_metrics(b, a)
                    m['br_at_1pct'] = score_metrics(b, a, .01)['br']
                    per_seed.append(m)
                    if s == SEEDS[0]:
                        ci = bootstrap_metrics(b, a, replicates, f'{label}|{scheme}|{model}|{cls}')
                cell = {k: mean_sd([m[k] for m in per_seed])
                        for k in ('auc', 'br', 'realized_fpr', 'asr', 'br_poisoned', 'br_at_1pct')}
                cell['seed0_ci95'] = ci
                cell['n_attack'], cell['n_benign'] = per_seed[0]['n_attack'], per_seed[0]['n_benign']
                cell['chosen_params'] = sorted({json.dumps(c) for c in chosen})
                cells.setdefault(model, {})[cls] = cell
                print(label, scheme, model, cls, {k: round(cell[k]['mean'], 3)
                      for k in ('auc', 'br', 'realized_fpr', 'asr')}, flush=True)
        out[scheme] = cells
    return out


def key_length_cells(u: dict) -> dict:
    """Reference: the key's word count alone (block iff words >= t, benign FPR <= 5%)."""
    words = np.array([len(k.split()) for k in u['key']], float)
    out = {}
    for cls in CLASSES:
        ai, bi = class_rows(u, cls)
        t, fpr = R.discrete_threshold(words[bi], .05)
        out[cls] = {'auc': R.auc(words[ai], words[bi]), 'threshold_words': t,
                    'br': float((words[ai] >= t).mean()), 'realized_fpr': fpr,
                    'median_words_attack': float(np.median(words[ai])),
                    'median_words_benign': float(np.median(words[bi]))}
    return out


def cmd_summarize(args) -> int:
    u = load_union(args.inputs)
    emb = np.load(args.embeddings, allow_pickle=False)
    result = {
        'encoder': ENCODER, 'pooling': 'cls, no prefix, l2',
        'features': '[E(k), E(q), E(k)-E(q), E(k)*E(q)], 1536 dims, standardised per training fold',
        'models': {'logreg': 'L2 logistic regression, C in ' + str(GRIDS['logreg']['clf__C']),
                   'mlp': 'one hidden layer of 256 (relu, adam), early_stopping=True, alpha in '
                          + str(GRIDS['mlp']['clf__alpha'])},
        'selection': f'inner {INNER_FOLDS}-fold GroupKFold by intent on the training fold, ROC AUC',
        'splits': f'{N_FOLDS}-fold GroupKFold by intent over the union, seeds {list(SEEDS)}',
        'threshold': 'benign 95th percentile of the pooled OOF scores of the class corpus; '
                     'block iff score > t',
        'asr_definition': 'mean(cos>=0.90 AND accepted AND poisoned) over all attack rows',
        'bootstrap': f'{args.replicates} intent-grouped replicates on seed 0, threshold refit',
        'n_rows': int(len(u['attack'])), 'n_intents': int(len(set(u['intent_id']))),
        'n_benign': {c: int((~u['attack'] & (u['corpus'] == c)).sum()) for c in ('comqa', 'nq')},
        'n_attack': {c: int((u['attack'] & (u['cls'] == c)).sum()) for c in CLASSES},
        'embedding_check_max_abs_cos_diff': float(emb['max_abs_cos_diff']),
        'inputs_sha256': {s: sha_file(args.inputs / f'rows_{s}.jsonl') for s, _ in CLASSES.values()},
        'oof_sha256': sha_file(args.oof),
        'schemes': scheme_cells(u, args.oof, args.replicates, 'pair'),
    }
    if args.oof_key_only:
        result['key_only_control'] = {
            'features': 'E(k) alone, 384 dims; same models, grids, splits and seeds',
            'oof_sha256': sha_file(args.oof_key_only),
            'schemes': scheme_cells(u, args.oof_key_only, args.replicates, 'key')}
    if args.ood_rows:
        result['ood_benign_fpr'] = ood_section(u, args.ood_rows, args.ood_scores, args.frozen_replay)
    result['key_length_reference'] = key_length_cells(u)
    result['baselines'] = {}
    for cls in CLASSES:
        result['baselines'][cls] = baseline_cells(u, cls, args.replicates)
        print('baselines', cls, {m: (round(v['auc'], 3), round(v['br'], 3), round(v['asr'], 3))
                                 for m, v in result['baselines'][cls].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('encode')
    e.add_argument('--inputs', type=Path, required=True)
    e.add_argument('--out', type=Path, required=True)
    f = sub.add_parser('fit')
    f.add_argument('--inputs', type=Path, required=True)
    f.add_argument('--embeddings', type=Path, required=True)
    f.add_argument('--jobs', type=int, default=32)
    f.add_argument('--features', choices=('pair', 'key'), default='pair',
                   help="'key' fits on E(k) alone: a control for what the pair adds")
    f.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('summarize')
    s.add_argument('--inputs', type=Path, required=True)
    s.add_argument('--embeddings', type=Path, required=True)
    s.add_argument('--oof', type=Path, required=True)
    s.add_argument('--oof-key-only', type=Path, help='OOF scores of the E(k)-only control')
    s.add_argument('--ood-rows', type=Path, help='ood-encode output (benign rows never trained on)')
    s.add_argument('--ood-scores', type=Path, help='ood-score output')
    s.add_argument('--frozen-replay', type=Path,
                   default=Path('experiments/paper/results/supp_controls_20260920/frozen_replay.json'))
    s.add_argument('--replicates', type=int, default=2000)
    s.add_argument('--out', type=Path, required=True)
    oe = sub.add_parser('ood-encode')
    oe.add_argument('--scores', type=Path, required=True,
                    help='answer_check_20260909/full/scores.jsonl')
    oe.add_argument('--qqp', type=Path, required=True, help='qqp_validated_records.jsonl')
    oe.add_argument('--out', type=Path, required=True)
    os_ = sub.add_parser('ood-score')
    os_.add_argument('--inputs', type=Path, required=True)
    os_.add_argument('--embeddings', type=Path, required=True)
    os_.add_argument('--ood-rows', type=Path, required=True)
    os_.add_argument('--oof', type=Path, required=True)
    os_.add_argument('--oof-key-only', type=Path, required=True)
    os_.add_argument('--jobs', type=int, default=32)
    os_.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    return {'encode': cmd_encode, 'fit': cmd_fit, 'summarize': cmd_summarize,
            'ood-encode': cmd_ood_encode, 'ood-score': cmd_ood_score}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
