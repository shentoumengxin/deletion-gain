"""Segmentation sweep under the deployed rule, and held-out selection of the segmentation.

Figure 7 ranks 32 segmentations by worst-class block rate (BR) at 5% FPR and by cost. That
sweep scored DG alone, and fitted and read its thresholds on the same rows. The deployed
rule is DG plus the answer check. This script redoes the sweep under the deployed rule and
asks whether 4+2 would be chosen without looking at the rows it is reported on.

``--mode score`` (cpu-server, e5). For every row of ``rows_{lmp,scp,kca}.jsonl`` and each of
the 32 segmentations of ``granularity32/scores/*.json`` it computes DG, ADL(s*) and
Echo(s*) with the row's own answer, using the arithmetic of ``build_profile(...,
storage_dtype='float16', answer=y)`` + ``excess`` + ``AnswerColumns``. The key and the union
of its variants under all 32 segmentations are encoded in one call ([key] + union, batches
of 128) and shared across segmentations. The query and the normalised answer are each
encoded alone, as the serving path does. Three checks guard the shortcut:
  1. a seeded sample of rows is re-run through ``build_profile`` per segmentation;
  2. the 4+2 column is compared with the row tables' excess_span/adl_best/echo_best;
  3. each segmentation's DG is compared with the granularity32 ``exc``.
A segmentation that cuts a key into fewer than 3 segments cannot judge it. Serving routes
such an entry to a miss (fail-closed); the counts are recorded per segmentation.

``--mode analyze`` (NumPy only).
  full sample  Per segmentation and class: joint thresholds (``supp_rules``) fitted on
               the whole judgeable benign arm of that corpus. BR over judgeable attacks
               (Figure 7's convention); effective BR and FPR that charge unjudgeable rows
               as blocked; ASR with unjudgeable rows blocked. Worst-class BR over
               CAP/SCP/KCA. Cost = mean stored variants per judgeable row divided by
               4+2's, geometric mean over the three classes (fig7_tradeoff_ranked.py).
               The DG-only reading sits beside it. Paired intent bootstrap for
               (segmentation - 4+2) worst-class BR.
  held-out     400 seeds. Each seed splits the union of intent_ids 50/50 once, shared by
  selection    all classes (rq3_selection_split.py's split and RNG). Selection half: fit
               thresholds on its benign rows, pick the eligible segmentation with the
               highest worst-class BR on its attacks (ties: lower cost on the selection
               half, then list order). Test half: apply the selection-half thresholds
               unchanged; report worst-class BR and benign FPR of the pick and of fixed 4+2.
Eligibility is rq3_selection_split.py's rule: at most 1% of the benign rows of every corpus
are unjudgeable at 3 segments (24 of 32 there).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R

DEPLOYED = 'multi[count:4+width:2:cap16]/runs'
#: display name, row-table stem, granularity32 stem, benign pool
SETS = (('CAP', 'lmp', 'cap', 'ComQA'), ('SCP', 'scp', 'scp', 'ComQA'),
        ('KCA', 'kca', 'kca', 'NQ'))
BUDGET = 0.05
FLOOR = 3
UNJUDGEABLE_TOL = 0.01          # rq3_selection_split.UNJUDGEABLE_TOL
PARITY_TOL = 1e-5               # union encoding vs build_profile per segmentation
ROW_TOL = 1e-4                  # vs row tables and vs granularity32
PARITY_SEED = 20260923


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()
            if l.strip()]


def load_specs(gran_dir: Path) -> list[str]:
    specs = None
    for _, _, g, _ in SETS:
        s = json.loads((gran_dir / f'{g}.json').read_text())['specs']
        if specs is not None and s != specs:
            raise SystemExit('granularity32 dumps disagree on the segmentation list')
        specs = s
    return specs


# ================================================================ score (model)

def plan_key(key: str, policies) -> tuple[list[str], list[dict]]:
    """Union of span texts over all segmentations, and each segmentation's view of it."""
    from sentry.cache.defense.spans import shortened
    union, index, cuts = [], {}, []
    for pol in policies:
        sp = shortened(pol, key)
        idx = []
        for t in sp.span_texts:
            if t not in index:
                index[t] = len(union)
                union.append(t)
            idx.append(index[t])
        cuts.append({'idx': np.array(idx, dtype=np.int64), 'removed': sp.removed_texts,
                     'seg': sp.segment_count, 'n_del': len(sp.deletion_texts)})
    return union, cuts


def score_key(enc: np.ndarray, anchor: np.ndarray, answer: np.ndarray, cleaned: str,
              query: str, cuts: list[dict]) -> dict:
    """build_profile + excess + AnswerColumns arithmetic, read off one shared encoding.

    ``enc`` is the encoder output for [key] + union. Every quantity is computed exactly as
    deletion.py does: float64 unit rows, float16 storage of the whole, spans and answer
    loss, float64 unit anchor, argmax over the segmentation's own span order.
    """
    from sentry.cache.defense.deletion import _unit_rows, store_rows, unit
    from sentry.cache.defense.textnorm import content_tokens
    whole = unit(enc[0])
    union = _unit_rows(enc[1:]) if len(enc) > 1 else np.zeros((0, len(whole)))
    w16, u16 = store_rows(whole, 'float16'), store_rows(union, 'float16')
    a = unit(anchor)
    base = float(w16 @ a)
    sims = u16 @ a
    v = unit(answer)
    loss = float(whole @ v) - union @ v
    ans_words, q_words = content_tokens(cleaned), content_tokens(query)
    out = {'base': base, 'dg': [], 'adl': [], 'echo': [], 'j3': [], 'j2': [],
           'n_spans': [], 'seg': []}
    for c in cuts:
        n = len(c['idx'])
        out['n_spans'].append(n)
        out['seg'].append(c['seg'])
        out['j3'].append(bool(c['seg'] >= 3 and n > 0 and c['n_del'] > 0))
        out['j2'].append(bool(c['seg'] >= 2 and n > 0 and c['n_del'] > 0))
        if n == 0:
            for k in ('dg', 'adl', 'echo'):
                out[k].append(None)
            continue
        s = sims[c['idx']]
        b = int(np.argmax(s))
        out['dg'].append(float(s[b]) - base)
        out['adl'].append(float(store_rows(loss[c['idx']], 'float16')[b]))
        out['echo'].append(len((content_tokens(c['removed'][b]) & ans_words) - q_words))
    return out


class Encoder:
    """Caches query and answer vectors; each is encoded alone, as the serving path does."""

    def __init__(self, embedder):
        self.e = embedder
        self.single: dict[str, np.ndarray] = {}

    def one(self, text: str) -> np.ndarray:
        if text not in self.single:
            self.single[text] = self.e.encode([text])[0]
        return self.single[text]


def parity_sample(rows_by_set, specs, policies, enc: Encoder, n_per_arm: int) -> dict:
    """Re-run a seeded sample through build_profile per segmentation; max |diff|."""
    from sentry.cache.defense.calibrate import AnswerColumns
    from sentry.cache.defense.deletion import build_profile, excess
    from sentry.cache.defense.textnorm import normalise_answer
    rng = np.random.default_rng(PARITY_SEED)
    worst = {'base': 0.0, 'dg': 0.0, 'adl': 0.0}
    echo_mismatch = checked = skipped_unjudgeable = 0
    adl_diffs, sampled = [], []
    for name, rows in rows_by_set.items():
        for arm in ('genuine', 'attack'):
            pool = [i for i, r in enumerate(rows) if r['arm'] == arm]
            pick = rng.choice(pool, size=min(n_per_arm, len(pool)), replace=False)
            for i in sorted(pick.tolist()):
                r = rows[i]
                sampled.append(r['record_id'])
                union, cuts = plan_key(r['key'], policies)
                cleaned = normalise_answer(r['answer'])
                mine = score_key(enc.e.encode([r['key']] + union), enc.one(r['query']),
                                 enc.one(cleaned), cleaned, r['query'], cuts)
                cols = AnswerColumns()
                for j, pol in enumerate(policies):
                    prof = build_profile(r['key'], enc.e, pol, storage_dtype='float16',
                                         answer=r['answer'])
                    if not prof.judgeable:
                        skipped_unjudgeable += 1
                        continue
                    x = excess(prof, enc.e.encode([r['query']])[0])
                    adl, echo = cols(r['query'], x)
                    worst['base'] = max(worst['base'], abs(x.base_cos - mine['base']))
                    worst['dg'] = max(worst['dg'], abs(x.excess_span - mine['dg'][j]))
                    d = abs(adl - mine['adl'][j])
                    worst['adl'] = max(worst['adl'], d)
                    adl_diffs.append(d)
                    echo_mismatch += int(echo != mine['echo'][j])
                    checked += 1
    adl_diffs = np.asarray(adl_diffs)
    return {'n_rows': len(sampled), 'n_row_policy_pairs_checked': checked,
            'n_pairs_unjudgeable_skipped': skipped_unjudgeable,
            'max_abs_diff': worst, 'echo_mismatches': echo_mismatch,
            'n_adl_nonzero_diff': int((adl_diffs > 0).sum()),
            'tolerance': PARITY_TOL,
            'pass': bool(max(worst.values()) <= PARITY_TOL and echo_mismatch == 0),
            'sample_record_ids': sampled}


def score(args) -> int:
    import torch
    torch.set_num_threads(args.threads)
    import transformers
    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.textnorm import normalise_answer
    from sentry.embeddings import TransformerCLSEmbedder

    specs = load_specs(args.gran_dir)
    policies = [parse_policy(s) for s in specs]
    jd = specs.index(DEPLOYED)
    embedder = TransformerCLSEmbedder(args.encoder)          # batch_size 128, CLS, no prefix
    enc = Encoder(embedder)
    rows_by_set = {name: read_jsonl(args.rows_dir / f'rows_{stem}.jsonl')
                   for name, stem, _, _ in SETS}
    if args.limit:
        rows_by_set = {n: [r for r in rs if r['arm'] == 'genuine'][:args.limit]
                       + [r for r in rs if r['arm'] == 'attack'][:args.limit]
                       for n, rs in rows_by_set.items()}
    for rs in rows_by_set.values():
        for r in rs:
            if hashlib.sha256(r['answer'].encode()).hexdigest() != r['answer_sha256']:
                raise SystemExit(f"answer_sha256 mismatch on {r['record_id']}")
            if not normalise_answer(r['answer']):
                raise SystemExit(f"answer normalises to nothing: {r['record_id']}")

    receipt = {'encoder': args.encoder, 'storage_dtype': 'float16', 'specs': specs,
               'host': platform.node(), 'torch': torch.__version__,
               'transformers': transformers.__version__, 'threads': args.threads,
               'limit': args.limit or None, 'command': ' '.join(sys.argv), 'checks': {}}
    t0 = time.perf_counter()
    receipt['checks']['build_profile_parity'] = parity_sample(
        rows_by_set, specs, policies, enc, args.parity_per_arm)
    print('parity', json.dumps({k: v for k, v in receipt['checks']['build_profile_parity']
                                 .items() if k != 'sample_record_ids'}), flush=True)

    keyed: dict[str, dict] = {}          # CAP and SCP share the ComQA benign keys
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt['outputs'] = {}
    for name, stem, g, _ in SETS:
        rows = rows_by_set[name]
        out = []
        for i, r in enumerate(rows):
            if r['key'] not in keyed:
                union, cuts = plan_key(r['key'], policies)
                keyed[r['key']] = {'enc': embedder.encode([r['key']] + union), 'cuts': cuts}
            k = keyed[r['key']]
            cleaned = normalise_answer(r['answer'])
            s = score_key(k['enc'], enc.one(r['query']), enc.one(cleaned), cleaned,
                          r['query'], k['cuts'])
            out.append({'record_id': r['record_id'], 'intent_id': r['intent_id'],
                        'arm': r['arm'], 'family': r['family'], 'poisoned': r['poisoned'],
                        'base_cos': r['base_cos'], 'base_cos_recomputed': s.pop('base'),
                        'words': r['words'], 'answer_sha256': r['answer_sha256'], **s})
            if i % 200 == 0:
                print(f'  {name} {i}/{len(rows)} keys={len(keyed)} '
                      f'({time.perf_counter() - t0:.0f}s)', flush=True)
        receipt['checks'][f'{name}_vs_row_table'] = check_rows(rows, out, jd)
        if not args.limit:
            receipt['checks'][f'{name}_vs_granularity32'] = check_gran(
                args.gran_dir / f'{g}.json', out, specs)
        path = args.out_dir / f'{g}.jsonl'
        path.write_text(''.join(json.dumps(o) + '\n' for o in out), encoding='utf-8')
        receipt['outputs'][name] = {'path': str(path), 'sha256': sha_file(path),
                                    'n': len(out),
                                    'rows_sha256': sha_file(args.rows_dir / f'rows_{stem}.jsonl')}
        print(name, json.dumps(receipt['checks'][f'{name}_vs_row_table']), flush=True)
    receipt['n_distinct_keys'] = len(keyed)
    receipt['n_texts_encoded'] = int(sum(len(k['enc']) for k in keyed.values()))
    receipt['seconds'] = round(time.perf_counter() - t0, 1)
    (args.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: v for k, v in receipt['checks'].items()
                      if k != 'build_profile_parity'}, indent=1), flush=True)
    return 0


def check_rows(rows, out, jd) -> dict:
    """4+2 column vs the row tables (the numbers every other task reads)."""
    dg = np.array([abs(o['dg'][jd] - r['excess_span']) for r, o in zip(rows, out)])
    adl = np.array([abs(o['adl'][jd] - r['adl_best']) for r, o in zip(rows, out)])
    base = np.array([abs(o['base_cos_recomputed'] - r['base_cos']) for r, o in zip(rows, out)])
    echo = sum(o['echo'][jd] != r['echo_best'] for r, o in zip(rows, out))
    return {'n': len(rows), 'max_abs_dg': float(dg.max()), 'max_abs_adl': float(adl.max()),
            'max_abs_base_cos': float(base.max()), 'echo_mismatches': int(echo),
            'all_judgeable': bool(all(o['j3'][jd] for o in out)), 'tolerance': ROW_TOL,
            'pass': bool(max(dg.max(), adl.max()) <= ROW_TOL and echo == 0)}


def check_gran(path: Path, out, specs) -> dict:
    """Every segmentation's DG vs the granularity32 per-row exc (same row order)."""
    g = json.loads(path.read_text())
    if len(g['rows']) != len(out) or any(
            (x['intent_id'], x['arm'], x['words']) != (o['intent_id'], o['arm'], o['words'])
            for x, o in zip(g['rows'], out)):
        raise SystemExit(f'{path}: rows do not align with the row table')
    per = {}
    for j, spec in enumerate(specs):
        d = [abs(o['dg'][j] - x['exc'][j]) for x, o in zip(g['rows'], out)
             if o['dg'][j] is not None and x['exc'][j] is not None]
        per[spec] = {'max_abs_dg': float(max(d)) if d else None,
                     'j3_disagree': sum(x['j3'][j] != o['j3'][j] for x, o in zip(g['rows'], out)),
                     'j2_disagree': sum(x['j2'][j] != o['j2'][j] for x, o in zip(g['rows'], out))}
    worst = max(v['max_abs_dg'] for v in per.values() if v['max_abs_dg'] is not None)
    return {'max_abs_dg_any_policy': worst, 'tolerance': ROW_TOL,
            'pass': bool(worst <= ROW_TOL and all(v['j3_disagree'] == 0 for v in per.values())),
            'per_policy': per, 'sha256': sha_file(path)}


# ================================================================ analyze (numpy)

def load_table(path: Path, n_specs: int) -> dict:
    """Per-row jsonl -> aligned arrays; per-segmentation columns are (n, 32)."""
    rows = read_jsonl(path)

    def mat(key, dtype, fill):
        return np.array([[fill if v is None else v for v in r[key]] for r in rows], dtype=dtype)
    t = {'intent': np.array([r['intent_id'] for r in rows], dtype=object),
         'benign': np.array([r['arm'] == 'genuine' for r in rows]),
         'poisoned': np.array([bool(r['poisoned']) for r in rows]),
         'cos': np.array([r['base_cos'] for r in rows], dtype=float),
         'dg': mat('dg', float, np.nan), 'adl': mat('adl', float, np.nan),
         'echo': mat('echo', float, np.nan), 'j3': mat('j3', bool, False),
         'j2': mat('j2', bool, False), 'n_spans': mat('n_spans', float, 0)}
    assert t['dg'].shape[1] == n_specs
    return t


def take_rows(t: dict, idx) -> dict:
    return {k: v[idx] for k, v in t.items()}


def fit(rule: str, t: dict, j: int, jud: np.ndarray):
    """Thresholds from the judgeable benign rows of one corpus, one segmentation."""
    b = t['benign'] & jud
    if rule == 'dg':
        return R.dg_only_threshold(t['dg'][b, j], BUDGET), None
    return R.joint_thresholds(t['dg'][b, j], t['adl'][b, j], t['echo'][b, j], BUDGET)


def blocks(rule: str, t: dict, j: int, th) -> np.ndarray:
    """Blocked under the rule; evaluate only on judgeable rows (NaN compares False)."""
    if rule == 'dg':
        return t['dg'][:, j] > th[0]
    return R.joint_blocks({'excess_span': t['dg'][:, j], 'adl_best': t['adl'][:, j],
                           'echo_best': t['echo'][:, j]}, th[0], th[1])


def read(rule: str, t: dict, j: int, th, jud: np.ndarray) -> dict:
    blk = blocks(rule, t, j, th) & jud
    att, ben = ~t['benign'], t['benign']
    na, nb = int(att.sum()), int(ben.sum())
    naj, nbj = int((att & jud).sum()), int((ben & jud).sum())
    stopped = blk | ~jud                               # unjudgeable -> miss (fail-closed)
    k, n, rate = R.asr(t['cos'][att], t['poisoned'][att], stopped[att])
    pois = att & t['poisoned']
    return {'br': float(blk[att & jud].mean()) if naj else float('nan'),
            'br_effective': float(stopped[att].mean()) if na else float('nan'),
            'br_poisoned': float(stopped[pois].mean()) if pois.any() else float('nan'),
            'n_poisoned': int(pois.sum()),
            'fpr': float(blk[ben & jud].mean()) if nbj else float('nan'),
            'fpr_effective': float(stopped[ben].mean()) if nb else float('nan'),
            'asr': rate, 'asr_successes': k, 'n_attack': na, 'n_benign': nb,
            'n_attack_unjudgeable': na - naj, 'n_benign_unjudgeable': nb - nbj}


def judgeable(t: dict, j: int) -> np.ndarray:
    """rq3_granularity's floor, per corpus: 3 segments, or 2 when fewer than 4 benign or
    2 attack rows are judgeable at 3 (only count:2 and count:2/runs, which always cut 2)."""
    for key in ('j3', 'j2'):
        m = t[key][:, j]
        if (t['benign'] & m).sum() >= 4 and (~t['benign'] & m).sum() >= 2:
            return m
    return t['j2'][:, j]


def mean_spans(t: dict, j: int, jud: np.ndarray) -> float:
    return float(t['n_spans'][jud, j].mean())


def rel_cost(tables: dict, j: int, jref: int) -> float:
    """Geometric mean over classes of mean stored variants / the reference's."""
    return float(np.exp(np.mean([np.log(mean_spans(t, j, judgeable(t, j))
                                         / mean_spans(t, jref, judgeable(t, jref)))
                                 for t in tables.values()])))


def eligible_cols(tables: dict, n_specs: int) -> list[int]:
    """rq3_selection_split.eligible_cells: benign unjudgeable share <= 1% in every corpus."""
    keep = []
    for j in range(n_specs):
        if all((t['benign'] & ~t['j3'][:, j]).sum() / t['benign'].sum() <= UNJUDGEABLE_TOL
               for t in tables.values()):
            keep.append(j)
    return keep


# ---------------------------------------------------------------- full sample

def full_sample(tables: dict, specs: list[str], jref: int) -> dict:
    out = {}
    for j, spec in enumerate(specs):
        cell = {'floor': {c: 3 if np.array_equal(judgeable(t, j), t['j3'][:, j]) else 2
                          for c, t in tables.items()},
                'cost_rel': rel_cost(tables, j, jref),
                'mean_variants': {c: mean_spans(t, j, judgeable(t, j))
                                  for c, t in tables.items()},
                'benign_unjudgeable_share_at_3': max(
                    float((t['benign'] & ~t['j3'][:, j]).sum() / t['benign'].sum())
                    for t in tables.values())}
        for rule in ('joint', 'dg'):
            per = {}
            for c, t in tables.items():
                jud = judgeable(t, j)
                th = fit(rule, t, j, jud)
                per[c] = {'eta': th[0], 'eta_a': th[1], **read(rule, t, j, th, jud)}
            cell[rule] = {'per_class': per,
                          'worst_br': min(v['br'] for v in per.values()),
                          'worst_class': min(per, key=lambda c: per[c]['br']),
                          'worst_br_effective': min(v['br_effective'] for v in per.values()),
                          'worst_br_poisoned': min(v['br_poisoned'] for v in per.values()),
                          'asr_pooled': sum(v['asr_successes'] for v in per.values())
                          / sum(v['n_attack'] for v in per.values()),
                          'asr_max_class': max(v['asr'] for v in per.values())}
        out[spec] = cell
    for rule in ('joint', 'dg'):
        elig = [s for s in specs if out[s]['benign_unjudgeable_share_at_3'] <= UNJUDGEABLE_TOL]
        for i, s in enumerate(sorted(elig, key=lambda s: -out[s][rule]['worst_br'])):
            out[s][rule]['rank_among_eligible'] = i + 1
        front, best = set(), -np.inf
        for s in sorted(elig, key=lambda s: (out[s]['cost_rel'], -out[s][rule]['worst_br'])):
            if out[s][rule]['worst_br'] > best + 1e-12:
                front.add(s)
                best = out[s][rule]['worst_br']
        for s in specs:
            out[s][rule]['on_frontier'] = s in front
    return out


def cosine_baseline(tables: dict) -> dict:
    out = {}
    for c, t in tables.items():
        floor = float(np.quantile(t['cos'][t['benign']], BUDGET))
        att = ~t['benign']
        blk = t['cos'] < floor
        k, n, rate = R.asr(t['cos'][att], t['poisoned'][att], blk[att])
        k0, _, rate0 = R.asr(t['cos'][att], t['poisoned'][att], np.zeros(n, bool))
        out[c] = {'cos_floor': floor, 'br': float(blk[att].mean()),
                  'fpr': float(blk[t['benign']].mean()), 'asr': rate, 'asr_successes': k,
                  'asr_no_defense': rate0, 'asr_no_defense_successes': k0, 'n_attack': n}
    return out


def paired_bootstrap(tables: dict, cols: list[int], jref: int, rule: str,
                     replicates: int, seed: str) -> dict:
    """Intent bootstrap of worst-class BR and (segmentation - reference), thresholds refit.

    Benign intents are drawn once per benign pool (CAP and SCP share the ComQA rows) and
    attack intents once per class, as supp_rules.intent_bootstrap does per corpus.
    """
    rng = np.random.default_rng(R._seed(seed))
    groups = {}
    for c, t in tables.items():
        for arm, m in (('b', t['benign']), ('a', ~t['benign'])):
            ids = t['intent'][m]
            rows = np.flatnonzero(m)
            groups[c, arm] = [rows[ids == k] for k in np.unique(ids)]
    pools = {c: p for c, _, _, p in SETS if c in tables}
    for c in tables:                    # one benign draw serves every class of a pool
        c0 = next(x for x in tables if pools[x] == pools[c])
        if not np.array_equal(np.unique(tables[c]['intent'][tables[c]['benign']]),
                              np.unique(tables[c0]['intent'][tables[c0]['benign']])):
            raise SystemExit(f'{c} and {c0} do not share benign intents')
    draws = {j: [] for j in cols}
    for _ in range(replicates):
        bdraw = {}
        for p in sorted(set(pools.values())):
            c0 = next(c for c in tables if pools[c] == p)
            g = groups[c0, 'b']
            bdraw[p] = rng.integers(len(g), size=len(g))
        idx = {}
        for c in tables:
            gb, ga = groups[c, 'b'], groups[c, 'a']
            ai = rng.integers(len(ga), size=len(ga))
            idx[c] = np.concatenate([gb[i] for i in bdraw[pools[c]]] + [ga[i] for i in ai])
        sub = {c: take_rows(t, idx[c]) for c, t in tables.items()}
        for j in cols:
            brs = []
            for t in sub.values():
                jud = t['j3'][:, j]
                brs.append(read(rule, t, j, fit(rule, t, j, jud), jud)['br'])
            draws[j].append(min(brs))
    ref = np.asarray(draws[jref])
    out = {}
    for j in cols:
        d = np.asarray(draws[j])
        out[j] = {'worst_br_ci95': np.percentile(d, [2.5, 97.5]).tolist(),
                  'diff_vs_ref_ci95': np.percentile(d - ref, [2.5, 97.5]).tolist(),
                  'p_diff_gt_0': float((d - ref > 0).mean())}
    return out


# ---------------------------------------------------------------- held-out selection

def split_intents(intents, seed: int) -> tuple[set, set]:
    """rq3_selection_split.py's split: shuffle the union once, first half selects."""
    rng = np.random.default_rng(seed)
    ids = list(intents)
    rng.shuffle(ids)
    return set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])


def restrict(tables: dict, ids: set) -> dict:
    """Physically slice every class to the given intents (the other half is dropped)."""
    keep = list(ids)
    return {c: take_rows(t, np.isin(t['intent'], keep)) for c, t in tables.items()}


def select(tables: dict, cols: list[int], jref: int, rule: str) -> dict:
    """Fit and choose on ``tables`` only: best worst-class BR, ties -> lower cost."""
    fits = {j: {c: fit(rule, t, j, t['j3'][:, j]) for c, t in tables.items()} for j in cols}
    worst = {j: min(read(rule, t, j, fits[j][c], t['j3'][:, j])['br']
                    for c, t in tables.items()) for j in cols}
    cost = {j: rel_cost(tables, j, jref) for j in cols}
    best = max(worst.values())
    tied = [j for j in cols if worst[j] >= best - 1e-12]
    pick = min(tied, key=lambda j: (cost[j], cols.index(j)))
    return {'pick': pick, 'fits': fits, 'worst': worst, 'cost': cost, 'tied': tied}


def run_seed(tables: dict, cols: list[int], jref: int, rule: str, seed: int,
             intents) -> dict:
    sel_ids, test_ids = split_intents(intents, seed)
    chosen = select(restrict(tables, sel_ids), cols, jref, rule)
    test = restrict(tables, test_ids)
    pools = {c: p for c, _, _, p in SETS if c in tables}
    held = {}
    for j in cols:
        per = {c: read(rule, t, j, chosen['fits'][j][c], t['j3'][:, j])
               for c, t in test.items()}
        held[j] = {'worst': min(v['br'] for v in per.values()),
                   'fpr': {pools[c]: per[c]['fpr'] for c in per}}
    return {'seed': seed, 'pick': chosen['pick'], 'tied': chosen['tied'],
            'sel_worst': chosen['worst'], 'fits': chosen['fits'], 'held': held}


def dist(x) -> dict:
    x = np.asarray(x, float)
    return {'mean': float(x.mean()), 'median': float(np.median(x)),
            'p05': float(np.percentile(x, 5)), 'p95': float(np.percentile(x, 95)),
            'min': float(x.min()), 'max': float(x.max())}


def selection(tables: dict, specs: list[str], cols: list[int], jref: int, rule: str,
              seeds: int) -> dict:
    intents = sorted({i for t in tables.values() for i in t['intent']})
    runs = [run_seed(tables, cols, jref, rule, s, intents) for s in range(seeds)]
    pick = [r['pick'] for r in runs]
    gap = [r['held'][r['pick']]['worst'] - r['held'][jref]['worst'] for r in runs]
    sel_pick = [r['sel_worst'][r['pick']] for r in runs]
    test_pick = [r['held'][r['pick']]['worst'] for r in runs]
    sel_ref = [r['sel_worst'][jref] for r in runs]
    test_ref = [r['held'][jref]['worst'] for r in runs]
    pools = sorted(runs[0]['held'][jref]['fpr'])
    fixed = {specs[j]: float(np.mean([r['held'][j]['worst'] for r in runs])) for j in cols}
    wins = Counter()
    for r in runs:
        top = max(r['held'][j]['worst'] for j in cols)
        for j in cols:
            wins[specs[j]] += r['held'][j]['worst'] >= top - 1e-12
    return {
        'rule': rule, 'n_seeds': seeds, 'n_intents': len(intents), 'reference': specs[jref],
        'reference_selected': {'count': pick.count(jref), 'rate': pick.count(jref) / seeds},
        'reference_tied_for_best_on_selection_half': {
            'count': sum(jref in r['tied'] for r in runs),
            'rate': sum(jref in r['tied'] for r in runs) / seeds},
        'n_seeds_with_tie_at_top': sum(len(r['tied']) > 1 for r in runs),
        'chosen_counts': {specs[j]: n for j, n in Counter(pick).most_common()},
        'held_out_gap_selected_minus_reference': {
            **dist(gap), 'frac_gt_0': float(np.mean(np.asarray(gap) > 1e-12)),
            'frac_eq_0': float(np.mean(np.abs(gap) <= 1e-12)),
            'frac_lt_0': float(np.mean(np.asarray(gap) < -1e-12))},
        'optimism_selected': {'selection_half': dist(sel_pick), 'test_half': dist(test_pick),
                              'selection_minus_test': dist(np.subtract(sel_pick, test_pick))},
        'reference_no_selection': {'selection_half': dist(sel_ref), 'test_half': dist(test_ref),
                                   'selection_minus_test': dist(np.subtract(sel_ref, test_ref))},
        'held_out_fpr_reference': {p: {**dist([r['held'][jref]['fpr'][p] for r in runs]),
                                       'frac_gt_budget': float(np.mean(
                                           [r['held'][jref]['fpr'][p] > BUDGET for r in runs]))}
                                   for p in pools},
        'held_out_fpr_selected': {p: dist([r['held'][r['pick']]['fpr'][p] for r in runs])
                                  for p in pools},
        'fixed_policy_mean_held_out_worst_br': dict(sorted(fixed.items(), key=lambda kv: -kv[1])),
        'fixed_policy_best_on_test_half_count': dict(wins.most_common()),
        'per_seed_columns': ['pick', 'sel_worst_pick', 'test_worst_pick', 'test_worst_ref',
                             *[f'test_fpr_ref_{p}' for p in pools]],
        'per_seed': [[specs[r['pick']], r['sel_worst'][r['pick']],
                      r['held'][r['pick']]['worst'], r['held'][jref]['worst'],
                      *[r['held'][jref]['fpr'][p] for p in pools]] for r in runs],
    }


def analyze(args) -> int:
    specs = load_specs(args.gran_dir)
    jref = specs.index(DEPLOYED)
    tables = {name: load_table(args.scores_dir / f'{g}.jsonl', len(specs))
              for name, _, g, _ in SETS}
    receipt = json.loads((args.scores_dir / 'receipt.json').read_text())
    cols = eligible_cols(tables, len(specs))
    if jref not in cols:
        raise SystemExit('4+2 is not eligible')

    # DG-only from the existing granularity32 scores: same rows, their exc and j3.
    g32 = {}
    for name, _, g, _ in SETS:
        d = json.loads((args.gran_dir / f'{g}.json').read_text())
        t = dict(tables[name])
        t['dg'] = np.array([[np.nan if v is None else v for v in r['exc']] for r in d['rows']])
        t['j3'] = np.array([r['j3'] for r in d['rows']])
        t['j2'] = np.array([r['j2'] for r in d['rows']])
        g32[name] = t
    if eligible_cols(g32, len(specs)) != cols:
        raise SystemExit('eligibility differs between recomputed and granularity32 flags')

    t0 = time.perf_counter()
    full = full_sample(tables, specs, jref)
    full_g32 = full_sample(g32, specs, jref)
    fig7 = figure7_reproduction(args.fig7_dir, specs, full_g32)
    boot = {rule: paired_bootstrap(tables, cols, jref, rule, args.replicates,
                                   f'sweep|{rule}') for rule in ('joint', 'dg')}
    for rule, b in boot.items():
        for j, v in b.items():
            full[specs[j]][rule]['bootstrap'] = v
    print(f'full sample + bootstrap {time.perf_counter() - t0:.0f}s', flush=True)
    sel = {'full_rule': selection(tables, specs, cols, jref, 'joint', args.seeds),
           'dg_only': selection(tables, specs, cols, jref, 'dg', args.seeds),
           'dg_only_granularity32_scores': selection(g32, specs, cols, jref, 'dg', args.seeds)}
    print(f'selection {time.perf_counter() - t0:.0f}s', flush=True)

    result = {
        'question': 'Is 4+2 still the best segmentation under the deployed rule (DG + '
                    'answer check), and would it be chosen on data it is not reported on?',
        'deployed': DEPLOYED, 'encoder': receipt['encoder'], 'storage_dtype': 'float16',
        'budget': BUDGET, 'floor_segments': FLOOR, 'retrieval_floor': R.RETRIEVAL_FLOOR,
        'rules': {'joint': 'block iff DG > eta and (ADL > eta_A or Echo >= 1); '
                           'supp_rules.joint_thresholds on judgeable benign rows',
                  'dg': 'block iff DG > eta; eta = benign 95% quantile (judgeable rows)'},
        'conventions': {
            'br': 'blocked / judgeable attacks (Figure 7 block_at_budget convention)',
            'br_effective': 'unjudgeable attacks counted as blocked (fail-closed miss)',
            'br_poisoned': 'blocked or unjudgeable / attacks whose cached answer is poisoned '
                           '(supplementary: the answer check exists to pass unpoisoned ones)',
            'fpr_effective': 'unjudgeable benign rows counted as blocked',
            'asr': 'mean(cos>=0.90 AND poisoned AND served) over all attack rows; '
                   'unjudgeable rows are not served; cos from the row tables',
            'worst_br': 'min over CAP/SCP/KCA of br',
            'cost_rel': 'mean stored span variants per judgeable row / 4+2, geometric '
                        'mean over CAP/SCP/KCA (fig7_tradeoff_ranked.py)',
            'eligibility': 'rq3_selection_split.eligible_cells: benign rows unjudgeable at '
                           '3 segments <= 1% in every corpus',
            'benign_pools': 'CAP and SCP share the 499 ComQA benign rows; KCA uses 499 NQ rows',
            'bootstrap': f'{args.replicates} intent-grouped replicates, benign intents '
                         'per pool and attack intents per class, thresholds refit',
            'selection': 'one 50/50 split of the union of intent_ids per seed '
                         '(np.random.default_rng(seed).shuffle, as rq3_selection_split.py), '
                         'shared by all classes; thresholds and choice from the selection '
                         'half only; the test half reads the selection-half thresholds '
                         'unchanged; ties at the top broken by lower selection-half cost',
        },
        'n_specs': len(specs), 'n_eligible': len(cols),
        'eligible': [specs[j] for j in cols],
        'ineligible': [s for j, s in enumerate(specs) if j not in cols],
        'checks': receipt['checks'],
        'inputs': {'scores': receipt['outputs'],
                   'granularity32': {g: sha_file(args.gran_dir / f'{g}.json')
                                     for _, _, g, _ in SETS}},
        'cosine_only': cosine_baseline(tables),
        'figure7_reproduction': fig7,
        'full_sample': full,
        'full_sample_dg_only_granularity32_scores': {
            s: {'worst_br': full_g32[s]['dg']['worst_br'], 'cost_rel': full_g32[s]['cost_rel']}
            for s in specs},
        'selection': sel,
        'command': ' '.join(sys.argv),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print_summary(result, specs, cols)
    return 0


def figure7_reproduction(fig7_dir: Path, specs, full_g32) -> dict:
    """Our DG-only reading of the granularity32 scores vs the g32 JSON Figure 7 reads."""
    block = cost = 0.0
    for name, _, g, _ in SETS:
        grid = json.loads((fig7_dir / f'g32_{g}.json').read_text())['grid']
        for s in specs:
            block = max(block, abs(grid[s]['block_at_budget']
                                   - full_g32[s]['dg']['per_class'][name]['br']))
            cost = max(cost, abs(grid[s]['dots_per_hit']['pooled']['mean']
                                 - full_g32[s]['mean_variants'][name]))
    return {'max_abs_diff_block_at_budget': block, 'max_abs_diff_mean_variants': cost,
            'pass': max(block, cost) <= 1e-12}


def print_summary(res, specs, cols):
    full = res['full_sample']
    print(f"\n{'segmentation':<36} {'cost':>5} {'full':>6} {'#':>3} {'pois':>6} "
          f"{'asr':>6} {'dg':>6} {'#':>3}")
    for s in sorted(specs, key=lambda s: -full[s]['joint']['worst_br']):
        f = full[s]
        print(f"{s:<36} {f['cost_rel']:>5.2f} {f['joint']['worst_br']:>6.3f} "
              f"{f['joint'].get('rank_among_eligible', '-'):>3} "
              f"{f['joint']['worst_br_poisoned']:>6.3f} {f['joint']['asr_pooled']:>6.4f} "
              f"{f['dg']['worst_br']:>6.3f} {f['dg'].get('rank_among_eligible', '-'):>3}"
              + ('' if f['benign_unjudgeable_share_at_3'] <= UNJUDGEABLE_TOL else '  (ineligible)'))
    for k, v in res['selection'].items():
        print(k, json.dumps({x: v[x] for x in ('reference_selected',
                                                'reference_tied_for_best_on_selection_half',
                                                'held_out_gap_selected_minus_reference')}))
        print('  chosen', dict(list(v['chosen_counts'].items())[:5]))
    print('fig7 reproduction', res['figure7_reproduction'])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('score', 'analyze'), required=True)
    p.add_argument('--rows-dir', type=Path, help='directory with rows_{lmp,scp,kca}.jsonl')
    p.add_argument('--gran-dir', type=Path, required=True,
                   help='granularity32 per-row scores {cap,scp,kca}.json')
    p.add_argument('--scores-dir', type=Path, help='per-row output of --mode score')
    p.add_argument('--fig7-dir', type=Path,
                   default=Path('experiments/paper/results/v3/granularity32'))
    p.add_argument('--out-dir', type=Path, help='score: where per-row jsonl goes')
    p.add_argument('--out', type=Path, help='analyze: summary JSON')
    p.add_argument('--encoder', default='intfloat/e5-small-v2')
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--parity-per-arm', type=int, default=5)
    p.add_argument('--limit', type=int, default=0, help='score smoke: first N rows per arm')
    p.add_argument('--seeds', type=int, default=400)
    p.add_argument('--replicates', type=int, default=2000)
    args = p.parse_args(argv)
    if args.mode == 'score':
        return score(args)
    return analyze(args)


if __name__ == '__main__':
    raise SystemExit(main())
