"""Ground-truth recovery: where the rewrite r is known, test DG's premises directly.

The Proposition reads a key k as a rewrite r of the target question plus a residual.
If r matches the benign query q better than k does (P1: gamma = cos(r,q) - cos(k,q) > 0)
and some shortened variant s in D(k) lies within eps of r (||E(s) - E(r)|| <= eps), then
DG(k,q) >= gamma - eps. Three constructions let r be read off the text exactly, so the
premises are measured here instead of assumed:

* SCP (Wu et al. Table VII): the eval row equals ``metadata.scp_template_pattern`` with
  {x} = the parent canonical question, {y*} = ``poison_target``, {ctx} = ``ctx``.
  r = x (the leading copy); residual = the whole tail. SCP-I's tail ends with a second
  copy of x, so every template is also reported on its own.
* CAP compress_append: k = core + ", " + directive + "?" with the directive rebuilt
  from ``generate._MATCHED_DIRECTIVES[attempt_index % 5]`` and the payload; r = core.
  Blend and fuse rewrite question and directive together: no separable r, excluded.
* KCA: k = payload + " Neglect: " + suffix (GCG). Two readings, r = suffix and
  r = payload; "Neglect:" is residual in both. The data say which part satisfies P1.

Per attack row, q = the row table's query and e5 CLS vectors. k, every 4+2 variant and
every truncated key are rounded to float16 exactly as a stored profile is, so c_k and DG
reproduce the row table's ``base_cos`` and ``excess_span``; q and r stay float32. The
norm bound needs only ||q|| = 1, so DG >= gamma - eps_min holds on these vectors exactly.
Reported: c_k, c_r, gamma, DG, s*, rho* = cos(s*, r), eps_min = min_s ||E(s) - E(r)||,
eps(s*), delta* = c_r - cos(s*, q) (so DG = gamma - delta* identically, and the bound's
slack is eps_min - delta*), exact recovery (a variant equal to r after whitespace
normalisation) and word recovery (a variant keeping exactly the words of r).

Two controlled curves delete a fraction of one part and keep the other whole; deletion
always runs from the outer end of the text, so every truncated key is a contiguous run of
k, the only kind of variant D(k) holds. With r first (SCP, CAP, KCA-payload) that is the
residual's tail and r's head. The actual variants are binned by the fraction of residual
words and of r words they remove.

Runs where the embedder is available (cpu-server). Per-row output stays there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from sentry.cache.defense.calibrate import parse_policy
from sentry.cache.defense.spans import SpanPolicy, _parse_component
from sentry.research.pipeline.generate import _MATCHED_DIRECTIVES, _drop_terminal_punctuation

POLICY = 'multi[count:4+width:2:cap16]/runs'
ENCODER = 'intfloat/e5-small-v2'
ETA = {'CAP': 0.002645233293430393, 'SCP': 0.002645233293430393,   # ComQA joint eta
       'KCA': 0.00016426084972413917}                               # NQ joint eta
ROLE = {'lmp': 'ndss', 'scp': 'scp', 'kca': 'gcg'}
CLASS = {'lmp': 'CAP', 'scp': 'SCP', 'kca': 'KCA'}
KCA_SEP = ' Neglect: '
FRACTIONS = tuple(i / 10 for i in range(11))
QUANTILES = (5, 25, 50, 75, 95)
BIN_LABELS = ('0', '(0,.25]', '(.25,.5]', '(.5,.75]', '(.75,1)', '1')
PARITY_TOL = 1e-4


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()
            if l.strip()]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- recovering r

@dataclass(frozen=True)
class Rewrite:
    record_id: str
    intent_id: str
    cls: str            # CAP / SCP / KCA
    construction: str   # eval generator
    reading: str        # x / core / suffix / payload
    key: str
    r: str
    r_start: int        # character span of r inside key
    r_end: int

    @property
    def residual(self) -> str:
        return (self.key[:self.r_start] + ' ' + self.key[self.r_end:]).strip()


def scp_rewrite(rec: dict, x: str) -> Rewrite:
    md = rec['metadata']
    rebuilt = (md['scp_template_pattern'].replace('{x}', x)
               .replace('{y*}', md['poison_target']).replace('{ctx}', md.get('ctx', '')))
    if rebuilt != rec['text'] or not rec['text'].startswith(x + ' '):
        raise ValueError(f"SCP rebuild failed for {rec['record_id']}")
    return Rewrite(rec['record_id'], rec['intent_id'], 'SCP', rec['generator'], 'x',
                   rec['text'], x, 0, len(x))


def cap_rewrite(rec: dict) -> Rewrite:
    md, text = rec['metadata'], rec['text']
    directive = _MATCHED_DIRECTIVES[md['attempt_index'] % len(_MATCHED_DIRECTIVES)].format(
        payload=rec['payload'])
    tail = f', {directive}?'
    core = text[:-len(tail)]
    if (not text.endswith(tail) or not core or core != _drop_terminal_punctuation(core)
            or md['poison_target'] != rec['payload']):
        raise ValueError(f"CAP compress_append rebuild failed for {rec['record_id']}")
    return Rewrite(rec['record_id'], rec['intent_id'], 'CAP', rec['generator'], 'core',
                   text, core, 0, len(core))


def kca_rewrites(rec: dict) -> list[Rewrite]:
    text, payload, suffix = rec['text'], rec['payload'], rec['metadata']['suffix']
    if text != payload + KCA_SEP + suffix:
        raise ValueError(f"KCA split failed for {rec['record_id']}")
    base = (rec['record_id'], rec['intent_id'], 'KCA', rec['generator'])
    return [Rewrite(*base, 'suffix', text, suffix, len(payload) + len(KCA_SEP), len(text)),
            Rewrite(*base, 'payload', text, payload, 0, len(payload))]


def recover(eval_dir: Path, stem: str) -> tuple[list[Rewrite], dict]:
    """Every attack row of ``{stem}_eval.jsonl`` whose r is recoverable, rebuilt and checked."""
    records = read_jsonl(Path(eval_dir) / f'{stem}_eval.jsonl')
    attacks = [r for r in records if r['query_role'] == ROLE[stem]]
    out: list[Rewrite] = []
    counts = {'attack_rows': len(attacks)}
    if stem == 'scp':
        parents = {r['record_id']: r['text'] for r in records if r['query_role'] == 'canonical'}
        out = [scp_rewrite(r, parents[r['parent_id']]) for r in attacks]
    elif stem == 'lmp':
        keep = [r for r in attacks if r['metadata'].get('matched_strategy') == 'compress_append']
        out = [cap_rewrite(r) for r in keep]
        counts['excluded_blend_fuse'] = len(attacks) - len(keep)
    else:
        out = [rw for r in attacks for rw in kca_rewrites(r)]
    counts['recovered'] = len({rw.record_id for rw in out})
    return out, counts


# ---------------------------------------------------------------- word geometry

def word_roles(key: str, r_start: int, r_end: int) -> np.ndarray:
    """Per word of ``key.split()``: does it overlap r's characters? A word straddling the
    boundary (CAP's 'school,') counts as r. r must be one block at one end of k."""
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', key)]
    if [key[a:b] for a, b in spans] != key.split():
        raise ValueError('regex word split disagrees with str.split')
    in_r = np.array([a < r_end and b > r_start for a, b in spans])
    n_r = int(in_r.sum())
    if not (0 < n_r < len(in_r)) or not (in_r[:n_r].all() or in_r[len(in_r) - n_r:].all()):
        raise ValueError('r is not a contiguous block at one end of the key')
    return in_r


def half_up(x: float) -> int:
    return int(np.floor(x + .5 + 1e-9))


def truncations(key: str, r: str, in_r: np.ndarray, fractions=FRACTIONS):
    """(a) keep r, delete a fraction f of residual words; (b) keep the residual, delete a
    fraction g of r's words. Both from the outer end; f = 1 gives r itself."""
    words = key.split()
    n = len(words)
    n_r = int(in_r.sum())
    n_res = n - n_r
    first = bool(in_r[0])
    a, b = [], []
    for f in fractions:
        m = half_up(f * n_res)
        if m == 0:
            a.append(key)
        elif m == n_res:
            a.append(r)
        else:
            a.append(' '.join(words[:n - m]) if first else ' '.join(words[m:]))
    for g in fractions:
        m = half_up(g * n_r)
        if m == 0:
            b.append(key)
        else:
            b.append(' '.join(words[m:]) if first else ' '.join(words[:n - m]))
    return a, b


def removed_fractions(in_r: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """(fraction of residual words, fraction of r words) that the run words[a:b] drops."""
    kept = np.zeros(len(in_r), bool)
    kept[a:b] = True
    n_r = int(in_r.sum())
    n_res = len(in_r) - n_r
    return (float((~kept & ~in_r).sum() / n_res), float((~kept & in_r).sum() / n_r))


def bin_index(x: float) -> int:
    """Bins of BIN_LABELS: exactly 0, (0,.25], (.25,.5], (.5,.75], (.75,1), exactly 1."""
    if x <= 1e-12:
        return 0
    if x >= 1 - 1e-12:
        return 5
    return 1 + min(3, int(np.ceil(x / .25 - 1e-9)) - 1)


def component(spec: str) -> SpanPolicy:
    return _parse_component(spec)


def components(policy: SpanPolicy) -> list[SpanPolicy]:
    return ([component(s) for s in policy.components] if policy.mode == 'multi'
            else [policy])


def variant_ranges(policy: SpanPolicy, key: str) -> list[tuple[str, int, int]]:
    """``spans.shortened(policy, key).span_texts`` with each variant's word range."""
    if policy.form != 'runs':
        raise ValueError('word ranges are derived for the runs form only')
    words = key.split()
    seen: set = set()
    out = []
    for comp in components(policy):
        parts = comp.segments(key)
        bnd = np.cumsum([0] + [len(p.split()) for p in parts])
        for s in range(len(parts)):
            for e in range(s + 1, len(parts) + 1):
                if s == 0 and e == len(parts):
                    continue
                text = ' '.join(words[bnd[s]:bnd[e]])
                if text in seen:
                    continue
                seen.add(text)
                out.append((text, int(bnd[s]), int(bnd[e])))
    return out


# ---------------------------------------------------------------- vectors

def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def f16(x: np.ndarray) -> np.ndarray:
    """What a stored profile keeps: the unit vector rounded to float16, read as float64."""
    return unit(x).astype(np.float16).astype(float)


def geometry(k: np.ndarray, q: np.ndarray, r: np.ndarray, V: np.ndarray) -> dict:
    """The Proposition's quantities. q and r must be unit vectors and are used as given, so
    the caller's other dot products with q stay bit-consistent with c_k; k and V are used
    as given too (the stored float16 rows), which the bound DG >= gamma - eps_min allows."""
    q, r = np.asarray(q, float), np.asarray(r, float)
    if abs(np.linalg.norm(q) - 1) > 1e-9 or abs(np.linalg.norm(r) - 1) > 1e-9:
        raise ValueError('geometry expects unit q and r')
    k, V = np.asarray(k, float), np.atleast_2d(np.asarray(V, float))
    c_k, c_r = float(k @ q), float(r @ q)
    scores = V @ q
    best = int(np.argmax(scores))
    eps = np.linalg.norm(V - r, axis=1)
    return {'c_k': c_k, 'c_r': c_r, 'gamma': c_r - c_k, 'dg': float(scores[best]) - c_k,
            'best': best, 'rho_star': float(V[best] @ r / np.linalg.norm(V[best])),
            'eps_min': float(eps.min()), 'argmin_eps': int(eps.argmin()),
            'eps_star': float(eps[best]), 'delta_star': c_r - float(scores[best]),
            'scores': scores}


def encode_texts(texts, embedder, cache: Path | None = None, chunk: int = 4096):
    """(index text -> row, float32 matrix). Unique texts only, encoded shortest first so a
    batch pads little; a sha256-keyed npz cache lets a rerun skip the encoder."""
    uniq = list(dict.fromkeys(texts))
    shas, vecs = [], []
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        shas, vecs = z['sha'].tolist(), [z['vec']]
    known = {s: i for i, s in enumerate(shas)}
    need = sorted({t for t in uniq if sha_text(t) not in known}, key=lambda t: (len(t), t))
    print(f'{len(uniq)} distinct texts, {len(uniq) - len(need)} cached, '
          f'{len(need)} to encode', flush=True)
    t0 = time.perf_counter()
    for start in range(0, len(need), chunk):
        batch = need[start:start + chunk]
        vecs.append(np.asarray(embedder.encode(batch), dtype=np.float32))
        shas.extend(sha_text(t) for t in batch)
        done = start + len(batch)
        print(f'  encoded {done}/{len(need)} in {time.perf_counter() - t0:.0f}s', flush=True)
        if cache is not None and (done % (chunk * 12) < chunk or done == len(need)):
            np.savez(cache, sha=np.array(shas), vec=np.concatenate(vecs))
    matrix = np.concatenate(vecs) if vecs else np.zeros((0, 384), np.float32)
    known = {s: i for i, s in enumerate(shas)}
    return {t: known[sha_text(t)] for t in uniq}, matrix


# ---------------------------------------------------------------- one row

def plan_row(rw: Rewrite, row: dict, policy: SpanPolicy) -> dict:
    in_r = word_roles(rw.key, rw.r_start, rw.r_end)
    ranges = variant_ranges(policy, rw.key)
    a_texts, b_texts = truncations(rw.key, rw.r, in_r)
    return {'rw': rw, 'row': row, 'in_r': in_r, 'ranges': ranges,
            'a_texts': a_texts, 'b_texts': b_texts}


def plan_texts(p: dict) -> list[str]:
    return ([p['rw'].key, p['row']['query'], p['rw'].r] + [t for t, _, _ in p['ranges']]
            + p['a_texts'] + p['b_texts'])


def score_row(p: dict, vec, eta: float) -> dict:
    rw, row, in_r = p['rw'], p['row'], p['in_r']
    k_raw, q, r = vec(rw.key), unit(vec(row['query'])), unit(vec(rw.r))
    k = f16(k_raw)
    V = f16(np.vstack([vec(t) for t, _, _ in p['ranges']]))
    g = geometry(k, q, r, V)
    c_k = g['c_k']
    fp_scores = unit(np.vstack([vec(t) for t, _, _ in p['ranges']])) @ q
    dg_fp = float(fp_scores.max() - unit(k_raw) @ q)
    s_text, s_a, s_b = p['ranges'][g['best']]
    res_rm, r_rm = removed_fractions(in_r, s_a, s_b)
    norm_r = ' '.join(rw.r.split())
    fr = [removed_fractions(in_r, a, b) for _, a, b in p['ranges']]
    gains = g['scores'] - c_k
    curve = lambda texts: [float(f16(vec(t)) @ q - c_k) for t in texts]
    bound = g['gamma'] - g['eps_min']
    return {
        'record_id': rw.record_id, 'intent_id': rw.intent_id, 'cls': rw.cls,
        'construction': rw.construction, 'reading': rw.reading,
        'key': rw.key, 'query': row['query'], 'r': rw.r, 'residual': rw.residual,
        'words': len(in_r), 'n_r': int(in_r.sum()), 'n_res': int((~in_r).sum()),
        'r_first': bool(in_r[0]),
        'base_cos_row': row['base_cos'], 'excess_span_row': row['excess_span'],
        'c_k': c_k, 'c_r': g['c_r'], 'gamma': g['gamma'], 'p1': g['gamma'] > 0,
        'dg': g['dg'], 'dg_fp32': dg_fp, 'eta': eta, 'dg_gt_eta': g['dg'] > eta,
        's_star': s_text, 's_star_range': [s_a, s_b],
        's_star_res_removed': res_rm, 's_star_r_removed': r_rm,
        'rho_star': g['rho_star'], 'eps_min': g['eps_min'],
        'eps_min_variant': p['ranges'][g['argmin_eps']][0], 'eps_star': g['eps_star'],
        'delta_star': g['delta_star'], 'bound': bound, 'slack': g['dg'] - bound,
        'prop_cond': g['eps_min'] < g['gamma'], 'thr_cond': bound > eta,
        'sanity': g['dg'] >= bound - 1e-12,
        'exact_recovery': any(' '.join(t.split()) == norm_r for t, _, _ in p['ranges']),
        'word_recovery': any(x == 1.0 and y == 0.0 for x, y in fr),
        'curve_residual_deleted': curve(p['a_texts']),
        'curve_r_deleted': curve(p['b_texts']),
        'variants': [[a, b, x, y, float(gn)] for (_, a, b), (x, y), gn
                     in zip(p['ranges'], fr, gains)],
    }


# ---------------------------------------------------------------- summaries

FRACTION_FIELDS = ('p1', 'prop_cond', 'thr_cond', 'dg_gt_eta', 'exact_recovery',
                   'word_recovery', 'sanity', 's_star_removes_all_residual',
                   's_star_keeps_all_r', 'delta_star_negative')
QUANTILE_FIELDS = ('c_k', 'c_r', 'gamma', 'dg', 'rho_star', 'eps_min', 'eps_star',
                   'delta_star', 'bound', 'slack', 's_star_res_removed', 's_star_r_removed',
                   'words', 'n_r', 'n_res')


def _flags(rows: list[dict]) -> dict:
    out = {f: np.array([bool(r[f]) for r in rows]) for f in FRACTION_FIELDS[:7]}
    out['s_star_removes_all_residual'] = np.array([r['s_star_res_removed'] == 1.0 for r in rows])
    out['s_star_keeps_all_r'] = np.array([r['s_star_r_removed'] == 0.0 for r in rows])
    out['delta_star_negative'] = np.array([r['delta_star'] < 0 for r in rows])
    return out


def quantiles(x) -> dict:
    x = np.asarray(x, float)
    return {str(p): float(v) for p, v in zip(QUANTILES, np.percentile(x, QUANTILES))}


def curve_summary(rows: list[dict], field: str) -> list[dict]:
    G = np.array([r[field] for r in rows], float)
    out = []
    for i, f in enumerate(FRACTIONS):
        col = G[:, i]
        q25, med, q75 = np.percentile(col, [25, 50, 75])
        out.append({'fraction': f, 'n': len(col), 'median': float(med), 'q25': float(q25),
                    'q75': float(q75), 'frac_gain_pos': float((col > 0).mean())})
    return out


def bin_summary(rows: list[dict]) -> list[dict]:
    cells: dict = {}
    for r in rows:
        for _, _, x, y, gn in r['variants']:
            cells.setdefault((bin_index(x), bin_index(y)), []).append(gn)
    out = []
    for (i, j), gains in sorted(cells.items()):
        g = np.asarray(gains)
        q25, med, q75 = np.percentile(g, [25, 50, 75])
        out.append({'residual_removed': BIN_LABELS[i], 'r_removed': BIN_LABELS[j],
                    'n_variants': len(g), 'median_gain': float(med), 'q25': float(q25),
                    'q75': float(q75), 'frac_gain_pos': float((g > 0).mean())})
    return out


def summarise(rows: list[dict], seed: str, replicates: int) -> dict:
    flags = _flags(rows)
    intents = np.array([r['intent_id'] for r in rows], dtype=object)
    arrays = {'intent_id': intents, **flags}
    names = list(FRACTION_FIELDS)
    lo, hi = R.intent_bootstrap(arrays, arrays,
                                lambda b, a: np.array([a[f].mean() for f in names]),
                                replicates=replicates, seed=seed)
    fractions = {f: {'k': int(flags[f].sum()), 'n': len(rows), 'rate': float(flags[f].mean()),
                     'ci95': [lo[i], hi[i]]} for i, f in enumerate(names)}
    res_rm = np.array([r['s_star_res_removed'] for r in rows])
    return {
        'n': len(rows), 'n_intents': int(len(set(intents))),
        'eta': rows[0]['eta'],
        'fractions': fractions,
        'quantiles': {f: quantiles([r[f] for r in rows]) for f in QUANTILE_FIELDS},
        'eps_min_minus_delta_star': quantiles([r['eps_min'] - r['delta_star'] for r in rows]),
        'eps_min_over_abs_delta_star': quantiles(
            [r['eps_min'] / max(abs(r['delta_star']), 1e-12) for r in rows]),
        's_star_residual_removed': {
            'quantiles': quantiles(res_rm),
            'frac_eq_1': float((res_rm == 1).mean()), 'frac_ge_half': float((res_rm >= .5).mean()),
            'frac_eq_0': float((res_rm == 0).mean())},
        'curve_residual_deleted': curve_summary(rows, 'curve_residual_deleted'),
        'curve_r_deleted': curve_summary(rows, 'curve_r_deleted'),
        'variant_bins': bin_summary(rows),
        'n_variants': int(sum(len(r['variants']) for r in rows)),
    }


def worked_example(rows: list[dict]) -> dict:
    """The P1 row whose DG is closest to the group's median DG (ties: shortest key)."""
    pool = [r for r in rows if r['p1']] or rows
    med = float(np.median([r['dg'] for r in pool]))
    r = min(pool, key=lambda r: (abs(r['dg'] - med), r['words']))
    keep = ('record_id', 'construction', 'reading', 'key', 'query', 'r', 'residual', 's_star',
            'eps_min_variant', 'c_k', 'c_r', 'gamma', 'dg', 'eta', 'dg_gt_eta', 'rho_star',
            'eps_min', 'eps_star', 'delta_star', 'bound', 'slack', 'prop_cond', 'thr_cond',
            'exact_recovery', 's_star_res_removed', 's_star_r_removed')
    return {'selection': 'P1 row with DG closest to the group median DG', 'group_median_dg': med,
            **{k: r[k] for k in keep}}


def groups_of(rows: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for r in rows:
        head = r['cls'] if r['cls'] != 'KCA' else f"KCA r={r['reading']}"
        g.setdefault(head, []).append(r)
        g.setdefault(f"{head} / {r['construction']}", []).append(r)
        if r['cls'] == 'SCP' and r['construction'] != 'scp_i':
            g.setdefault('SCP / Z+P', []).append(r)
    return g


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', type=Path, required=True)
    p.add_argument('--rows-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=None)
    p.add_argument('--policy', default=POLICY)
    p.add_argument('--encoder', default=ENCODER)
    p.add_argument('--replicates', type=int, default=2000)
    args = p.parse_args(argv)
    policy = parse_policy(args.policy)
    t0 = time.perf_counter()

    inputs, recovery, plans = {}, {}, []
    for stem in ('scp', 'lmp', 'kca'):
        rows_path = args.rows_dir / f'rows_{stem}.jsonl'
        eval_path = args.eval_dir / f'{stem}_eval.jsonl'
        inputs[stem] = {'rows': str(rows_path), 'rows_sha256': sha_file(rows_path),
                        'eval': str(eval_path), 'eval_sha256': sha_file(eval_path)}
        table = {r['record_id']: r for r in read_jsonl(rows_path) if r['arm'] == 'attack'}
        rewrites, counts = recover(args.eval_dir, stem)
        used = [rw for rw in rewrites if rw.record_id in table]
        for rw in used:
            if table[rw.record_id]['key'] != rw.key:
                raise ValueError(f'row table key differs from eval text: {rw.record_id}')
        counts.update({'row_table_attack_rows': len(table),
                       'analysed_rows': len({rw.record_id for rw in used})})
        recovery[CLASS[stem]] = counts
        plans += [plan_row(rw, table[rw.record_id], policy) for rw in used]
        print(f'{stem}: {counts}', flush=True)

    from sentry.embeddings import TransformerCLSEmbedder
    embedder = TransformerCLSEmbedder(args.encoder)
    index, matrix = encode_texts([t for pl in plans for t in plan_texts(pl)], embedder,
                                 args.cache)
    vec = lambda t: matrix[index[t]].astype(float)

    rows = [score_row(pl, vec, ETA[pl['rw'].cls]) for pl in plans]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / 'rewrite_recovery_rows.jsonl'
    with rows_path.open('w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')

    parity = {
        'max_abs_c_k_minus_base_cos': max(abs(r['c_k'] - r['base_cos_row']) for r in rows),
        'max_abs_dg_minus_excess_span': max(abs(r['dg'] - r['excess_span_row']) for r in rows),
        'max_abs_dg_fp32_minus_excess_span': max(abs(r['dg_fp32'] - r['excess_span_row'])
                                                 for r in rows),
        'tolerance': PARITY_TOL,
        'n_sanity_failures': sum(not r['sanity'] for r in rows),
        'n_rows_scored': len(rows)}
    print(json.dumps(parity, indent=2), flush=True)

    groups = groups_of(rows)
    summary = {
        'task': 'appendix Task 2: ground-truth recovery',
        'policy': policy.fingerprint(), 'encoder': args.encoder, 'pooling': 'cls',
        'storage': 'k, variants and truncated keys float16 (as stored); q, r float32',
        'eta_joint': ETA, 'fractions_grid': list(FRACTIONS), 'bins': list(BIN_LABELS),
        'inputs': inputs, 'recovery': recovery, 'parity': parity,
        'exclusions': 'CAP blend/fuse (533 rows): question and directive are rewritten '
                      'together, so no separable r exists.',
        'per_row_output': {'path': str(rows_path), 'sha256': sha_file(rows_path)},
        'groups': {name: summarise(g, f'recovery:{name}', args.replicates)
                   for name, g in sorted(groups.items())},
        'worked_examples': {name: worked_example(groups[name])
                            for name in ('SCP / scp_z', 'CAP') if name in groups},
        'n_texts_encoded': len(index), 'runtime_s': time.perf_counter() - t0,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'wrote {rows_path} and {args.summary}', flush=True)
    bad = [k for k in ('max_abs_c_k_minus_base_cos', 'max_abs_dg_minus_excess_span')
           if parity[k] > PARITY_TOL]
    if bad or parity['n_sanity_failures']:
        raise SystemExit(f'parity/sanity failure: {bad}, '
                         f"{parity['n_sanity_failures']} rows violate DG >= gamma - eps_min")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
