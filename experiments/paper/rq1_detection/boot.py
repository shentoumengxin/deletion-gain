"""Intent-grouped cluster bootstrap shared by ci_table1.py and table4.py.

Each (set, method, budget, arm) cell draws from its own generator seeded from its key, so a
cell's interval does not depend on which other cells are computed or in what order.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.paper.paths import REPO_ROOT, RESULTS_ROOT, data_root, perrow_root
import collections, hashlib
import numpy as np

B = 2000


def rng_for(key: str):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))


def _groups(ints):
    g = collections.defaultdict(list)
    for i, x in enumerate(ints): g[x].append(i)
    return [np.array(v) for v in g.values()]


def boot_br(key, att, att_int, ben, ben_int, budget=0.05, mask=None):
    """nominal BR and its 95% percentile interval; mask restricts the attack arm (Succ.)."""
    att = np.asarray(att, float); ben = np.asarray(ben, float)
    att_int = np.asarray(att_int); ben_int = np.asarray(ben_int)
    if mask is not None:
        m = np.asarray(mask, bool); att = att[m]; att_int = att_int[m]
    thr = np.quantile(ben, 1 - budget)
    nominal = float((att > thr).mean())
    a_idx = _groups(att_int); b_idx = _groups(ben_int)
    rng = rng_for(key)
    out = np.empty(B)
    for b in range(B):
        bs = np.concatenate([b_idx[j] for j in rng.integers(0, len(b_idx), len(b_idx))])
        as_ = np.concatenate([a_idx[j] for j in rng.integers(0, len(a_idx), len(a_idx))])
        t = np.quantile(ben[bs], 1 - budget)
        out[b] = (att[as_] > t).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return nominal, float(lo), float(hi)


def boot_rate(key, vals, ints):
    """grouped bootstrap of a mean (ISR / ASR)."""
    vals = np.asarray(vals, float); ints = np.asarray(ints)
    idx = _groups(ints)
    rng = rng_for(key)
    out = np.empty(B)
    for b in range(B):
        s = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        out[b] = vals[s].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def boot_stat(key, fn, a_int, b_int):
    """grouped bootstrap of an arbitrary statistic fn(attack_idx, benign_idx)."""
    a_idx = _groups(a_int); b_idx = _groups(b_int)
    rng = rng_for(key)
    out = []
    for b in range(B):
        bs = np.concatenate([b_idx[j] for j in rng.integers(0, len(b_idx), len(b_idx))])
        as_ = np.concatenate([a_idx[j] for j in rng.integers(0, len(a_idx), len(a_idx))])
        out.append(fn(as_, bs))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]
