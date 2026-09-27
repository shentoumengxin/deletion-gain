"""Decision rules and statistics for the appendix experiments (NumPy only).

Every threshold here is fitted on benign rows alone. The joint rule reproduces
``paper/data/recompute_e2e_asr.py::decisions`` exactly: eta_A is the benign
1-budget quantile of the answer loss, and eta is refitted on the benign rows the answer
check fires on so that the conjunction spends the whole budget. ASR intersects retrieval,
acceptance and the poisoned label on each record before averaging over all attack records.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Mapping

import numpy as np

from sentry.cache.defense.textnorm import content_tokens, normalise_answer

RETRIEVAL_FLOOR = 0.90
_NUMERIC = ('base_cos', 'excess_span', 'adl_best', 'echo_best', 'words')


def split_arms(rows: list[dict]) -> tuple[dict, dict]:
    """Dump rows -> (benign, attack) dicts of aligned arrays."""
    def pack(sub):
        out = {k: np.array([r[k] for r in sub], dtype=float) for k in _NUMERIC if k in sub[0]}
        for k in ('intent_id', 'record_id', 'family'):
            out[k] = np.array([r.get(k) for r in sub], dtype=object)
        if 'poisoned' in sub[0]:
            out['poisoned'] = np.array([bool(r['poisoned']) for r in sub])
        return out
    benign = [r for r in rows if r['arm'] == 'genuine']
    attack = [r for r in rows if r['arm'] == 'attack']
    return pack(benign), pack(attack)


def take(arrays: Mapping[str, np.ndarray], idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in arrays.items()}


# ---------------------------------------------------------------- thresholds

def dg_only_threshold(b_dg, budget: float = .05) -> float:
    return float(np.quantile(np.asarray(b_dg, float), 1 - budget))


def joint_thresholds(b_dg, b_adl, b_echo, budget: float = .05, echo_min: int = 1):
    b_dg, b_adl, b_echo = (np.asarray(x) for x in (b_dg, b_adl, b_echo))
    eta_a = float(np.quantile(b_adl, 1 - budget))
    fires = (b_adl > eta_a) | (b_echo >= echo_min)
    if fires.sum() <= budget * len(b_dg):      # the check alone already fits the budget
        return float('-inf'), eta_a
    eta = float(np.quantile(b_dg[fires], 1 - budget * len(b_dg) / fires.sum()))
    return eta, eta_a


def joint_blocks(a: Mapping[str, np.ndarray], eta: float, eta_a: float, echo_min: int = 1):
    return (a['excess_span'] > eta) & ((a['adl_best'] > eta_a) | (a['echo_best'] >= echo_min))


def ac_only_thresholds(b_adl, b_echo, budget: float = .05):
    """Answer check without the DG gate: block if ADL > eta_A or Echo >= e.

    e is the smallest integer whose benign firing rate fits the budget; eta_A then
    spends what is left on the rows Echo does not already block. Benign rows only.
    """
    b_adl, b_echo = np.asarray(b_adl, float), np.asarray(b_echo)
    e = next(e for e in range(1, int(b_echo.max()) + 2) if (b_echo >= e).mean() <= budget)
    room = int(np.floor(budget * len(b_adl) + 1e-9)) - int((b_echo >= e).sum())
    rest = np.sort(b_adl[b_echo < e])[::-1]
    eta_a = float(rest[room]) if room < len(rest) else float('-inf')   # strict '>' keeps ties out
    return eta_a, e


def discrete_threshold(b_score, budget: float = .05, candidates=None):
    """Smallest t with mean(b >= t) <= budget (score >= t blocks)."""
    b = np.asarray(b_score, float)
    cand = np.unique(b) if candidates is None else np.unique(np.asarray(candidates, float))
    cand = np.append(cand, cand.max() + 1)
    for t in cand:
        fpr = float((b >= t).mean())
        if fpr <= budget + 1e-12:
            return float(t), fpr
    raise AssertionError('unreachable: the top candidate blocks nothing')


def randomized_br(b_score, a_score, budget: float = .05) -> float:
    """Neyman-Pearson randomisation between the two operating points that bracket the
    budget, so a discrete score spends exactly the budget. Reported beside, never
    instead of, the deterministic operating point."""
    b, a = np.asarray(b_score, float), np.asarray(a_score, float)
    cand = np.unique(np.concatenate([b, a]))
    t_hi, f_hi = discrete_threshold(b, budget, cand)
    lower = cand[cand < t_hi]
    br_hi = float((a >= t_hi).mean())
    if len(lower) == 0 or f_hi >= budget - 1e-12:
        return br_hi
    t_lo = lower.max()
    f_lo, br_lo = float((b >= t_lo).mean()), float((a >= t_lo).mean())
    p = (budget - f_hi) / (f_lo - f_hi)
    return p * br_lo + (1 - p) * br_hi


# ---------------------------------------------------------------- ranking

def auc(pos, neg) -> float:
    """Mann-Whitney AUC; ties count one half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r = ranks[:len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def disjunction_envelope_auc(b_adl, b_echo, a_adl, a_echo) -> float:
    """Area under the upper ROC envelope of the family {ADL > t or Echo >= e}.

    The envelope is the best member at every FPR, chosen with the attack labels, so it
    upper-bounds any benign-calibrated member: a generous score for an ablation.
    """
    b_adl, a_adl = np.asarray(b_adl, float), np.asarray(a_adl, float)
    b_echo, a_echo = np.asarray(b_echo), np.asarray(a_echo)
    ts = np.concatenate([[-np.inf], np.unique(np.concatenate([b_adl, a_adl])), [np.inf]])
    pts = []
    for e in range(0, int(max(b_echo.max(), a_echo.max())) + 2):
        eb, ea = b_echo >= e, a_echo >= e
        if e == 0:
            continue
        for t in ts:
            pts.append(((eb | (b_adl > t)).mean(), (ea | (a_adl > t)).mean()))
    pts.append((0.0, 0.0))
    pts.append((1.0, 1.0))
    pts = np.array(sorted(set(pts)))
    fpr, tpr = pts[:, 0], np.maximum.accumulate(pts[:, 1])
    # keep the highest TPR at each FPR, then integrate the linear interpolation
    ufpr = np.unique(fpr)
    utpr = np.array([tpr[fpr == f].max() for f in ufpr])
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(utpr, ufpr))


# ---------------------------------------------------------------- outcomes

def asr(cos, poisoned, blocked, floor: float = RETRIEVAL_FLOOR):
    success = (np.asarray(cos) >= floor) & np.asarray(poisoned, bool) & ~np.asarray(blocked, bool)
    n = len(success)
    k = int(success.sum())
    return k, n, k / n


def full_key_echo(key: str, answer: str, query: str) -> int:
    """|(W(k) ∩ W(y)) \\ W(q)| with W = textnorm.content_tokens."""
    return len((content_tokens(key) & content_tokens(normalise_answer(answer)))
               - content_tokens(query))


# ---------------------------------------------------------------- uncertainty

def _seed(seed) -> int:
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], 'little')


def intent_bootstrap(b: Mapping[str, np.ndarray], a: Mapping[str, np.ndarray],
                     stat_fn: Callable[[dict, dict], float], replicates: int = 2000,
                     seed='reviewer') -> tuple:
    """95% percentile interval; benign and attack intents are resampled separately and
    ``stat_fn`` refits its own thresholds on every replicate."""
    rng = np.random.default_rng(_seed(seed))
    bg = [np.flatnonzero(b['intent_id'] == k) for k in np.unique(b['intent_id'])]
    ag = [np.flatnonzero(a['intent_id'] == k) for k in np.unique(a['intent_id'])]
    draws = []
    for _ in range(replicates):
        bi = np.concatenate([bg[j] for j in rng.integers(len(bg), size=len(bg))])
        ai = np.concatenate([ag[j] for j in rng.integers(len(ag), size=len(ag))])
        draws.append(stat_fn(take(b, bi), take(a, ai)))
    lo, hi = np.percentile(np.asarray(draws, float), [2.5, 97.5], axis=0)
    if np.ndim(lo) == 0:
        return float(lo), float(hi)
    return lo.tolist(), hi.tolist()
