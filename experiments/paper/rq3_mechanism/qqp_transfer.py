"""QQP transfer of the deployed 4+2 rule, with FPR and BR read under one rule.

Population. Attacks: the 3,586 CAP entries on QQP (``datasets/qqp/c01_records.jsonl``;
3,290 distinct texts over 160 intents), each scored against its intent's human-labelled
duplicate question. Benign: all 2,636 human-labelled QQP duplicate pairs of
``validated_records.jsonl`` (cached entry = the canonical question, incoming query = its
duplicate); the 200 pairs of c01 are among them. Answers: Qwen3-8B generated with the
main-table victim settings (``gpu/victim_answers_offline.py``), joined by sha256 of the
entry text.

Two readings, both for Cosine, DG only and DG + answer check:
1. frozen: the ComQA main-table thresholds applied unchanged to QQP;
2. local: 200 intent-grouped half splits of QQP (``main_table_holdout.split_masks``),
   thresholds fitted on one half's benign pairs, FPR and BR read on the other half.
Plus cosine-matched AUC of DG with its support. Scoring needs the embedder (cpu-server);
``--summarize`` re-reads the saved per-row scores without it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.rq1_detection import supp_rules as R
from experiments.paper.rq1_detection.main_table_holdout import aggregate, split_masks
from experiments.paper.rq1_detection.v3_detect import load_rows

POLICY = 'multi[count:4+width:2:cap16]/runs'
FROZEN = {'eta': 0.002645233293430393, 'eta_a': 0.01965637207031249,
          'eta_dg_only': 0.006797628467845549}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def population(c01: Path, validated: Path) -> list[dict]:
    rows, _ = load_rows(c01, ['ndss'], ['human_qqp'])
    attacks = [dict(r, arm='attack') for r in rows if r['arm'] == 'attack']
    val = read_jsonl(validated)
    canon = {r['record_id']: r for r in val if r['query_role'] == 'canonical'}
    benign = [{'text': canon[r['parent_id']]['text'], 'anchor': r['text'], 'arm': 'genuine',
               'family': 'human_qqp', 'intent_id': r['intent_id'], 'record_id': r['record_id']}
              for r in val if r['query_role'] == 'benign_query']
    return benign + attacks


def score(rows, answers_path: Path, out_path: Path) -> None:
    from sentry.cache.defense.calibrate import AnswerColumns, parse_policy
    from sentry.cache.defense.deletion import build_profile, excess
    from sentry.embeddings import TransformerCLSEmbedder
    answers = {r['prompt_sha']: r['response'] for r in read_jsonl(answers_path)}
    embedder = TransformerCLSEmbedder('intfloat/e5-small-v2')
    policy = parse_policy(POLICY)
    columns = AnswerColumns()
    with open(out_path, 'w') as f:
        for r in rows:
            y = answers[sha(r['text'])]
            profile = build_profile(r['text'], embedder, policy, storage_dtype='float16', answer=y)
            rec = {k: r[k] for k in ('record_id', 'intent_id', 'arm', 'family')}
            rec['judgeable'] = bool(profile.judgeable)
            if profile.judgeable:
                x = excess(profile, embedder.encode([r['anchor']])[0])
                adl, echo = columns(r['anchor'], x)
                rec.update(base_cos=float(x.base_cos), words=int(x.words),
                           excess_span=float(x.excess_span),
                           adl_best=None if adl is None else float(adl),
                           echo_best=None if echo is None else int(echo))
            f.write(json.dumps(rec) + '\n')


def summarize(scored: Path, comqa_rows: Path, splits: int) -> dict:
    rows = read_jsonl(scored)
    unjudgeable = {arm: sum(1 for r in rows if r['arm'] == arm and not r['judgeable'])
                   for arm in ('genuine', 'attack')}
    b, a = R.split_arms([r for r in rows if r['judgeable']])
    comqa_b, _ = R.split_arms(read_jsonl(comqa_rows))
    frozen_floor = float(np.quantile(comqa_b['base_cos'], .05))

    def rules(bb, fit=True):
        if not fit:
            return {'Cosine': lambda x: x['base_cos'] < frozen_floor,
                    'DG only': lambda x: x['excess_span'] > FROZEN['eta_dg_only'],
                    'DG + answer check': lambda x: R.joint_blocks(x, FROZEN['eta'], FROZEN['eta_a'])}
        floor = float(np.quantile(bb['base_cos'], .05))
        eta_dg = R.dg_only_threshold(bb['excess_span'])
        eta, eta_a = R.joint_thresholds(bb['excess_span'], bb['adl_best'], bb['echo_best'])
        return {'Cosine': lambda x: x['base_cos'] < floor,
                'DG only': lambda x: x['excess_span'] > eta_dg,
                'DG + answer check': lambda x: R.joint_blocks(x, eta, eta_a)}

    out = {'n_benign': int(len(b['base_cos'])), 'n_attack': int(len(a['base_cos'])),
           'n_attack_intents': int(len(set(a['intent_id']))),
           'unjudgeable_fail_closed': unjudgeable,
           'frozen_thresholds': {**FROZEN, 'cosine_floor': frozen_floor},
           'frozen': {}, 'local_in_sample': {}, 'local_held_out': {}}
    for name, rule in rules(None, fit=False).items():
        out['frozen'][name] = {'fpr': float(rule(b).mean()), 'br': float(rule(a).mean())}
    for name, rule in rules(b).items():
        out['local_in_sample'][name] = {'fpr': float(rule(b).mean()), 'br': float(rule(a).mean())}
    per = {name: [] for name in out['frozen']}
    bi, ai = list(b['intent_id']), list(a['intent_id'])
    for seed in range(splits):
        fit_b, test_b, test_a = split_masks(bi, ai, seed)
        fitted = rules(R.take(b, np.flatnonzero(fit_b)))
        for name, rule in fitted.items():
            per[name].append({'fpr': float(rule(b)[test_b].mean()), 'br': float(rule(a)[test_a].mean())})
    for name, values in per.items():
        out['local_held_out'][name] = {k: aggregate([v[k] for v in values]) for k in ('fpr', 'br')}
    from sentry.cache.defense.calibrate import matched_auroc
    auc, support = matched_auroc(a['excess_span'], b['excess_span'], a['base_cos'], b['base_cos'],
                                 width=0.01, min_per_bin=20)
    out['dg_auc'] = R.auc(a['excess_span'], b['excess_span'])
    out['dg_cosine_matched_auc'] = {'auc': auc, 'support': support}
    out['cosine_auc'] = R.auc(-a['base_cos'], -b['base_cos'])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c01', type=Path, required=True)
    p.add_argument('--validated', type=Path, required=True)
    p.add_argument('--answers', type=Path, help='victim_answers.jsonl (prompt_sha, response)')
    p.add_argument('--scored', type=Path, required=True, help='per-row output / input with --summarize')
    p.add_argument('--comqa-rows', type=Path, required=True, help='rows_lmp.jsonl (frozen cosine floor)')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--splits', type=int, default=200)
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    if not a.summarize:
        rows = population(a.c01, a.validated)
        score(rows, a.answers, a.scored)
        print('scored', len(rows), flush=True)
        return 0
    result = summarize(a.scored, a.comqa_rows, a.splits)
    result['scored_sha256'] = hashlib.sha256(a.scored.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('n_benign', 'n_attack', 'frozen', 'dg_cosine_matched_auc')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
