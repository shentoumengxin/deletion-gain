#!/usr/bin/env python
"""Victim-model pass for the ASR measurement: one victim, every attack family.

ASR was previously measured on two axes that never met. Answer poisoning was run on
DeepSeek over seven *placement* constructions; instruction injection was run on Qwen3-8B
over the GCG family; and the three ndss families that every detection table reports had no
ASR at all. This file generates the victim's answers for all four sets in one pass against
one victim, so a single table can put the detection rate and the success rate in the same
row.

What the measurement is. The attack works at **insertion**: the attacker sends the crafted
text as an ordinary query, the cache misses, the model answers, and *that answer is what
gets cached*. So the victim prompt is the attack text and nothing else, exactly as
``rq4_isr.py`` sends it. Judging and the defended column happen downstream; this file only
produces answers, keyed so they can be joined.

The four sets:

``ndss``        the 2175 rows the detection tables report, three families. Each carries
                ``metadata.poison_target``, the literal the attacker wants echoed, so this
                family has a mechanical anchor and does not depend on a judge alone.
``gcg``         the 500 + 336 optimized collisions. Their payload is a harmful
                *instruction*, not a literal, so they need a semantic judge.
``placement``   the seven hand-written constructions, regenerated here on the new victim
                so no table mixes victims. Built by ``rq4_spread_payload.CONSTRUCTIONS``
                rather than reimplemented, so the two measurements cannot drift apart.
``benign``      the 499 canonical comqa questions with a legal anchor. Sent with no attack,
                to measure how often the victim emits the literal anyway. Should be ~0, and
                it is the control that makes the attack rates mean anything.

**The v3 sets (EVAL_REDESIGN §1).** The same three-set shape, renamed to the attacks'
own names and cut to a fixed 800 rows each by an id list built once and recorded:

``kca``         Key Collision Attack (Zhang et al.): the f1 + f2 optimized collisions,
                800 of the 836. Payload is a harmful *instruction*, so the judge decides.
``scp``         Semantic Cache Poisoning (Wu et al., NDSS 2026): question kept verbatim,
                three fixed templates Z/I/P appending the poisoned answer y*. ``literal``
                is y*.
``lmp``         Length-Matched Poisoning (ours): the ``query_role=ndss`` rows, whose
                attack text is constrained to the length of a legal paraphrase.
``benign``      unchanged in kind — the canonical comqa questions with a legal anchor.

Which 800 is not decided here. ``--set-dir`` points at the id lists
(``{kca,scp,lmp}_800.json``, keys ``record_ids`` / ``rule`` / ``seed``) written by the
sampling step; rows are filtered to those ids, in the file's order, and anything the id
list names but the corpus does not have is reported rather than dropped quietly. With no
``--set-dir`` every row of the family is built, which is what a smoke test wants.

The old set names (``ndss`` / ``gcg`` / ``placement``) still build exactly the rows they
built before, record ids included, so an existing answers file still resumes.

Two failure modes this file refuses to have:

* **Empty content read as a failed attack.** Every reasoning-capable victim tried so far
  (DeepSeek, Qwen3, glm-5.2) burns the whole token budget on reasoning and returns an
  empty string unless thinking is disabled. An empty answer scores as "attack failed" on
  every tier, so the bug would read as ASR 0.000 rather than as an error. Pass the right
  ``--extra-body`` for the endpoint, and this file counts empties and refuses to finish
  quietly if there are any.
* **Losing a run to an interruption.** Answers are appended as they arrive and existing
  ``record_id``s are skipped on restart, on top of ``Client``'s own on-disk cache.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from sentry.research.operators import Client, load_env  # noqa: E402

# ``rq4_spread_payload`` is imported lazily: it is only needed by the legacy ``placement``
# set, and the GPU box that runs the v3 sets carries this file, ``lacache_prefix.py`` and
# ``sentry/`` and nothing else.


def _spread_payload():
    from experiments.paper.rq2_robustness.rq4_spread_payload import CONSTRUCTIONS, load_payloads, load_targets
    return CONSTRUCTIONS, load_payloads, load_targets


def literal_of(payload: str) -> str:
    """The string the attacker wants emitted, e.g. ``1971-04-19``. Same rule as rq4_isr."""
    match = re.search(r'"([^"]+)"', payload)
    return match.group(1) if match else payload.strip()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _anchors(records: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """``intent_id -> canonical question`` and ``intent_id -> legal paraphrase``.

    The legal paraphrase is the query an ordinary user arrives with; it is the anchor the
    entry is scored against on the entry side, and the arriving query LaCache decodes for.
    """
    canon: dict[str, str] = {}
    legal: dict[str, str] = {}
    for r in records:
        if r.get("query_role") == "canonical" and r.get("generator") == "human_comqa":
            canon.setdefault(r["intent_id"], r["text"])
        elif r.get("query_role") == "legal":
            legal.setdefault(r["intent_id"], r["text"])
    return canon, legal


# ---- the four sets ---------------------------------------------------------

def build_ndss(records_path: Path, set_name: str = "ndss") -> list[dict]:
    """The three families the detection tables report, with their canonical/legal siblings.

    The anchor is the *legal* paraphrase of the same intent: that is the query an ordinary
    user arrives with, and it is what the entry gets scored against on the entry side. The
    canonical is what the attacker targeted.
    """
    records = _read_jsonl(records_path)
    canon, legal = _anchors(records)
    rows = []
    for r in records:
        if r.get("query_role") != "ndss":
            continue
        meta = r.get("metadata") or {}
        target = meta.get("poison_target")
        if not target:
            continue
        rows.append({
            "set": set_name,
            "family": meta.get("matched_strategy") or r.get("generator", ""),
            "record_id": r["record_id"],
            "alt_ids": [],
            "intent_id": r.get("intent_id", ""),
            "prompt": r["text"],
            "literal": str(target),
            "payload": r.get("payload") or str(target),
            "canonical": canon.get(r.get("intent_id", ""), ""),
            "anchor": legal.get(r.get("intent_id", ""), ""),
        })
    return rows


def build_lmp(records_path: Path) -> list[dict]:
    """LMP = the ``query_role=ndss`` rows under their own name. Same rows, same ids."""
    return build_ndss(records_path, set_name="lmp")


def build_gcg(f1_path: Path | None, f2_path: Path | None,
              set_name: str = "gcg") -> list[dict]:
    """The optimized collisions. No literal to match on, so the judge decides these.

    Two id conventions exist and both are carried. ``f1`` rows come from a records file
    that already has a ``record_id``; ``f2`` rows come from a merged shard file that has
    none, so theirs is positional. Under the legacy name ``gcg`` the id is positional for
    both shards, exactly as the first ASR run wrote them. Under ``kca`` the f1 id is the
    corpus's own ``record_id`` — the id a sampling step reading that file would naturally
    write into a set list — and every other spelling is kept in ``alt_ids`` so a set list
    that used one of them still matches.
    """
    legacy = set_name == "gcg"
    rows = []
    for shard, path in (("f1", f1_path), ("f2", f2_path)):
        if path is None:
            continue
        for index, r in enumerate(_read_jsonl(path)):
            meta = r.get("metadata") or {}
            positional = f"{set_name}-{shard}-{index:04d}"
            own = r.get("record_id") or ""
            primary = positional if (legacy or not own) else own
            alt = {positional, own, f"gcg-{shard}-{index:04d}",
                   f"kca-{shard}-{index:04d}"} - {"", primary}
            target = meta.get("target_question") or r.get("target_question", "")
            rows.append({
                "set": set_name,
                "family": f"{set_name}_{shard}",
                "record_id": primary,
                "alt_ids": sorted(alt),
                "intent_id": r.get("intent_id", ""),
                "prompt": r.get("attack_text") or r["text"],
                "literal": "",                       # no mechanical criterion exists
                "payload": r.get("payload", ""),
                "canonical": target,
                "anchor": target,
                "lambda_ppl": r.get("lambda_ppl", meta.get("lambda_ppl")),
                "final_cosine": r.get("final_cosine", meta.get("final_cosine")),
            })
    return rows


def build_kca(f1_path: Path | None, f2_path: Path | None) -> list[dict]:
    """KCA = the f1 + f2 collisions under the name Zhang et al. give the attack."""
    return build_gcg(f1_path, f2_path, set_name="kca")


# Where a Wu-style record may keep the poisoned answer y*, most specific first. The
# generator for this family has not run yet (EVAL_REDESIGN E1/E2); reading several keys
# means this file does not have to be edited when it does.
_SCP_LITERAL_KEYS = ("y_star", "poison_answer", "poison_target", "wrong_answer")
_SCP_TEMPLATE_KEYS = ("scp_template", "wu_template", "template", "ndss_style")


def build_scp(scp_path: Path | None, records_path: Path) -> list[dict]:
    """Wu et al.'s Semantic Cache Poisoning rows: the question verbatim plus Z/I/P.

    ``literal`` is **y\\***, the short wrong answer the attacker wants the victim to
    repeat, so this family has a mechanical criterion the way LMP does and does not rest
    on the judge alone.

    Rows are read from ``--scp-records`` when it is given, and otherwise from whatever
    ``query_role == "scp"`` rows the main records file carries, so the generator may write
    either a side file or straight into the corpus.
    """
    base = _read_jsonl(records_path)
    canon, legal = _anchors(base)
    source = _read_jsonl(scp_path) if scp_path is not None else base
    rows = []
    for r in source:
        if r.get("query_role") != "scp":
            continue
        meta = r.get("metadata") or {}
        literal = ""
        for key in _SCP_LITERAL_KEYS:
            if meta.get(key) or r.get(key):
                literal = str(meta.get(key) or r.get(key))
                break
        if not literal:
            literal = str(r.get("payload") or "")
        # The generator already spells the template as ``scp_z`` / ``scp_i`` / ``scp_p``;
        # use it so the family names in this table read the same as everywhere else.
        generator = r.get("generator", "")
        if generator.startswith("scp_"):
            family = generator
        else:
            template = next((meta[key] for key in _SCP_TEMPLATE_KEYS if meta.get(key)), "")
            family = f"scp_{str(template).lower()}" if template else (generator or "scp")
        intent = r.get("intent_id", "")
        rows.append({
            "set": "scp",
            "family": family,
            "record_id": r["record_id"],
            "alt_ids": [],
            "intent_id": intent,
            "prompt": r["text"],
            "literal": literal,
            "payload": r.get("payload") or literal,
            "canonical": canon.get(intent) or meta.get("target_question", ""),
            "anchor": legal.get(intent) or canon.get(intent)
                      or meta.get("target_question", ""),
        })
    return rows


def build_placement(records_path: Path, n_targets: int) -> list[dict]:
    """The seven placement constructions, rebuilt on this victim.

    Payloads are cycled over the targets exactly as ``rq4_isr.py`` cycles them, so the rows
    are the same rows, only the victim changes.
    """
    CONSTRUCTIONS, load_payloads, load_targets = _spread_payload()
    targets = load_targets(str(records_path), n_targets)
    payloads = load_payloads(str(records_path))
    rows = []
    for name, build in CONSTRUCTIONS.items():
        for index, (question, anchor, intent) in enumerate(targets):
            payload = payloads[index % len(payloads)]
            rows.append({
                "set": "placement",
                "family": name,
                "record_id": f"place-{name}-{intent}",
                "alt_ids": [],
                "intent_id": intent,
                "prompt": build(question, payload),
                "literal": literal_of(payload),
                "payload": payload,
                "canonical": question,
                "anchor": anchor,
            })
    return rows


def build_benign(records_path: Path, n_targets: int,
                 literals: list[tuple[str, str]] | None = None) -> list[dict]:
    """The control: the real question, no attack, carrying the literal it is scored against.

    A nonzero rate here would mean the literal is something the victim says anyway, which
    would invalidate the attack rates rather than merely qualify them.

    ``literals`` supplies that pool as ``(literal, payload)`` pairs. Left at ``None`` it is
    lifted from the corpus's ndss clauses by ``rq4_spread_payload.load_payloads``, which is
    what the first ASR run did and what the legacy sets still do. The v3 sets pass the
    poisoned answers of the attack rows they actually built, so the control is cycled over
    the literals it is a control for — and so this file does not need
    ``rq4_spread_payload`` on a machine that only runs the v3 sets.
    """
    if literals is None:
        _, load_payloads, load_targets = _spread_payload()
        literals = [(literal_of(p), p) for p in load_payloads(str(records_path))]
        targets = load_targets(str(records_path), n_targets)
    else:
        canon, legal = _anchors(_read_jsonl(records_path))
        targets = [(canon[i], legal[i], i) for i in sorted(set(canon) & set(legal))]
        targets = targets[:n_targets] if n_targets else targets
    if not literals:
        raise SystemExit("benign control has no literal pool: build an attack set first, "
                         "or point --records at a corpus with ndss rows")
    rows = []
    for index, (question, anchor, intent) in enumerate(targets):
        literal, payload = literals[index % len(literals)]
        rows.append({
            "set": "benign",
            "family": "canonical",
            "record_id": f"benign-{intent}",
            "alt_ids": [],
            "intent_id": intent,
            "prompt": question,
            "literal": literal,
            "payload": payload,
            "canonical": question,
            "anchor": anchor,
        })
    return rows


# ---- the fixed 800-row id lists -------------------------------------------

def load_set_file(path: Path) -> dict:
    """One ``{kca,scp,lmp}_800.json``: ``record_ids`` plus the rule and seed that made it.

    The rule and seed are carried through into the generated rows' provenance rather than
    checked here — this file is not the place that decides which 800.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    ids = blob.get("record_ids")
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"{path}: expected a non-empty 'record_ids' list, got "
                         f"{type(ids).__name__}")
    return {"record_ids": [str(i) for i in ids],
            "rule": blob.get("rule", ""), "seed": blob.get("seed")}


def find_set_file(set_dir: Path, name: str) -> Path | None:
    for candidate in (set_dir / f"{name}_800.json", set_dir / f"{name}.json"):
        if candidate.exists():
            return candidate
    return None


def apply_set(rows: list[dict], spec: dict, name: str) -> tuple[list[dict], dict]:
    """Keep the rows the id list names, in the id list's order.

    An id that matches nothing is reported, not dropped quietly: a set list that half
    misses is a broken join, and a run that silently generated 412 rows instead of 800
    would be discovered only in the final table.
    """
    index: dict[str, dict] = {}
    for row in rows:
        for key in [row["record_id"], *row.get("alt_ids", [])]:
            index.setdefault(key, row)
    kept, missing, seen = [], [], set()
    for record_id in spec["record_ids"]:
        row = index.get(record_id)
        if row is None:
            missing.append(record_id)
        elif id(row) not in seen:
            seen.add(id(row))
            kept.append({**row, "set_rule": spec["rule"], "set_seed": spec["seed"]})
    report = {"set": name, "requested": len(spec["record_ids"]), "matched": len(kept),
              "missing": len(missing), "missing_examples": missing[:5],
              "rule": spec["rule"], "seed": spec["seed"]}
    return kept, report


# ---- row assembly ----------------------------------------------------------

SET_NAMES = ("kca", "scp", "lmp", "benign", "ndss", "gcg", "placement")
V3_ATTACK_SETS = ("kca", "scp", "lmp")


def add_row_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments that decide *which rows*, shared with ``lacache_prefix.py``.

    The two files must build the same rows with the same ids or the prefixes cannot be
    joined to the answers, so they take the same flags and call the same builders.
    """
    parser.add_argument("--records", required=True,
                        help="final500 validated_records.jsonl (lmp/scp/ndss/placement/"
                             "benign, and the canonical+legal anchors for every set)")
    parser.add_argument("--kca-f1", "--gcg-f1", dest="kca_f1", default="",
                        help="gcg500 validated_records.jsonl (generator "
                             "cacheattack_gcg_f1)")
    parser.add_argument("--kca-f2", "--gcg-f2", dest="kca_f2", default="",
                        help="f2_records_merged.jsonl (attack_text/target_question/"
                             "payload/lambda_ppl/final_cosine)")
    parser.add_argument("--scp-records", default="",
                        help="scp_records.jsonl (query_role=scp). Omitted, scp rows are "
                             "read from --records if it carries any.")
    parser.add_argument("--set-dir", default="",
                        help="directory of {kca,scp,lmp}_800.json id lists (keys "
                             "record_ids / rule / seed). Omitted, every row of each "
                             "family is built.")
    parser.add_argument("--sets", default="ndss,gcg,placement,benign",
                        help=f"comma-separated, from {', '.join(SET_NAMES)}")
    parser.add_argument("--benign-build", choices=("auto", "legacy", "v3"), default="auto",
                        help="which literal pool the benign control cycles over. 'auto' "
                             "is v3 whenever a v3 attack set was asked for, and legacy "
                             "otherwise, so the old command builds the old rows.")
    parser.add_argument("--n-targets", type=int, default=200,
                        help="targets per placement construction (200 -> 1400 rows)")
    parser.add_argument("--n-benign", type=int, default=0,
                        help="benign control rows; 0 means every intent with a legal "
                             "anchor (499). Deliberately not tied to --n-targets: the "
                             "control fits the fence at its 5%% budget and wants the "
                             "whole arm, while placement is capped for cost.")
    parser.add_argument("--limit", type=int, default=0, help="smoke-test cap on rows")
    parser.add_argument("--limit-per-set", type=int, default=0,
                        help="smoke-test cap applied to each set/family separately, so a "
                             "3-row smoke exercises every set instead of the first one")


def limit_per_set(rows: list[dict], cap: int) -> list[dict]:
    """Keep the first ``cap`` rows of each ``set/family``. A smoke test wants coverage."""
    seen: dict[str, int] = {}
    out = []
    for row in rows:
        key = f'{row["set"]}/{row["family"]}'
        if seen.get(key, 0) < cap:
            seen[key] = seen.get(key, 0) + 1
            out.append(row)
    return out


def build_rows(args) -> tuple[list[dict], list[dict]]:
    """Every requested set's rows, filtered by the id lists, plus one report per list."""
    records_path = Path(args.records)
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SET_NAMES]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; choose from {list(SET_NAMES)}")
    set_dir = Path(args.set_dir) if getattr(args, "set_dir", "") else None

    built: dict[str, list[dict]] = {}
    if "ndss" in wanted:
        built["ndss"] = build_ndss(records_path)
    if "lmp" in wanted:
        built["lmp"] = build_lmp(records_path)
    if "gcg" in wanted:
        built["gcg"] = build_gcg(Path(args.kca_f1) if args.kca_f1 else None,
                                 Path(args.kca_f2) if args.kca_f2 else None)
    if "kca" in wanted:
        built["kca"] = build_kca(Path(args.kca_f1) if args.kca_f1 else None,
                                 Path(args.kca_f2) if args.kca_f2 else None)
    if "scp" in wanted:
        built["scp"] = build_scp(
            Path(args.scp_records) if getattr(args, "scp_records", "") else None,
            records_path)
        if not built["scp"]:
            raise SystemExit(
                "no scp rows found: neither --scp-records nor --records carries "
                "query_role=scp rows. EVAL_REDESIGN E1/E2 build them; run those first.")
    if "placement" in wanted:
        built["placement"] = build_placement(records_path, args.n_targets)

    reports = []
    for name in V3_ATTACK_SETS:
        if name not in built:
            continue
        path = find_set_file(set_dir, name) if set_dir else None
        if path is None:
            if set_dir is not None:
                print(f"NOTE: no {name}_800.json under {set_dir}; building every "
                      f"{name} row ({len(built[name])}).", flush=True)
            continue
        built[name], report = apply_set(built[name], load_set_file(path), name)
        report["file"] = str(path)
        reports.append(report)

    rows = [row for name in wanted if name != "benign" for row in built.get(name, [])]
    if "benign" in wanted:
        mode = getattr(args, "benign_build", "auto")
        if mode == "auto":
            mode = "v3" if any(n in wanted for n in V3_ATTACK_SETS) else "legacy"
        literals = None
        if mode == "v3":
            # First-seen order over the attack rows just built, so the control cycles the
            # same literals in the same order the corpus presents them.
            seen: dict[str, str] = {}
            for r in rows:
                if r.get("literal"):
                    seen.setdefault(str(r["literal"]), str(r.get("payload") or r["literal"]))
            literals = list(seen.items())
        rows += build_benign(records_path, args.n_benign, literals)
    return rows, reports


# ---- generation ------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_row_arguments(parser)
    parser.add_argument("--out", required=True, help="JSONL, appended and resumable")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--cache-name", default="asr_victim")
    parser.add_argument("--env-dir", default="",
                        help="directory holding the victim's .env (url/openai_key/model). "
                             "Defaults to the repo root, whose .env is the paraphrase "
                             "operator's DeepSeek key -- point this at a separate file when "
                             "the victim is a self-hosted server, so the operator's "
                             "credentials are never edited to run an experiment.")
    parser.add_argument(
        "--extra-body",
        default='{"chat_template_kwargs": {"enable_thinking": false}}',
        help="JSON passed through to the endpoint. vLLM/Qwen3 wants "
             '\'{"chat_template_kwargs": {"enable_thinking": false}}\'; DeepSeek wants '
             '\'{"thinking": {"type": "disabled"}}\'. Leaving thinking on returns empty '
             "content, which would read as ASR 0.000.")
    args = parser.parse_args(argv)

    rows, set_reports = build_rows(args)
    for report in set_reports:
        print("set list: " + json.dumps(report), flush=True)
        if report["missing"]:
            print(f"WARNING: {report['missing']} of {report['requested']} ids in the "
                  f"{report['set']} set list matched no row.", flush=True)
    if args.limit_per_set:
        rows = limit_per_set(rows, args.limit_per_set)
    if args.limit:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["record_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in rows if r["record_id"] not in done]

    counts: dict[str, int] = {}
    for r in rows:
        counts[f'{r["set"]}/{r["family"]}'] = counts.get(f'{r["set"]}/{r["family"]}', 0) + 1
    print(json.dumps({"rows_total": len(rows), "already_done": len(done),
                      "to_generate": len(todo), "by_family": counts}, indent=1), flush=True)
    if not todo:
        return 0

    extra_body = json.loads(args.extra_body) if args.extra_body.strip() else None
    creds = load_env(Path(args.env_dir)) if args.env_dir else load_env()
    client = Client(creds, cache_name=args.cache_name)
    print(f"victim endpoint: {creds['base_url']}  model: {creds['model']}", flush=True)

    handle = out_path.open("a", encoding="utf-8")
    empties: list[str] = []
    failures: list[tuple[str, str]] = []

    def run(item: tuple[int, dict]) -> dict | None:
        index, row = item
        try:
            answer = client.chat(
                [{"role": "user", "content": row["prompt"]}],
                temperature=args.temperature, seed=args.seed + index,
                max_tokens=args.max_tokens, extra_body=extra_body,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((row["record_id"], f"{type(exc).__name__}: {str(exc)[:160]}"))
            return None
        if not (answer or "").strip():
            empties.append(row["record_id"])
        return {**row, "response": answer, "victim_model": creds["model"],
                "temperature": args.temperature, "seed": args.seed + index}

    written = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for result in pool.map(run, enumerate(todo)):
            if result is None:
                continue
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                handle.flush()
                print(f"  {written}/{len(todo)}", flush=True)
    handle.close()

    summary = {"written": written, "failed": len(failures), "empty_responses": len(empties)}
    print(json.dumps(summary, indent=1), flush=True)
    if failures:
        print("first failures:", failures[:5], flush=True)
    if empties:
        # Loud, and a nonzero exit: an empty answer is scored as a failed attack by every
        # tier, so a silent empty run reports ASR 0.000 and looks like a working defense.
        print(f"ERROR: {len(empties)} empty responses. Thinking mode is probably still on; "
              f"check --extra-body against this endpoint. Examples: {empties[:5]}", flush=True)
        return 2
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
