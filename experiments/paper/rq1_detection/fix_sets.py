#!/usr/bin/env python
"""Fix the three attack sets (KCA / SCP / LMP) with one stated, seeded sampling rule.

The evaluation redesign asks every family to be exactly 800 rows drawn the same way, so
that a difference between two rows of a table can only come from how the attack was
built and never from which questions it happened to land on. This script is the whole
of that decision: it reads the attack corpora, applies one rule, and writes the chosen
``record_id`` lists together with the coverage statistics the appendix has to quote.

**The rule, in one sentence.** Within each family (a *generator*), rows are grouped by
the intent they attack and drawn round-robin over intents in a seeded order — every
intent that has a row in that family contributes one before any intent contributes a
second — until the family's quota is met, and no attack text is ever drawn twice.

Nothing here is random at run time. The order of intents, and of rows inside an intent,
is ``sha256(seed | salt | id)``, so the same seed reproduces the same 800 rows on any
machine and in any language, without depending on a Python RNG implementation.

**Distinct on text.** A set of 800 rows is only 800 attacks if the 800 texts differ. The
LMP corpus does not have that property for free: its generator re-emitted identical
strings under different ``metadata.attempt_index``, so 2,175 rows carry 1,040 distinct
texts, and an earlier round-robin that looked only at ``record_id`` produced an LMP set
with 194 duplicate rows. Every draw here is therefore filtered on the whitespace-
normalised text: within one ``(family, intent)`` group only the hash-first row of each
text survives, and a text already chosen anywhere in the *set* — any family — is skipped.

The filter turns the quota table into a feasibility question, so the script answers it
before drawing. Families are processed in ascending order of distinct-text headroom
(``distinct texts available − quota``), which lets the tightest family claim a text two
families share before a roomier one spends it; LMP needs exactly this, because ``fuse``
has 266 distinct texts for a quota of 266 and shares 16 of them with ``blend``. If a
family still cannot fill its quota it is taken to its distinct ceiling and the shortfall
is made up from whichever family has the most spare distinct texts, so the set reaches
800 whenever the corpus holds 800 distinct texts at all. Every one of these is reported
per family in the output, whether or not it fired.

Three sets, three quota tables:

``kca``
    Zhang et al.'s key-collision attack. Family ``f1`` (500 plain GCG rows) is taken
    whole; family ``f2`` (336 PPL-regularised rows) is cut to 300. ``f2`` rows carry no
    ``record_id`` and no ``intent_id`` — only ``target_question`` — so each is given the
    stable id ``gcg-f2-NNNN`` from its line index in the merged file plus a hash of its
    ``attack_text``, and its intent is recovered by matching ``target_question`` to a
    canonical under casefolding and whitespace normalization, keeping punctuation.

``lmp``
    This paper's length-matched poisoning: 267 / 267 / 266 over compress-append /
    blend / fuse, drawn from the 2,175 ``query_role=ndss`` rows.

``scp``
    Wu et al.'s semantic cache poisoning, built by a separate step. It arrives already
    at 800 (267 / 267 / 266 over the Z / I / P templates), so the rule is a no-op that
    takes every row — but it runs through the same code and emits the same statistics,
    which is the point of the ``--scp`` flag.

Usage (on cpu-server, where the corpora live)::

    python fix_sets.py --out-dir <server-workdir>/final500/sets
    python fix_sets.py --scp --scp-records .../datasets/scp_records.jsonl \\
        --out-dir <server-workdir>/final500/sets

Standard library only (``json``, ``hashlib``, ``argparse``); no numpy, no corpus stays
behind in the output — only ids, hashes and counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

#: Fixed for the paper. Quoted in the appendix; changing it changes the three sets.
SEED = 20260827

#: Defaults are the cpu-server paths verified on 2026-08-27; every one is overridable.
DEFAULT_LMP_RECORDS = "<server-workdir>/final500/datasets/validated_records.jsonl"
DEFAULT_KCA_F1 = "<server-workdir>/gcg500/datasets/validated_records.jsonl"
DEFAULT_KCA_F2 = "<server-workdir>/out/rq3/f2_records_merged.jsonl"
DEFAULT_KCA_ANCHORS = (
    "<server-workdir>/analysis-ws/f1e5/datasets/validated_records.jsonl")
DEFAULT_SCP_RECORDS = "<server-workdir>/final500/datasets/scp_records.jsonl"

#: LMP: the three fusion levels, in the order the paper lists them.
LMP_QUOTAS = {
    "ndss_matched_compress_append": 267,
    "ndss_matched_blend": 267,
    "ndss_matched_fuse": 266,
}

#: SCP: Wu's three black-box templates. The generator names the E2 step writes.
SCP_QUOTAS = {"scp_z": 267, "scp_i": 267, "scp_p": 266}

#: KCA: f1 whole, f2 cut. 500 + 300 = 800.
KCA_QUOTAS = {"cacheattack_gcg_f1": 500, "cacheattack_gcg_f2": 300}

RULE_TEXT = (
    "Within each family, rows are grouped by the intent they attack and drawn "
    "round-robin over intents in an order fixed by sha256(seed|family|id): every "
    "intent holding a row in that family contributes one row before any intent "
    "contributes a second, and rows inside an intent are ordered by the same hash. "
    "Drawing stops when the family's quota is met. Seed 20260827. No run-time "
    "randomness is involved, so the selection reproduces exactly from the seed alone."
)

#: The distinct-text constraint, in the one sentence the appendix quotes.
DEDUPE_RULE_TEXT = (
    "The draw is distinct on text: a row whose attack text (lowercase-preserving, "
    "whitespace-normalised) repeats one already selected anywhere in the set is "
    "skipped, families are processed in ascending order of distinct-text headroom so "
    "that the family with the least slack claims a text two families share before a "
    "roomier one spends it, and a family that still cannot fill its quota is taken to "
    "its distinct ceiling with the shortfall made up from whichever family has the "
    "most spare distinct texts."
)

#: Internal key on a pool row: full sha256 of the whitespace-normalised attack text.
#: Stripped from the JSON (only its first 16 hex are kept, as ``norm_text_sha256_16``).
DEDUPE_KEY = "_dedupe_key"


# --------------------------------------------------------------------------- helpers

def normalise(text: object) -> str:
    """Text key used to join an attack to the canonical it targets.

    Casefold and collapse whitespace, keeping punctuation. That join is exact for
    this corpus (verified: 500/500 f1 and
    336/336 f2 targets resolve against the ``comqa_f1`` canonicals).
    """
    return " ".join(str(text or "").casefold().split())


def normalise_text(text: object) -> str:
    """Identity key for an attack text: strip and collapse whitespace, nothing else.

    Case is *kept*, unlike :func:`normalise` — two attacks differing only in case are
    two different strings to a victim model and to an encoder, so they are two rows.
    Only whitespace is folded, because the generators emit the same sentence with
    different trailing newlines and that is not a second attack.
    """
    return " ".join(str(text or "").split())


def dedupe_key(text: object) -> str:
    """Full sha256 of the normalised text — the set's exact-match distinctness key."""
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def order_key(seed: int, salt: str, ident: str) -> str:
    """Deterministic sort key. Replaces a seeded RNG so the order is language-agnostic."""
    return hashlib.sha256(f"{seed}|{salt}|{ident}".encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    """Short content hash, so a row selected by line index can be re-verified later."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def round_robin(rows: list[dict], quota: int, seed: int, salt: str,
                used_texts: set[str], skip_ids: set[str] | None = None) -> list[dict]:
    """Draw ``quota`` rows, spreading them over intents as evenly as possible.

    Rows must carry ``id``, ``stratum`` and :data:`DEDUPE_KEY`. Intents are visited in
    hash order; each pass takes at most one row per intent, so an intent gets a second
    row only once every intent has had a first.

    Two filters sit on top of that order, and neither changes it:

    * within a ``(family, intent)`` group, rows are hash-sorted and then reduced to one
      row per distinct text — the hash-first one — so an intent can never contribute the
      same string twice;
    * a candidate whose text is already in ``used_texts`` is consumed and skipped, and
      the pass moves to that intent's next candidate. ``used_texts`` is owned by the
      caller and shared across families, so distinctness holds over the whole set;
      every row this function returns has added its text to it.

    Returns fewer than ``quota`` rows only when the family runs out of distinct texts —
    the caller reports that and tops the set up rather than silently reallocating.
    """
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if skip_ids and row["id"] in skip_ids:
            continue
        by_stratum[row["stratum"]].append(row)
    for stratum in list(by_stratum):
        group = sorted(by_stratum[stratum],
                       key=lambda r: order_key(seed, f"{salt}|row", r["id"]))
        seen: set[str] = set()
        deduped = []
        for row in group:
            if row[DEDUPE_KEY] in seen:
                continue
            seen.add(row[DEDUPE_KEY])
            deduped.append(row)
        by_stratum[stratum] = deduped
    strata = sorted(by_stratum, key=lambda s: order_key(seed, f"{salt}|stratum", s))

    cursors = {stratum: 0 for stratum in strata}
    chosen: list[dict] = []
    while len(chosen) < quota:
        took_any = False
        for stratum in strata:
            group = by_stratum[stratum]
            index = cursors[stratum]
            while index < len(group) and group[index][DEDUPE_KEY] in used_texts:
                index += 1
            cursors[stratum] = index
            if index >= len(group):
                continue
            row = group[index]
            cursors[stratum] = index + 1
            used_texts.add(row[DEDUPE_KEY])
            chosen.append(row)
            took_any = True
            if len(chosen) == quota:
                return chosen
        if not took_any:
            break
    return chosen


def coverage(rows: list[dict]) -> dict:
    """Intent coverage of a selection: how many intents, and how deep on each."""
    per_intent = Counter(row["intent_id"] for row in rows)
    histogram = Counter(per_intent.values())
    return {
        "n_rows": len(rows),
        "n_distinct_texts": len({row[DEDUPE_KEY] for row in rows if DEDUPE_KEY in row}),
        "n_intents_covered": len(per_intent),
        "rows_per_intent_histogram": {
            str(k): histogram[k] for k in sorted(histogram)},
        "rows_per_intent_min": min(per_intent.values()) if per_intent else 0,
        "rows_per_intent_max": max(per_intent.values()) if per_intent else 0,
        "per_intent_counts": dict(sorted(per_intent.items())),
    }


def sample_families(pool: dict[str, list[dict]], quotas: dict[str, int],
                    seed: int) -> tuple[list[dict], list[dict], dict, list[str]]:
    """Apply the rule family by family.

    Returns ``(selected, dropped, per-family stats, processing order)``.

    Families are *processed* in ascending order of distinct-text headroom (ties by
    name) so that a family with no slack claims a shared text first; they are *emitted*
    in sorted-name order regardless, so the output does not depend on which family
    happened to be tight. If the processing pass leaves the set short of the summed
    quota, the deficit is drawn from the families with spare distinct texts, most spare
    first.
    """
    used_texts: set[str] = set()
    picks: dict[str, list[dict]] = {family: [] for family in quotas}
    from_quota: dict[str, int] = {}

    distinct_available = {
        family: len({row[DEDUPE_KEY] for row in pool.get(family, [])})
        for family in quotas}
    order = sorted(quotas, key=lambda f: (distinct_available[f] - quotas[f], f))
    for family in order:
        picked = round_robin(pool.get(family, []), quotas[family], seed, family,
                             used_texts)
        picks[family] = picked
        from_quota[family] = len(picked)

    # Top-up. Only fires when a family hit its distinct ceiling; the deficit goes to
    # whichever family still holds the most unused distinct texts.
    target = sum(quotas.values())
    topped_up = {family: 0 for family in quotas}
    while sum(len(rows) for rows in picks.values()) < target:
        deficit = target - sum(len(rows) for rows in picks.values())
        spare = {family: len({row[DEDUPE_KEY] for row in pool.get(family, [])}
                             - used_texts) for family in quotas}
        donors = sorted((f for f in quotas if spare[f] > 0),
                        key=lambda f: (-spare[f], f))
        if not donors:
            break
        donor = donors[0]
        extra = round_robin(pool.get(donor, []), min(deficit, spare[donor]), seed,
                            donor, used_texts,
                            skip_ids={row["id"] for row in picks[donor]})
        if not extra:
            break
        picks[donor].extend(extra)
        topped_up[donor] += len(extra)

    selected: list[dict] = []
    dropped: list[dict] = []
    stats: dict = {}
    for family in sorted(quotas):
        rows = pool.get(family, [])
        picked = picks[family]
        picked_ids = {row["id"] for row in picked}
        rest = [row for row in rows if row["id"] not in picked_ids]
        selected.extend(picked)
        dropped.extend(rest)
        family_cov = coverage(picked)
        stats[family] = {
            "available_rows": len(rows),
            "distinct_available": distinct_available[family],
            "duplicate_rows_in_family": len(rows) - distinct_available[family],
            "available_intents": len({row["stratum"] for row in rows}),
            "quota": quotas[family],
            "selected": len(picked),
            "selected_from_quota": from_quota.get(family, 0),
            "selected_as_topup": topped_up.get(family, 0),
            "shortfall": max(0, quotas[family] - from_quota.get(family, 0)),
            "dropped": len(rest),
            "distinct_texts_selected": family_cov["n_distinct_texts"],
            "n_intents_covered": family_cov["n_intents_covered"],
            "rows_per_intent_histogram": family_cov["rows_per_intent_histogram"],
        }
    unknown = sorted(set(pool) - set(quotas))
    for family in unknown:
        dropped.extend(pool[family])
        stats[family] = {
            "available_rows": len(pool[family]), "quota": 0, "selected": 0,
            "dropped": len(pool[family]), "note": "family not in the quota table",
        }
    return selected, dropped, stats, list(order)


def public_rows(rows: list[dict]) -> list[dict]:
    """Drop the internal ``_``-prefixed keys before a row is serialised.

    The full dedupe key is a working value; what the file keeps is its first 16 hex as
    ``norm_text_sha256_16``, which is enough for a reader to re-check that the 800 rows
    carry 800 distinct texts without the file carrying any attack text.
    """
    return [{key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows]


def write_set(out_dirs: list[Path], name: str, payload: dict) -> list[Path]:
    written = []
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}_800.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        written.append(path)
    return written


# ------------------------------------------------------------------------------ sets

def build_kca(f1_path: Path, f2_path: Path, anchors_path: Path, seed: int) -> dict:
    """KCA = every f1 row + 300 of the 336 f2 rows, stratified by target question.

    f2 rows have no id of their own. They are given ``gcg-f2-NNNN`` from the line index
    of the merged file (the file is append-only and never re-sorted) and the first 16
    hex of ``sha256(attack_text)``, so a selection survives a rebuild of the file: match
    on the hash, fall back to the index.
    """
    notes: list[str] = []

    f1_rows_raw = [r for r in read_jsonl(f1_path)
                   if r.get("generator") == "cacheattack_gcg_f1"]
    # Intent -> canonical text map, from the anchor workspace the f1 targets were drawn
    # from. f2 carries only the question text, so this is the only way back to an intent.
    intent_by_text: dict[str, str] = {}
    anchor_source = None
    if anchors_path.exists():
        anchors = [r for r in read_jsonl(anchors_path)
                   if r.get("query_role") == "canonical"]
        intent_by_text = {normalise(r["text"]): r["intent_id"] for r in anchors}
        anchor_source = str(anchors_path)
    # f1 metadata carries both the target text and the intent, so it is a second, always
    # available source for the same map. Cross-check the two rather than trust one.
    from_f1 = {normalise((r.get("metadata") or {}).get("target_question")): r["intent_id"]
               for r in f1_rows_raw}
    disagree = sum(1 for k, v in from_f1.items()
                   if k in intent_by_text and intent_by_text[k] != v)
    if disagree:
        notes.append(f"{disagree} target questions map to different intents in the "
                     f"anchor file and in the f1 rows; the anchor file wins")
    for key, value in from_f1.items():
        intent_by_text.setdefault(key, value)

    pool: dict[str, list[dict]] = defaultdict(list)
    for record in f1_rows_raw:
        target = (record.get("metadata") or {}).get("target_question")
        pool["cacheattack_gcg_f1"].append({
            "id": record["record_id"],
            "stratum": normalise(target),
            "intent_id": record["intent_id"],
            "family": "f1",
            "generator": "cacheattack_gcg_f1",
            "attack_text_sha256_16": text_hash(record.get("text")),
            "norm_text_sha256_16": dedupe_key(record.get("text"))[:16],
            "final_cosine": (record.get("metadata") or {}).get("final_cosine"),
            "lambda_ppl": (record.get("metadata") or {}).get("lambda_ppl"),
            DEDUPE_KEY: dedupe_key(record.get("text")),
        })

    unresolved = 0
    for index, record in enumerate(read_jsonl(f2_path)):
        key = normalise(record.get("target_question"))
        intent = intent_by_text.get(key)
        if intent is None:
            unresolved += 1
        pool["cacheattack_gcg_f2"].append({
            "id": f"gcg-f2-{index:04d}",
            "line_index": index,
            "stratum": key,
            "intent_id": intent,
            "family": "f2",
            "generator": "cacheattack_gcg_f2",
            "attack_text_sha256_16": text_hash(record.get("attack_text")),
            "norm_text_sha256_16": dedupe_key(record.get("attack_text"))[:16],
            "final_cosine": record.get("final_cosine"),
            "lambda_ppl": record.get("lambda_ppl"),
            DEDUPE_KEY: dedupe_key(record.get("attack_text")),
        })
    if unresolved:
        notes.append(f"{unresolved} f2 rows could not be matched to a canonical by "
                     f"normalised target_question and carry intent_id=null")

    selected, dropped, per_family, family_order = sample_families(
        pool, KCA_QUOTAS, seed)
    resolved = [row for row in selected if row["intent_id"] is not None]
    distinct = len({row[DEDUPE_KEY] for row in selected})
    notes.append(
        f"distinct-text check: {distinct} distinct attack texts among "
        f"{len(selected)} selected rows"
        + ("" if distinct == len(selected) else " — DUPLICATES PRESENT"))
    notes.append(
        "KCA targets the 500-question comqa_f1 corpus (CacheAttack/data/cleaned_qa.jsonl, "
        "intent ids comqaf1-*), which is a different question set from the 500 ComQA "
        "canonicals the benign arm and LMP use — the GCG suffixes were optimised against "
        "those anchors and cannot be re-pointed without re-running the attack.")
    return {
        "set": "kca",
        "attack": "Key Collision Attack (Zhang et al. 2026)",
        "rule": RULE_TEXT + (
            " For KCA, f1 (500 plain GCG rows) is taken whole and f2 (336 "
            "PPL-regularised rows) is cut to 300; f2 holds exactly one row per target "
            "question, so the rule reduces there to a seeded draw of 300 of the 336 "
            "target questions.") + " " + DEDUPE_RULE_TEXT + (
            " KCA's 836 rows already carry 836 distinct texts, so the constraint binds "
            "nothing here and the set is the same one the un-filtered rule drew."),
        "seed": seed,
        "target_size": sum(KCA_QUOTAS.values()),
        "n_selected": len(selected),
        "n_distinct_texts": distinct,
        "family_processing_order": family_order,
        "sources": {
            "f1": str(f1_path), "f2": str(f2_path),
            "intent_anchors": anchor_source or "(absent; intents taken from f1 metadata)",
        },
        "id_scheme": {
            "cacheattack_gcg_f1": "record_id from the gcg500 workspace",
            "cacheattack_gcg_f2": ("gcg-f2-NNNN from the 0-based line index of "
                                   "f2_records_merged.jsonl, plus attack_text_sha256_16 "
                                   "(first 16 hex of sha256 of attack_text) to rejoin"),
        },
        "per_family": per_family,
        "coverage": coverage(resolved),
        "record_ids": [row["id"] for row in selected],
        "records": public_rows(selected),
        "dropped_ids": [row["id"] for row in dropped],
        "notes": notes,
    }


def build_from_records(records_path: Path, query_role: str, quotas: dict[str, int],
                       seed: int, set_name: str, attack: str, seed_notes: list[str],
                       rule_suffix: str) -> dict:
    """LMP and SCP: both are plain ``query_role``-filtered slices of a records file."""
    notes = list(seed_notes)
    rows = [r for r in read_jsonl(records_path) if r.get("query_role") == query_role]
    if not rows:
        raise SystemExit(
            f"no rows with query_role={query_role!r} in {records_path}")

    present = sorted({r.get("generator") for r in rows})
    effective = dict(quotas)
    if not set(present) & set(quotas):
        if len(present) != 3:
            raise SystemExit(
                f"expected the 3 generators {sorted(quotas)}, found {present}")
        effective = dict(zip(present, (267, 267, 266)))
        notes.append(f"generator names {present} did not match the configured "
                     f"{sorted(quotas)}; quotas 267/267/266 assigned in sorted order")

    pool: dict[str, list[dict]] = defaultdict(list)
    for record in rows:
        pool[record["generator"]].append({
            "id": record["record_id"],
            "stratum": record["intent_id"],
            "intent_id": record["intent_id"],
            "generator": record["generator"],
            "attack_text_sha256_16": text_hash(record.get("text")),
            "norm_text_sha256_16": dedupe_key(record.get("text"))[:16],
            DEDUPE_KEY: dedupe_key(record.get("text")),
        })

    # Feasibility, stated before the draw: can the quota table be met with distinct
    # texts at all? The answer goes into the file whether it is yes or no.
    distinct_by_family = {family: len({row[DEDUPE_KEY] for row in group})
                          for family, group in pool.items()}
    distinct_union = len({row[DEDUPE_KEY] for group in pool.values() for row in group})
    notes.append(
        "distinct-text pool: "
        + ", ".join(f"{family} {distinct_by_family.get(family, 0)}/{len(pool.get(family, []))}"
                    f" (quota {effective[family]})" for family in sorted(effective))
        + f"; union {distinct_union} distinct texts over "
          f"{sum(len(g) for g in pool.values())} rows")
    tight = {f: distinct_by_family.get(f, 0) for f in sorted(effective)
             if distinct_by_family.get(f, 0) < effective[f]}
    if tight:
        notes.append(f"families whose distinct-text ceiling is below their quota: "
                     f"{tight}; each is taken to its ceiling and the set is topped up "
                     f"from the families with spare distinct texts")

    selected, dropped, per_family, family_order = sample_families(
        pool, effective, seed)
    short = {f: s["shortfall"] for f, s in per_family.items() if s.get("shortfall")}
    if short:
        notes.append(f"families short of their quota after the distinct-text filter: "
                     f"{short}")
    topups = {f: s["selected_as_topup"] for f, s in per_family.items()
              if s.get("selected_as_topup")}
    if topups:
        notes.append(f"top-up rows drawn beyond quota to reach the target: {topups}")
    distinct = len({row[DEDUPE_KEY] for row in selected})
    notes.append(
        f"distinct-text check: {distinct} distinct attack texts among "
        f"{len(selected)} selected rows"
        + ("" if distinct == len(selected) else " — DUPLICATES PRESENT"))
    return {
        "set": set_name,
        "attack": attack,
        "rule": RULE_TEXT + rule_suffix + " " + DEDUPE_RULE_TEXT,
        "seed": seed,
        "target_size": sum(effective.values()),
        "n_selected": len(selected),
        "n_distinct_texts": distinct,
        "family_processing_order": family_order,
        "sources": {"records": str(records_path), "query_role": query_role},
        "id_scheme": {"all": "record_id from the records file"},
        "per_family": per_family,
        "coverage": coverage(selected),
        "record_ids": [row["id"] for row in selected],
        "records": public_rows(selected),
        "dropped_ids": [row["id"] for row in dropped],
        "notes": notes,
    }


# ------------------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", action="append", default=None,
                        help="repeatable; each gets a full copy of the set files")
    parser.add_argument("--lmp-records", default=DEFAULT_LMP_RECORDS)
    parser.add_argument("--kca-f1", default=DEFAULT_KCA_F1)
    parser.add_argument("--kca-f2", default=DEFAULT_KCA_F2)
    parser.add_argument("--kca-anchors", default=DEFAULT_KCA_ANCHORS)
    parser.add_argument("--scp-records", default=DEFAULT_SCP_RECORDS)
    parser.add_argument("--scp", action="store_true",
                        help="build scp_800.json instead of kca/lmp (needs E2's "
                             "scp_records.jsonl to exist)")
    args = parser.parse_args()

    out_dirs = [Path(p) for p in (args.out_dir or ["."])]
    summary = {}

    if args.scp:
        payload = build_from_records(
            Path(args.scp_records), "scp", SCP_QUOTAS, args.seed, "scp",
            "Semantic Cache Poisoning (Wu et al., NDSS 2026)",
            ["SCP is built at exactly 800 by the E2 generation step, so this rule "
             "selects every row it can; it runs anyway to emit the same coverage "
             "statistics and to enforce the same distinct-text constraint."],
            " For SCP the quota equals the family size, so no row is dropped unless "
            "two rows carry the same text.")
        for path in write_set(out_dirs, "scp", payload):
            print(f"wrote {path}")
        summary["scp"] = payload
    else:
        kca = build_kca(Path(args.kca_f1), Path(args.kca_f2),
                        Path(args.kca_anchors), args.seed)
        for path in write_set(out_dirs, "kca", kca):
            print(f"wrote {path}")
        lmp = build_from_records(
            Path(args.lmp_records), "ndss", LMP_QUOTAS, args.seed, "lmp",
            "Length-Matched Poisoning (this paper)",
            ["The LMP corpus holds rows for 191 of the 500 ComQA intents; the "
             "remaining intents have no length-matched attack, so coverage is bounded "
             "at 191 by the corpus, not by the sampling rule.",
             "The LMP generator re-emitted identical strings under different "
             "metadata.attempt_index: 2,175 rows carry 1,040 distinct texts. The "
             "distinct-text constraint is what makes this set 800 attacks rather than "
             "800 rows; without it the same draw yielded 606 distinct texts."],
            " For LMP the quotas are 267 / 267 / 266 over compress-append, blend and "
            "fuse.")
        for path in write_set(out_dirs, "lmp", lmp):
            print(f"wrote {path}")
        summary["kca"], summary["lmp"] = kca, lmp

    for name, payload in summary.items():
        distinct = payload["n_distinct_texts"]
        flag = "ok" if distinct == payload["n_selected"] else "DUPLICATES"
        print(f"\n{name}: {payload['n_selected']}/{payload['target_size']} rows, "
              f"{distinct} distinct texts ({flag}), "
              f"{payload['coverage']['n_intents_covered']} intents, "
              f"histogram {payload['coverage']['rows_per_intent_histogram']}")
        print(f"  family processing order (tightest headroom first): "
              f"{payload['family_processing_order']}")
        for family, stat in payload["per_family"].items():
            print(f"  {family}: {stat['selected']}/{stat['quota']} of "
                  f"{stat['available_rows']} rows "
                  f"({stat.get('distinct_available')} distinct) available, "
                  f"{stat.get('n_intents_covered')} intents")


if __name__ == "__main__":
    main()
