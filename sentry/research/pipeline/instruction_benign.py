"""Deterministic benign instruction stress test; never modifies the serving rule.

Run build/score/analyze with ``python -m ...instruction_benign``. Artifacts must
live outside the checkout. Scores use the entry's shortened variants exclusively.

``score --answers`` attaches the victim's own cached answer to every entry it scores,
joined by the sha256 of the entry text — the key
``experiments/paper/analysis/gen_answers.py`` writes. That adds the answer-side
columns (``adl_best``, ``echo_best``, the two truncation ablations and ``adl_delta_best``)
next to Deletion Gain; it does not move Deletion Gain, which is computed by code that has
no answer to read. An entry with no answer is scored anyway, marked ``has_answer=false``
and counted in ``embedding_provenance.json``. No threshold is fitted here — the rules are
applied in ``instruction_benign_analysis``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import socket
import sqlite3
import time

import numpy as np

from sentry.cache.defense.calibrate import parse_policy
from sentry.cache.defense.deletion import unit, store_rows
from sentry.cache.defense.spans import shortened
from sentry.cache.defense.textnorm import content_tokens, normalise_answer
from .config import ExperimentConfig


@dataclass(frozen=True)
class InstructionExperimentConfig(ExperimentConfig):
    """ExperimentConfig extension keeps this study's statistical settings explicit."""
    fpr_budget: float = 0.05
    calibration_fraction: float = 0.5
    cosine_bin_width: float = 0.01
    word_bin_width: float = 4.0
    min_bin_intents: int = 20
    policies: list[str] = field(default_factory=lambda: [
        "multi[count:4+width:2:cap16]/runs",
        "multi[count:6+width:2:cap16]/runs"])
    pooling: str = "cls"
    text_prefix: str = ""
    storage_dtype: str = "float16"

    def validate(self):
        super().validate()
        if not 0 < self.fpr_budget < 1 or not 0 < self.calibration_fraction < 1:
            raise ValueError("budget and calibration_fraction must be in (0, 1)")
        if self.cosine_bin_width <= 0 or self.word_bin_width <= 0 or self.min_bin_intents < 2:
            raise ValueError("invalid matching settings")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive")
        for policy in self.policies:
            parse_policy(policy)


# Fixed BEFORE scoring. Each pair expresses the same instruction in different words.
# No answer facts, semantic substitutions, generation, or score-based selection.
TEMPLATES = (
    ("polite_tell", "polite", "Please tell me.", "Could you tell me?"),
    ("polite_answer", "polite", "Please answer this question.",
     "Kindly answer this question."),
    ("polite_help", "polite", "I would appreciate your help with this question.",
     "Could you help me by answering this question?"),
    ("one_sentence", "constraint", "Answer in one sentence.",
     "Use a single sentence for your answer."),
    ("brief", "constraint", "Please answer briefly.", "Give a concise answer."),
    ("two_sentences", "constraint", "Answer in exactly two sentences.",
     "Use exactly two sentences in your response."),
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def intent_split(intent_id: str, seed: int, fraction: float = .5) -> str:
    value = int(digest(f"instruction-benign:{seed}:{intent_id}")[:16], 16) / 2**64
    return "calibration" if value < fraction else "test"


def wrap(core: str, instruction: str, position: str) -> str:
    # Keep core bytes intact, including punctuation and capitalization.
    return f"{instruction}\n{core}" if position == "prefix" else f"{core}\n{instruction}"


def construct(sources: list[dict], seed: int, fraction: float = .5) -> list[dict]:
    rows = []
    for src in sorted(sources, key=lambda r: (r["corpus"], r["intent_id"], r["record_id"])):
        common = {**src, "base_text": src["text"], "base_anchor": src["anchor"],
                  "split": intent_split(src["intent_id"], seed, fraction), "malicious": False}

        def add(condition, template, kind, position, entry, anchor):
            ambiguous = kind == "constraint" and condition in (
                "entry_only", "query_only", "exact_core")
            rows.append({**common, "sample_id": digest(
                f"{src['corpus']}:{src['record_id']}:{template}:{position}:{condition}")[:24],
                "condition": condition, "template": template, "kind": kind,
                "position": position, "text": entry, "anchor": anchor,
                "reuse_label": ("answer_dependent" if ambiguous else
                                "compatible_by_construction" if kind == "constraint" else
                                "inherited_from_source_pair"),
                "pair_basis": "identical_core" if condition == "exact_core" else "source_paraphrase"})

        add("bare", "bare", "bare", "none", src["text"], src["anchor"])
        for name, kind, instruction, paraphrase in TEMPLATES:
            for pos in ("prefix", "suffix"):
                entry = wrap(src["text"], instruction, pos)
                query = wrap(src["anchor"], instruction, pos)
                alternate = wrap(src["anchor"], paraphrase, pos)
                add("entry_only", name, kind, pos, entry, src["anchor"])
                add("query_only", name, kind, pos, src["text"], query)
                add("both_same", name, kind, pos, entry, query)
                add("both_paraphrase", name, kind, pos, entry, alternate)
                add("exact_core", name, kind, pos, entry, src["text"])
    return rows


def read_jsonl(path):
    # Explicit UTF-8: write_jsonl writes it, and a manifest that hashes a file's UTF-8
    # bytes must describe the same text this parsed, whatever the machine's locale is.
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_sources(records_path, corpus, benign_generator, attack_role, attack_class, seed):
    """Use the existing evaluator's first-legal-anchor rule; preserve IDs for auditing."""
    records = read_jsonl(records_path)
    anchors = {}
    for role in ("legal", "benign_query"):
        for rec in records:
            if rec.get("query_role") == role:
                anchors.setdefault(rec["intent_id"], rec)
    genuine, attacks, dropped = [], [], Counter()
    for rec in records:
        is_benign = rec.get("query_role") == "canonical" and rec.get("generator") == benign_generator
        is_attack = rec.get("query_role") == attack_role
        if not (is_benign or is_attack):
            continue
        anchor = anchors.get(rec["intent_id"])
        if anchor is None or anchor["text"] == rec["text"]:
            dropped["benign" if is_benign else "attack"] += 1
            continue
        row = {"record_id": rec["record_id"], "intent_id": rec["intent_id"],
               "corpus": corpus, "text": rec["text"], "anchor": anchor["text"],
               "anchor_id": anchor["record_id"], "anchor_generator": anchor.get("generator"),
               "source_split": rec.get("split"), "source_file": str(records_path),
               "generator": rec.get("generator")}
        if is_benign:
            genuine.append(row)
        else:
            attacks.append({**row, "sample_id": digest(f"{attack_class}:{rec['record_id']}")[:24],
                            "split": intent_split(rec["intent_id"], seed), "malicious": True,
                            "kind": "attack", "template": attack_class, "condition": "attack",
                            "position": "native", "reuse_label": "attack_intended",
                            "attack_class": attack_class, "base_text": None, "base_anchor": None})
    return genuine, attacks, dict(dropped)


def _ordered(texts):
    # Length sorting reduces transformer padding; stable order permits exact replay.
    return sorted(texts, key=lambda t: (len(t.split()), t))


def text_pool(rows, policies, extra=()):
    texts = set(extra)
    entries = {row["text"] for row in rows}
    for row in rows:
        texts.update((row["text"], row["anchor"]))
        if row.get("base_text"):
            texts.add(row["base_text"])
    for entry in entries:
        for policy in policies:
            texts.update(shortened(policy, entry).span_texts)
    return _ordered(texts)


#: End of the first sentence: a full stop, question mark or exclamation mark that a space
#: or the end of the text follows. Decimals and abbreviations mid-number survive, which is
#: the point — an answer that opens "It cost 3.5 million" must not be truncated to "It cost 3."
_SENTENCE_END = re.compile(r"[.?!](?=\s|$)")

#: Everything the answer side adds to a scored row. Kept in one place so the blank record
#: an unanswered entry gets cannot drift from the filled one.
ANSWER_FIELDS = ("answer_sha", "answer_words", "adl_best", "echo_best", "echo_tokens_best",
                 "adl_best_first20", "adl_best_first_sentence", "adl_delta_best")


def answer_variants(answer):
    """The three answer texts a scored row reads, or ``{}`` when there is no answer.

    ``full`` is the answer as :func:`~sentry.cache.defense.textnorm.normalise_answer`
    leaves it — the same text :func:`~sentry.cache.defense.deletion.build_profile`
    embeds, so ``adl_best`` here is the ``answer_loss`` the deployed profile stores. The
    two truncations are the ablation: if the loss survives cutting the answer to its
    opening, it is not being carried by a long tail the victim appended.

    An answer that normalises to nothing is no answer, exactly as ``build_profile``
    treats it.
    """
    cleaned = normalise_answer(answer) if answer is not None else ""
    if not cleaned:
        return {}
    end = _SENTENCE_END.search(cleaned)
    return {"full": cleaned,
            "first20": " ".join(cleaned.split()[:20]),
            "first_sentence": cleaned[:end.end()] if end else cleaned}


def _deletion_gain(row, policy, variants, vectors, storage_dtype):
    """Deletion Gain for one (row, policy). **Takes no answer and never sees one.**

    Separated from :func:`score_row` so that the guarantee is structural rather than a
    promise: attaching an answer cannot move ``dg``, ``base_cos``, ``best_name`` or
    ``best_text``, because the code that computes them has no way to read one.
    """
    anchor = unit(vectors[row["anchor"]])
    whole = store_rows(unit(vectors[row["text"]]), storage_dtype)
    base = float(whole @ anchor)
    out = {"policy": policy.fingerprint(), "base_cos": base,
           "retrieval_cos": float(unit(vectors[row["text"]]) @ anchor),
           "words": len(row["text"].split()), "n_variants": len(variants.span_texts),
           "judgeable": bool(variants.segment_count >= policy.min_segments and
                              variants.span_texts and variants.deletion_texts),
           "dg": None, "best_name": None, "best_text": None, "best_index": None}
    if not out["judgeable"]:
        return out
    matrix = store_rows(np.vstack([unit(vectors[t]) for t in variants.span_texts]), storage_dtype)
    scores = matrix @ anchor
    best = int(np.argmax(scores))
    out.update(dg=float(scores[best]) - base, best_name=variants.span_names[best],
               best_text=variants.span_texts[best], best_index=best)
    if row.get("base_text"):
        core = row["base_text"]
        # Diagnostic oracle only: it is never substituted for the deployed score.
        out["core_only_gain"] = float(store_rows(unit(vectors[core]), storage_dtype) @ anchor) - base
        out["best_is_core"] = " ".join(core.split()) == " ".join(out["best_text"].split())
    return out


def answer_fields(row, variants, vectors, storage_dtype, answer, best_index):
    """What the entry's own cached answer says about the content DG chose to drop.

    Reads two things and only two: which variant won, and therefore what that variant
    dropped (``Δ*``). Everything else comes from the answer.

    - ``adl_best`` — ``cos(entry, y) − cos(s*, y)``: how much the entry's match to its own
      answer falls when the winning variant's complement goes. Arithmetic mirrors
      ``build_profile``: float64 dot products on unrounded unit vectors, rounded once at
      the end, so this equals the ``answer_loss`` a deployed profile stores.
    - ``echo_best`` / ``echo_tokens_best`` — content words ``Δ*`` and the answer share,
      **net of the arriving query's own vocabulary**, which is the row's ``anchor``: a word
      the query itself asked about is not something the query never needed. Same
      subtraction :func:`~sentry.cache.defense.deletion.answer_check` performs, so
      ``len(echo_tokens_best) == echo_best`` always.
    - ``adl_best_first20`` / ``adl_best_first_sentence`` — the same loss against a truncated
      answer.
    - ``adl_delta_best`` — ``cos(Δ*, y) − cos(s*, y)``: is the *removed* content closer to
      the answer than what survived? A different question from ``adl_best`` and reported
      beside it, never in place of it.

    ``has_answer`` means what :attr:`DeletionProfile.has_answer` means: *the answer fields
    on this row are readable*. An unjudgeable entry has no winning variant, so there is
    nothing to read even when the victim did answer it, and the row says ``false`` — which
    is the answer the fail-closed rule needs, since an unjudgeable entry is vetoed on DG
    alone. ``readings_answered_but_unjudgeable`` in ``embedding_provenance.json`` counts
    how often that happened, so the two populations can be told apart.
    """
    texts = answer_variants(answer)
    blank = {"has_answer": False, **{name: None for name in ANSWER_FIELDS}}
    if not texts or best_index is None:
        return blank
    whole = unit(vectors[row["text"]])
    kept = unit(vectors[variants.span_texts[best_index]])
    gone = variants.removed_texts[best_index]
    dropped = unit(vectors[gone])
    full = unit(vectors[texts["full"]])

    def loss(key):
        answer_vector = unit(vectors[texts[key]])
        return float(store_rows(float(whole @ answer_vector) - float(kept @ answer_vector),
                                storage_dtype))

    echo = sorted((content_tokens(gone) & content_tokens(texts["full"]))
                  - content_tokens(row["anchor"]))
    return {"has_answer": True,
            "answer_sha": hashlib.sha256(texts["full"].encode("utf-8")).hexdigest(),
            "answer_words": len(texts["full"].split()),
            "adl_best": loss("full"), "echo_best": len(echo), "echo_tokens_best": echo,
            "adl_best_first20": loss("first20"),
            "adl_best_first_sentence": loss("first_sentence"),
            "adl_delta_best": float(store_rows(float(dropped @ full) - float(kept @ full),
                                               storage_dtype))}


def score_row(row, policy, vectors, storage_dtype, answer=None):
    variants = shortened(policy, row["text"])
    out = _deletion_gain(row, policy, variants, vectors, storage_dtype)
    out.update(answer_fields(row, variants, vectors, storage_dtype, answer,
                             out["best_index"]))
    return out


def fit_thresholds(rows, budget, retrieval_floor):
    cal = [r for r in rows if r["split"] == "calibration" and r["condition"] == "bare"
           and r["judgeable"] and r.get("retrieval_cos", r["base_cos"]) >= retrieval_floor]
    if len(cal) < 2:
        raise ValueError("fewer than two judgeable bare calibration hits")
    return {"dg": float(np.quantile([r["dg"] for r in cal], 1-budget)),
            "neg_cos": float(np.quantile([-r["base_cos"] for r in cal], 1-budget)), "n": len(cal)}


def cluster_rate(values, groups, iterations, seed):
    """Fixed-threshold, intent-cluster percentile interval; all templates stay together."""
    if len(values) == 0:
        return {"rate": None, "ci95": None, "n": 0, "n_intents": 0}
    grouped = defaultdict(list)
    for group, value in zip(groups, values):
        grouped[group].append(float(value))
    totals = np.array([sum(grouped[k]) for k in sorted(grouped)])
    counts = np.array([len(grouped[k]) for k in sorted(grouped)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(totals), size=(iterations, len(totals)))
    boot = totals[idx].sum(1) / counts[idx].sum(1)
    return {"rate": float(np.mean(values)), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "n": len(values), "n_intents": len(grouped)}


def operating_rates(rows, thresholds, retrieval_floor, iterations, seed):
    hits = [r for r in rows if r.get("retrieval_cos", r["base_cos"]) >= retrieval_floor]
    def rate(values, rs):
        return cluster_rate(values, [r["intent_id"] for r in rs], iterations, seed)
    dg = [not r["judgeable"] or r["dg"] > thresholds["dg"] for r in hits]
    cos = [-r["base_cos"] > thresholds["neg_cos"] for r in hits]
    return {"n": len(rows), "n_intents": len({r["intent_id"] for r in rows}),
            "n_hits": len(hits), "n_hit_intents": len({r["intent_id"] for r in hits}),
            "retrieval_rate": len(hits)/len(rows) if rows else None,
            "n_unjudgeable_hits": sum(not r["judgeable"] for r in hits),
            "dg_veto_given_hit": rate(dg, hits), "cosine_veto_given_hit": rate(cos, hits),
            "dg_veto_all_pairs": rate([not r["judgeable"] or r["dg"] > thresholds["dg"]
                                        for r in rows], rows),
            "cosine_veto_all_pairs": rate([-r["base_cos"] > thresholds["neg_cos"] for r in rows], rows),
            "dg_served_fraction": (len(hits)-sum(dg))/len(rows) if rows else None,
            "cosine_served_fraction": (len(hits)-sum(cos))/len(rows) if rows else None}


def _jsonl_files(path):
    """``path`` itself if it is a file, else every ``*.jsonl`` under it, in a stable order."""
    path = Path(path)
    return [path] if path.is_file() else sorted(path.rglob("*.jsonl"))


def load_answer_files(paths):
    """Victim answers keyed by the sha256 of the prompt they answered.

    Same rule as ``experiments/paper/analysis/gen_answers.py:load_answers``: those
    files are append-only, so the **last non-empty** response for a hash wins and a hash
    whose rows all came back empty has no answer at all. Later ``--answers`` paths
    override earlier ones for the same hash.

    ``gen_answers`` owns that rule, and it is *not* imported here: it sits in
    ``experiments/`` rather than in a package, and importing it drags in
    ``sentry.research.operators`` and therefore ``requests`` — a network client this
    scoring run has no use for and a new way for a multi-hour job to die at import. The
    agreement is pinned by a test instead
    (``tests/test_instruction_benign.py::test_answer_loading_agrees_with_gen_answers``),
    which imports both functions and compares them on a file holding a repeated hash and
    an all-empty one. Change the rule there and that test fails here.

    The stored ``prompt_sha`` is re-derived from the prompt and a disagreement raises: a
    wrong hash joins silently to the wrong entry, and every answer-side number after that
    would be about some other question.
    """
    answers, files = {}, []
    for entry in paths or ():
        root = Path(entry)
        if not root.exists():
            raise ValueError(f"--answers path does not exist: {root}")
        for path in _jsonl_files(root):
            rows = usable = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:  # tolerate a partial line left by a killed generator
                    record = json.loads(line)
                    sha = record["prompt_sha"]
                except (json.JSONDecodeError, KeyError):
                    continue
                rows += 1
                if "prompt" in record and digest(record["prompt"]) != sha:
                    raise ValueError(f"{path}: prompt_sha does not match its prompt ({sha})")
                if (record.get("response") or "").strip():
                    answers[sha] = record["response"]
                    usable += 1
            files.append({"path": str(path), "n_rows": rows, "n_non_empty": usable,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return answers, files


def load_extra_sets(directories):
    """Extra sample rows another builder wrote, read as they are.

    ``answer_check_sets.py`` emits one JSONL per set with the shape of ``samples.jsonl``
    except that the entry text is called ``prompt``. Only what scoring needs is checked
    here — an entry text and an anchor — so a set carrying its own extra columns passes
    through untouched and a missing analysis column is the analysis stage's complaint,
    not a crash five hours into an embedding run. A directory that is absent or holds no
    JSONL contributes nothing: the built samples are the experiment, these are additions.
    """
    rows, files = [], []
    for entry in directories or ():
        root = Path(entry)
        if not root.exists():
            print(f"extra-sets: {root} does not exist; skipping", flush=True)
            continue
        for path in _jsonl_files(root):
            loaded = read_jsonl(path)
            for index, row in enumerate(loaded):
                text = row.get("text") or row.get("prompt")
                if not text or not row.get("anchor"):
                    raise ValueError(f"{path} row {index}: needs a text/prompt and an anchor")
                rows.append({**row, "text": text, "set": row.get("set") or path.stem})
            files.append({"path": str(path), "n_rows": len(loaded),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows, files


def answer_token_report(texts, cfg, smoke):
    """How many answers the embedder will silently cut, and where its limit is.

    ``TransformerCLSEmbedder`` tokenises with ``truncation=True`` and the tokenizer's own
    ``model_max_length``, so an over-long answer is embedded as its opening while
    ``answer_words`` still reports the whole thing. Counting it is what lets the write-up
    say what "full answer" means rather than assume it. Only the tokenizer is loaded, not
    the model weights; the hash embedder has no limit at all.

    Never fatal. A tokenizer that will not load or reports a sentinel instead of a limit
    leaves a note in the provenance — losing a diagnostic is not worth killing a run that
    is otherwise about to spend hours embedding.
    """
    report = {"cap": None, "n_answer_texts": len(texts), "n_at_cap": None,
              "max_tokens": None, "note": None}
    if smoke or not texts:
        return {**report, "n_at_cap": 0, "note": "hash embedder has no length cap"}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model)
        cap = int(getattr(tokenizer, "model_max_length", 0) or 0)
        if not 0 < cap < 1_000_000:
            return {**report, "note": f"tokenizer reports no usable cap ({cap})"}
        lengths = [len(tokenizer(cfg.text_prefix + text)["input_ids"]) for text in texts]
    except Exception as exc:  # noqa: BLE001 - diagnostic only, see docstring
        return {**report, "note": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {**report, "cap": cap, "n_at_cap": int(sum(n >= cap for n in lengths)),
            "max_tokens": int(max(lengths))}


def complement_pool(rows, policies, vectors, storage_dtype, answered):
    """Texts the winning variants dropped — the ``adl_delta_best`` ablation's ``Δ*``.

    Which variant wins depends on the anchor, so this cannot be known before Deletion
    Gain has run: the pass below *is* Deletion Gain, read for nothing but ``best_index``,
    and it runs only over entries that actually have an answer. Most complements cost
    nothing — the complement of a prefix run is a suffix run, which is already a variant —
    so what comes back is essentially the interior winners.
    """
    needed = set()
    for row in rows:
        if row["text"] not in answered:
            continue
        for policy in policies:
            variants = shortened(policy, row["text"])
            best = _deletion_gain(row, policy, variants, vectors, storage_dtype)["best_index"]
            if best is not None:
                needed.add(variants.removed_texts[best])
    return [text for text in _ordered(needed) if text not in vectors]


def encode_pool(texts, cfg, workspace, threads, batch_size, smoke, vectors=None, extra=None):
    """Content-addressed SQLite memoization allows safe resumption after interruption.

    ``vectors`` lets a second pass reuse what a first one already holds, so the answer
    texts and the winning variants' complements are encoded without a second cache and
    without re-reading the whole table. ``extra`` is merged into the provenance file and
    collects one entry per pass under ``"passes"``.
    """
    identity = {"model": cfg.embedding_model, "pooling": cfg.pooling,
                "prefix": cfg.text_prefix, "smoke": smoke,
                "embed_source_sha256": digest(Path(__file__).with_name("embed.py").read_text())}
    cache = workspace / f"embeddings-{digest(json.dumps(identity, sort_keys=True))[:16]}.sqlite3"
    db = sqlite3.connect(cache)
    db.execute("CREATE TABLE IF NOT EXISTS vectors (text TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    vectors = {} if vectors is None else vectors
    want = {t for t in texts if t not in vectors}
    for text, blob in db.execute("SELECT text,vector FROM vectors"):
        if text in want:
            vectors[text] = np.frombuffer(blob, dtype=np.float32)
    missing = [t for t in texts if t not in vectors]
    print(f"Embedding pool: {len(texts)} texts, {len(missing)} missing", flush=True)
    if smoke:
        from sentry.embeddings import HashEmbedder
        embedder = HashEmbedder(64)
    else:
        import torch
        torch.set_num_threads(threads)
        from sentry.embeddings import TransformerCLSEmbedder
        embedder = TransformerCLSEmbedder(cfg.embedding_model, batch_size=batch_size,
                                          pooling=cfg.pooling, text_prefix=cfg.text_prefix)
    t0 = time.perf_counter()
    for start in range(0, len(missing), 2048):
        chunk = missing[start:start+2048]
        matrix = embedder.encode(chunk)
        if not np.isfinite(matrix).all():
            raise ValueError("nonfinite model embedding")
        db.executemany("INSERT INTO vectors VALUES (?, ?)", [(t, v.astype(np.float32).tobytes())
                                                            for t, v in zip(chunk, matrix)])
        db.commit()
        vectors.update(zip(chunk, matrix))
        print(f"encoded {min(start+2048,len(missing))}/{len(missing)} in {time.perf_counter()-t0:.1f}s", flush=True)
    db.close()
    report = {**identity, "hostname": socket.gethostname(), "cache": str(cache),
              "unique_texts": len(texts), "new_texts": len(missing), "threads": threads,
              "seconds": time.perf_counter()-t0}
    extra = {} if extra is None else extra
    extra.setdefault("passes", []).append(
        {"unique_texts": len(texts), "new_texts": len(missing),
         "seconds": report["seconds"]})
    # Passes are handed disjoint text lists, so the top-level totals stay the totals.
    for key in ("unique_texts", "new_texts", "seconds"):
        report[key] = sum(entry[key] for entry in extra["passes"])
    (workspace / "embedding_provenance.json").write_text(
        json.dumps({**report, **extra}, indent=2))
    return vectors


def entry_prompts(rows, set_name="wrapped_benign"):
    """One row per unique entry text, in file order.

    The victim is asked the entry and nothing else, so the prompt list is the set of
    distinct ``text`` values — bare and wrapped, both corpora. Identical entries are
    answered once and joined back by the prompt hash.
    """
    prompts = {}
    for row in rows:
        text = row["text"]
        if text not in prompts:
            prompts[text] = {"prompt": text, "corpus": row.get("corpus", ""), "set": set_name}
    return list(prompts.values())


def _write_prompt_list(args, parser):
    workspace = Path(args.workspace).resolve()
    repo = Path(__file__).resolve().parents[3]
    if workspace == repo or repo in workspace.parents:
        raise ValueError("experiment workspace must be outside the git checkout")
    if not args.out:
        parser.error("prompts requires --out")
    samples = Path(args.samples) if args.samples else workspace / "full" / "samples.jsonl"
    if not samples.exists():
        parser.error(f"no samples at {samples}; pass --samples")
    rows = entry_prompts(read_jsonl(samples))
    write_jsonl(args.out, rows)
    print(json.dumps({"samples": str(samples), "n_prompts": len(rows), "out": str(args.out)}),
          flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "score", "analyze", "prompts"))
    parser.add_argument("--config")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", help="prompts: JSONL victim prompt list to write")
    parser.add_argument("--samples", help="prompts: samples file, default WORKSPACE/full/samples.jsonl")
    parser.add_argument("--eval-dir", help="directory containing lmp/scp/kca_eval.jsonl")
    parser.add_argument("--answers", action="append", metavar="PATH_OR_DIR",
                        help="score: victim answers JSONL (or a directory of them), "
                             "joined to an entry by the sha256 of its text; repeatable")
    parser.add_argument("--extra-sets", action="append", metavar="DIR",
                        help="score: directory of additional sample JSONL files to score "
                             "alongside the built samples; repeatable, absent is fine")
    parser.add_argument("--poisoned-flags", action="append", metavar="PATH_OR_DIR",
                        help="analyze: asr_judge --flags-out JSONL (or a directory of "
                             "them), joined to an attack row by record_id; adds a "
                             "poisoned-only arm beside every all-planted attack rate. "
                             "Repeatable.")
    parser.add_argument("--eta-a", type=float, default=None,
                        help="analyze: an extra answer-loss threshold to report as an "
                             "ablation column beside the calibrated one. It never "
                             "replaces the calibrated eta_a.")
    parser.add_argument("--limit", type=int, default=0, help="diagnostic pilot, first N stable intent IDs per corpus")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-local-heavy", action="store_true",
                        help="explicitly authorized server CPU exception to the existing heavy-stage gate")
    args = parser.parse_args(argv)
    if args.stage == "prompts":
        # Reads an existing samples file only; it needs no config and writes no workspace state.
        return _write_prompt_list(args, parser)
    if not args.config:
        parser.error(f"{args.stage} requires --config")
    cfg = InstructionExperimentConfig.from_json(args.config)
    workspace = Path(args.workspace).resolve()
    repo = Path(__file__).resolve().parents[3]
    if workspace == repo or repo in workspace.parents:
        raise ValueError("experiment workspace must be outside the git checkout")
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "config.json"
    effective = asdict(cfg)
    if config_path.exists() and json.loads(config_path.read_text()) != effective:
        raise ValueError("workspace config differs; choose a new workspace")
    config_path.write_text(json.dumps(effective, indent=2))
    if args.stage == "build":
        if not args.eval_dir:
            parser.error("build requires --eval-dir")
        if (workspace / "samples.jsonl").exists():
            raise ValueError("samples already exist; use another workspace")
        all_benign, all_attacks, provenance = {}, [], {}
        for name, corpus, gen, role, attack_class in (
            ("lmp", "comqa", "human_comqa", "ndss", "CAP"),
            ("scp", "comqa", "human_comqa", "scp", "SCP"),
            ("kca", "nq", "cacheattack_cleaned_qa", "gcg", "KCA")):
            path = Path(args.eval_dir) / f"{name}_eval.jsonl"
            benign, attacks, dropped = load_sources(path, corpus, gen, role, attack_class, cfg.seed)
            if args.limit:
                ids = set(sorted({r["intent_id"] for r in benign})[:args.limit])
                benign = [r for r in benign if r["intent_id"] in ids]
                attacks = [r for r in attacks if r["intent_id"] in ids]
            provenance[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "n_benign": len(benign), "n_attack": len(attacks), "dropped": dropped}
            for r in benign:
                key = (r["corpus"], r["record_id"])
                if key in all_benign:
                    old = all_benign[key]
                    if (r["text"], r["anchor"]) != (old["text"], old["anchor"]):
                        raise ValueError("CAP and SCP benign controls disagree")
                else:
                    all_benign[key] = r
            all_attacks.extend(attacks)
        rows = construct(list(all_benign.values()), cfg.seed, cfg.calibration_fraction)
        for row in all_attacks:
            row["split"] = intent_split(row["intent_id"], cfg.seed, cfg.calibration_fraction)
        rows += all_attacks
        write_jsonl(workspace / "samples.jsonl", rows)
        (workspace / "manifest.json").write_text(json.dumps(
            {"sources": provenance, "pilot_limit": args.limit, "n_rows": len(rows),
             "n_base_pairs": len(all_benign), "templates": TEMPLATES,
             "construction_sha256": digest(Path(__file__).read_text()),
             "label_note": "Harmless instructions are benign. Unilateral response constraints are answer-dependent for reuse. Source paraphrase labels are inherited, not independently revalidated."}, indent=2))
        print(json.dumps({"n_rows": len(rows), "n_base_pairs": len(all_benign), "sources": provenance}, indent=2))
    elif args.stage == "score":
        from .cli import _require_server
        _require_server("instruction-benign embedding", args.smoke, args.allow_local_heavy)
        rows = [{**row, "set": row.get("set") or "instruction_benign"}
                for row in read_jsonl(workspace / "samples.jsonl")]
        extra_rows, extra_files = load_extra_sets(args.extra_sets)
        rows += extra_rows
        policies = [parse_policy(spec) for spec in cfg.policies]

        # An entry text is answered once, so the join is by text; identical entries
        # (`query_only` reuses the bare one) share that answer, as the victim run did.
        answers, answer_files = load_answer_files(args.answers)
        entries = {row["text"] for row in rows}
        by_text = {}
        unusable = 0
        for text in entries:
            response = answers.get(digest(text))
            if response is None:
                continue
            if answer_variants(response):
                by_text[text] = response
            else:
                unusable += 1
        answered_rows = sum(row["text"] in by_text for row in rows)
        provenance = {"answers": {
            "files": answer_files, "hashes_loaded": len(answers),
            "extra_set_files": extra_files, "n_extra_rows": len(extra_rows),
            "entry_texts": len(entries), "entry_texts_with_answer": len(by_text),
            "entry_texts_without_answer": len(entries) - len(by_text),
            "unusable_answers": unusable,
            "rows_with_answer": answered_rows,
            "rows_without_answer": len(rows) - answered_rows,
            "answer_texts_embedded": 0, "complement_texts_embedded": 0}}

        answer_texts = {t for response in by_text.values()
                        for t in answer_variants(response).values()}
        provenance["answers"]["answer_texts_embedded"] = len(answer_texts)
        provenance["answers"]["tokens"] = answer_token_report(
            sorted(answer_texts), cfg, args.smoke)
        # Say what was joined *before* paying for the embedding pass, and refuse a
        # --answers path that matched nothing: that is a typo, not an empty result.
        print(json.dumps({k: v for k, v in provenance["answers"].items()
                          if not isinstance(v, list)}), flush=True)
        if args.answers and not by_text:
            raise ValueError(
                f"--answers matched no entry text ({len(answers)} hashes loaded from "
                f"{len(answer_files)} file(s)); check the paths before embedding")
        vectors = encode_pool(text_pool(rows, policies, extra=answer_texts), cfg, workspace,
                              args.threads, args.batch_size, args.smoke, extra=provenance)
        if by_text:
            # Second pass: Δ* is chosen by the anchor, so it is only knowable once DG ran.
            delta = complement_pool(rows, policies, vectors, cfg.storage_dtype, set(by_text))
            provenance["answers"]["complement_texts_embedded"] = len(delta)
            encode_pool(delta, cfg, workspace, args.threads, args.batch_size, args.smoke,
                        vectors=vectors, extra=provenance)

        target = workspace / ("scores_smoke.jsonl" if args.smoke else "scores.jsonl")
        temporary = target.with_suffix(".partial")
        unreadable = 0
        with temporary.open("w") as handle:
            for i, row in enumerate(rows):
                answer = by_text.get(row["text"])
                for policy in policies:
                    record = {**row, **score_row(row, policy, vectors, cfg.storage_dtype, answer)}
                    unreadable += answer is not None and not record["has_answer"]
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                if i % 2000 == 0:
                    print(f"scored {i}/{len(rows)} pairs", flush=True)
        temporary.replace(target)
        # An answered entry with too few segments has no winning variant, so its answer
        # fields are unreadable and the row says has_answer=false. Count it rather than
        # let it hide inside entry_texts_without_answer, which it is not.
        provenance["answers"]["readings_answered_but_unjudgeable"] = int(unreadable)
        record_path = workspace / "embedding_provenance.json"
        record_path.write_text(json.dumps(
            {**json.loads(record_path.read_text()), **provenance}, indent=2))
        print(f"Wrote {target}", flush=True)
        print(json.dumps({k: v for k, v in provenance["answers"].items()
                          if not isinstance(v, list)}), flush=True)
    else:
        from .instruction_benign_analysis import analyze
        analyze(workspace, cfg, args.smoke, poisoned_flags=args.poisoned_flags,
                eta_a_override=args.eta_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
