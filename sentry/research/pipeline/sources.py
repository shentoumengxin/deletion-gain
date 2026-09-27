from __future__ import annotations

import ast
import csv
import io
import json
import random
import re
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig
from .io import atomic_write_json, atomic_write_jsonl, file_sha256, stable_id
from .schema import QueryRecord

COMQA_URLS = {
    "train": "https://qa.mpi-inf.mpg.de/comqa/comqa_train.json",
    "dev": "https://qa.mpi-inf.mpg.de/comqa/comqa_dev.json",
}


@dataclass(frozen=True)
class Intent:
    intent_id: str
    source: str
    questions: list[str]
    answers: list[str]
    metadata: dict[str, Any]


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _flatten_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean_text(value)] if _clean_text(value) else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        for key in ("text", "answer", "label", "name", "value", "normalized"):
            if key in value:
                return _flatten_answers(value[key])
        answers: list[str] = []
        for nested in value.values():
            answers.extend(_flatten_answers(nested))
        return answers
    answers = []
    for item in value:
        answers.extend(_flatten_answers(item))
    return list(dict.fromkeys(answer for answer in answers if answer))


def _questions_from_row(row: dict[str, Any]) -> list[str]:
    for key in ("questions", "paraphrases", "utterances"):
        if key in row:
            values = row[key]
            if isinstance(values, str):
                values = [values]
            normalized = []
            for item in values:
                if isinstance(item, dict):
                    item = item.get("question", item.get("query", item.get("text", "")))
                text = _clean_text(item)
                if text:
                    normalized.append(text)
            return list(dict.fromkeys(normalized))
    for key in ("question", "query", "text"):
        if key in row:
            text = _clean_text(row[key])
            return [text] if text else []
    return []


def _cluster_key(row: dict[str, Any], fallback: str) -> str:
    for key in ("cluster_id", "cluster", "paraphrase_cluster_id", "id"):
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, dict):
                value = value.get("id", value.get("cluster_id", fallback))
            return str(value)
    return fallback


def _answer_from_row(row: dict[str, Any]) -> list[str]:
    for key in (
        "answers",
        "answer",
        "answer_text",
        "normalized_answer",
        "gold_answers",
    ):
        if key in row:
            return _flatten_answers(row[key])
    return []


def parse_comqa_file(path: str | Path, source_split: str) -> list[Intent]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("data", "clusters", "examples", "questions"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if payload and all(
                isinstance(value, (dict, list)) for value in payload.values()
            ):
                mapped_rows = []
                for cluster_id, value in payload.items():
                    if isinstance(value, dict):
                        mapped_rows.append({"cluster_id": cluster_id, **value})
                    else:
                        mapped_rows.append(
                            {"cluster_id": cluster_id, "questions": value}
                        )
                payload = mapped_rows
            else:
                payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"unsupported ComQA structure in {path}")

    grouped_questions: dict[str, list[str]] = defaultdict(list)
    grouped_answers: dict[str, list[str]] = defaultdict(list)
    grouped_metadata: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(payload):
        if not isinstance(raw_row, dict):
            continue
        cluster = _cluster_key(raw_row, fallback=f"{source_split}-{index}")
        grouped_questions[cluster].extend(_questions_from_row(raw_row))
        grouped_answers[cluster].extend(_answer_from_row(raw_row))
        grouped_metadata.setdefault(cluster, {"source_split": source_split})

    intents = []
    for cluster in sorted(grouped_questions):
        questions = list(dict.fromkeys(q for q in grouped_questions[cluster] if q))
        answers = list(dict.fromkeys(a for a in grouped_answers[cluster] if a))
        if not questions or not answers:
            continue
        intents.append(
            Intent(
                intent_id=f"comqa-{stable_id(source_split, cluster, length=16)}",
                source="comqa",
                questions=questions,
                answers=answers,
                metadata=grouped_metadata[cluster],
            )
        )
    return intents


def fetch_comqa(raw_dir: str | Path, allow_network: bool) -> dict[str, Path]:
    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in COMQA_URLS.items():
        target = raw_root / f"comqa_{split}.json"
        paths[split] = target
        if target.exists():
            continue
        if not allow_network:
            raise FileNotFoundError(
                f"{target} is missing. Run the build stage on the CPU server with "
                "--allow-network, or place the official ComQA files in the raw directory."
            )
        import requests

        response = requests.get(url, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
    atomic_write_json(
        raw_root / "comqa_manifest.json",
        {
            split: {"url": COMQA_URLS[split], "sha256": file_sha256(path)}
            for split, path in paths.items()
        },
    )
    return paths


def deterministic_split(intent_id: str, config: ExperimentConfig) -> str:
    bucket = int(stable_id(config.seed, intent_id, length=12), 16) / float(16**12)
    if bucket < config.split_train:
        return "train"
    if bucket < config.split_train + config.split_val:
        return "val"
    return "test"


def select_intents(intents: Iterable[Intent], config: ExperimentConfig) -> list[Intent]:
    unique = {intent.intent_id: intent for intent in intents}
    ranked = sorted(
        unique.values(),
        key=lambda intent: stable_id(config.seed, intent.intent_id, "selection"),
    )
    if len(ranked) < config.n_intents:
        raise ValueError(
            f"only {len(ranked)} answerable intents are available; "
            f"{config.n_intents} requested"
        )
    return ranked[: config.n_intents]


def build_comqa_records(
    raw_dir: str | Path,
    dataset_dir: str | Path,
    config: ExperimentConfig,
    allow_network: bool = False,
) -> tuple[list[Intent], list[QueryRecord]]:
    paths = fetch_comqa(raw_dir, allow_network=allow_network)
    all_intents: list[Intent] = []
    for source_split, path in paths.items():
        all_intents.extend(parse_comqa_file(path, source_split))
    intents = select_intents(all_intents, config)

    records: list[QueryRecord] = []
    intent_rows = []
    for intent in intents:
        split = deterministic_split(intent.intent_id, config)
        canonical = intent.questions[0]
        canonical_id = stable_id(intent.intent_id, "canonical")
        records.append(
            QueryRecord(
                record_id=canonical_id,
                intent_id=intent.intent_id,
                source=intent.source,
                split=split,
                query_role="canonical",
                text=canonical,
                generator="human_comqa",
                seed=config.seed,
                semantic_label="equivalent",
                canonical_answer=intent.answers,
                metadata=intent.metadata,
            )
        )
        for index, question in enumerate(intent.questions[1:]):
            records.append(
                QueryRecord(
                    record_id=stable_id(intent.intent_id, "human", index, question),
                    intent_id=intent.intent_id,
                    source=intent.source,
                    split=split,
                    query_role="legal_candidate",
                    text=question,
                    generator="human_comqa",
                    seed=config.seed,
                    semantic_label="equivalent",
                    parent_id=canonical_id,
                    canonical_answer=intent.answers,
                    metadata={"gold_cluster": True, **intent.metadata},
                )
            )
        intent_rows.append(
            {
                "intent_id": intent.intent_id,
                "source": intent.source,
                "split": split,
                "canonical_record_id": canonical_id,
                "canonical_query": canonical,
                "canonical_answer": intent.answers,
                "human_question_count": len(intent.questions),
                "source_metadata": intent.metadata,
            }
        )

    dataset_root = Path(dataset_dir)
    atomic_write_jsonl(dataset_root / "intents.jsonl", intent_rows)
    atomic_write_jsonl(
        dataset_root / "records.jsonl",
        (record.to_dict() for record in records),
    )
    atomic_write_json(
        dataset_root / "build_manifest.json",
        {
            "config": config.to_dict(),
            "intent_count": len(intents),
            "record_count": len(records),
            "source_hashes": {
                split: file_sha256(path) for split, path in paths.items()
            },
        },
    )
    return intents, records


def build_smoke_records(
    dataset_dir: str | Path, config: ExperimentConfig
) -> list[QueryRecord]:
    examples = [
        (
            "smoke-capital",
            "What is the capital of France?",
            ["Paris"],
            ["Which city is France's capital?", "Name the capital city of France."],
        ),
        (
            "smoke-author",
            "Who wrote Pride and Prejudice?",
            ["Jane Austen"],
            [
                "Who is the author of Pride and Prejudice?",
                "Who penned Pride and Prejudice?",
            ],
        ),
        (
            "smoke-year",
            "In what year did Apollo 11 land on the Moon?",
            ["1969"],
            [
                "When did Apollo 11 reach the Moon?",
                "What year was the Apollo 11 Moon landing?",
            ],
        ),
    ][: config.n_intents]
    records: list[QueryRecord] = []
    intent_rows = []
    for intent_id, canonical, answers, paraphrases in examples:
        split = deterministic_split(intent_id, config)
        canonical_id = stable_id(intent_id, "canonical")
        records.append(
            QueryRecord(
                record_id=canonical_id,
                intent_id=intent_id,
                source="synthetic_smoke",
                split=split,
                query_role="canonical",
                text=canonical,
                generator="synthetic",
                seed=config.seed,
                semantic_label="equivalent",
                canonical_answer=answers,
            )
        )
        for index, paraphrase in enumerate(paraphrases):
            records.append(
                QueryRecord(
                    record_id=stable_id(intent_id, "human", index),
                    intent_id=intent_id,
                    source="synthetic_smoke",
                    split=split,
                    query_role="legal_candidate",
                    text=paraphrase,
                    generator="synthetic_human",
                    seed=config.seed,
                    semantic_label="equivalent",
                    parent_id=canonical_id,
                    canonical_answer=answers,
                    metadata={"gold_cluster": True},
                )
            )
        intent_rows.append(
            {
                "intent_id": intent_id,
                "source": "synthetic_smoke",
                "split": split,
                "canonical_record_id": canonical_id,
                "canonical_query": canonical,
                "canonical_answer": answers,
                "human_question_count": 1 + len(paraphrases),
            }
        )
    dataset_root = Path(dataset_dir)
    atomic_write_jsonl(dataset_root / "intents.jsonl", intent_rows)
    atomic_write_jsonl(
        dataset_root / "records.jsonl", (record.to_dict() for record in records)
    )
    atomic_write_json(
        dataset_root / "build_manifest.json",
        {"config": config.to_dict(), "intent_count": len(examples), "smoke": True},
    )
    return records


def import_external_jsonl(
    source_path: str | Path,
    output_path: str | Path,
    source_name: str,
    limit: int | None = None,
) -> int:
    rows = []
    with Path(source_path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and len(rows) >= limit:
                break
            payload = json.loads(line)
            text = _clean_text(
                payload.get("prompt", payload.get("text", payload.get("sentence1")))
            )
            if not text:
                continue
            rows.append(
                QueryRecord(
                    record_id=stable_id(source_name, index, text),
                    intent_id=str(payload.get("cluster_id", f"{source_name}-{index}")),
                    source=source_name,
                    split="external",
                    query_role="external",
                    text=text,
                    generator="external_dataset",
                    seed=0,
                    semantic_label="pending",
                    metadata=payload,
                ).to_dict()
            )
    atomic_write_jsonl(output_path, rows)
    return len(rows)


def fetch_huggingface_external(
    dataset_name: str,
    output_path: str | Path,
    limit: int,
    seed: int,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "install the project research and models extras"
        ) from exc

    rows: list[dict] = []
    if dataset_name == "vcache":
        dataset = load_dataset(
            "vCache/SemBenchmarkSearchQueries",
            split="train",
        )
        cluster_ids = sorted(
            {str(row["cluster_id"]) for row in dataset},
            key=lambda value: stable_id(seed, "vcache", value),
        )[:limit]
        selected = set(cluster_ids)
        for index, row in enumerate(dataset):
            cluster_id = str(row["cluster_id"])
            if cluster_id not in selected:
                continue
            text = _clean_text(row["text"])
            rows.append(
                QueryRecord(
                    record_id=stable_id("vcache", cluster_id, index, text),
                    intent_id=f"vcache-{cluster_id}",
                    source="vcache_search_queries",
                    split="external",
                    query_role="external",
                    text=text,
                    generator="external_dataset",
                    seed=seed,
                    semantic_label="pending",
                    metadata={"cluster_id": cluster_id, "original_id": row.get("id")},
                ).to_dict()
            )
    elif dataset_name == "paws":
        dataset = load_dataset(
            "google-research-datasets/paws",
            "labeled_final",
            split="train",
        )
        negatives = [row for row in dataset if int(row["label"]) == 0]
        negatives.sort(
            key=lambda row: stable_id(seed, "paws", row["sentence1"], row["sentence2"])
        )
        for index, row in enumerate(negatives[:limit]):
            text = _clean_text(row["sentence1"])
            rows.append(
                QueryRecord(
                    record_id=stable_id("paws", index, text, row["sentence2"]),
                    intent_id=f"paws-{index:05d}",
                    source="paws_labeled_final",
                    split="external",
                    query_role="hard_negative",
                    text=text,
                    generator="external_dataset",
                    seed=seed,
                    semantic_label="non_equivalent",
                    metadata={
                        "paired_text": _clean_text(row["sentence2"]),
                        "label": 0,
                    },
                ).to_dict()
            )
    else:
        raise ValueError(f"unsupported external dataset: {dataset_name}")
    atomic_write_jsonl(output_path, rows)
    return len(rows)


# --- Independent benign population (benign-population plan, T1.1) -------------------
#
# The previous benign set was paraphrases from an operator closely related to
# the one generating orbits, so a low rho on it was partly a property of the
# setup. This population instead pairs questions that *two different people*
# wrote to mean the same thing. q1 is the cache entry and q2 the incoming query.
# PAWS-QQP negatives are human-labelled pairs with high lexical overlap but
# different meaning.

QQP_URL = "https://dl.fbaipublicfiles.com/glue/data/QQP.zip"
# PAWS-QQP cannot be redistributed (QQP license); Google ships an *index* file
# of token positions that is reconstructed against the original QQP corpus. The
# GCS bucket went access-denied in 2026, so the primary URL is the Wayback
# Machine capture of the same object.
PAWS_QQP_INDEX_URLS = [
    "https://storage.googleapis.com/paws/english/paws_qqp.tar.gz",
    "https://web.archive.org/web/20250731054122id_/https://storage.googleapis.com/paws/english/paws_qqp.tar.gz",
]


def _download_file(url: str, target: Path, allow_network: bool) -> Path:
    if target.exists():
        return target
    if not allow_network:
        raise FileNotFoundError(
            f"{target} is missing. Run the build stage on the CPU server with "
            "--allow-network, or place the file there manually."
        )
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=600, stream=True) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    return target


def fetch_qqp(raw_dir: str | Path, allow_network: bool) -> Path:
    """Return the path to the QQP train.tsv, downloading + extracting if needed."""
    raw_root = Path(raw_dir)
    target = raw_root / "qqp_train.tsv"
    if target.exists():
        return target
    archive = _download_file(QQP_URL, raw_root / "QQP.zip", allow_network)
    with zipfile.ZipFile(archive) as bundle:
        for member in ("QQP/train.tsv", "QQP/dev.tsv"):
            if member not in bundle.namelist():
                continue
            with bundle.open(member) as source:
                suffix = "train" if "train" in member else "dev"
                with (raw_root / f"qqp_{suffix}.tsv").open("wb") as out:
                    out.write(source.read())
    return target


def _qqp_token_index(raw_root: Path, allow_network: bool) -> dict[int, list[str]]:
    """qid -> NLTK tokens, from the GLUE QQP train+dev files.

    The PAWS-QQP index addresses tokens of the original Quora release by qid;
    the GLUE packaging preserves qids and question text, so the same map can be
    rebuilt without the (no longer distributed) original file.
    """
    import nltk

    fetch_qqp(raw_root, allow_network)
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:  # older nltk has no punkt_tab
        pass
    tokens_by_qid: dict[int, list[str]] = {}
    for name in ("qqp_train.tsv", "qqp_dev.tsv"):
        path = raw_root / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader, None)
            for row in reader:
                if len(row) < 5:
                    continue
                for qid_field, text_field in ((1, 3), (2, 4)):
                    try:
                        qid = int(row[qid_field])
                    except ValueError:
                        continue
                    if qid not in tokens_by_qid:
                        tokens_by_qid[qid] = nltk.word_tokenize(row[text_field])
    return tokens_by_qid


def _reconstruct_paws_sentence(
    spec: str, qid: int, tokens_by_qid: dict[int, list[str]]
) -> str | None:
    """Rebuild one PAWS-QQP sentence from its token-position spec."""
    tokens: list[str] = []
    own = tokens_by_qid.get(qid)
    if own is None:
        return None
    for part in spec.split("/"):
        if part.startswith("("):
            # "(qid,index)": a token borrowed from a different question
            try:
                other_qid, index = ast.literal_eval(part)
            except (SyntaxError, ValueError):
                return None
            source = tokens_by_qid.get(other_qid)
            if source is None or not 0 <= index < len(source):
                return None
            tokens.append(source[index])
        else:
            try:
                index = int(part)
            except ValueError:
                return None
            if not 0 <= index < len(own):
                return None
            tokens.append(own[index])
    return " ".join(tokens)


def fetch_paws_qqp(raw_dir: str | Path, allow_network: bool) -> Path:
    """Return the path to the reconstructed PAWS-QQP final train tsv.

    Produces ``paws_qqp_final_train.tsv`` (columns id, sentence1, sentence2,
    label) from the official index file plus the GLUE QQP corpus. Pairs whose
    token positions cannot be resolved are skipped and counted in the manifest
    sidecar.
    """
    raw_root = Path(raw_dir)
    target = raw_root / "paws_qqp_final_train.tsv"
    if target.exists():
        return target
    archive = None
    last_error: Exception | None = None
    for url in PAWS_QQP_INDEX_URLS:
        try:
            archive = _download_file(url, raw_root / "paws_qqp_index.tar.gz", allow_network)
            with tarfile.open(archive):
                break
        except Exception as exc:  # try the next mirror
            last_error = exc
            archive = None
            if (raw_root / "paws_qqp_index.tar.gz").exists():
                (raw_root / "paws_qqp_index.tar.gz").unlink()
    if archive is None:
        raise FileNotFoundError(
            f"could not fetch the PAWS-QQP index from any mirror: {last_error}"
        )
    tokens_by_qid = _qqp_token_index(raw_root, allow_network)
    kept = skipped = 0
    with tarfile.open(archive) as bundle:
        member = next(
            name for name in bundle.getnames() if name.endswith("train.tsv")
        )
        with bundle.extractfile(member) as source:
            index_rows = csv.DictReader(
                io.TextIOWrapper(source, encoding="utf-8"), delimiter="\t"
            )
            with target.open("w", encoding="utf-8", newline="") as out:
                writer = csv.writer(out, delimiter="\t")
                writer.writerow(["id", "sentence1", "sentence2", "label"])
                for row in index_rows:
                    try:
                        qid1, qid2 = int(row["qid1"]), int(row["qid2"])
                    except (KeyError, ValueError):
                        skipped += 1
                        continue
                    s1 = _reconstruct_paws_sentence(row["sentence1"], qid1, tokens_by_qid)
                    s2 = _reconstruct_paws_sentence(row["sentence2"], qid2, tokens_by_qid)
                    label = str(row.get("label", "")).strip()
                    if not s1 or not s2 or label not in {"0", "1"}:
                        skipped += 1
                        continue
                    writer.writerow([row.get("id", kept), s1, s2, label])
                    kept += 1
    atomic_write_json(
        raw_root / "paws_qqp_reconstruction_manifest.json",
        {"kept": kept, "skipped": skipped, "source": "paws_qqp_index + GLUE QQP"},
    )
    return target


def _read_qqp_pairs(path: Path) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        for row in reader:
            if len(row) < 6 or row[5] not in {"0", "1"}:
                continue  # QQP has malformed lines with embedded tabs; skip them
            q1, q2 = _clean_text(row[3]), _clean_text(row[4])
            if q1 and q2 and q1 != q2:
                pairs.append((q1, q2, int(row[5])))
    return pairs


def _read_paws_qqp_pairs(path: Path) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            q1, q2 = _clean_text(row["sentence1"]), _clean_text(row["sentence2"])
            label = str(row.get("label", "")).strip()
            if q1 and q2 and q1 != q2 and label in {"0", "1"}:
                pairs.append((q1, q2, int(label)))
    return pairs


def _select_pairs(
    pairs: list[tuple[str, str, int]], count: int, seed: int, tag: str
) -> list[tuple[str, str, int]]:
    ordered = sorted(
        pairs, key=lambda pair: stable_id(seed, tag, pair[0], pair[1])
    )
    return ordered[:count]


def build_benign_pair_records(
    raw_dir: str | Path,
    config: ExperimentConfig,
    allow_network: bool = False,
    existing_texts: set[str] | None = None,
) -> tuple[list[dict], list[QueryRecord]]:
    """Build the independent benign population as (intent_rows, records).

    Positive pairs (QQP is_duplicate=1 plus PAWS-QQP positives, up to
    ``config.qqp_benign_pairs`` total) become a canonical entry (q1) plus a
    ``benign_query`` record (q2, semantic_label ``pending`` — it must pass the
    same V pass as every other equivalence claim). PAWS-QQP negatives become a
    canonical entry plus a ``hard_negative`` record (F3), keeping the human
    non-equivalence label. Pairs whose q1 or q2 already appears in
    ``existing_texts`` are skipped (top-up runs after a V pass rejection)."""
    skip = {t.casefold() for t in (existing_texts or set())}
    paws_pairs = _read_paws_qqp_pairs(fetch_paws_qqp(raw_dir, allow_network))
    qqp_pairs = _read_qqp_pairs(fetch_qqp(raw_dir, allow_network))

    paws_pos = _select_pairs(
        [pair for pair in paws_pairs if pair[2] == 1],
        config.paws_qqp_positive_pairs,
        config.seed,
        "paws-qqp-pos",
    )
    paws_neg = _select_pairs(
        [pair for pair in paws_pairs if pair[2] == 0],
        config.paws_qqp_negative_pairs,
        config.seed,
        "paws-qqp-neg",
    )
    qqp_pos = _select_pairs(
        [pair for pair in qqp_pairs if pair[2] == 1],
        config.qqp_benign_pairs - len(paws_pos),
        config.seed,
        "qqp-pos",
    )

    intent_rows: list[dict] = []
    records: list[QueryRecord] = []

    def emit_pair(source: str, q1: str, q2: str, kind: str) -> None:
        if q1.casefold() in skip or q2.casefold() in skip:
            return
        intent_id = f"{source}-{stable_id(config.seed, source, q1, q2, length=12)}"
        split = deterministic_split(intent_id, config)
        canonical_id = stable_id(intent_id, "canonical")
        records.append(
            QueryRecord(
                record_id=canonical_id,
                intent_id=intent_id,
                source=source,
                split=split,
                query_role="canonical",
                text=q1,
                generator=f"human_{source}",
                seed=config.seed,
                semantic_label="equivalent",
                canonical_answer=[],
                metadata={"benign_pair": True, "pair_kind": kind},
            )
        )
        if kind == "positive":
            records.append(
                QueryRecord(
                    record_id=stable_id(intent_id, "benign_query", q2),
                    intent_id=intent_id,
                    source=source,
                    split=split,
                    query_role="benign_query",
                    text=q2,
                    generator=f"human_{source}",
                    seed=config.seed,
                    semantic_label="pending",
                    parent_id=canonical_id,
                    metadata={"benign_pair": True, "pair_kind": kind},
                )
            )
        else:
            records.append(
                QueryRecord(
                    record_id=stable_id(intent_id, "hard_negative", q2),
                    intent_id=intent_id,
                    source=source,
                    split=split,
                    query_role="hard_negative",
                    text=q2,
                    generator=f"human_{source}",
                    seed=config.seed,
                    semantic_label="non_equivalent",
                    parent_id=canonical_id,
                    metadata={"benign_pair": True, "pair_kind": kind, "f3": True},
                )
            )
        intent_rows.append(
            {
                "intent_id": intent_id,
                "source": source,
                "split": split,
                "canonical_record_id": canonical_id,
                "canonical_query": q1,
                "canonical_answer": [],
                "human_question_count": 2,
                "benign_pair": True,
                "pair_kind": kind,
            }
        )

    for q1, q2, _ in paws_pos:
        emit_pair("paws", q1, q2, "positive")
    for q1, q2, _ in qqp_pos:
        emit_pair("qqp", q1, q2, "positive")
    for q1, q2, _ in paws_neg:
        emit_pair("paws", q1, q2, "negative")
    return intent_rows, records


def append_benign_pair_records(
    dataset_dir: str | Path,
    raw_dir: str | Path,
    config: ExperimentConfig,
    allow_network: bool = False,
) -> dict:
    """Append the benign-pair population to an existing comqa build.

    Idempotent: texts already present are skipped, so a top-up run with a
    larger ``qqp_benign_pairs`` only adds the new tail of the deterministic
    selection (used after a V pass rejects part of the first batch)."""
    from .io import read_records, write_records

    dataset_root = Path(dataset_dir)
    existing = read_records(dataset_root / "records.jsonl")
    existing_texts = {record.text for record in existing}
    intent_rows, records = build_benign_pair_records(
        raw_dir, config, allow_network=allow_network, existing_texts=existing_texts
    )
    write_records(dataset_root / "records.jsonl", [*existing, *records])
    intents_path = dataset_root / "intents.jsonl"
    with intents_path.open("r", encoding="utf-8") as handle:
        existing_intents = [json.loads(line) for line in handle if line.strip()]
    atomic_write_jsonl(intents_path, [*existing_intents, *intent_rows])
    manifest_path = dataset_root / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benign_pairs"] = {
        "positive_pairs": sum(1 for row in intent_rows if row["pair_kind"] == "positive"),
        "negative_pairs": sum(1 for row in intent_rows if row["pair_kind"] == "negative"),
        "paws_positive_pairs": len(
            [r for r in records if r.query_role == "benign_query" and r.source == "paws"]
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest["benign_pairs"]
