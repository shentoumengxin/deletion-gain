from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .config import ExperimentConfig
from .io import atomic_write_json, read_records, stable_id, write_records
from .schema import QueryRecord

VALID_ANNOTATION_LABELS = {"equivalent", "non_equivalent", "uncertain"}


def export_annotation_template(
    records_path: str | Path,
    output_path: str | Path,
    roles: set[str],
    sample_limit: int | None = None,
    seed: int = 0,
) -> int:
    records = [
        record
        for record in read_records(records_path)
        if record.query_role in roles and record.semantic_label == "pending"
    ]
    records.sort(key=lambda record: stable_id(seed, record.record_id, "annotation"))
    if sample_limit is not None:
        records = records[:sample_limit]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "intent_id",
                "query_role",
                "generator",
                "parent_id",
                "text",
                "annotator_1_label",
                "annotator_2_label",
                "adjudicated_label",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "intent_id": record.intent_id,
                    "query_role": record.query_role,
                    "generator": record.generator,
                    "parent_id": record.parent_id or "",
                    "text": record.text,
                    "annotator_1_label": "",
                    "annotator_2_label": "",
                    "adjudicated_label": "",
                }
            )
    return len(records)


def load_annotations(path: str | Path | None) -> tuple[dict[str, str], dict]:
    if path is None or not Path(path).exists():
        return {}, {"rows": 0, "annotated": 0, "double_annotated": 0, "kappa": None}
    labels: dict[str, str] = {}
    first: list[str] = []
    second: list[str] = []
    row_count = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            record_id = row["record_id"].strip()
            label_1 = row.get("annotator_1_label", "").strip()
            label_2 = row.get("annotator_2_label", "").strip()
            adjudicated = row.get("adjudicated_label", "").strip()
            for label in (label_1, label_2, adjudicated):
                if label and label not in VALID_ANNOTATION_LABELS:
                    raise ValueError(
                        f"invalid annotation label {label!r} for {record_id}"
                    )
            if label_1 and label_2:
                first.append(label_1)
                second.append(label_2)
            final = adjudicated or (label_1 if label_1 and label_1 == label_2 else "")
            if final in {"equivalent", "non_equivalent"}:
                labels[record_id] = final
    kappa = (
        float(cohen_kappa_score(first, second))
        if first and len(set(first + second)) > 1
        else (1.0 if first else None)
    )
    return labels, {
        "rows": row_count,
        "annotated": len(labels),
        "double_annotated": len(first),
        "kappa": kappa,
    }


def _apply_annotations(
    records: list[QueryRecord], labels: dict[str, str]
) -> list[QueryRecord]:
    for record in records:
        if record.record_id in labels:
            record.semantic_label = labels[record.record_id]
            record.metadata = {
                **record.metadata,
                "human_annotation": labels[record.record_id],
            }
    return records


def _balanced_legal_selection(
    candidates: list[QueryRecord],
    count: int,
    seed: int,
) -> list[QueryRecord]:
    equivalent = [
        record for record in candidates if record.semantic_label == "equivalent"
    ]
    groups: dict[str, list[QueryRecord]] = defaultdict(list)
    for record in equivalent:
        groups[record.generator].append(record)
    for generator in groups:
        groups[generator].sort(
            key=lambda record: stable_id(seed, record.record_id, "legal-rank")
        )

    selected: list[QueryRecord] = []
    human = groups.pop("human_comqa", []) + groups.pop("synthetic_human", [])
    selected.extend(human[:count])
    generated_names = [name for name in ("qcpg", "api") if name in groups]
    while len(selected) < count and generated_names:
        progress = False
        for name in generated_names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop(0))
                progress = True
        if not progress:
            break
    leftovers = [
        record for generator_records in groups.values() for record in generator_records
    ]
    leftovers.sort(
        key=lambda record: stable_id(seed, record.record_id, "legal-leftover")
    )
    selected.extend(leftovers[: max(0, count - len(selected))])
    return selected[:count]


def validate_base_records(
    records_path: str | Path,
    output_path: str | Path,
    config: ExperimentConfig,
    annotations_path: str | Path | None = None,
) -> dict:
    records = read_records(records_path)
    labels, annotation_stats = load_annotations(annotations_path)
    pending_legal = [
        record
        for record in records
        if record.query_role in {"legal_candidate", "benign_query"}
        and record.semantic_label == "pending"
    ]
    if pending_legal and (
        annotation_stats["rows"] == 0
        or annotation_stats["double_annotated"] != annotation_stats["rows"]
        or annotation_stats["annotated"] != annotation_stats["rows"]
    ):
        raise RuntimeError(
            "generated legal candidates and benign-query pairs require completed "
            "annotations: two completed labels per row and an adjudicated label "
            "for every disagreement"
        )
    records = _apply_annotations(records, labels)
    by_intent: dict[str, list[QueryRecord]] = defaultdict(list)
    for record in records:
        by_intent[record.intent_id].append(record)

    validated: list[QueryRecord] = []
    audit_rows = []
    for intent_id in sorted(by_intent):
        intent_records = by_intent[intent_id]
        canonicals = [
            record for record in intent_records if record.query_role == "canonical"
        ]
        if len(canonicals) != 1:
            raise ValueError(f"{intent_id} has {len(canonicals)} canonical records")
        canonical = canonicals[0]
        validated.append(canonical)
        selected = _balanced_legal_selection(
            [
                record
                for record in intent_records
                if record.query_role == "legal_candidate"
            ],
            count=config.min_paraphrases,
            seed=config.seed,
        )
        for record in selected:
            record.query_role = "legal"
            validated.append(record)
        wu_records = [record for record in intent_records if record.query_role == "wu"]
        gcg_records = [
            record
            for record in intent_records
            if record.query_role == "gcg"
            and record.metadata.get("accepted_band") is True
        ]
        ndss_records = [
            record for record in intent_records if record.query_role == "ndss"
        ]
        validated.extend(wu_records)
        validated.extend(gcg_records)
        validated.extend(ndss_records)
        # Independent benign population (T1.1): the incoming side of a benign
        # pair is carried only after the V pass confirms equivalence; the
        # human-labelled F3 hard negatives carry their non-equivalence label.
        benign_validated = [
            record
            for record in intent_records
            if record.query_role == "benign_query"
            and record.semantic_label == "equivalent"
        ]
        benign_dropped = [
            record
            for record in intent_records
            if record.query_role == "benign_query"
            and record.semantic_label != "equivalent"
        ]
        hard_negatives = [
            record for record in intent_records if record.query_role == "hard_negative"
        ]
        validated.extend(benign_validated)
        validated.extend(hard_negatives)
        audit_rows.append(
            {
                "intent_id": intent_id,
                "split": canonical.split,
                "legal_selected": len(selected),
                "legal_by_generator": dict(
                    Counter(record.generator for record in selected)
                ),
                "wu_count": len(wu_records),
                "gcg_accepted_count": len(gcg_records),
                "complete_legal": len(selected) >= config.min_paraphrases,
                "dual_attack": (
                    len(wu_records) >= config.min_attacks_per_type
                    and len(gcg_records) >= config.min_attacks_per_type
                ),
                "benign_query_validated": len(benign_validated),
                "benign_query_dropped": len(benign_dropped),
                "hard_negative_count": len(hard_negatives),
            }
        )

    write_records(output_path, validated)
    complete = sum(row["complete_legal"] for row in audit_rows)
    dual_attack = sum(row["dual_attack"] for row in audit_rows)
    report = {
        "intent_count": len(audit_rows),
        "record_count": len(validated),
        "complete_legal_intents": complete,
        "dual_attack_intents": dual_attack,
        "benign_query_validated": sum(row["benign_query_validated"] for row in audit_rows),
        "benign_query_dropped": sum(row["benign_query_dropped"] for row in audit_rows),
        "hard_negative_count": sum(row["hard_negative_count"] for row in audit_rows),
        "annotation": annotation_stats,
        "audit": audit_rows,
    }
    atomic_write_json(Path(output_path).with_name("validation_report.json"), report)
    return report
