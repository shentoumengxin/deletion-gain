from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ExperimentConfig
from .embed import embed_records
from .generate import (
    generate_gcg,
    generate_legal_candidates,
    generate_mock_gcg,
    merge_gcg_shards,
    generate_responses,
    generate_wu,
    generate_long_legit,
    generate_ndss,
    generate_ndss_adaptive,
    generate_ndss_matched,
)
from .io import ensure_workspace
from .sources import (
    append_benign_pair_records,
    build_comqa_records,
    build_smoke_records,
    fetch_huggingface_external,
    import_external_jsonl,
)
from .validate import (
    export_annotation_template,
    validate_base_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper corpus construction, validation and embedding pipeline"
    )
    parser.add_argument("--config", required=True, help="Experiment JSON config")
    parser.add_argument(
        "--work-root", "--workspace", dest="workspace",
        required=True,
        help="External work directory for generated artifacts; SENTRY_DATA_ROOT identifies frozen inputs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--source", choices=["comqa", "smoke"], default="comqa")
    build.add_argument("--allow-network", action="store_true")
    build.add_argument(
        "--with-benign-qqp",
        action="store_true",
        help="append the independent QQP/PAWS-QQP benign-pair population (T1.1)",
    )
    build.add_argument(
        "--only-benign-qqp",
        action="store_true",
        help="skip the comqa build; only (re)run the benign-pair append "
        "(top-up after a V pass rejected part of the first batch)",
    )

    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--task",
        required=True,
        choices=[
            "legal-qcpg",
            "legal-api",
            "wu",
            "gcg",
            "merge-gcg",
            "responses",
            "ndss",
            "ndss-adaptive",
            "ndss-matched",
            "long-legit",
        ],
    )
    generate.add_argument("--smoke", action="store_true")
    generate.add_argument("--allow-local-heavy", action="store_true")
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--shard-count", type=int, default=1)
    # long-legit only. The target length band is defined by the attacks this run
    # must be length-matched against, and pooling families of very different
    # lengths (a 288-char GCG string with a 47-char matched injection) makes the
    # band incoherent, so match against one family at a time.
    generate.add_argument(
        "--attack-generators",
        default=None,
        help="comma-separated generator names defining the long-legit target "
        "length band (e.g. 'gcg'); default: every attack record",
    )

    validate = subparsers.add_parser("validate")
    validate.add_argument("--phase", choices=["base"], default="base")
    validate.add_argument("--annotations")
    validate.add_argument("--export-template", action="store_true")
    validate.add_argument(
        "--roles",
        default=None,
        help="comma-separated query_role values for --export-template "
        "(default: legal_candidate)",
    )

    embed = subparsers.add_parser("embed")
    embed.add_argument("--smoke", action="store_true")
    embed.add_argument("--allow-local-heavy", action="store_true")
    embed.add_argument(
        "--records", help="Explicit input records; default: WORK_ROOT/datasets/validated_records.jsonl"
    )

    external = subparsers.add_parser("import-external")
    external.add_argument("--input", required=True)
    external.add_argument("--source-name", required=True)
    external.add_argument("--limit", type=int)

    fetch_external = subparsers.add_parser("fetch-external")
    fetch_external.add_argument("--dataset", choices=["vcache", "paws"], required=True)
    fetch_external.add_argument("--limit", type=int, default=1000)

    subparsers.add_parser("run-smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    workspace = ensure_workspace(args.workspace)
    repository_root = Path(__file__).resolve().parents[3]
    if repository_root in (
        workspace,
        *workspace.parents,
    ):
        raise RuntimeError(
            "the experiment workspace must be outside the Git checkout; "
            f"received {workspace}"
        )
    if args.command == "build":
        if args.source == "smoke":
            records = build_smoke_records(workspace / "datasets", config)
            result = {"records": len(records), "source": "smoke"}
        elif args.only_benign_qqp:
            result = {
                "benign_pairs": append_benign_pair_records(
                    workspace / "datasets",
                    workspace / "raw",
                    config,
                    allow_network=args.allow_network,
                ),
                "source": "qqp_topup",
            }
        else:
            intents, records = build_comqa_records(
                workspace / "raw",
                workspace / "datasets",
                config,
                allow_network=args.allow_network,
            )
            result = {
                "intents": len(intents),
                "records": len(records),
                "source": "comqa",
            }
            if args.with_benign_qqp:
                result["benign_pairs"] = append_benign_pair_records(
                    workspace / "datasets",
                    workspace / "raw",
                    config,
                    allow_network=args.allow_network,
                )
    elif args.command == "generate":
        result = _run_generate(args, workspace, config)
    elif args.command == "validate":
        result = _run_validate(args, workspace, config)
    elif args.command == "embed":
        _require_server("embedding", args.smoke, args.allow_local_heavy)
        result = embed_records(
            Path(args.records) if args.records else workspace / "datasets" / "validated_records.jsonl",
            workspace / "embeddings" / "query_embeddings.npz",
            workspace / "datasets" / "embedded_records.jsonl",
            config,
            smoke=args.smoke,
        )
    elif args.command == "import-external":
        count = import_external_jsonl(
            args.input,
            workspace / "datasets" / f"external_{args.source_name}.jsonl",
            args.source_name,
            args.limit,
        )
        result = {"imported": count, "source": args.source_name}
    elif args.command == "fetch-external":
        count = fetch_huggingface_external(
            args.dataset,
            workspace / "datasets" / f"external_{args.dataset}.jsonl",
            args.limit,
            config.seed,
        )
        result = {"downloaded": count, "dataset": args.dataset}
    elif args.command == "run-smoke":
        result = run_smoke(workspace, config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def _run_generate(args, workspace: Path, config: ExperimentConfig) -> dict:
    records_path = workspace / "datasets" / "records.jsonl"
    if args.task == "legal-qcpg":
        _require_server("QCPG generation", args.smoke, args.allow_local_heavy)
        added = generate_legal_candidates(
            records_path, config, "qcpg", smoke=args.smoke
        )
    elif args.task == "legal-api":
        added = generate_legal_candidates(records_path, config, "api", smoke=args.smoke)
    elif args.task == "wu":
        added = generate_wu(records_path, config)
    elif args.task == "ndss":
        added = generate_ndss(records_path, config)
    elif args.task == "ndss-adaptive":
        added = generate_ndss_adaptive(records_path, config)
    elif args.task == "ndss-matched":
        # API rewriting, not GPU: same placement as legal-api, so no
        # _require_server gate. Returns a dict (attempt log + yield), not a count.
        # Length budgets come from validated_records.jsonl (role "legal" only
        # exists there); new ndss records are appended to records.jsonl so the
        # next base validate carries them.
        return generate_ndss_matched(
            records_path,
            config,
            smoke=args.smoke,
            validated_path=workspace / "datasets" / "validated_records.jsonl",
        )
    elif args.task == "long-legit":
        generators = getattr(args, "attack_generators", None)
        return generate_long_legit(
            records_path,
            config,
            smoke=args.smoke,
            attack_generators=(
                tuple(part.strip() for part in generators.split(",") if part.strip())
                if generators
                else None
            ),
        )
    elif args.task == "gcg":
        _require_server("GCG generation", args.smoke, args.allow_local_heavy)
        if args.smoke:
            added = generate_mock_gcg(records_path, config)
        else:
            shard_dir = workspace / "datasets" / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            added = generate_gcg(
                records_path,
                config,
                output_path=shard_dir
                / f"gcg_s{args.shard_count:03d}_{args.shard_index:03d}.jsonl",
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
    elif args.task == "merge-gcg":
        added = merge_gcg_shards(
            records_path,
            workspace / "datasets" / "shards",
            args.shard_count,
        )
    elif args.task == "responses":
        return generate_responses(workspace / "datasets" / "validated_records.jsonl")
    else:
        raise AssertionError(args.task)
    return {"task": args.task, "records_added": added}


def _run_validate(args, workspace: Path, config: ExperimentConfig) -> dict:
    annotations = Path(args.annotations) if args.annotations else None
    export_roles = (
        {role.strip() for role in args.roles.split(",") if role.strip()}
        if args.roles
        else None
    )
    if args.export_template:
        count = export_annotation_template(
            workspace / "datasets" / "records.jsonl",
            workspace / "annotations" / "legal_candidates.csv",
            export_roles or {"legal_candidate"},
            seed=config.seed,
        )
        return {"annotation_rows": count}
    return validate_base_records(
        workspace / "datasets" / "records.jsonl",
        workspace / "datasets" / "validated_records.jsonl",
        config,
        annotations,
    )


def run_smoke(workspace: Path, config: ExperimentConfig) -> dict:
    """Verify benign corpus IO using authored fixtures and the hash embedder.

    No candidate/attack generation, network calls, or learned models run here.
    Separate annotation tests enforce the independent validator's requirements.
    """
    build_smoke_records(workspace / "datasets", config)
    validation = validate_base_records(
        workspace / "datasets" / "records.jsonl",
        workspace / "datasets" / "validated_records.jsonl",
        config,
    )
    manifest = embed_records(
        workspace / "datasets" / "validated_records.jsonl",
        workspace / "embeddings" / "query_embeddings.npz",
        workspace / "datasets" / "embedded_records.jsonl",
        config,
        smoke=True,
    )
    return {"smoke": True, "validation": validation, "embedding": manifest}


def _require_server(task: str, smoke: bool, allow_local_heavy: bool) -> None:
    if smoke or allow_local_heavy:
        return
    if (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("ORTHO_ALLOW_LOCAL_HEAVY") == "1"
    ):
        return
    raise RuntimeError(
        f"{task} is blocked outside a scheduled server job. Submit the provided "
        "Slurm script or pass --allow-local-heavy explicitly."
    )


if __name__ == "__main__":
    raise SystemExit(main())
