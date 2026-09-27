"""Locate and verify the external experiment-data worktree without model calls.

Set SENTRY_DATA_ROOT on every machine or pass --data-root explicitly. The release
bundles the paper's data under ``data/``; any other location must be outside the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def data_root(root: str | Path | None = None) -> Path:
    """Return the configured external data root; do not create it."""
    value = root if root is not None else os.environ.get("SENTRY_DATA_ROOT")
    if not value:
        raise ValueError("Set SENTRY_DATA_ROOT or pass an explicit data root")
    path = Path(value).expanduser().resolve()
    code_root = Path(__file__).resolve().parents[1]
    if path.is_relative_to(code_root) and path != code_root / "data":
        raise ValueError("SENTRY_DATA_ROOT must be outside the code checkout "
                         "(or the bundled data/ directory)")
    return path


def data_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a relative artifact path within the data root."""
    base = data_root(root)
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Artifact paths must stay inside SENTRY_DATA_ROOT")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: str | Path | None = None, manifest: str | Path | None = None) -> dict:
    """Check registered local artifacts and report remote/LFS availability honestly."""
    base = data_root(root)
    source = Path(manifest) if manifest else base / "manifests" / "inventory.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    failures, unavailable = [], []
    checked = 0
    for item in document["artifacts"]:
        if item["status"] in {"remote-only", "remote-unverified", "missing"}:
            remote = item.get("canonical_remote", {})
            verified_remote = (remote.get("status") == "verified-remote-canonical"
                               and bool(item.get("sha256"))
                               and remote.get("sha256") == item["sha256"]
                               and remote.get("bytes") == item.get("bytes"))
            unavailable.append({"id": item["id"], "status": item["status"],
                                "required_for_paper": item.get("required_for_paper", True),
                                "verified_remote": verified_remote})
            continue
        path = data_path(item["path"], root=base)
        if not path.is_file():
            failures.append({"id": item["id"], "error": "missing local file"})
            continue
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append({"id": item["id"], "error": "size or SHA256 mismatch"})
            continue
        if item["status"] == "lfs-pointer":
            unavailable.append({"id": item["id"], "status": "lfs-pointer",
                                "required_for_paper": item.get("required_for_paper", True),
                                "verified_remote": False})
        if item.get("lines") is not None:
            with path.open("rb") as stream:
                lines = sum(1 for _ in stream)
            if lines != item["lines"]:
                failures.append({"id": item["id"], "error": "line count mismatch"})
                continue
        checked += 1
    return {"root": str(base), "checked": checked, "failures": failures,
            "unavailable": unavailable,
            "local_complete": not failures and not any(x["required_for_paper"] for x in unavailable),
            "complete": not failures and not any(x["required_for_paper"] and not x["verified_remote"]
                                                  for x in unavailable)}


def verify_evidence(path: str | Path, root: str | Path | None = None,
                    paper_root: str | Path | None = None) -> dict:
    """Validate the recorded paper sources and JSON pointers without running experiments."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    code = Path(__file__).resolve().parents[1]
    roots = {"code": code, "data": data_root(root),
             "paper": Path(paper_root or os.environ.get("SENTRY_PAPER_ROOT")
                           or code.parent / "paper").expanduser().resolve()}
    failures, resolved = [], {}
    receipt_path = roots["data"] / "manifests" / "verified_remote_sources.json"
    receipts = (json.loads(receipt_path.read_text(encoding="utf-8"))["records"]
                if receipt_path.is_file() else [])
    remote_receipts = {item["source"]: item for item in receipts}
    for artifact_id, item in document["artifacts"].items():
        if item["namespace"] == "remote":
            receipt = remote_receipts.get(item["path"], {})
            if (receipt.get("status") != "verified-remote-canonical"
                    or not item.get("sha256")
                    or any(receipt.get(key) != item.get(key)
                           for key in ("sha256", "bytes", "canonical_path"))):
                failures.append({"id": artifact_id,
                                 "error": "remote verification receipt missing or inconsistent"})
            continue
        source = (roots[item["namespace"]] / item["path"]).resolve()
        if not source.is_relative_to(roots[item["namespace"]]):
            failures.append({"id": artifact_id, "error": "source escapes its root"})
        elif not source.is_file():
            failures.append({"id": artifact_id, "error": "source missing"})
        elif sha256(source) != item["sha256"]:
            failures.append({"id": artifact_id, "error": "source SHA256 changed"})
        else:
            resolved[artifact_id] = source
    checked_fields = 0
    for evidence in document["evidence"]:
        references = (evidence.get("paper", []) + evidence.get("source_code", [])
                      + evidence.get("inputs", [])
                      + [field["artifact"] for field in evidence.get("result_fields", [])])
        for artifact_id in set(references) - document["artifacts"].keys():
            failures.append({"id": evidence["id"], "artifact": artifact_id,
                             "error": "unregistered evidence artifact"})
        for field in evidence.get("result_fields", []):
            artifact_id = field["artifact"]
            if artifact_id not in resolved or "json_pointer" not in field:
                continue
            try:
                value = json.loads(resolved[artifact_id].read_text(encoding="utf-8"))
                pointer = field["json_pointer"]
                for part in pointer.split("/")[1:] if pointer else []:
                    key = part.replace("~1", "/").replace("~0", "~")
                    value = value[int(key)] if isinstance(value, list) else value[key]
                if "value" in field and value != field["value"]:
                    raise ValueError("recorded value changed")
                checked_fields += 1
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                failures.append({"id": evidence["id"], "artifact": artifact_id,
                                 "field": field["json_pointer"], "error": str(exc)})
    return {"artifacts_checked": len(resolved), "fields_checked": checked_fields,
            "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("path", "verify"))
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--evidence", help="Also verify a paper evidence manifest")
    parser.add_argument("--paper-root", help="Override the separate paper checkout location")
    parser.add_argument("--require-complete", action="store_true",
                        help="Also fail when a paper dependency has no verified local or remote copy")
    parser.add_argument("--require-local", action="store_true",
                        help="Also fail when a paper dependency is not materialized locally")
    args = parser.parse_args()
    if args.command == "path":
        print(data_root(args.data_root))
        return 0
    result = verify(args.data_root, args.manifest)
    if args.evidence:
        result["evidence"] = verify_evidence(args.evidence, args.data_root, args.paper_root)
        if result["evidence"]["failures"]:
            result["complete"] = False
            result["local_complete"] = False
    summary = dict(result)
    summary["optional_unavailable"] = sum(not x["required_for_paper"] for x in result["unavailable"])
    summary["unavailable"] = [x for x in result["unavailable"] if x["required_for_paper"]]
    print(json.dumps(summary, indent=2))
    return int(bool(result["failures"]) or bool(result.get("evidence", {}).get("failures"))
               or (args.require_complete and not result["complete"])
               or (args.require_local and not result["local_complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
