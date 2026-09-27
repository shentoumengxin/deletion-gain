import hashlib
import json

import pytest

from sentry.artifacts import data_path, data_root, verify, verify_evidence


def test_data_root_requires_explicit_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("SENTRY_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="SENTRY_DATA_ROOT"):
        data_root()
    monkeypatch.setenv("SENTRY_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path
    assert data_path("datasets", "records.jsonl") == tmp_path / "datasets/records.jsonl"
    with pytest.raises(ValueError, match="inside"):
        data_path("../outside")


def test_verifier_detects_corruption_and_missing_materialization(tmp_path):
    payload = b'{"id": 1}\n'
    (tmp_path / "records.jsonl").write_bytes(payload)
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 900\n"
    (tmp_path / "vectors.npz").write_bytes(pointer)
    entries = [
        {"id": name, "path": name, "status": status, "bytes": len(content),
         "sha256": hashlib.sha256(content).hexdigest()}
        for name, content, status in [
            ("records.jsonl", payload, "present"), ("vectors.npz", pointer, "lfs-pointer")]
    ]
    entries[0]["lines"] = 1
    entries.append({"id": "remote", "status": "remote-only"})
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"artifacts": entries}))
    result = verify(tmp_path, manifest)
    assert result["checked"] == 2
    assert result["failures"] == []
    assert result["complete"] is False
    assert {x["status"] for x in result["unavailable"]} == {"lfs-pointer", "remote-only"}
    (tmp_path / "records.jsonl").write_bytes(b"changed")
    assert verify(tmp_path, manifest)["failures"][0]["error"] == "size or SHA256 mismatch"
    (tmp_path / "records.jsonl").unlink()
    assert verify(tmp_path, manifest)["failures"][0]["error"] == "missing local file"


def test_verified_server_copy_does_not_claim_local_materialization(tmp_path):
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"artifacts": [{
        "id": "large-scores", "status": "remote-only", "required_for_paper": True,
        "bytes": 100, "sha256": "a" * 64,
        "canonical_remote": {"host": "example-server", "path": "/data/scores.jsonl",
                             "bytes": 100, "sha256": "a" * 64,
                             "status": "verified-remote-canonical"},
    }]}))
    result = verify(tmp_path, manifest)
    assert result["complete"] is True
    assert result["local_complete"] is False


def test_evidence_rejects_changed_fields_and_unregistered_inputs(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"count": 42}')
    manifest = tmp_path / "evidence.json"
    document = {"artifacts": {"summary": {
        "namespace": "data", "path": "summary.json",
        "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }}, "evidence": [{"id": "table", "inputs": ["summary"], "result_fields": [
        {"artifact": "summary", "json_pointer": "/count", "value": 42},
    ]}]}
    manifest.write_text(json.dumps(document))
    result = verify_evidence(manifest, tmp_path)
    assert result["failures"] == []
    assert result["fields_checked"] == 1
    document["evidence"][0]["inputs"].append("unknown")
    document["evidence"][0]["result_fields"][0]["value"] = 43
    manifest.write_text(json.dumps(document))
    errors = {x["error"] for x in verify_evidence(manifest, tmp_path)["failures"]}
    assert errors == {"unregistered evidence artifact", "recorded value changed"}
