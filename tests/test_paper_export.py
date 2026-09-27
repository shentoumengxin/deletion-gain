import json
from pathlib import Path

import pytest

from experiments.paper.figures import export
from sentry.artifacts import sha256


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / 'data'
    (root / 'manifests').mkdir(parents=True)
    summary = root / 'summary.json'
    summary.write_text(json.dumps({'metrics': {'rate': 0.25}, 'rows': ['not exported']}))
    artifact = {'id': 'summary', 'path': 'summary.json', 'status': 'local',
                'bytes': summary.stat().st_size, 'sha256': sha256(summary)}
    (root / 'manifests/inventory.json').write_text(json.dumps({'artifacts': [artifact]}))
    manifest = tmp_path / 'evidence.json'
    manifest.write_text(json.dumps({
        'artifacts': {'data:summary': {**artifact, 'namespace': 'data'}},
        'evidence': [{'id': 'table', 'result_fields': [
            {'artifact': 'data:summary', 'json_pointer': '/metrics/rate', 'value': 0.25}]}],
    }))
    paper = tmp_path / 'paper'
    (paper / 'figures').mkdir(parents=True)
    return root, manifest, paper


def test_export_summary_contains_only_verified_result_fields(evidence):
    root, manifest, paper = evidence
    index, summary = export.prepare_evidence(manifest, root, paper)
    assert summary['evidence']['table'][0]['value'] == 0.25
    assert 'not exported' not in json.dumps(summary)
    assert index['verification']['complete']
    assert index['evidence_manifest_sha256'] == sha256(manifest)


def test_whole_artifact_with_row_data_cannot_be_published_as_a_summary(evidence):
    root, manifest, paper = evidence
    document = json.loads(manifest.read_text())
    document['evidence'][0]['result_fields'] = [{'artifact': 'data:summary', 'json_pointer': ''}]
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='row-level'):
        export.prepare_evidence(manifest, root, paper)


def test_corrupt_evidence_aborts_before_rendering_or_publishing(evidence, monkeypatch):
    root, manifest, paper = evidence
    original = paper / 'figures/fig6_panels.pdf'
    original.write_bytes(b'%PDF-original')
    (root / 'summary.json').write_text('{}')
    monkeypatch.setattr('sys.argv', ['export', '--paper-dir', str(paper),
                                    '--data-root', str(root), '--evidence-manifest', str(manifest)])
    def forbidden(*args, **kwargs):
        pytest.fail('rendering started before evidence verification')
    monkeypatch.setattr(export.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='Evidence verification failed'):
        export.main()
    assert original.read_bytes() == b'%PDF-original'
    assert not (paper / 'data').exists()


def test_failed_publication_preserves_previous_file_and_cleans_stage(tmp_path, monkeypatch):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_text('new')
    target.write_text('old')
    def failure(*args):
        raise OSError('fixture publication failure')
    monkeypatch.setattr(export.os, 'replace', failure)
    with pytest.raises(OSError):
        export.publish_file(source, target)
    assert target.read_text() == 'old'
    assert set(tmp_path.iterdir()) == {source, target}


def test_undefined_statistics_are_explicitly_recorded_without_changing_finite_values():
    payload = {'cell/a': {'matched': float('nan'), 'rate': 0.25, 'n': 0}}
    result = export.json_payload(payload)
    json.dumps(result, allow_nan=False)
    assert result['cell/a'] == {'matched': None, 'rate': 0.25, 'n': 0}
    assert result['nonfinite_values'] == [
        {'json_pointer': '/cell~1a/matched', 'source_value': 'NaN'}]
    assert payload['cell/a']['matched'] != payload['cell/a']['matched']
