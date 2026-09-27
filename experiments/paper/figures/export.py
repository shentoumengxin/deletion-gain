"""Verify paper evidence, then export approved figures, summaries and provenance."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from experiments.paper.paths import REPO_ROOT
from sentry.artifacts import data_root, sha256, verify, verify_evidence

APPROVED = (
    ('fig6_panels', 'fig6_panels.pdf'),
    ('fig3_coexistence', 'fig3_coexistence.pdf'),
    ('fig7_tradeoff_ranked', 'fig7_tradeoff.pdf'),
)

def contains_rows(value) -> bool:
    if isinstance(value, dict):
        if 'intent_id' in value:
            return True
        return any((key in {'rows', 'records', 'samples', 'responses'}
                    and isinstance(item, (list, dict))) or contains_rows(item)
                   for key, item in value.items())
    return isinstance(value, list) and any(contains_rows(item) for item in value)


def prepare_evidence(manifest: Path, root: Path, paper: Path) -> tuple[dict, dict]:
    """Resolve checked aggregate fields; never export the underlying row files."""
    data_check = verify(root)
    source_check = verify_evidence(manifest, root, paper)
    if not data_check['complete'] or source_check['failures']:
        raise ValueError('Evidence verification failed; no paper files were published: '
                         + json.dumps({'data_failures': data_check['failures'],
                                       'sources': source_check['failures'],
                                       'complete': data_check['complete']}))
    document = json.loads(manifest.read_text())
    roots = {'code': REPO_ROOT, 'data': root, 'paper': paper}
    summaries = {}
    for evidence in document['evidence']:
        fields = []
        for field in evidence.get('result_fields', []):
            if 'json_pointer' not in field:
                continue
            item = document['artifacts'][field['artifact']]
            source = roots[item['namespace']] / item['path']
            value = json.loads(source.read_text())
            for part in field['json_pointer'].split('/')[1:]:
                key = part.replace('~1', '/').replace('~0', '~')
                value = value[int(key)] if isinstance(value, list) else value[key]
            if contains_rows(value):
                raise ValueError('A summary field contains row-level data: '
                                 + field['artifact'] + '#' + field['json_pointer'])
            fields.append({'artifact': field['artifact'],
                           'json_pointer': field['json_pointer'], 'value': value})
        summaries[evidence['id']] = fields
    index = {'schema_version': 1, 'evidence_manifest_sha256': sha256(manifest),
             'exporter_sha256': sha256(Path(__file__)),
             'source_evidence': document,
             'verification': {'local_artifacts_checked': data_check['checked'],
                              'local_complete': data_check['local_complete'],
                              'complete': data_check['complete'], **source_check},
             'exported_figures': {}}
    return index, {'schema_version': 1, 'evidence': summaries}


def publish_file(source: Path, target: Path) -> None:
    """Replace one file atomically after staging; the whole export is not a transaction."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.sentry-' + uuid.uuid4().hex)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def json_payload(payload: dict) -> dict:
    """Represent undefined source statistics as null with an explicit lossless index."""
    nonfinite = []
    def convert(value, pointer=''):
        if isinstance(value, float) and not math.isfinite(value):
            label = 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
            nonfinite.append({'json_pointer': pointer, 'source_value': label})
            return None
        if isinstance(value, dict):
            return {key: convert(item, pointer + '/' + key.replace('~', '~0').replace('/', '~1'))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item, pointer + '/' + str(i)) for i, item in enumerate(value)]
        return value
    result = convert(payload)
    result['nonfinite_values'] = nonfinite
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--evidence-manifest', type=Path,
                        default=REPO_ROOT / 'experiments/paper/results/evidence_manifest.json')
    args = parser.parse_args()
    paper = args.paper_dir.expanduser().resolve()
    target = paper / 'figures'
    if not target.is_dir():
        parser.error('--paper-dir must be an existing paper checkout with figures/')
    root = data_root(args.data_root)
    index, summary = prepare_evidence(args.evidence_manifest, root, paper)
    with tempfile.TemporaryDirectory(prefix='sentry-paper-') as tmp:
        stage = Path(tmp)
        (stage / 'figures').mkdir()
        env = {**os.environ, 'SENTRY_PAPER_DIR': str(stage),
               'MPLCONFIGDIR': str(stage / 'matplotlib'),
               'MPLBACKEND': 'Agg', 'SOURCE_DATE_EPOCH': '0'}
        env['SENTRY_DATA_ROOT'] = str(root)
        for module, name in APPROVED:
            subprocess.run([sys.executable, '-m', f'experiments.paper.figures.{module}'],
                           cwd=REPO_ROOT, env=env, check=True)
            output = stage / 'figures' / name
            if not output.read_bytes().startswith(b'%PDF-'):
                raise RuntimeError(f'Invalid rendered PDF: {name}')
            index['exported_figures'][name] = {'sha256': sha256(output),
                                                'bytes': output.stat().st_size}
        for name, payload in [('evidence_index.json', index), ('result_summary.json', summary)]:
            (stage / name).write_text(json.dumps(json_payload(payload), ensure_ascii=False,
                                                 indent=2, allow_nan=False) + '\n')
        for _, name in APPROVED:
            publish_file(stage / 'figures' / name, target / name)
            print(target / name)
        for name in ('evidence_index.json', 'result_summary.json'):
            publish_file(stage / name, paper / 'data' / name)
            print(paper / 'data' / name)

if __name__ == '__main__':
    main()
