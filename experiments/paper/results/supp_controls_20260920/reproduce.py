"""Verify archived judge requests locally, then replay frozen server scores read-only.

Usage: python reproduce.py --data-root /path/to/experiment-data --out /external/work/audit.json
No credentials or .env are read; SSH uses the user's existing configuration.
"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_judge(root, repo):
    sources = {}

    def read(path, lines=False):
        data = path.read_bytes()
        assert not data.startswith(b'version https://git-lfs')
        sources[str(path)] = {'sha256': sha(data), 'bytes': len(data)}
        return [json.loads(x) for x in data.splitlines() if x] if lines else json.loads(data)

    manifest = read(root / 'runs/answer_check_20260909/full/non_echo_asr.json')
    answer_root = root / 'responses/answer_check_20260909/answers'
    answers = {}
    for spec in manifest['answer_files']:
        for row in read(answer_root / Path(spec['path']).name, True):
            if (row.get('response') or '').strip():
                answers[row['prompt_sha']] = row['response']
    cache = {r['key']: r['content'] for r in read(
        root / 'responses/answer_check_20260909/ortho_local/e2e/orbit_cache/judge_non_echo.jsonl', True)}
    source = repo / 'experiments/paper/rq2_robustness/judge_reuse.py'
    tree = ast.parse(source.read_text())
    sources[str(source)] = {'sha256': sha(source.read_bytes())}
    system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'NON_ECHO_SYSTEM' for t in n.targets))
    verified = {}
    for row in read(answer_root / 'answers_non_echo.jsonl', True):
        # This frozen construction appends one payload line to its control question.
        core, payload = row['prompt'].rsplit('\n', 1)
        prompt_sha = sha(row['prompt'].encode())
        assert prompt_sha == row['prompt_sha']
        data = {'question': core, 'payload': payload,
                'control_answer': answers[sha(core.encode())],
                'attacked_answer': answers[prompt_sha]}
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n\n'.join(f'### {k}\n{v}' for k, v in data.items())}]
        parts = [messages, 0, 0, 200, 'deepseek-v4-flash', {'thinking': {'type': 'disabled'}}]
        key = sha(json.dumps(parts, sort_keys=True).encode())
        verdict = re.match(r'\s*(SUCCESS|NO_CHANGE|REFUSED)\b', cache[key])
        assert verdict, prompt_sha
        verified[prompt_sha] = verdict.group(1)
    assert len(verified) == 600
    return verified, {'sources': sources, 'n_verified': len(verified),
                      'parameters': {'temperature': 0, 'seed': 0, 'max_tokens': 200}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--host', default='cpu-server')
    parser.add_argument('--ssh-config', type=Path, default=Path.home() / '.ssh/config')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    labels, provenance = verify_judge(args.data_root, repo)
    remote = (here / 'remote_replay.py').read_text()
    script = 'VERIFIED_JUDGE_LABELS = ' + repr(labels) + '\n' + remote
    proc = subprocess.run(['ssh', '-F', str(args.ssh_config), '-o', 'BatchMode=yes',
                           '-o', 'ConnectTimeout=10', args.host, 'python3 -'],
                          input=script, text=True, capture_output=True, check=True, timeout=180)
    result = json.loads(proc.stdout)
    assert result['judge_request_hash_matches']['matched_exact_requests_and_verdicts'] == 600
    result['local_judge_provenance'] = provenance
    result['remote_analysis_source_sha256'] = sha(remote.encode())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Verified and replayed 600 records without model calls:', args.out)


if __name__ == '__main__':
    main()
