"""Replay inspected submitted analyses with path-only adaptations; no inference."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('--lab', type=Path, required=True)
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--python', required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parent
src = root / 'submitted'
a.out.mkdir(exist_ok=False)
checks = []
for folder in ['spatial-persistence', 'grid-localisation']:
    for line in (src / folder / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        p = src / folder / name
        assert sha(p) == digest, str(p)
        checks.append(str(p.relative_to(src)))
for name, digest in json.loads((src / 'completion-checks/SHA256.json').read_text()).items():
    p = src / 'completion-checks' / name
    assert sha(p) == digest, str(p)
    checks.append(str(p.relative_to(src)))
inputs = a.out / 'input'
inputs.mkdir()
(inputs / 'valid.npy').symlink_to(a.lab / 'reports/fortysixth-experiment/work/valid.npy')
(inputs / 'restored').symlink_to(a.lab)
prior = a.repo / 'research/reviews/sam-pherc343-20261010/original'
replacements = {
    '/workspace/shared/pherc343-input': str(inputs),
    '/workspace/shared/pherc343-review-output': str(prior),
    '/workspace/shared/pherc343-persistence-output': str(a.out / 'spatial-persistence'),
    '/workspace/shared/pherc343-grid-output': str(a.out / 'grid-localisation'),
}
numeric_count = 0
maximum_difference = 0.0


def compare(x, y, where):
    global numeric_count, maximum_difference
    if isinstance(x, dict):
        assert x.keys() == y.keys(), where
        for k in x:
            compare(x[k], y[k], where + '/' + k)
    elif isinstance(x, list):
        assert len(x) == len(y), where
        for i, (xx, yy) in enumerate(zip(x, y)):
            compare(xx, yy, where + '/' + str(i))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        error = abs(x - y)
        assert error < 1e-10, (where, x, y)
        numeric_count += 1
        maximum_difference = max(maximum_difference, error)
    else:
        assert x == y, (where, x, y)


stages = []
for folder in ['spatial-persistence', 'grid-localisation']:
    out = a.out / folder
    out.mkdir()
    for p in (src / folder).glob('*PLAN.json'):
        shutil.copy2(p, out / p.name)
    if (src / folder / 'source').exists():
        shutil.copytree(src / folder / 'source', out / 'source')
    scripts = ['analyze.py', 'verify.py']
    if folder == 'spatial-persistence':
        scripts.append('write_report.py')
    for name in scripts:
        original = (src / folder / name).read_text()
        adapted = original
        for before, after in replacements.items():
            adapted = adapted.replace(before, after)
        p = out / name
        p.write_text(adapted)
        with (out / (name + '.log')).open('w') as log:
            result = subprocess.run([a.python, '-B', str(p)], stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, str(out / (name + '.log'))
        stages.append({'script': folder + '/' + name, 'original_sha256': sha(src / folder / name),
                       'adapted_sha256': sha(p), 'exit_code': result.returncode})
        print('PASS', folder, name, flush=True)
    compare(json.loads((out / 'metrics.json').read_text()),
            json.loads((src / folder / 'metrics.json').read_text()), folder)
    if folder == 'spatial-persistence':
        for name in ['READ_ME.txt', 'comparisons.csv', 'selections.csv']:
            assert (out / name).read_bytes() == (src / folder / name).read_bytes(), name
manifest = [{'path': str(p.relative_to(src)), 'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(src.rglob('*')) if p.is_file()]
report = {'status': 'PASS', 'submitted_files': manifest, 'submitted_hash_entries_verified': len(checks),
          'path_replacements_only': replacements, 'stages': stages,
          'numeric_values_compared': numeric_count, 'maximum_absolute_difference': maximum_difference,
          'scope': 'Full numerical replay and submitted verification scripts; no model inference or scientific accuracy claim.'}
(a.out / 'REPLAY.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['submitted_files', 'stages', 'path_replacements_only']}, indent=2))
