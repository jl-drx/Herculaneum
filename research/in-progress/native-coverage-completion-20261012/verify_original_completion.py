"""Replay the unchanged independent verifier against the new original result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.current.resolve()
    completed = root / 'work/completion-20261012'
    expected = json.loads((completed / 'ORIGINAL_VERIFICATION.json').read_text())
    verifier = root / 'code/verify_original.py'
    assert sha(verifier) == expected['implementation_sha256']
    assert sha(completed / 'qualification/ORIGINAL_SUPPORT.csv') == expected['result_csv_sha256']
    if a.output.exists():
        raise FileExistsError(a.output)
    # Only the path view changes. Original sources and both dated results stay intact.
    with tempfile.TemporaryDirectory(prefix='plinius-original-replay-') as folder:
        view = Path(folder)
        (view / 'original').symlink_to(root / 'work/original', target_is_directory=True)
        (view / 'qualification').symlink_to(completed / 'qualification', target_is_directory=True)
        subprocess.run([sys.executable, str(verifier), '--work', str(view), '--output', str(a.output.resolve())], check=True)
    actual = json.loads(a.output.read_text())
    assert actual == expected, 'Completed original numeric replay changed'


if __name__ == '__main__':
    main()
