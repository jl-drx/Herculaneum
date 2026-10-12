"""Package explicit private research inputs; no downloads or model execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.archive.exists() or a.receipt.exists():
        raise FileExistsError('Use new archive and receipt paths')
    native = 'native-coverage-preparation-20261012'
    geometry = 'geometry-corridor-preparation-20261012'
    selected = {}

    def add(source, dest):
        if not source.is_file() or source.is_symlink() or dest in selected:
            raise ValueError((source, dest))
        selected[dest] = source

    allowed_native = [
        'COHORT_CANDIDATES.json', 'FIRST_TRAINING_ATTEMPT.json',
        'GEOMETRY_COVERAGE.json', 'MESH_ARRAY_REQUESTS.json',
        'MESH_METADATA_REQUESTS.json', 'MESH_SOURCES.json',
        'METADATA_REQUESTS.json', 'SOURCE_CROP_ACQUISITION.json',
        'SOURCE_CROP_GO.json', 'SOURCE_FIRST_ATTEMPT.json',
        'SOURCE_QUALIFICATION.json', 'SOURCE_REVIEW.png',
        'SOURCE_REVIEW_RECEIPT.json', 'SUPPORT_VERIFICATION.json',
        'TRAINING_FIXTURE.npz', 'TRAINING_GO.json',
        'TRAINING_QUALIFICATION.json', 'TRAINING_STATUS.json',
        'mechanics-step1.json', 'mesh-arrays', 'mesh-metadata', 'metadata',
        'qualification-v1', 'source-assembly.log', 'source-crops',
        'source-fixtures', 'support-reservation', 'support-reservation-v2',
        'teacher-metadata', 'training-check.log']
    for name in allowed_native:
        path = a.reports / native / name
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for source in files:
            if source.is_file() and '__pycache__' not in source.parts:
                add(source, 'native/' + str(source.relative_to(a.reports / native)))
    for source in sorted((a.reports / geometry).rglob('*')):
        if source.is_file() and source.suffix in {'.json', '.png'}:
            add(source, 'geometry/' + str(source.relative_to(a.reports / geometry)))
    add(a.reports / 'frontier-intake-20261012/native-contexts/PHerc0814-06.npy',
        'geometry/native-contexts/PHerc0814-06.npy')
    for packet in [native, geometry]:
        for source in sorted((a.repo / 'research/in-progress' / packet).iterdir()):
            if source.suffix == '.py' or source.name in {'README.md', 'PLAN.md'}:
                add(source, 'repo/' + str(source.relative_to(a.repo)))
    dependencies = [
        'research/in-progress/frontier-intake-20261012/registry.py',
        'research/in-progress/frontier-intake-20261012/models/MATCHED_COVERAGE.md',
        'research/in-progress/geometry-reference-20261012/check_paths.py',
        'research/in-progress/geometry-reference-20261012/CANDIDATE_PATHS.json',
        'research/in-progress/geometry-reference-20261012/PLAN.json',
        'research/in-progress/geometry-reference-20261012/README.md',
        'research/in-progress/native-common-support-20261012/verify_bundle.py']
    for rel in dependencies:
        add(a.repo / rel, 'repo/' + rel)
    for rel in [
        'frontier-intake-20261012/sources/current-catalog.raw',
        'plinius-frontier-programme-20261012/sources/reader-config.raw',
        'plinius-frontier-programme-20261012/sources/reader-evaluation-code.raw']:
        add(a.reports / rel, 'sources/' + Path(rel).name)

    with tempfile.TemporaryDirectory(prefix='plinius-coverage-package-', dir='/tmp') as folder:
        root = Path(folder)
        for name, source in selected.items():
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            assert sha(dest) == sha(source)
        (root / 'README.md').write_text(
            '# Private native-coverage preparation evidence\n\n'
            'Start with repo/research/in-progress/native-coverage-preparation-20261012/README.md.\n'
            'native/ contains bounded source bodies, meshes, frozen plans, numerical receipts,\n'
            'real training fixtures and preserved first attempts. geometry/ contains the\n'
            'single already-exposed PHerc0814 CT context and all new review images.\n'
            'sources/current-catalog.raw is gzip-encoded JSON; the Reader sources are pinned\n'
            'public code/config from commit 02c7b46987813c78f56dd99f38c5a84b65a9cd2a.\n\n'
            'The checksum manifest verifies saved bytes. The bundled verify_support.py\n'
            'recomputes all reserved candidates and physical-context proxy counts with\n'
            '--work /path/to/extracted/native --output /new/support-check.json.\n'
            'That check needs NumPy and tifffile, no model or network.\n\n'
            'Training qualification is a saved CPU mechanics result. Model initialization,\n'
            'runtime and the disposable mechanics-step1.pth are deliberately not included.\n'
            'The latter has a retained byte/SHA receipt; no claim is made that this package\n'
            'alone can redo the neural forwards or read that excluded checkpoint.\n'
            'The four real input/target/mask fixtures are included. Historical absolute\n'
            'paths in GO/receipt files identify the producer workspace, not portable paths.\n'
            'No full-cohort admission, new learning result, physically registered teacher\n'
            'labels, independent geometry truth or GPU readiness follows from this package.\n\n'
            'CT/annotation/prediction sources are attributed to the Vesuvius Challenge\n'
            'open-data catalogue and released Reader configuration. Catalogue CT licensing\n'
            'is CC BY-NC 4.0; retain source metadata and code provenance. Keep this packet\n'
            'private. It includes no credentials and launches no cloud resources.\n')
        files = {}
        for source in sorted(root.rglob('*')):
            if source.is_file():
                files[str(source.relative_to(root))] = dict(bytes=source.stat().st_size, sha256=sha(source))
        manifest = dict(schema=1, scope='Private source/mechanics/reference preparation',
                        files=files, excluded=['mechanics-step1.pth', 'released model weights', 'Python environment'],
                        paid_compute=False, new_held_out_predictions=0)
        (root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (root / 'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {k}\n" for k, v in files.items()))
        a.archive.parent.mkdir(parents=True, exist_ok=True)
        with a.archive.open('xb') as out:
            with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode='w') as tf:
                    for source in sorted(root.rglob('*')):
                        if not source.is_file():
                            continue
                        info = tf.gettarinfo(str(source), arcname=str(source.relative_to(root)))
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        info.mode = 0o644
                        with source.open('rb') as stream:
                            tf.addfile(info, stream)
        receipt = dict(status='PACKAGED_PENDING_RESTORE', archive=str(a.archive.relative_to(a.repo)),
                       bytes=a.archive.stat().st_size, sha256=sha(a.archive),
                       manifest_files=len(files), payload_bytes=sum(v['bytes'] for v in files.values()),
                       manifest_sha256=sha(root / 'MANIFEST.json'), implementation_sha256=sha(Path(__file__)),
                       includes_raw_ct_contexts=1, mechanics_checkpoint_included=False,
                       model_weights_included=False, paid_compute=False)
        a.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
