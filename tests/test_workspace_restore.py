"""Restore boundaries: integrity, idempotency and preservation of existing files."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'research/tools/workspace.py'
SPEC = importlib.util.spec_from_file_location('portable_workspace', SOURCE)
WORKSPACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKSPACE)


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / 'restored'; self.dest.mkdir()

    def archive(self, name='reports/example.npy', body=b'fixture', kind=None):
        path = self.root / 'fixture.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            info = tarfile.TarInfo(name)
            if kind is not None:
                info.type = kind; info.linkname = '/outside'
                tar.addfile(info)
            else:
                info.size = len(body); tar.addfile(info, io.BytesIO(body))
        return path, [{'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}]

    def test_exact_restore_and_idempotency(self):
        archive, records = self.archive()
        self.assertEqual(WORKSPACE.restore_bundle(archive, self.dest, records), {'restored': 1, 'reused': 0})
        self.assertEqual(WORKSPACE.restore_bundle(archive, self.dest, records), {'restored': 0, 'reused': 1})

    def test_conflicting_existing_file_preserved(self):
        archive, records = self.archive(); target = self.dest / records[0]['path']
        target.parent.mkdir(); target.write_bytes(b'user data')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            WORKSPACE.restore_bundle(archive, self.dest, records)
        self.assertEqual(target.read_bytes(), b'user data')

    def test_bad_digest_not_published(self):
        archive, records = self.archive(); records[0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            WORKSPACE.restore_bundle(archive, self.dest, records)
        self.assertFalse((self.dest / records[0]['path']).exists())
        self.assertEqual(list(self.dest.rglob('*.plinius-partial')), [])

    def test_path_escape_rejected(self):
        archive, records = self.archive('../outside')
        with self.assertRaisesRegex(ValueError, 'Unsafe artifact path'):
            WORKSPACE.restore_bundle(archive, self.dest, records)
        self.assertFalse((self.root / 'outside').exists())

    def test_archive_symlink_rejected(self):
        archive, records = self.archive(kind=tarfile.SYMTYPE)
        with self.assertRaisesRegex(ValueError, 'nonregular'):
            WORKSPACE.restore_bundle(archive, self.dest, records)

    def test_existing_partial_is_preserved(self):
        archive, records = self.archive(); partial = self.dest / (records[0]['path'] + '.plinius-partial')
        partial.parent.mkdir(); partial.write_bytes(b'existing partial')
        with self.assertRaises(FileExistsError):
            WORKSPACE.restore_bundle(archive, self.dest, records)
        self.assertEqual(partial.read_bytes(), b'existing partial')


if __name__ == '__main__':
    unittest.main()
