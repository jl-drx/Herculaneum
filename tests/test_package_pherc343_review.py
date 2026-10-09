"""Check selective extraction and integrity failures before artifact publication."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/package_pherc343_review.py"
SPEC = importlib.util.spec_from_file_location("package_pherc343_review", SCRIPT)
PACKAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGER)


class PackagingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mask = b"a fixture, not a model input"
        self.record = {"bytes": len(self.mask), "sha256": hashlib.sha256(self.mask).hexdigest()}

    def archive(self, duplicate=False, symlink=False):
        archive = self.root / "input.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            # Unrequested input files, even unsafe names, must never be extracted.
            other = tarfile.TarInfo("../../outside"); other.size = 7
            tar.addfile(other, io.BytesIO(b"ignored"))
            other = tarfile.TarInfo("work/hecate-plain.zarr/chunk"); other.size = 7
            tar.addfile(other, io.BytesIO(b"ignored"))
            for _ in range(2 if duplicate else 1):
                item = tarfile.TarInfo(PACKAGER.MASK_MEMBER)
                if symlink:
                    item.type = tarfile.SYMTYPE; item.linkname = "/outside"
                    tar.addfile(item)
                else:
                    item.size = len(self.mask); tar.addfile(item, io.BytesIO(self.mask))
        return archive

    def test_only_mask_is_extracted(self):
        archive = self.archive(); destination = self.root / "valid.npy"
        PACKAGER.extract_mask(archive, destination, self.record)
        self.assertEqual(destination.read_bytes(), self.mask)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"input.tar.gz", "valid.npy"})

    def test_wrong_mask_hash_rejected(self):
        record = dict(self.record, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            PACKAGER.extract_mask(self.archive(), self.root / "valid.npy", record)

    def test_duplicate_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            PACKAGER.extract_mask(self.archive(duplicate=True), self.root / "valid.npy", self.record)

    def test_mask_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            PACKAGER.extract_mask(self.archive(symlink=True), self.root / "valid.npy", self.record)

    def test_package_failure_does_not_publish_partial_output(self):
        archive = self.archive()
        source = {"bytes": archive.stat().st_size, "sha256": PACKAGER.sha256(archive)}
        cache = self.root / "cache"
        target = PACKAGER.object_path(cache, source); target.parent.mkdir(parents=True)
        target.write_bytes(archive.read_bytes())
        bundles = {name: dict(source, name=name) for name in PACKAGER.SELECTED_BUNDLES}
        output = self.root / "output"
        with patch.object(PACKAGER, "object_cache", return_value=cache):
            with self.assertRaisesRegex(ValueError, "checksum"):
                PACKAGER.package(self.root, output, bundles, dict(self.record, sha256="0" * 64))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".pherc343-review-*")), [])

    def test_existing_output_is_preserved(self):
        output = self.root / "output"; output.mkdir(); (output / "mine").write_text("keep")
        with self.assertRaisesRegex(ValueError, "new directory"):
            PACKAGER.package(self.root, output, {}, self.record)
        self.assertEqual((output / "mine").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
