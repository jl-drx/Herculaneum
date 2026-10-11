"""Status after a file-only restore must not depend on a vanished process record."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from status_phase52 import inspect

REAL = Path('/workspace/herculaneum-lab/reports/fiftysecond-experiment')

class RestoredStatus(unittest.TestCase):
    def prepare(self, folder):
        root = Path(folder)
        for name in ('INFERENCE_GO.json', 'INFERENCE_RESULTS.json'):
            shutil.copyfile(REAL / name, root / name)
        for source in (REAL / 'work/inference').glob('*/COMPLETE.json'):
            target = root / source.relative_to(REAL)
            target.parent.mkdir(parents=True)
            shutil.copyfile(source, target)
        return root

    def test_completed_restore_without_run_state(self):
        with tempfile.TemporaryDirectory() as folder:
            r = inspect(self.prepare(folder))
            self.assertEqual(r['status'], 'COMPLETE_RECEIPT_PRESENT')
            self.assertEqual(r['completed_maps'], 32)
            self.assertFalse(r['worker_identity_live'])
            self.assertFalse(r['local_process_state_present'])

    def test_final_receipt_cannot_hide_missing_map(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.prepare(folder)
            next((root / 'work/inference').glob('*/COMPLETE.json')).unlink()
            self.assertEqual(inspect(root)['status'], 'NO_LOCAL_PROCESS_STATE_INCOMPLETE')

    def test_32_receipts_with_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.prepare(folder)
            a, b = list((root / 'work/inference').glob('*/COMPLETE.json'))[:2]
            shutil.copyfile(a, b)
            with self.assertRaisesRegex(AssertionError, 'Duplicate'):
                inspect(root)

    def test_absent_workspace_never_reported_running(self):
        with tempfile.TemporaryDirectory() as folder:
            r = inspect(folder)
            self.assertEqual(r['status'], 'NO_LOCAL_PROCESS_STATE_INCOMPLETE')
            self.assertFalse(r['worker_identity_live'])

if __name__ == '__main__':
    if not REAL.exists():
        raise SystemExit('Restore the checksummed study52 receipts before running this integration check.')
    unittest.main()
