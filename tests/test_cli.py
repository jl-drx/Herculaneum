from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from herculaneum.cli import main


class CommandIntegrationTests(unittest.TestCase):
    def invoke(self, args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(args)
        return status, output.getvalue(), errors.getvalue()

    def test_publish_verify_and_detect_corruption_through_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, store = root / "source", root / "store"
            source.mkdir()
            (source / "metrics.json").write_text('{"tiles": 1}')
            shared = ["--run-id", "command-smoke", "--store-dir", str(store)]
            status, output, _ = self.invoke(["artifacts", "publish", *shared, "--input-dir", str(source)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output)["artifact_count"], 1)
            self.assertEqual(self.invoke(["artifacts", "verify", *shared])[0], 0)
            (store / "runs/command-smoke/artifacts/metrics.json").write_text("changed")
            self.assertEqual(self.invoke(["artifacts", "verify", *shared])[0], 2)

    def test_directory_links_are_not_uploaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "link").symlink_to(root)
            status, _, _ = self.invoke(["artifacts", "publish", "--run-id", "unsafe",
                                       "--store-dir", str(root / "store"), "--input-dir", str(source)])
            self.assertEqual(status, 2)
            self.assertFalse((root / "store/runs/unsafe/manifest.json").exists())

    def test_unready_job_returns_nonzero_and_readable_blockers(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = Path(temporary) / "job.json"
            spec.write_text('{"kind":"scientific-plan","ready_for_gpu":false}')
            status, output, _ = self.invoke(["job", "validate", str(spec)])
            self.assertEqual(status, 2)
            result = json.loads(output)
            self.assertFalse(result["ready"])
            self.assertTrue(result["errors"])
            self.assertFalse(result["launch_performed"])
