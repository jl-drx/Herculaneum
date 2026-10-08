"""Artifact integrity and publication tests; no R2 account or network needed."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from herculaneum.artifacts import (
    ArtifactCollisionError,
    ArtifactIntegrityError,
    ArtifactTransportError,
    LocalArtifactStore,
    R2ArtifactStore,
    fingerprint_file,
    publish_run,
    verify_run,
)


class FakeClientError(Exception):
    def __init__(self, code, status):
        super().__init__("SENSITIVE URL AND TOKEN MUST NOT APPEAR")
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class TrackingBody(io.BytesIO):
    def __init__(self, data, sizes):
        super().__init__(data)
        self.sizes = sizes

    def read(self, size=-1):
        self.sizes.append(size)
        return super().read(size)


class FakeR2:
    """Strict fake: conditional writes, streaming reads, injected failures."""

    def __init__(self):
        self.objects = {}
        self.events = []
        self.read_sizes = []
        self.corrupt_key = None
        self.fail_put_key = None
        self.get_error = None
        self.race_key = None
        self.commit_then_fail_key = None

    def get_object(self, *, Bucket, Key):
        self.events.append(("GET", Key))
        if self.get_error:
            raise self.get_error
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey", 404)
        return {"Body": TrackingBody(self.objects[Key], self.read_sizes), "ETag": "not-a-sha256"}

    def put_object(self, *, Bucket, Key, Body, ContentLength, IfNoneMatch):
        self.events.append(("PUT", Key))
        assert IfNoneMatch == "*", "An unconditional overwrite was attempted"
        if Key == self.fail_put_key:
            raise FakeClientError("ServiceUnavailable", 503)
        if Key == self.race_key:
            self.objects[Key] = b"competing writer"
        if Key in self.objects:
            raise FakeClientError("PreconditionFailed", 412)
        parts = []
        while part := Body.read(65536):
            parts.append(part)
        data = b"".join(parts)
        assert len(data) == ContentLength
        self.objects[Key] = b"corrupted upload" if Key == self.corrupt_key else data
        if Key == self.commit_then_fail_key:
            raise FakeClientError("ServiceUnavailable", 503)
        return {"ETag": "also-not-a-sha256"}


class RecordingLocal(LocalArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.writes = []
        self.fail_key = None

    def put_if_absent(self, key, stream, size_bytes):
        self.writes.append(key)
        if key == self.fail_key:
            raise OSError("Simulated interruption")
        return super().put_if_absent(key, stream, size_bytes)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = self.root / "a.dat"
        self.b = self.root / "b.dat"
        self.a.write_bytes(b"first artifact")
        self.b.write_bytes(b"second artifact")
        self.inputs = {"outputs/a.dat": self.a, "b.dat": self.b}

    def local(self):
        return RecordingLocal(self.root / "store")

    def r2(self):
        client = FakeR2()
        return R2ArtifactStore(client, "test-bucket"), client

    def test_local_roundtrip_hash_length_and_manifest_last(self):
        store = self.local()
        manifest = publish_run(store, "run1", self.inputs, metadata={"git_commit": "abc"})
        self.assertEqual(manifest, verify_run(store, "run1"))
        self.assertEqual(store.writes[0], "runs/run1/plan.json")
        self.assertEqual(store.writes[-1], "runs/run1/manifest.json")
        self.assertEqual(len(store.writes), 4)
        for record in manifest["artifacts"]:
            data = self.inputs[record["path"]].read_bytes()
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(record["size_bytes"], len(data))

    def test_exact_retry_verifies_without_writes(self):
        store = self.local()
        first = publish_run(store, "run1", self.inputs)
        store.writes.clear()
        self.assertEqual(first, publish_run(store, "run1", dict(reversed(list(self.inputs.items())))))
        self.assertEqual(store.writes, [])

    def test_interruption_resume_and_no_partial_completion(self):
        store = self.local()
        store.fail_key = "runs/run1/artifacts/outputs/a.dat"
        with self.assertRaises(OSError):
            publish_run(store, "run1", self.inputs)
        self.assertTrue((store.root / "runs/run1/artifacts/b.dat").exists())
        self.assertFalse((store.root / "runs/run1/manifest.json").exists())
        with self.assertRaises(ArtifactIntegrityError):
            verify_run(store, "run1")
        store.fail_key = None
        store.writes.clear()
        publish_run(store, "run1", self.inputs)
        self.assertEqual(store.writes, ["runs/run1/artifacts/outputs/a.dat", "runs/run1/manifest.json"])
        verify_run(store, "run1")

    def test_interrupted_plan_rejects_changed_artifact_set(self):
        store = self.local()
        store.fail_key = "runs/run1/artifacts/b.dat"
        with self.assertRaises(OSError):
            publish_run(store, "run1", self.inputs)
        store.writes.clear()
        with self.assertRaises(ArtifactCollisionError):
            publish_run(store, "run1", {"new.dat": self.a})
        self.assertEqual(store.writes, [])
        self.assertFalse((store.root / "runs/run1/manifest.json").exists())

    def test_completed_run_rejects_changed_input_or_metadata(self):
        store = self.local()
        publish_run(store, "run1", self.inputs, metadata={"seed": 42})
        store.writes.clear()
        with self.assertRaises(ArtifactCollisionError):
            publish_run(store, "run1", self.inputs, metadata={"seed": 43})
        self.a.write_bytes(b"new bytes")
        with self.assertRaises(ArtifactCollisionError):
            publish_run(store, "run1", self.inputs, metadata={"seed": 42})
        self.assertEqual(store.writes, [])

    def test_completed_remote_corruption_is_not_repaired_or_hidden(self):
        store = self.local()
        publish_run(store, "run1", self.inputs)
        (store.root / "runs/run1/artifacts/b.dat").write_bytes(b"corruption")
        store.writes.clear()
        with self.assertRaises(ArtifactIntegrityError):
            verify_run(store, "run1")
        with self.assertRaises(ArtifactIntegrityError):
            publish_run(store, "run1", self.inputs)
        self.assertEqual(store.writes, [])

    def test_missing_completed_artifact_is_not_reuploaded(self):
        store = self.local()
        publish_run(store, "run1", self.inputs)
        (store.root / "runs/run1/artifacts/b.dat").unlink()
        store.writes.clear()
        with self.assertRaises(ArtifactIntegrityError):
            publish_run(store, "run1", self.inputs)
        self.assertEqual(store.writes, [])

    def test_atomic_local_create_does_not_overwrite_existing_bytes(self):
        store = self.local()
        self.assertTrue(store.put_if_absent("object", io.BytesIO(b"one"), 3))
        self.assertFalse(store.put_if_absent("object", io.BytesIO(b"two"), 3))
        self.assertEqual((store.root / "object").read_bytes(), b"one")

    def test_local_staging_cleanup_on_stream_interruption(self):
        store = self.local()

        class BrokenStream:
            def read(self, size):
                raise OSError("interrupted source")

        with self.assertRaises(OSError):
            store.put_if_absent("folder/object", BrokenStream(), 3)
        self.assertEqual(list((store.root / "folder").iterdir()), [])

    def test_local_short_source_is_not_published(self):
        store = self.local()
        with self.assertRaises(ArtifactIntegrityError):
            store.put_if_absent("folder/object", io.BytesIO(b"short"), 99)
        self.assertEqual(list((store.root / "folder").iterdir()), [])

    def test_empty_artifact_roundtrip(self):
        self.a.write_bytes(b"")
        store = self.local()
        record = publish_run(store, "empty", {"empty": self.a})["artifacts"][0]
        self.assertEqual(record["size_bytes"], 0)
        self.assertEqual(record["sha256"], hashlib.sha256(b"").hexdigest())
        verify_run(store, "empty")

    def test_unsafe_paths_and_empty_artifact_set_fail(self):
        store = self.local()
        for run_id in ("../escape", "/absolute", "a/b", ".", ".."):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                publish_run(store, run_id, self.inputs)
        for name in ("../escape", "/absolute", "a//b", "a/../b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                publish_run(store, "run1", {name: self.a})
        with self.assertRaises(ValueError):
            publish_run(store, "run1", {})
        self.assertEqual(store.writes, [])

    def test_symlink_escape_rejected(self):
        store = self.local()
        outside = self.root / "outside"
        outside.mkdir()
        (store.root / "runs").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactIntegrityError):
            publish_run(store, "run1", self.inputs)
        self.assertEqual(list(outside.iterdir()), [])

    def test_plan_corruption_breaks_completed_verification(self):
        store = self.local()
        publish_run(store, "run1", self.inputs)
        (store.root / "runs/run1/plan.json").write_text("{}")
        with self.assertRaises(ArtifactIntegrityError):
            verify_run(store, "run1")

    def test_forged_manifest_cannot_reference_another_run(self):
        store = self.local()
        manifest = publish_run(store, "run1", self.inputs)
        manifest["artifacts"][0]["key"] = "runs/other/artifacts/b.dat"
        (store.root / "runs/run1/manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ArtifactIntegrityError):
            verify_run(store, "run1")

    def test_fingerprint_uses_actual_bytes(self):
        self.assertEqual(fingerprint_file(self.a), {"sha256": hashlib.sha256(self.a.read_bytes()).hexdigest(), "size_bytes": self.a.stat().st_size})

    def test_r2_streams_readback_and_manifest_is_last_put(self):
        store, client = self.r2()
        publish_run(store, "run1", self.inputs)
        puts = [key for method, key in client.events if method == "PUT"]
        self.assertEqual(puts[-1], "runs/run1/manifest.json")
        for key in puts:
            index = client.events.index(("PUT", key))
            self.assertIn(("GET", key), client.events[index + 1 :])
        self.assertTrue(all(0 < size <= 1024 * 1024 for size in client.read_sizes))
        verify_run(store, "run1")

    def test_r2_corrupt_upload_blocks_completion(self):
        store, client = self.r2()
        client.corrupt_key = "runs/run1/artifacts/b.dat"
        with self.assertRaises(ArtifactIntegrityError):
            publish_run(store, "run1", self.inputs)
        self.assertNotIn("runs/run1/manifest.json", client.objects)
        client.corrupt_key = None
        with self.assertRaises(ArtifactCollisionError):
            publish_run(store, "run1", self.inputs)

    def test_r2_failed_put_is_secret_safe_and_resumable(self):
        store, client = self.r2()
        client.fail_put_key = "runs/run1/artifacts/outputs/a.dat"
        with self.assertRaises(ArtifactTransportError) as context:
            publish_run(store, "run1", self.inputs)
        self.assertNotIn("SENSITIVE", str(context.exception))
        self.assertNotIn("runs/run1/manifest.json", client.objects)
        client.fail_put_key = None
        publish_run(store, "run1", self.inputs)
        verify_run(store, "run1")

    def test_r2_uncertain_completion_is_reconciled_without_overwrite(self):
        store, client = self.r2()
        client.commit_then_fail_key = "runs/run1/manifest.json"
        with self.assertRaises(ArtifactTransportError):
            publish_run(store, "run1", self.inputs)
        self.assertIn("runs/run1/manifest.json", client.objects)
        verify_run(store, "run1")
        client.events.clear()
        publish_run(store, "run1", self.inputs)
        self.assertTrue(all(method == "GET" for method, key in client.events))

    def test_r2_stream_failure_sanitizes_error_and_closes_body(self):
        store, client = self.r2()

        class BrokenBody(io.BytesIO):
            def read(self, size=-1):
                raise RuntimeError("SENSITIVE URL AND TOKEN")

        body = BrokenBody()
        with patch.object(client, "get_object", return_value={"Body": body}):
            with self.assertRaises(ArtifactTransportError) as context:
                publish_run(store, "run1", self.inputs)
        self.assertTrue(body.closed)
        self.assertNotIn("SENSITIVE", str(context.exception))

    def test_r2_read_denied_and_missing_bucket_are_not_absent_objects(self):
        store, client = self.r2()
        for code, status in (("AccessDenied", 403), ("NoSuchBucket", 404)):
            client.get_error = FakeClientError(code, status)
            with self.subTest(code=code), self.assertRaises(ArtifactTransportError) as context:
                publish_run(store, "run1", self.inputs)
            self.assertNotIn("SENSITIVE", str(context.exception))
        self.assertFalse(any(method == "PUT" for method, key in client.events))

    def test_r2_malformed_sdk_error_is_still_secret_safe(self):
        store, client = self.r2()
        error = RuntimeError("SENSITIVE URL AND TOKEN")
        error.response = None
        client.get_error = error
        with self.assertRaises(ArtifactTransportError) as context:
            publish_run(store, "run1", self.inputs)
        self.assertNotIn("SENSITIVE", str(context.exception))
        self.assertTrue(context.exception.__suppress_context__)

    def test_r2_conditional_race_never_overwrites_competing_object(self):
        store, client = self.r2()
        client.race_key = "runs/run1/artifacts/b.dat"
        with self.assertRaises(ArtifactCollisionError):
            publish_run(store, "run1", self.inputs)
        self.assertEqual(client.objects[client.race_key], b"competing writer")
        self.assertNotIn("runs/run1/manifest.json", client.objects)

    def test_r2_size_limit_fails_before_any_request(self):
        store, client = self.r2()
        store.max_artifact_size = 4
        with self.assertRaises(ValueError):
            publish_run(store, "run1", self.inputs)
        self.assertEqual(client.events, [])

    def test_r2_prefix_and_exact_retry(self):
        client = FakeR2()
        store = R2ArtifactStore(client, "test-bucket", prefix="herculaneum/private")
        first = publish_run(store, "run1", self.inputs)
        client.events.clear()
        self.assertEqual(first, publish_run(store, "run1", self.inputs))
        self.assertTrue(all(method == "GET" for method, key in client.events))
        self.assertTrue(all(key.startswith("herculaneum/private/") for key in client.objects))

    def test_r2_credentials_never_fall_back_to_aws(self):
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "AWS_SECRET_MARKER", "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_MARKER"}, clear=True):
            with self.assertRaises(ValueError) as context:
                R2ArtifactStore.from_env()
        self.assertIn("R2_ACCESS_KEY_ID", str(context.exception))
        self.assertNotIn("AWS_SECRET_MARKER", str(context.exception))

    def test_sdk_initialization_ignores_unrelated_aws_profile_and_makes_no_connection(self):
        try:
            import botocore  # noqa: F401
        except ImportError:
            self.skipTest("Optional botocore SDK is not installed")
        environment = {"R2_ACCOUNT_ID": "0" * 32, "R2_BUCKET": "test-bucket", "R2_ACCESS_KEY_ID": "dummy-r2-key", "R2_SECRET_ACCESS_KEY": "dummy-r2-secret", "AWS_PROFILE": "nonexistent-unrelated-aws-profile", "AWS_DEFAULT_PROFILE": "also-nonexistent-aws-profile"}
        with patch.dict(os.environ, environment, clear=True), patch("socket.socket.connect", side_effect=AssertionError("Unexpected network connection")):
            store = R2ArtifactStore.from_env()
            self.assertEqual(store.client.meta.endpoint_url, "https://" + "0" * 32 + ".r2.cloudflarestorage.com")
            self.assertEqual(store.client.meta.config.retries["total_max_attempts"], 3)
            store.client.close()

    def _sign_r2_put_headers(self):
        try:
            from botocore.awsrequest import AWSRequest
        except ImportError:
            self.skipTest("Optional botocore SDK is not installed")
        store = R2ArtifactStore.from_env()
        try:
            request = AWSRequest(
                method="PUT",
                url=f"{store.client.meta.endpoint_url}/{store.bucket}/offline-test.bin",
                data=b"offline signing payload",
            )
            # Sign a real request locally without invoking the transport.
            store.client._request_signer.sign("PutObject", request)
            return {name.lower(): value for name, value in request.headers.items()}
        finally:
            store.client.close()

    def test_r2_session_token_is_signed_and_changes_authorization(self):
        try:
            import botocore.auth  # noqa: F401
        except ImportError:
            self.skipTest("Optional botocore SDK is not installed")
        environment = {
            "R2_ACCOUNT_ID": "0" * 32,
            "R2_BUCKET": "test-bucket",
            "R2_ACCESS_KEY_ID": "dummy-r2-key",
            "R2_SECRET_ACCESS_KEY": "dummy-r2-secret",
            "R2_SESSION_TOKEN": "dummy-r2-session-token",
            "AWS_SESSION_TOKEN": "unrelated-aws-session-token",
            "AWS_PROFILE": "nonexistent-unrelated-aws-profile",
            "AWS_DEFAULT_PROFILE": "also-nonexistent-aws-profile",
        }
        fixed_time = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        with patch.dict(os.environ, environment, clear=True), patch(
            "socket.socket.connect", side_effect=AssertionError("Unexpected network connection")
        ), patch("botocore.auth.get_current_datetime", return_value=fixed_time):
            first = self._sign_r2_put_headers()
            os.environ["R2_SESSION_TOKEN"] = "different-dummy-r2-session-token"
            second = self._sign_r2_put_headers()
        self.assertEqual(first["x-amz-security-token"], "dummy-r2-session-token")
        self.assertEqual(second["x-amz-security-token"], "different-dummy-r2-session-token")
        signed_headers = first["authorization"].split("SignedHeaders=", 1)[1].split(",", 1)[0].split(";")
        self.assertIn("x-amz-security-token", signed_headers)
        self.assertEqual(first["x-amz-date"], "20261008T120000Z")
        other_headers = lambda headers: {
            name: value for name, value in headers.items()
            if name not in {"authorization", "x-amz-security-token"}
        }
        self.assertEqual(other_headers(first), other_headers(second))
        self.assertNotEqual(first["authorization"], second["authorization"])
        self.assertIn("Credential=dummy-r2-key/", first["authorization"])

    def test_r2_long_lived_pair_ignores_unrelated_aws_session_token(self):
        environment = {
            "R2_ACCOUNT_ID": "0" * 32,
            "R2_BUCKET": "test-bucket",
            "R2_ACCESS_KEY_ID": "dummy-r2-key",
            "R2_SECRET_ACCESS_KEY": "dummy-r2-secret",
            "AWS_SESSION_TOKEN": "unrelated-aws-session-token",
            "AWS_PROFILE": "nonexistent-unrelated-aws-profile",
            "AWS_DEFAULT_PROFILE": "also-nonexistent-aws-profile",
        }
        for token in (None, ""):
            with self.subTest(r2_session_token="unset" if token is None else "empty"):
                values = dict(environment)
                if token is not None:
                    values["R2_SESSION_TOKEN"] = token
                with patch.dict(os.environ, values, clear=True), patch(
                    "socket.socket.connect", side_effect=AssertionError("Unexpected network connection")
                ):
                    headers = self._sign_r2_put_headers()
                self.assertNotIn("x-amz-security-token", headers)
                signed_headers = headers["authorization"].split("SignedHeaders=", 1)[1].split(",", 1)[0].split(";")
                self.assertNotIn("x-amz-security-token", signed_headers)
                self.assertIn("Credential=dummy-r2-key/", headers["authorization"])


if __name__ == "__main__":
    unittest.main()
