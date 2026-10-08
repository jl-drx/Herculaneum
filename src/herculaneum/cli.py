"""Small command interface for acquisition, artifact verification and job review."""
import argparse
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="Fetch one bounded source object")
    cache_commands = cache.add_subparsers(dest="operation", required=True)
    for operation in ["fetch", "unpin"]:
        fetch = cache_commands.add_parser(operation)
        fetch.add_argument("url")
        fetch.add_argument("--metadata-sha256", required=True)
        fetch.add_argument("--expected-sha256")
        fetch.add_argument("--etag")
        fetch.add_argument("--cache-dir", type=Path, required=True)
        if operation == "fetch":
            fetch.add_argument("--pin", help="Named owner protecting an active input from eviction")
        else:
            fetch.add_argument("--owner", required=True)
        fetch.add_argument("--max-object-bytes", type=int, default=16 * 1024**2)
        fetch.add_argument("--max-cache-bytes", type=int, default=8 * 1024**3)
        fetch.add_argument("--min-free-bytes", type=int, default=8 * 1024**3)
    artifacts = commands.add_parser("artifacts", help="Publish or verify immutable artifacts")
    artifact_commands = artifacts.add_subparsers(dest="operation", required=True)
    for operation in ["publish", "verify"]:
        command = artifact_commands.add_parser(operation)
        command.add_argument("--store", choices=["local", "r2"], default="local")
        command.add_argument("--store-dir", type=Path)
        command.add_argument("--run-id", required=True)
        if operation == "publish":
            command.add_argument("--input-dir", required=True, type=Path)
            command.add_argument("--metadata", type=Path)
    job = commands.add_parser("job", help="Review a GPU specification without launching it")
    job_commands = job.add_subparsers(dest="operation", required=True)
    validate = job_commands.add_parser("validate")
    validate.add_argument("spec", type=Path)
    return root


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "cache":
            from .chunk_cache import ChunkCache, SourceObject
            source = SourceObject(args.url, args.metadata_sha256, etag=args.etag,
                                  expected_sha256=args.expected_sha256)
            with ChunkCache(args.cache_dir, max_object_bytes=args.max_object_bytes,
                            max_cache_bytes=args.max_cache_bytes, min_free_bytes=args.min_free_bytes) as cache:
                if args.operation == "unpin":
                    result = {"released": cache.unpin(source, owner=args.owner)}
                else:
                    item = cache.fetch(source, pin=args.pin)
                    result = {"path": str(item.path), "sha256": item.sha256, "size_bytes": item.size_bytes,
                              "source_id": item.source_id, "reused": item.reused}
        elif args.command == "artifacts":
            from .artifacts import LocalArtifactStore, R2ArtifactStore, publish_run, verify_run
            if args.store == "local":
                if args.store_dir is None:
                    raise ValueError("--store-dir is required for a local store")
                store = LocalArtifactStore(args.store_dir)
            else:
                store = R2ArtifactStore.from_env()
            if args.operation == "publish":
                directory = args.input_dir.resolve(strict=True)
                if not directory.is_dir():
                    raise ValueError("--input-dir must be an artifact directory")
                files = {}
                for path in sorted(directory.rglob("*")):
                    relative = path.relative_to(directory)
                    if path.is_symlink() or any(part.startswith(".") for part in relative.parts):
                        raise ValueError("Artifact directory must not contain links or hidden files")
                    if path.is_file():
                        files[relative.as_posix()] = path
                if not files:
                    raise ValueError("Artifact directory contains no files")
                metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
                if not isinstance(metadata, dict):
                    raise ValueError("Metadata must be a JSON object")
                manifest = publish_run(store, args.run_id, artifacts=files, metadata=metadata)
            else:
                manifest = verify_run(store, args.run_id)
            result = {"run_id": manifest["run_id"], "state": manifest["state"],
                      "artifact_count": len(manifest["artifacts"])}
        else:
            from .jobs import validate_gpu_job
            spec = json.loads(args.spec.read_text())
            if not isinstance(spec, dict):
                raise ValueError("A job specification must be a JSON object")
            result = validate_gpu_job(spec)
            print(json.dumps(result, indent=2))
            return 0 if result["ready"] else 2
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        # SDK and HTTP exceptions can include signed URLs or credential-bearing responses.
        # Report only the category here; backend validation messages are available in tests.
        print(f"Operation failed: {type(exc).__name__}", file=sys.stderr)
        return 2
