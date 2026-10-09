"""Package retained PHerc343 review files without loading models or scan arrays."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REPOSITORY = "jl-drx/Herculaneum"
SNAPSHOT_SHA = "f85c6ab579da054d3ca94c3b7e4b58699f7c6a3a"
MANIFEST_SHA256 = "b5b4904eda36f91ce243af24fbfcd6aebbd64072be7d3f88340dcdd3151109c9"
MANIFEST_PATH = "research/workspace/MANIFEST.json"
BUNDLE_PATH = "research/workspace/bundles"
RESULT_BUNDLE = "pherc343-results-001.tar.gz"
REFERENCE_BUNDLE = "pherc343-reference-001.tar.gz"
MASK_BUNDLE = "pherc343-inputs-004.tar.gz"
MASK_MEMBER = "reports/fortysixth-experiment/work/valid.npy"
SELECTED_BUNDLES = (RESULT_BUNDLE, REFERENCE_BUNDLE, MASK_BUNDLE)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, record):
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]):
        raise ValueError(f"File checksum/length mismatch: {path.name}")


def git(snapshot, *args, env=None):
    result = subprocess.run(
        ["git", "-C", str(snapshot), *args], env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        # A fetch may use a credential header. Never echo captured diagnostics.
        raise RuntimeError(
            f"Git operation failed (exit {result.returncode}); "
            "check repository access, Actions policy and the LFS allowance."
        )
    return result.stdout.strip()


def load_plan(snapshot):
    if git(snapshot, "rev-parse", "HEAD") != SNAPSHOT_SHA:
        raise ValueError("Snapshot checkout is not the pinned commit")
    manifest_path = snapshot / MANIFEST_PATH
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Snapshot manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    bundles = {}
    for name in SELECTED_BUNDLES:
        matches = [b for b in manifest["bundles"] if b["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicated bundle: {name}")
        record = matches[0]
        expected = ("version https://git-lfs.github.com/spec/v1\n"
                    f"oid sha256:{record['sha256']}\nsize {record['bytes']}\n")
        if (snapshot / BUNDLE_PATH / name).read_text() != expected:
            raise ValueError(f"Bundle must remain its exact LFS pointer: {name}")
        bundles[name] = record
    masks = [r for r in manifest["files"] if r["path"] == MASK_MEMBER]
    if len(masks) != 1 or masks[0]["bundle"] != MASK_BUNDLE:
        raise ValueError("The pinned mask is missing or belongs to another bundle")
    return bundles, masks[0]


def object_cache(snapshot):
    for line in git(snapshot, "lfs", "env").splitlines():
        if line.startswith("LocalMediaDir="):
            return Path(line.removeprefix("LocalMediaDir="))
    raise RuntimeError("Git LFS did not report its local object cache")


def object_path(cache, record):
    oid = record["sha256"]
    return cache / oid[:2] / oid[2:4] / oid


def request_lfs(url, headers, data=None):
    try:
        return urlopen(Request(url, data=data, headers=headers), timeout=30)
    except HTTPError as exc:
        # Signed download URLs and authorization headers must not reach logs.
        raise RuntimeError(f"LFS request returned HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("LFS request failed or timed out") from None


def fetch(snapshot, bundles):
    remote = git(snapshot, "remote", "get-url", "origin")
    if remote.removesuffix(".git") != f"https://github.com/{REPOSITORY}":
        raise ValueError("Unexpected snapshot remote")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("A read-only GITHUB_TOKEN is required for the private LFS fetch")
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    batch = {"operation": "download", "transfers": ["basic"],
             "objects": [{"oid": b["sha256"], "size": b["bytes"]}
                         for b in bundles.values()]}
    headers = {"Authorization": f"Basic {encoded}",
               "Accept": "application/vnd.git-lfs+json",
               "Content-Type": "application/vnd.git-lfs+json"}
    endpoint = f"https://github.com/{REPOSITORY}.git/info/lfs/objects/batch"
    with request_lfs(endpoint, headers, json.dumps(batch).encode()) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Unexpectedly large LFS batch response")
    objects = json.loads(raw)["objects"]
    expected = {b["sha256"]: b for b in bundles.values()}
    if len(objects) != len(expected) or {o["oid"] for o in objects} != set(expected):
        raise ValueError("LFS response does not match the three requested objects")
    cache = object_cache(snapshot)
    for item in objects:
        record = expected[item["oid"]]
        if "error" in item or item["size"] != record["bytes"]:
            raise RuntimeError(f"LFS object unavailable or mismatched: {record['name']}")
        action = item.get("actions", {}).get("download")
        if not action or urlsplit(action["href"]).scheme != "https":
            raise ValueError("LFS response lacks an HTTPS download action")
        destination = object_path(cache, record)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".download-partial")
            created = False
            try:
                # Only the storage service's supplied headers accompany this URL.
                # The repository token is confined to the batch request above.
                with request_lfs(action["href"], action.get("header", {})) as source:
                    with partial.open("xb") as target:
                        created = True
                        size = 0
                        while chunk := source.read(1024 * 1024):
                            size += len(chunk)
                            if size > record["bytes"]:
                                raise ValueError("LFS download exceeds its pinned size")
                            target.write(chunk)
                verify_file(partial, record)
                partial.rename(destination)
            finally:
                if created:
                    partial.unlink(missing_ok=True)
        verify_file(destination, record)
        print(json.dumps({"verified_object": record["name"], "bytes": record["bytes"]}), flush=True)
    print(json.dumps({"status": "PASS", "operation": "selective LFS fetch",
                      "objects": len(bundles),
                      "bytes": sum(b["bytes"] for b in bundles.values())}))


def extract_mask(archive, destination, record):
    """Stream one regular member. Never extract other input-bundle members."""
    found = False
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name != MASK_MEMBER:
                continue
            if found or not member.isfile() or member.size != record["bytes"]:
                raise ValueError("Invalid or duplicated valid.npy archive member")
            found = True
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    if not found:
        raise ValueError("valid.npy is missing from the input bundle")
    verify_file(destination, record)


def package(snapshot, output, bundles, mask):
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new directory")
    cache = object_cache(snapshot)
    objects = {name: object_path(cache, record) for name, record in bundles.items()}
    for name, source in objects.items():
        verify_file(source, bundles[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pherc343-review-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name in (RESULT_BUNDLE, REFERENCE_BUNDLE):
            shutil.copyfile(objects[name], staging / name)
            verify_file(staging / name, bundles[name])
        extract_mask(objects[MASK_BUNDLE], staging / "valid.npy", mask)
        payloads = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                    for p in sorted(staging.iterdir())]
        provenance = {
            "repository": REPOSITORY,
            "snapshot_commit": SNAPSHOT_SHA,
            "snapshot_manifest_sha256": MANIFEST_SHA256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_lfs_objects": list(bundles.values()),
            "valid_npy_source": mask,
            "payloads": payloads,
            "scope": "Retained reference/results archives and the valid mask only. "
                     "No inference, model loading, live-process access or Hecate work.",
        }
        (staging / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
        checksum_lines = [f"{sha256(p)}  {p.name}\n" for p in sorted(staging.iterdir())]
        (staging / "SHA256SUMS").write_text("".join(checksum_lines))
        staging.rename(output)
    print(json.dumps({"status": "PASS", "operation": "package", "directory": str(output),
                      "files": sorted(p.name for p in output.iterdir()),
                      "payload_bytes": sum(p["bytes"] for p in payloads)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "package"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    bundles, mask = load_plan(snapshot)
    if args.command == "fetch":
        fetch(snapshot, bundles)
    else:
        if args.output is None:
            parser.error("package requires --output")
        package(snapshot, args.output.absolute(), bundles, mask)


if __name__ == "__main__":
    main()
