"""Verify retained adapters and upstream modules without acquiring scan data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"
VENDOR = BASELINE / "vendor/villa-ink-baseline"


def verify(root, path, expected):
    target = (root / path).resolve(strict=True)
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Manifest path leaves its source directory")
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {path}")


def main():
    adapters = json.loads((BASELINE / "source-manifest.json").read_text())
    upstream = json.loads((VENDOR / "source-manifest.json").read_text())
    for entry in adapters["scripts"]:
        verify(BASELINE, entry["path"], entry["sha256"])
    for entry in upstream["files"]:
        verify(VENDOR, entry["path"], entry["sha256"])
    verify(VENDOR, "UPSTREAM-LICENSE.txt", upstream["license_sha256"])
    additional = VENDOR / "ADDITIONAL-LICENSES"
    notices = json.loads((additional / "sources.json").read_text())
    for entry in notices["sources"]:
        verify(additional, entry["local_file"], entry["sha256"])
    print(f"Verified {len(adapters['scripts'])} unchanged adapters, {len(upstream['files'])} upstream modules, and {1 + len(notices['sources'])} licenses.")


if __name__ == "__main__":
    main()
