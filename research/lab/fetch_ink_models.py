"""Download two pinned official model checkpoints, verifying their LFS SHA-256."""
import hashlib
import json
from pathlib import Path
import tempfile

import requests

LAB = Path(__file__).resolve().parent
root = LAB / "data/models"
manifest = json.loads((root / "ink_9um.hf-manifest.json").read_text())
revision = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"
assert manifest["sha"] == revision
records = []
for seed in (42, 43):
    name = f"hybrid_3d2d-seed{seed}/step-075000.pth"
    metadata = next(s for s in manifest["siblings"] if s["rfilename"] == name)
    expected = metadata["lfs"]["sha256"]
    size = metadata["lfs"]["size"]
    target = root / f"ink9um-seed{seed}-step075000.pth"
    if target.exists():
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if digest != expected:
            raise SystemExit(f"Existing checkpoint mismatch, preserved for review: {target.name}")
        print(f"Verified retained seed{seed} checkpoint")
        records.append({"seed": seed, "file": target.name, "sha256": digest, "size_bytes": size})
        continue
    url = f"https://huggingface.co/scrollprize/ink_9um/resolve/{revision}/{name}"
    temporary = None
    try:
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            if not response.url.startswith("https://"):
                raise ValueError("Non-HTTPS checkpoint redirect")
            digest, count = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                temporary = Path(stream.name)
                for block in response.iter_content(1024 * 1024):
                    count += len(block)
                    if count > size:
                        raise ValueError("Checkpoint exceeded declared size")
                    stream.write(block)
                    digest.update(block)
        if count != size or digest.hexdigest() != expected:
            raise ValueError("Checkpoint size or SHA-256 differs from official LFS metadata")
        temporary.replace(target)
        records.append({"seed": seed, "file": target.name, "sha256": digest.hexdigest(),
                        "size_bytes": count, "source_url": url})
        print(f"Downloaded and verified seed{seed}: {count} bytes")
    except Exception as exc:
        if temporary and temporary.exists():
            temporary.unlink()
        # Signed redirect query strings are omitted from error messages.
        raise SystemExit(f"Checkpoint download failed for seed{seed}: {type(exc).__name__}") from None
(root / "verified-checkpoints.json").write_text(json.dumps({"model_revision": revision, "checkpoints": records}, indent=2) + "\n")
