"""Exercise relocated inference on one retained genuine scan patch, without network."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, default=Path("/workspace/herculaneum-lab"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.lab_root / "data/experiments/pherc0800-larger1536"
    source = source_dir / "surface.npy"
    source_mask = source_dir / "geometry-mask.npy"
    checkpoint = args.lab_root / "data/models/ink9um-seed42-step075000.pth"
    provenance = json.loads((source_dir / "provenance.json").read_text())
    before = {"surface": sha(source), "mask": sha(source_mask), "checkpoint": sha(checkpoint)}
    if before["surface"] != provenance["array_sha256"] or before["mask"] != provenance["derived_outputs"]["geometry-mask.npy"]["sha256"]:
        raise ValueError("Retained input integrity failed")
    surface = np.load(source, mmap_mode="r", allow_pickle=False)
    mask = np.load(source_mask, mmap_mode="r", allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    patch = args.output_dir / "surface.npy"
    patch_mask = args.output_dir / "geometry-mask.npy"
    np.save(patch, surface[:, 704:832, 704:832], allow_pickle=False)
    np.save(patch_mask, mask[704:832, 704:832], allow_pickle=False)
    description = {"array_sha256": sha(patch), "shape_zyx": [31, 128, 128],
                   "role": "Infrastructure integration check; not an ink accuracy experiment",
                   "parent_array_sha256": before["surface"], "bbox_parent_xywh": [704, 704, 128, 128]}
    sidecar = args.output_dir / "provenance.json"
    sidecar.write_text(json.dumps(description, indent=2) + "\n")
    command = [sys.executable, str(ROOT / "baseline/run_ink_large.py"), str(patch),
               "--provenance", str(sidecar), "--mask", str(patch_mask),
               "--seed42-checkpoint", str(checkpoint), "--max-total-tiles", "1",
               "--output-dir", str(args.output_dir / "inference")]
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(args.output_dir / "cache")
    environment["MPLCONFIGDIR"] = str(args.output_dir / "matplotlib")
    positive = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    if positive.returncode != 0:
        raise RuntimeError(f"Packaged inference failed (exit {positive.returncode})")
    metadata = json.loads((args.output_dir / "inference/metadata.json").read_text())
    scored = np.load(args.output_dir / "inference/seed42-probabilities.npy", allow_pickle=False)
    evaluated = np.load(args.output_dir / "inference/seed42-evaluated-mask.npy", allow_pickle=False)
    values = scored[evaluated]
    if not values.size or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Model integration did not produce finite bounded scores")
    if metadata["models"][0]["scored_tile_count"] != 1 or not metadata["models"][0]["strict_state_match"]:
        raise ValueError("Expected one real strict-checkpoint model forward")
    invalid = dict(description, array_sha256="0" * 64)
    bad_sidecar = args.output_dir / "invalid-provenance.json"
    bad_sidecar.write_text(json.dumps(invalid))
    rejected_output = args.output_dir / "must-not-exist"
    negative_command = command.copy()
    negative_command[negative_command.index("--provenance") + 1] = str(bad_sidecar)
    negative_command[negative_command.index("--output-dir") + 1] = str(rejected_output)
    negative = subprocess.run(negative_command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    if negative.returncode != 2 or rejected_output.exists() or "does not match retained input provenance" not in negative.stderr:
        raise ValueError("Changed provenance was not rejected before output creation")
    after = {"surface": sha(source), "mask": sha(source_mask), "checkpoint": sha(checkpoint)}
    if before != after:
        raise ValueError("Original input changed during integration verification")
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "checks": ["Relocated focused-source and strict checkpoint loading", "One genuine CPU model forward",
                         "Incorrect source hash rejected before output creation", "Original inputs unchanged"],
              "scored_tiles": 1, "evaluated_pixels": int(values.size), "device": "cpu",
              "gpu_verified": False, "reading_or_translation": False}
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
