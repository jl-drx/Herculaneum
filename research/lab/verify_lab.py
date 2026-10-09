"""Exercise installed tools on published review images, never infer ancient text."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import imagecodecs  # noqa: F401; TIFF LZW round trip also exercises this dependency.
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import tifffile
import zarr

LAB = Path(__file__).resolve().parent
manifest = json.loads((LAB / "data" / "manifest.json").read_text())
run = Path(tempfile.mkdtemp(prefix="verification-", dir=LAB / "reports"))
checks = []
for asset in manifest["assets"]:
    source = LAB / asset["local_path"]
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual == asset["sha256"], f"Source checksum mismatch: {source.name}"
    checks.append(f"Verified original SHA-256: {source.name}")
    command = [sys.executable, str(LAB / "inspect_image.py"), str(source),
               "--output-dir", str(run / asset["id"]),
               "--source-url", asset["source_url"], "--sha256", asset["sha256"],
               "--provenance-note", asset["description"]]
    if asset.get("review_crop_xywh"):
        command += ["--crop", *map(str, asset["review_crop_xywh"])]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((run / asset["id"] / "metadata.json").read_text())
    assert metadata["source"]["sha256"] == asset["sha256"]
    for output in metadata["outputs"].values():
        with Image.open(run / asset["id"] / output["path"]) as view:
            assert list(view.size) == output["size_wh"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == actual
    checks.append(f"Generated previews and provenance without changing source: {source.name}")

source = LAB / manifest["assets"][0]["local_path"]
with Image.open(source) as image:
    array = np.array(image.convert("L"))
tiff_path = run / "reference-roundtrip.tif"
tifffile.imwrite(tiff_path, array, compression="lzw")
assert np.array_equal(tifffile.imread(tiff_path), array)
checks.append("TIFF LZW encode/decode exactly preserves reference pixels")
zarr_path = run / "reference-roundtrip.zarr"
store = zarr.open_array(str(zarr_path), mode="w", shape=array.shape,
                        dtype=array.dtype, chunks=(64, 64), zarr_format=2)
store[:] = array
assert np.array_equal(zarr.open_array(str(zarr_path), mode="r")[:], array)
checks.append("Chunked local Zarr write/read exactly preserves reference pixels")
filtered = gaussian_filter(array.astype(np.float32), sigma=1)
assert filtered.shape == array.shape and np.isfinite(filtered).all()
checks.append("SciPy processes a real reference array with finite, shape-preserving output")

for extra, label in [(["--sha256", "0" * 64], "Incorrect checksum"),
                     (["--crop", "-1", "0", "10", "10"], "Out-of-bounds crop")]:
    output = run / label.lower().replace(" ", "-")
    result = subprocess.run([sys.executable, str(LAB / "inspect_image.py"), str(source),
                             "--output-dir", str(output), *extra], capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()
    checks.append(f"{label} rejected before writing outputs")

packages = ["numpy", "scipy", "Pillow", "matplotlib", "tifffile", "scikit-image",
            "zarr", "fsspec", "s3fs", "imagecodecs", "jupyterlab"]
result = {"status": "passed", "checks": checks, "check_count": len(checks),
          "python": sys.version.split()[0],
          "packages": {p: importlib.metadata.version(p) for p in packages},
          "run_directory": str(run),
          "scope": "Published reference-image inspection and local data I/O only",
          "unverified": ["Remote CT access", "GPU inference", "Ink detection accuracy",
                         "New transcription or translation", "Prize eligibility"]}
(run / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
(LAB / "reports" / "latest-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"PASS: {len(checks)} functional checks. Results: {run / 'verification.json'}")
