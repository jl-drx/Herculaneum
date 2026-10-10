"""Fetch the complete small 980-pixel surface within a fixed 34 MiB byte budget."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

from fetch_zarr_chunk import fetch

LAB = Path(__file__).resolve().parent
output = LAB / "data/experiments/pherc0800-surface980"
if output.exists():
    raise SystemExit("Full surface already retained; preserve it and inspect provenance")
catalog = json.loads((LAB / "data/live-metadata.min.json").read_text())
segment_id = "20251029010146"
segment = catalog["samples"]["PHerc0800"]["segments"][segment_id]
item = next(d for d in segment["data"] if d["type"] == "layers-zarr")
assert item["parameters"]["target_volume"] == "20250521135224"
root = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + item["origins"][0]["path"]
attrs, attrs_record = fetch(root + ".zattrs", 512000)
metadata, metadata_record = fetch(root + "0/.zarray", 512000)
meta = json.loads(metadata)
assert meta["shape"] == [31, 980, 980] and meta["chunks"] == [31, 128, 128]
assert meta["dtype"] == "|u1" and meta["compressor"] is None and meta["order"] == "C" and not meta.get("filters")
assert meta["fill_value"] == 0
keys = [(row, col) for row in range(8) for col in range(8)]
budget = 64 * 31 * 128 * 128
assert budget < 34 * 1024 * 1024


def download(key):
    row, col = key
    url = root + f"0/0/{row}/{col}"
    try:
        data, record = fetch(url, 31 * 128 * 128)
    except requests.HTTPError as error:
        if error.response.status_code != 404 or not getattr(error, "zarr_missing_key", False):
            raise
        # Zarr v2 explicitly represents unstored chunks with the declared fill value.
        # Authentication and other HTTP errors never become empty data.
        data = bytes(31 * 128 * 128)
        record = {"url": url, "http_status": 404, "source_absence": "S3 NoSuchKey",
                  "size_bytes": 0, "reconstruction": "Zarr v2 fill_value=0",
                  "reconstructed_sha256": hashlib.sha256(data).hexdigest()}
    if len(data) != 31 * 128 * 128:
        raise ValueError(f"Unexpected native chunk length at {key}: {len(data)}")
    return key, data, record


# Retrievals are independent; save results only after every bounded read succeeds.
with ThreadPoolExecutor(max_workers=4) as pool:
    downloaded = list(pool.map(download, keys))
output.mkdir(parents=True)
(output / "source.zattrs.json").write_bytes(attrs)
(output / "source.zarray.json").write_bytes(metadata)
surface = np.zeros((31, 980, 980), dtype=np.uint8)
records = []
for (row, col), data, record in downloaded:
    block = np.frombuffer(data, dtype=np.uint8).reshape(31, 128, 128)
    y, x = row * 128, col * 128
    height, width = min(128, 980 - y), min(128, 980 - x)
    surface[:, y:y + height, x:x + width] = block[:, :height, :width]
    filename = f"source-chunk-0-{row}-{col}.bin"
    (output / filename).write_bytes(data)
    record.update({"chunk_index_zyx": [0, row, col], "local_file": filename,
                   "local_payload_role": "reconstructed declared fill" if record.get("http_status") == 404 else "observed source"})
    etag = record.get("etag", "").strip('"')
    if len(etag) == 32 and "-" not in etag:
        record["etag_matches_payload_md5"] = hashlib.md5(data).hexdigest() == etag
    records.append(record)
small = np.load(LAB / "data/experiments/pherc0800-roi256/surface.npy", allow_pickle=False)
assert np.array_equal(surface[:, 256:512, 256:512], small)
np.save(output / "surface.npy", surface, allow_pickle=False)
result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_id": "PHerc0800",
          "target_volume_id": "20250521135224", "segment_id": segment_id,
          "pyramid_level": 0, "bbox_source_xywh": [0, 0, 980, 980],
          "shape_zyx": list(surface.shape), "dtype": str(surface.dtype),
          "voxel_spacing_zyx_um": [8.64, 8.64, 8.64],
          "physical_canvas_xy_size_mm": [8.4672, 8.4672],
          "metadata_sources": [attrs_record, metadata_record], "source_chunks": records,
          "downloaded_chunk_bytes": sum(r["size_bytes"] for r in records),
          "array_sha256": hashlib.sha256((output / "surface.npy").read_bytes()).hexdigest(),
          "matches_retained_first_roi": True,
          "missing_chunks_reconstructed_as_declared_fill": sum(r.get("http_status") == 404 for r in records),
          "nonzero_fraction": float(np.count_nonzero(surface) / surface.size),
          "status": "Observed complete small surface; invalid geometry requires separate mask"}
(output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["shape_zyx", "downloaded_chunk_bytes", "matches_retained_first_roi", "nonzero_fraction"]}, indent=2))
