"""Retain four native surface chunks for a fixed, mesh-traceable first experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from fetch_zarr_chunk import fetch

LAB = Path(__file__).resolve().parent
output = LAB / "data/experiments/pherc0800-roi256"
if output.exists():
    raise SystemExit("Native ROI already retained; inspect its provenance rather than overwrite it")
catalog = json.loads((LAB / "data/live-metadata.min.json").read_text())
segment_id = "20251029010146"
segment = catalog["samples"]["PHerc0800"]["segments"][segment_id]
item = next(d for d in segment["data"] if d["type"] == "layers-zarr")
assert item["parameters"]["target_volume"] == "20250521135224"
root = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + item["origins"][0]["path"]
attrs_bytes, attrs_record = fetch(root + ".zattrs", 512000)
meta_bytes, meta_record = fetch(root + "0/.zarray", 512000)
meta = json.loads(meta_bytes)
assert meta["shape"] == [31, 980, 980] and meta["chunks"] == [31, 128, 128]
assert meta["compressor"] is None and meta["dtype"] == "|u1" and meta["order"] == "C"
assert not meta.get("filters")
output.mkdir(parents=True)
(output / "source.zattrs.json").write_bytes(attrs_bytes)
(output / "source.zarray.json").write_bytes(meta_bytes)
roi = np.empty((31, 256, 256), dtype=np.uint8)
records = []
for row in range(2, 4):
    for col in range(2, 4):
        data, record = fetch(root + f"0/0/{row}/{col}", 31 * 128 * 128)
        assert len(data) == 31 * 128 * 128
        block = np.frombuffer(data, dtype=np.uint8).reshape(31, 128, 128)
        rr, cc = (row - 2) * 128, (col - 2) * 128
        roi[:, rr:rr + 128, cc:cc + 128] = block
        filename = f"source-chunk-0-{row}-{col}.bin"
        (output / filename).write_bytes(data)
        record.update({"chunk_index_zyx": [0, row, col], "local_file": filename})
        etag = record.get("etag", "").strip('"')
        if len(etag) == 32 and "-" not in etag:
            record["etag_matches_payload_md5"] = hashlib.md5(data).hexdigest() == etag
        records.append(record)
np.save(output / "surface.npy", roi, allow_pickle=False)
result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_id": "PHerc0800",
          "target_volume_id": "20250521135224", "segment_id": segment_id,
          "pyramid_level": 0, "bbox_source_xywh": [256, 256, 256, 256],
          "shape_zyx": list(roi.shape), "dtype": str(roi.dtype),
          "voxel_spacing_zyx_um": [8.64, 8.64, 8.64],
          "physical_xy_size_mm": [2.21184, 2.21184],
          "metadata_sources": [attrs_record, meta_record], "source_chunks": records,
          "downloaded_chunk_bytes": sum(r["size_bytes"] for r in records),
          "array_sha256": hashlib.sha256((output / "surface.npy").read_bytes()).hexdigest(),
          "nonzero_fraction": float(np.count_nonzero(roi) / roi.size),
          "layer_mean_intensities": roi.mean(axis=(1, 2)).tolist(),
          "status": "Observed input retained; no model inference, letter reading, or translation"}
(output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["segment_id", "shape_zyx", "physical_xy_size_mm",
                                        "downloaded_chunk_bytes", "nonzero_fraction"]}, indent=2))
