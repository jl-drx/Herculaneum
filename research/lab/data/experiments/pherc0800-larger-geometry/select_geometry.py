"""Refresh bounded metadata and rank PHerc0800 crops using geometry alone.

Never reads a surface/CT chunk or model output. New files stay in this directory
and reports/third-experiment/geometry-selection.json.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.ndimage import label, minimum_filter
import tifffile

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[2]
REPORT = LAB / "reports/third-experiment"
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
MESH_METADATA_BUDGET = 2 * 1024 * 1024
SURFACE_BUDGET = 128 * 1024 * 1024
OLD = json.loads((LAB / "reports/live-target-inventory.json").read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def retrieve(url, relative, maximum=128 * 1024, expected_sha=None, expected_etag=None):
    records_path = ROOT / "source-records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    path = ROOT / relative
    if path.exists():
        record = next(record for record in records if record["local_file"] == relative)
        data = path.read_bytes()
        assert digest(data) == record["sha256"] and record["url"] == url
        return data, record
    remaining = MESH_METADATA_BUDGET - sum(record["size_bytes"] for record in records)
    maximum = min(maximum, remaining)
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=45) as response:
        assert response.status == 200
        advertised = response.headers.get("Content-Length")
        if advertised and int(advertised) > maximum:
            raise ValueError("Object exceeds bounded metadata/mesh budget")
        data = response.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("Object exceeds bounded metadata/mesh budget")
        record = {"url": url, "final_url": response.url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "local_file": relative, "size_bytes": len(data), "sha256": digest(data),
                  "status": response.status, "headers": dict(response.headers),
                  "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
    if expected_sha and record["sha256"] != expected_sha:
        raise ValueError("Source metadata differs from previous snapshot; audit before proceeding")
    if expected_etag and record["etag"] != expected_etag:
        raise ValueError("Mesh object changed since previous snapshot; audit before proceeding")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    records.append(record)
    write_json(records_path, records)
    return data, record


def prefix(array):
    return np.pad(np.cumsum(np.cumsum(array, axis=0, dtype=np.int64), axis=1, dtype=np.int64), ((1, 0), (1, 0)))


def rectangle_sum(integral, y, x, size):
    return int(integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x])


def derive(segment, render):
    sid = segment["segment_id"]
    sub = ROOT / sid
    coords = np.stack([tifffile.imread(sub / f"mesh.{axis}.tif") for axis in "xyz"], axis=-1)
    meta = json.loads((sub / "mesh.meta.json").read_text())
    spec = json.loads((sub / "surface.level0.zarray.json").read_text())
    assert coords.dtype == np.float32
    height, width = spec["shape"][1:]
    scale_x, scale_y = meta["scale"]
    assert tuple(np.rint(np.array(coords.shape[:2]) / [scale_y, scale_x]).astype(int)) == (height, width)
    vertex = np.isfinite(coords).all(axis=-1) & (coords > 0).all(axis=-1)
    assert np.array_equal(vertex, np.isfinite(coords).all(axis=-1) & (coords != -1).all(axis=-1))
    assert np.allclose(np.stack([coords[vertex].min(axis=0), coords[vertex].max(axis=0)]), meta["bbox"], atol=.002, rtol=0)
    mask = np.zeros((height, width), dtype=bool)
    mapping = np.full((height, width, 3), np.nan, dtype=np.float32)
    for row in range(0, height, 128):
        yy, xx = np.meshgrid(np.arange(row, min(row + 128, height), dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
        gy, gx = (yy + .5) * scale_y, (xx + .5) * scale_x
        iy, ix = np.floor(gy).astype(int), np.floor(gx).astype(int)
        inside = (iy >= 0) & (ix >= 0) & (iy + 1 < coords.shape[0]) & (ix + 1 < coords.shape[1])
        yc, xc = np.clip(iy, 0, coords.shape[0] - 2), np.clip(ix, 0, coords.shape[1] - 2)
        fy, fx = gy - iy, gx - ix
        ok = inside & vertex[yc, xc] & vertex[yc+1, xc] & vertex[yc, xc+1] & vertex[yc+1, xc+1]
        value = (((1-fy)*(1-fx))[..., None]*coords[yc, xc] + ((1-fy)*fx)[..., None]*coords[yc, xc+1]
                 + (fy*(1-fx))[..., None]*coords[yc+1, xc] + (fy*fx)[..., None]*coords[yc+1, xc+1])
        mask[row:row+ok.shape[0]] = ok
        mapping[row:row+ok.shape[0]] = np.where(ok[..., None], value, np.nan).astype(np.float32)
    assert np.isfinite(mapping[mask]).all() and np.isnan(mapping[~mask]).all()
    np.save(sub / "full-surface-valid-mask.npy", mask, allow_pickle=False)
    np.save(sub / "full-surface-ct-coordinates.xyz.npy", mapping, allow_pickle=False)
    # Minimum filter checks an exact 128-square geometry footprint. Candidate
    # scoring also removes a 64-pixel crop border, so support is inside the ROI.
    context = minimum_filter(mask.astype(np.uint8), size=128, mode="constant", cval=0) > 0
    integral, context_integral = prefix(mask), prefix(context)
    _, component_count = label(mask)
    vy, vx = np.where(mask)
    summary = {"segment_id": sid, "long_id": segment["long_id"],
               "surface_url": segment["surface_url"], "mesh_url": segment["mesh_url"],
               "surface_shape_zyx": spec["shape"], "mesh_shape_yx": list(coords.shape[:2]),
               "mesh_scale_xy": meta["scale"], "native_spacing_um": 8.64,
               "positive_finite_vertices": int(vertex.sum()), "total_vertices": int(vertex.size),
               "valid_pixels": int(mask.sum()), "valid_fraction": float(mask.mean()),
               "valid_bbox_yx_half_open": [[int(vy.min()), int(vy.max()+1)], [int(vx.min()), int(vx.max()+1)]],
               "connected_four_neighbor_components": int(component_count),
               "valid_base_ct_bbox_xyz": np.stack([mapping[mask].min(axis=0), mapping[mask].max(axis=0)]).tolist(),
               "context128_supported_pixels": int(context.sum()),
               "edge_without_full_bilinear_support": {"rows": int(np.sum((np.arange(height)+.5)*scale_y >= coords.shape[0]-1)), "columns": int(np.sum((np.arange(width)+.5)*scale_x >= coords.shape[1]-1))},
               "render_commit": render["creation_info"]["provenance"]["parameters"]["commit"],
               "render_parameters": render["creation_info"]["provenance"]["parameters"],
               "outputs": {name: {"path": str(sub / name), "sha256": digest((sub / name).read_bytes()), "size_bytes": (sub / name).stat().st_size}
                           for name in ["full-surface-valid-mask.npy", "full-surface-ct-coordinates.xyz.npy"]}}
    write_json(sub / "geometry-summary.json", summary)
    candidates = []
    for size in (1536, 2048):
        if min(height, width) < size:
            continue
        side_chunks = size // 128
        for y in range(0, height-size+1, 128):
            for x in range(0, width-size+1, 128):
                valid = rectangle_sum(integral, y, x, size)
                supported = rectangle_sum(context_integral, y+64, x+64, size-128)
                tile_starts = range(0, size-128+1, 64)
                fully_valid = sum(rectangle_sum(integral, y+ty, x+tx, 128) == 128*128 for ty in tile_starts for tx in tile_starts)
                total_tiles = len(tile_starts)**2
                chunk_payload_bytes = side_chunks**2 * 31 * 128 * 128
                assert chunk_payload_bytes <= SURFACE_BUDGET
                candidates.append({"segment_id": sid, "size": size, "bbox_source_xywh": [x,y,size,size],
                                   "valid_pixels": valid, "valid_fraction": valid/(size*size),
                                   "context128_supported_crop_interior_pixels": supported,
                                   "fully_valid_model_tiles": fully_valid, "scheduled_model_tiles": total_tiles,
                                   "fully_valid_model_tile_fraction": fully_valid/total_tiles,
                                   "native_surface_chunk_indices_zyx": [[0,cy,cx] for cy in range(y//128,y//128+side_chunks) for cx in range(x//128,x//128+side_chunks)],
                                   "surface_chunk_count": side_chunks**2, "surface_payload_bound_bytes": chunk_payload_bytes,
                                   "source_array_url": segment["surface_url"].rstrip("/") + "/0",
                                   "nominal_crop_size_mm": [size*.00864, size*.00864]})
    print(json.dumps({key:summary[key] for key in ["segment_id","surface_shape_zyx","valid_pixels","valid_fraction","context128_supported_pixels"]}), flush=True)
    return summary, candidates


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    raw, min_record = retrieve(BASE + "/metadata.min.json", "metadata.min.json.gz", 256*1024)
    current = json.loads(gzip.decompress(raw))
    old_min = json.loads((LAB / "data/live-metadata.min.json").read_text())
    assert current["samples"]["PHerc0800"] == old_min["samples"]["PHerc0800"], "PHerc0800 inventory changed; audit differences before proceeding"
    write_json(ROOT / "pherc0800-current-min-metadata.json", current["samples"]["PHerc0800"])
    old_full_file = LAB / "data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz"
    old_full_provenance = json.loads((old_full_file.parent / "metadata-provenance.json").read_text())
    with urllib.request.urlopen(urllib.request.Request(BASE + "/metadata.json", method="HEAD"), timeout=30) as response:
        full_head = {"url": BASE + "/metadata.json", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "method":"HEAD", "status":response.status, "headers":dict(response.headers)}
        assert response.headers.get("ETag") == old_full_provenance["headers"]["etag"], "Full provenance snapshot changed; fresh bounded retrieval required"
        assert response.headers.get("Last-Modified") == old_full_provenance["headers"]["last-modified"]
    write_json(ROOT / "full-metadata-revalidation-head.json", full_head)
    old_full_bytes = old_full_file.read_bytes()
    assert digest(old_full_bytes) == old_full_provenance["sha256"]
    full = json.loads(gzip.decompress(old_full_bytes))["samples"]["PHerc0800"]
    summaries, candidates = [], []
    for segment in OLD["segments"]:
        sid = segment["segment_id"]
        if sid == "20251029010146":
            continue
        prefix_path = sid + "/"
        retrieve(segment["mesh_url"] + "meta.json", prefix_path + "mesh.meta.json", expected_sha=segment["transformed_mesh_metadata"]["sha256"])
        for axis in "xyz":
            old_head = next(probe for probe in segment["head_probes"] if probe["kind"] == "mesh_" + axis)
            retrieve(segment["mesh_url"] + axis + ".tif", prefix_path + "mesh." + axis + ".tif", expected_etag=old_head["headers"]["etag"])
        retrieve(segment["surface_url"] + "0/.zarray", prefix_path + "surface.level0.zarray.json", expected_sha=segment["native_surface_metadata"]["sha256"])
        attrs, _ = retrieve(segment["surface_url"] + ".zattrs", prefix_path + "surface.zattrs.json", expected_sha=segment["surface_attributes"]["sha256"])
        spec = json.loads((ROOT / prefix_path / "surface.level0.zarray.json").read_text())
        assert spec["chunks"] == [31,128,128] and spec["dtype"] == "|u1" and spec["compressor"] is None
        assert spec["dimension_separator"] == "/" and spec["fill_value"] == 0
        assert json.loads(attrs)["slice_step"] == 1
        record = full["segments"][sid]
        render = [entry for entry in record["data"] if entry["type"] == "layers-zarr" and entry["parameters"]["target_volume"] == "20250521135224"]
        assert len(render) == 1
        params = render[0]["creation_info"]["provenance"]["parameters"]
        assert params["commit"] == "530584554c031c982ba3aa243e1ea39c4bef9818"
        assert params["num-slices"] == "31" and params["flip-normals"] == "true" and params["scale"] == "1"
        write_json(ROOT / prefix_path / "source-render-export-metadata.json", render[0])
        summary, options = derive(segment, render[0])
        summaries.append(summary)
        candidates.extend(options)
    candidates.sort(key=lambda c:(-c["context128_supported_crop_interior_pixels"], -c["fully_valid_model_tiles"], -c["valid_pixels"], -c["size"], c["segment_id"], c["bbox_source_xywh"][1], c["bbox_source_xywh"][0]))
    for rank, candidate in enumerate(candidates,1):
        candidate["rank"] = rank
    # Parent requests a 1536-square first experiment. Keep larger alternatives
    # in the same complete geometry ranking for an explicit cost/gain review.
    best = next(candidate for candidate in candidates if candidate["size"] == 1536)
    alternative2048 = next((candidate for candidate in candidates if candidate["size"] == 2048), None)
    sub = ROOT / best["segment_id"]
    x,y,size,_ = best["bbox_source_xywh"]
    mask = np.load(sub / "full-surface-valid-mask.npy", mmap_mode="r")[y:y+size,x:x+size]
    mapping = np.load(sub / "full-surface-ct-coordinates.xyz.npy", mmap_mode="r")[y:y+size,x:x+size]
    np.save(ROOT / "recommended-roi-valid-mask.npy", mask, allow_pickle=False)
    np.save(ROOT / "recommended-roi-ct-coordinates.xyz.npy", mapping, allow_pickle=False)
    best["valid_base_ct_bbox_xyz"] = np.stack([mapping[mask].min(axis=0),mapping[mask].max(axis=0)]).tolist()
    records = json.loads((ROOT / "source-records.json").read_text())
    result = {"frozen_at_utc":datetime.now(timezone.utc).isoformat(), "sample_id":"PHerc0800", "target_volume_id":"20250521135224",
              "selection_basis":"Geometry only; no new surface/CT pixels or model outputs read. Maximize128-square-supported crop interior pixels, then full-valid scheduled128-square tile count, total valid pixels, size; ties segment ID,Y,X ascending.",
              "crop_sizes_tested":[1536,2048], "origins_step":128, "native_spacing_um":8.64,
              "mesh_mapping_convention":"grid_x=(native_x+0.5)*scale_x; grid_y=(native_y+0.5)*scale_y; bilinear float64 arithmetic stored float32. All four XYZ corners finite and positive. Archive C++530584... convention previously verified locally at93/93native intensity samples.",
              "source_provenance_revalidation":{"small_inventory_sha256":min_record["sha256"], "previous_small_inventory_sha256":OLD["inventory_wire_sha256"], "pherc0800_inventory_unchanged":True,"full_metadata_head_matches_previous_etag_and_date":True, "retained_full_snapshot_sha256":digest(old_full_bytes),"retained_full_snapshot_path":str(old_full_file)},
              "segments":summaries, "ranked_candidates":candidates, "recommended":best,
              "recommendation_scope":"Best1536square candidate first per parent instruction;2048alternatives retained without authorizing surface reads.",
              "best2048_alternative":alternative2048,
              "metadata_mesh_downloaded_bytes":sum(r["size_bytes"] for r in records), "metadata_mesh_budget_bytes":MESH_METADATA_BUDGET,
              "surface_chunk_reads_performed":0, "ct_volume_chunk_reads_performed":0,
              "surface_chunk_payload_budget_bytes":SURFACE_BUDGET,
              "recommended_outputs":{name:{"path":str(ROOT/name),"sha256":digest((ROOT/name).read_bytes()),"size_bytes":(ROOT/name).stat().st_size}
                                     for name in ["recommended-roi-valid-mask.npy","recommended-roi-ct-coordinates.xyz.npy"]},
              "helper_sha256":digest(Path(__file__).read_bytes()),
              "limits":["Geometry support does not prove ink or source intensity quality; adjacent surfaces may overlap physically.","Base XYZ map excludes normal/depth offsets and differs from float32 tiled renderer math by subvoxel rounding previously bounded locally; numeric equality not established for these five meshes.","Minimum-filter128context is a geometry support score, not guarantee every overlapping inference tile contributing to a pixel is valid."]}
    write_json(REPORT / "geometry-selection.json", result)
    write_json(ROOT / "geometry-selection.json", result)
    print(json.dumps({"recommended":{k:v for k,v in best.items() if k != "native_surface_chunk_indices_zyx"}, "candidate_count":len(candidates), "metadata_mesh_downloaded_bytes":result["metadata_mesh_downloaded_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
