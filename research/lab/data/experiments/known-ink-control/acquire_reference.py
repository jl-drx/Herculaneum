"""Acquire a bounded, label-selected PHerc1667 training control without inference.

Run inspect first; acquire reads the fixed selection and downloads four scan chunks.
All new files remain within this directory. Source inputs elsewhere are read only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

import numpy as np
from numcodecs import get_codec

ROOT = Path(__file__).resolve().parent
CASE = "pherc1667-w013"
SEGMENT = "20240304141531-w013_20240304141531_flatboi"
SOURCE = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
          f"PHerc1667/segments/{SEGMENT}/surface-volumes/"
          "2.399um-0.22m-78keV-volume-20251217075048.zarr")
LABELS = ("https://huggingface.co/buckets/scrollprize/datasets/resolve/"
          f"ink_9um/labels/aligned-scrollprizeorg-21slices/{CASE}")
COMMIT = "e0bbb8b40a2db58b1d71864f286eb85717e59e64"
BUDGET = 64 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def download(url, filename, limit=1024 * 1024):
    manifest_path = ROOT / "acquisition-sources.json"
    entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    path = ROOT / filename
    previous = [entry for entry in entries if entry["local_file"] == filename]
    if path.exists():
        if len(previous) != 1 or previous[0]["url"] != url:
            raise ValueError(f"Untracked or inconsistent retained source: {filename}")
        data = path.read_bytes()
        if sha(data) != previous[0]["sha256"]:
            raise ValueError(f"Retained source changed: {filename}")
        return data
    downloaded = sum(entry["size_bytes"] for entry in entries)
    maximum = min(limit, BUDGET - downloaded)
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=45) as response:
        advertised = response.headers.get("Content-Length")
        if advertised and int(advertised) > maximum:
            raise ValueError("Advertised response exceeds remaining budget")
        data = response.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("Response exceeds remaining budget")
        entry = {"url": url, "final_url": response.url,
                 "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                 "status": response.status, "size_bytes": len(data),
                 "sha256": sha(data), "local_file": filename,
                 "headers": dict(response.headers),
                 "etag": response.headers.get("ETag"),
                 "last_modified": response.headers.get("Last-Modified")}
    path.write_bytes(data)
    entries.append(entry)
    save_json(manifest_path, entries)
    return data


def metadata(url, filename):
    return json.loads(download(url, filename))


def chunk(base, spec, y, x, stem):
    key = spec.get("dimension_separator", ".").join(map(str, (0, y, x)))
    payload = download(base + "/" + key, f"{stem}-chunk-0-{y}-{x}.bin", 3 * 1024 * 1024)
    decoded = get_codec(spec["compressor"]).decode(payload) if spec.get("compressor") else payload
    if len(decoded) != int(np.prod(spec["chunks"])):
        raise ValueError("Decoded chunk does not have the declared uint8 size")
    return np.frombuffer(decoded, dtype=spec["dtype"]).reshape(spec["chunks"], order="C")


def label_crop(y0, x0, ink_spec, supervision_spec):
    ink = np.zeros((21, 256, 256), dtype=np.uint8)
    supervision = np.zeros_like(ink)
    for dy in range(2):
        for dx in range(2):
            cy, cx = y0 // 128 + dy, x0 // 128 + dx
            target = (slice(None), slice(dy * 128, (dy + 1) * 128),
                      slice(dx * 128, (dx + 1) * 128))
            ink[target] = chunk(LABELS + f"/{CASE}_inklabels.zarr/0", ink_spec, cy, cx, "inklabels")
            supervision[target] = chunk(LABELS + f"/{CASE}_supervision_mask.zarr/0", supervision_spec, cy, cx, "supervision")
    if np.any(ink[np.arange(21) != 10]) or np.any(supervision[np.arange(21) != 10]):
        raise ValueError("Annotations extend outside the declared center plane")
    return ink[10], supervision[10]


def inspect():
    download("https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/README.md", "upstream-dataset-README.md")
    label_attrs = metadata(LABELS + f"/{CASE}_inklabels.zarr/.zattrs", "inklabels.zattrs.json")
    supervision_attrs = metadata(LABELS + f"/{CASE}_supervision_mask.zarr/.zattrs", "supervision.zattrs.json")
    ink_spec = metadata(LABELS + f"/{CASE}_inklabels.zarr/0/.zarray", "inklabels.level0.zarray.json")
    supervision_spec = metadata(LABELS + f"/{CASE}_supervision_mask.zarr/0/.zarray", "supervision.level0.zarray.json")
    source_spec = metadata(SOURCE + "/2/.zarray", "surface.level2.zarray.json")
    metadata(SOURCE + "/.zattrs", "surface.zattrs.json")
    if label_attrs != supervision_attrs:
        raise ValueError("Label and supervision mapping attributes differ")
    assert ink_spec == supervision_spec
    assert label_attrs["source_z_slice"] == [13, 97]
    assert label_attrs["source_surface_volume_level"] == 2
    assert label_attrs["source_shape_zyx"] == source_spec["shape"] == [109, 10400, 19900]
    assert ink_spec["shape"] == [21, 10400, 19900]
    assert ink_spec["chunks"] == [21, 128, 128]
    assert source_spec["chunks"] == [109, 128, 128]
    assert ink_spec["dtype"] == source_spec["dtype"] == "|u1"
    # First candidate is the chunk-aligned canvas center, fixed before reading labels.
    candidates = [(5120, 9856), (4992, 9728), (5120, 9728), (4992, 9856)]
    probes = []
    for y0, x0 in candidates:
        ink, supervision = label_crop(y0, x0, ink_spec, supervision_spec)
        valid = supervision > 0
        positives = (ink > 0) & valid
        negatives = (ink == 0) & valid
        probe = {"bbox_source_xywh": [x0, y0, 256, 256],
                 "ink_values": np.unique(ink).tolist(),
                 "supervision_values": np.unique(supervision).tolist(),
                 "annotated_positive_pixels": int(positives.sum()),
                 "annotated_negative_pixels": int(negatives.sum()),
                 "unknown_pixels": int((~valid).sum()),
                 "positive_outside_supervision": int(((ink > 0) & ~valid).sum())}
        probes.append(probe)
        print(json.dumps(probe), flush=True)
        if probe["annotated_positive_pixels"] >= 256 and probe["annotated_negative_pixels"] >= 256:
            selection = {"case": CASE, "selected_before_model_predictions": True,
                         "selection_rule": "First predefined canvas-center candidate with at least 256 supervised positive and 256 supervised negative pixels; no model or presentation image used",
                         "candidate_order_yx": candidates, "label_probes": probes,
                         "selected": probe, "source_url": SOURCE,
                         "source_depth_slice": [13, 97], "pool_z": 4,
                         "output_shape_zyx": [21, 256, 256],
                         "model_layer_slice": [2, 19],
                         "maximum_wire_bytes": BUDGET,
                         "decoded_source_chunk_bytes": 4 * 109 * 128 * 128}
            save_json(ROOT / "selection.json", selection)
            return
    save_json(ROOT / "unsuccessful-label-probes.json", probes)
    # Sparse training annotations need not cover the canvas center. Search only
    # the official label object inventory, never a model score or rendered image.
    # The first adequate labelled chunk in server path order fixes the crop.
    listing_url = ("https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
                   f"ink_9um/labels/aligned-scrollprizeorg-21slices/{CASE}/"
                   f"{CASE}_inklabels.zarr/0?recursive=false&limit=1000")
    for page in range(4):
        listing_file = f"label-chunk-listing-page-{page}.json"
        entries = json.loads(download(listing_url, listing_file, 1024 * 1024))
        for entry in entries:
            # A retained all-zero label chunk is 78 bytes. Its larger compressed
            # counterparts are only candidates; decoded positives decide.
            key = entry["path"].rsplit("/", 1)[-1]
            if entry.get("type") != "file" or key.startswith(".") or entry["size"] <= 78:
                continue
            cz, cy, cx = map(int, key.split("."))
            if cz != 0:
                raise ValueError("Unexpected label depth chunk index")
            y0, x0 = min(cy, (10400 - 256) // 128) * 128, min(cx, (19900 - 256) // 128) * 128
            ink, supervision = label_crop(y0, x0, ink_spec, supervision_spec)
            valid = supervision > 0
            probe = {"bbox_source_xywh": [x0, y0, 256, 256],
                     "ink_values": np.unique(ink).tolist(),
                     "supervision_values": np.unique(supervision).tolist(),
                     "annotated_positive_pixels": int(((ink > 0) & valid).sum()),
                     "annotated_negative_pixels": int(((ink == 0) & valid).sum()),
                     "unknown_pixels": int((~valid).sum()),
                     "positive_outside_supervision": int(((ink > 0) & ~valid).sum()),
                     "trigger_source_label_object": entry}
            probes.append(probe)
            print(json.dumps(probe), flush=True)
            if probe["annotated_positive_pixels"] >= 256 and probe["annotated_negative_pixels"] >= 256:
                selection = {"case": CASE, "selected_before_model_predictions": True,
                             "selection_rule": "Canvas-center candidates lacked annotations; select first larger-than-zero compressed label chunk in bounded official server path-order listing whose 256 square crop has at least 256 supervised positive and negative pixels. No model or presentation image used.",
                             "candidate_order_yx": candidates, "label_probes": probes,
                             "selected": probe, "source_url": SOURCE,
                             "source_depth_slice": [13, 97], "pool_z": 4,
                             "output_shape_zyx": [21, 256, 256], "model_layer_slice": [2, 19],
                             "maximum_wire_bytes": BUDGET,
                             "decoded_source_chunk_bytes": 4 * 109 * 128 * 128}
                save_json(ROOT / "selection.json", selection)
                return
        records = json.loads((ROOT / "acquisition-sources.json").read_text())
        record = next(item for item in records if item["local_file"] == listing_file)
        link = record["headers"].get("link", record["headers"].get("Link", ""))
        next_url = re.search(r'<([^>]+)>; rel="next"', link)
        if not next_url:
            break
        listing_url = next_url.group(1)
    save_json(ROOT / "unsuccessful-label-probes.json", probes)
    raise ValueError("Bounded label inventory did not supply an adequate control crop")


def acquire():
    selection = json.loads((ROOT / "selection.json").read_text())
    x0, y0, width, height = selection["selected"]["bbox_source_xywh"]
    spec = json.loads((ROOT / "surface.level2.zarray.json").read_text())
    ink_spec = json.loads((ROOT / "inklabels.level0.zarray.json").read_text())
    supervision_spec = json.loads((ROOT / "supervision.level0.zarray.json").read_text())
    raw = np.zeros((109, height, width), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            block = chunk(SOURCE + "/2", spec, y0 // 128 + dy, x0 // 128 + dx, "surface")
            raw[:, dy * 128:(dy + 1) * 128, dx * 128:(dx + 1) * 128] = block
    # Exact pinned upstream preparation: np.float32 mean, np.rint, then uint8.
    pooled = np.rint(raw[13:97].astype(np.float32).reshape(21, 4, height, width).mean(axis=1)).astype(np.uint8)
    ink, supervision = label_crop(y0, x0, ink_spec, supervision_spec)
    annotated = supervision > 0
    positive = (ink > 0) & annotated
    assert np.all(annotated[positive])
    assert not np.any((ink > 0) & ~annotated)
    derived = {"surface.npy": pooled, "inklabels.npy": ink,
               "supervision-mask.npy": supervision,
               "annotation-valid-mask.npy": annotated,
               "ink-positive-mask.npy": positive,
               "source-observed-nonzero-mask.npy": np.any(raw[13:97] != 0, axis=0)}
    outputs = {}
    for name, value in derived.items():
        path = ROOT / name
        if path.exists():
            raise FileExistsError(f"Refusing to replace derived input {path}")
        np.save(path, value, allow_pickle=False)
        outputs[name] = {"sha256": sha(path.read_bytes()), "size_bytes": path.stat().st_size,
                         "shape": list(value.shape), "dtype": str(value.dtype)}
    pool_source = "vesuvius/src/vesuvius/ink_detection/preprocessing/prepare_9um_isotropic_input.py"
    pool_code = subprocess.check_output(["git", "show", COMMIT + ":" + pool_source], cwd="/tmp/herculaneum-villa-geometry-source")
    (ROOT / "upstream-prepare_9um_isotropic_input.py").write_bytes(pool_code)
    full_path = Path("/workspace/herculaneum-lab/data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz")
    full_bytes = full_path.read_bytes()
    official = json.loads(gzip.decompress(full_bytes))
    segment = official["samples"]["PHerc1667"]["segments"]["20240304141531"]
    render = [entry for entry in segment["data"] if entry["type"] == "layers-zarr" and entry["parameters"]["target_volume"] == "20251217075048"]
    assert len(render) == 1
    params = render[0]["creation_info"]["provenance"]["parameters"]
    assert params["flip-normals"] == "true" and params["num-slices"] == "109"
    save_json(ROOT / "source-render-export-metadata.json", render[0])
    sources = json.loads((ROOT / "acquisition-sources.json").read_text())
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "role": "Source-derived supervised training control; no model inference or independent ground-truth claim",
                  "sample_id": "PHerc1667", "target_volume_id": "20251217075048",
                  "segment_id": "20240304141531", "segment_long_id": SEGMENT,
                  "training_representation": "public_2p4_level2_zmean4:pherc1667-w013",
                  "training_overlap": "Explicitly listed in released ink_9um hybrid recipe; not held out",
                  "bbox_source_xywh": [x0, y0, width, height], "pyramid_level": 2,
                  "shape_zyx": list(pooled.shape), "dtype": str(pooled.dtype),
                  "array_sha256": outputs["surface.npy"]["sha256"],
                  "voxel_spacing_zyx_um": [9.596, 9.596, 9.596],
                  "source_surface_url": SOURCE, "source_shape_zyx": spec["shape"],
                  "source_depth_slice": [13, 97], "depth_aggregation": "float32 mean of consecutive groups of 4, np.rint nearest-even, cast uint8; exact pinned upstream algorithm",
                  "pooled_plane_source_ranges": [[13 + 4 * i, 17 + 4 * i] for i in range(21)],
                  "model_layer_slice": [2, 19], "model_source_depth_slice": [21, 89],
                  "annotation_center_layer": 10, "annotation_center_source_planes": [53, 57],
                  "annotation_center_source_mean_index": 54.5,
                  "depth_orientation": "Increasing published source Z retained; archived render metadata records flip-normals=true. No reversal or fitted offset applied.",
                  "render_source_commit": params["commit"],
                  "full_metadata_source": {"url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json", "retained_local_path": str(full_path), "sha256": sha(full_bytes)},
                  "selection": selection,
                  "annotation_semantics": {"positive": "inklabels > 0 and supervision > 0", "negative": "inklabels == 0 and supervision > 0", "unknown": "supervision == 0; exclude", "encoding_observed": selection["selected"]},
                  "observed_mask_limit": "source-observed-nonzero-mask records CT occupancy, not a mesh validity guarantee; no independent geometry-valid mask has been acquired",
                  "outputs": outputs, "source_records": sources,
                  "downloaded_bytes": sum(entry["size_bytes"] for entry in sources),
                  "downloaded_scan_chunk_bytes": sum(entry["size_bytes"] for entry in sources if entry["local_file"].startswith("surface-chunk-")),
                  "preprocessing_source": {"commit": COMMIT, "url": "https://github.com/ScrollPrize/villa/blob/" + COMMIT + "/" + pool_source, "sha256": sha(pool_code)}}
    save_json(ROOT / "provenance.json", provenance)
    print(json.dumps({"shape_zyx": provenance["shape_zyx"], "array_sha256": provenance["array_sha256"], "downloaded_bytes": provenance["downloaded_bytes"], "selection": selection["selected"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inspect", "acquire"))
    phase = parser.parse_args().phase
    inspect() if phase == "inspect" else acquire()
