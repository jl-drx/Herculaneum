"""Execute a separately reviewed, SHA-pinned LABEL-ONLY w029 selection plan.

No source surface, CT, checkpoint, model, or prediction is read by this script.
Unknown/missing objects stop the stage. There are no automatic retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
from numcodecs import get_codec


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_inputs(plan_path, expected_sha, selector_sha):
    if sha_file(plan_path) != expected_sha or sha_file(__file__) != selector_sha:
        raise ValueError("Frozen plan or selector changed")
    plan = json.loads(Path(plan_path).read_text())
    for r in list(plan["metadata_files"].values()) + plan["inventory_files"]:
        p = Path(plan_path).parent / r["path"]
        if sha_file(p) != r["sha256"]:
            raise ValueError("Frozen metadata/inventory changed: " + r["path"])
    return plan


def fetch_object(obj, output, plan_sha, *, cap, ledger_name="label-request-ledger.json"):
    """Read one exact-length, optionally ETag-pinned object, retaining failures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / ledger_name
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "plan_sha256": plan_sha, "response_body_cap_bytes": cap, "body_bytes": 0, "requests": []}
    if ledger["plan_sha256"] != plan_sha or ledger["response_body_cap_bytes"] != cap:
        raise ValueError("Output belongs to a different frozen plan/budget")
    expected = int(obj["expected_byte_length"])
    key = hashlib.sha256(obj["url"].encode()).hexdigest()
    retained = output / (key + ".bin")
    previous = [r for r in ledger["requests"] if r.get("url") == obj["url"] and r.get("verified")]
    if retained.exists():
        if len(previous) != 1 or sha_file(retained) != previous[0]["sha256"] or retained.stat().st_size != expected:
            raise ValueError("Retained object is untracked or corrupt")
        return retained.read_bytes(), previous[0]
    remaining = cap - ledger["body_bytes"]
    if expected > remaining:
        raise ValueError("Object exceeds remaining response-body budget")
    entry = {"url": obj["url"], "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
             "expected_byte_length": expected, "body_bytes": 0, "verified": False}
    response = None
    body = b""
    try:
        try:
            response = urllib.request.urlopen(urllib.request.Request(obj["url"], headers={"Accept-Encoding": "identity"}), timeout=45)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            entry.update({"status": response.status, "headers": dict(response.headers),
                          "final_url_without_query": response.url.split("?", 1)[0]})
            if response.status != 200:
                body = response.read(min(4096, remaining))
                raise ValueError("Non-200 source object; no fill or retry")
            length = response.headers.get("Content-Length")
            if length is None or int(length) != expected:
                raise ValueError("Content-Length differs from frozen object metadata")
            etag = response.headers.get("ETag", "").strip('"')
            if obj.get("expected_etag") and etag != obj["expected_etag"]:
                raise ValueError("ETag differs from frozen object inventory")
            body = response.read(expected)
            if len(body) != expected:
                raise ValueError("Truncated source object")
            entry.update({"verified": True, "sha256": hashlib.sha256(body).hexdigest(), "local_file": retained.name})
    except Exception as exc:
        entry["error"] = str(exc)
        raise
    finally:
        entry["body_bytes"] = len(body)
        ledger["body_bytes"] += len(body)
        ledger["requests"].append(entry)
        if body:
            # Raw errors stay inspectable but cannot be reused as successful data.
            destination = retained if entry["verified"] else output / (key + ".error.bin")
            if destination.exists():
                raise FileExistsError("Refusing to replace retained response")
            destination.write_bytes(body)
        write_json(ledger_path, ledger)
    return body, entry


def decode_chunk(body, spec):
    if spec["dtype"] != "|u1" or spec["order"] != "C" or spec.get("filters") is not None:
        raise ValueError("Unsupported chunk contract")
    raw = get_codec(spec["compressor"]).decode(body) if spec.get("compressor") else body
    expected = int(np.prod(spec["chunks"]))
    if len(raw) != expected:
        raise ValueError("Decoded uint8 chunk length differs from metadata")
    return np.frombuffer(raw, dtype=np.uint8).reshape(spec["chunks"])


def binary_foreground(array, role):
    unique = set(np.unique(array).tolist())
    if unique <= {0, 1}:
        return 1
    if unique <= {0, 255}:
        return 255
    raise ValueError("Nonbinary annotation values for " + role)


def load_label_crop(plan, plan_path, output, plan_sha, candidate):
    y0, x0 = candidate["origin_yx"]
    objects = {(o["kind"], tuple(o["chunk_zyx"])): o for o in plan["label_objects"]}
    arrays = {}
    encodings = {}
    for role in ["inklabels", "supervision_mask", "validation_mask"]:
        r = plan["metadata_files"][f"w029-{role}-level0-zarray.json"]
        spec = json.loads((Path(plan_path).parent / r["path"]).read_text())
        if spec["shape"] != [21, 9500, 7830] or spec["chunks"] != [21, 128, 128]:
            raise ValueError("Unexpected label shape/chunks")
        assembled = np.zeros((21, 256, 256), dtype=np.uint8)
        for dy in [0, 1]:
            for dx in [0, 1]:
                obj = objects[(role, (0, y0 // 128 + dy, x0 // 128 + dx))]
                body, _ = fetch_object(obj, output / "raw-labels", plan_sha, cap=plan["label_probe_response_body_cap_bytes"])
                assembled[:, dy*128:(dy+1)*128, dx*128:(dx+1)*128] = decode_chunk(body, spec)
        if np.any(assembled[:10]) or np.any(assembled[11:]):
            raise ValueError("Annotations extend outside declared z10")
        arrays[role] = assembled[10].copy()
        encodings[role] = binary_foreground(arrays[role], role)
    return arrays, encodings


def crop_statistics(arrays):
    ink = arrays["inklabels"] > 0
    supervision = arrays["supervision_mask"] > 0
    validation = arrays["validation_mask"] > 0
    interior = np.zeros((256, 256), dtype=bool)
    interior[64:192, 64:192] = True
    primary = validation & interior
    return {"validation_pixels": int(validation.sum()), "raw_supervision_pixels": int(supervision.sum()),
            "raw_supervision_validation_overlap_pixels": int((supervision & validation).sum()),
            "conditional_current_loader_label_overlap_pixels": int(((supervision & ~validation) & validation).sum()),
            "primary_validation_pixels": int(primary.sum()), "primary_positive_pixels": int((primary & ink).sum()),
            "primary_negative_pixels": int((primary & ~ink).sum()),
            "full_validation_positive_pixels": int((validation & ink).sum()), "full_validation_negative_pixels": int((validation & ~ink).sum()),
            "unique_values": {k: np.unique(v).tolist() for k, v in arrays.items()},
            "actual_historical_training_overlap": {"status": "unknown", "mask_file": None}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--selector-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = verify_inputs(args.plan, args.plan_sha256, args.selector_sha256)
    if (args.output / "selection.json").exists():
        raise FileExistsError("Selection already published; inspect it without rerunning")
    args.output.mkdir(parents=True, exist_ok=True)
    probes = []
    for rank, candidate in enumerate(plan["candidates"], 1):
        arrays, encodings = load_label_crop(plan, args.plan, args.output, args.plan_sha256, candidate)
        stats = crop_statistics(arrays)
        probe = {"rank": rank, **candidate, **stats}
        probes.append(probe)
        write_json(args.output / "label-probes.json", {"plan_sha256": args.plan_sha256, "probes": probes, "completion_claim": False})
        print(json.dumps(probe), flush=True)
        if stats["primary_positive_pixels"] < plan["minimum_primary_positive_pixels"] or stats["primary_negative_pixels"] < plan["minimum_primary_negative_pixels"]:
            continue
        retained = {}
        for role, array in arrays.items():
            name = {"inklabels": "inklabels.npy", "supervision_mask": "supervision-mask.npy", "validation_mask": "validation-mask.npy"}[role]
            target = args.output / name
            if target.exists():
                raise FileExistsError("Refusing to replace derived label array")
            np.save(target, array, allow_pickle=False)
            retained[role] = {"path": name, "sha256": sha_file(target), "byte_length": target.stat().st_size,
                              "shape": list(array.shape), "dtype": str(array.dtype), "binary_foreground": encodings[role]}
        verify_inputs(args.plan, args.plan_sha256, args.selector_sha256)
        selection = {"status": "label_selection_complete_scan_not_acquired", "plan_sha256": args.plan_sha256,
                     "selector_sha256": args.selector_sha256, "selected_at_utc": datetime.now(timezone.utc).isoformat(),
                     "selected_before_source_scan_and_predictions": True, "case": plan["case"], "selected": probe,
                     "arrays": retained, "binary_foreground": encodings, "probes": probes,
                     "geometry_validity": plan["geometry_validity"], "actual_historical_training_overlap": plan["actual_historical_training_overlap"]}
        write_json(args.output / "selection.json", selection)  # publication last
        return
    raise ValueError("No predefined crop meets both fixed class-count gates; stop without relaxation")


if __name__ == "__main__":
    main()
