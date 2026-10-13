"""Pure, bounded decoder for one uint8 YX Zarr v3 indexed shard; no I/O.

Payload presence is a storage diagnostic, never a participation/geometry mask.
Missing entries in a valid shard use declared fill=0; malformed data raises.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

import google_crc32c
import numpy as np
from numcodecs import Blosc
from numcodecs.blosc import cbuffer_complib

UINT64_MAX = (1 << 64) - 1


def _pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " must contain two positive integers")
    if any(type(v) is not int or v <= 0 for v in value):
        raise ValueError(name + " must contain two positive integers")
    return tuple(value)


def validate_metadata(metadata):
    """Accept only the observed uint8/bytes/Blosc-zstd/CRC32C contract."""
    if (metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array"
            or metadata.get("data_type") != "uint8" or type(metadata.get("fill_value")) is not int
            or metadata.get("fill_value") != 0
            or metadata.get("dimension_names") != ["y", "x"]
            or metadata.get("storage_transformers", []) != []):
        raise ValueError("Unsupported array metadata")
    shape = _pair(metadata.get("shape"), "shape")
    grid = metadata.get("chunk_grid", {})
    if grid.get("name") != "regular" or set(grid) != {"name", "configuration"}:
        raise ValueError("Unsupported chunk grid")
    outer = _pair(grid.get("configuration", {}).get("chunk_shape"), "outer chunk shape")
    if any(s > o for s, o in zip(shape, outer)):
        raise ValueError("Decoder supports only a single outer shard")
    if metadata.get("chunk_key_encoding") != {"name": "default", "configuration": {"separator": "/"}}:
        raise ValueError("Unsupported chunk key encoding")
    codecs = metadata.get("codecs", [])
    if len(codecs) != 1 or codecs[0].get("name") != "sharding_indexed":
        raise ValueError("Expected indexed sharding only")
    config = codecs[0].get("configuration", {})
    if set(codecs[0]) != {"name", "configuration"} or set(config) != {
            "chunk_shape", "codecs", "index_codecs", "index_location"}:
        raise ValueError("Unsupported indexed-sharding configuration")
    inner = _pair(config.get("chunk_shape"), "inner chunk shape")
    if any(o % i for o, i in zip(outer, inner)):
        raise ValueError("Outer shape is not divisible by inner shape")
    inner_codecs = config.get("codecs", [])
    byte_variants = ({"name": "bytes"}, {"name": "bytes", "configuration": {"endian": "little"}})
    blosc = {"name": "blosc", "configuration": {"typesize": 1, "cname": "zstd", "clevel": 3,
                                               "shuffle": "bitshuffle", "blocksize": 0}}
    if len(inner_codecs) != 2 or inner_codecs[0] not in byte_variants or inner_codecs[1] != blosc:
        raise ValueError("Unsupported inner codec chain")
    if config.get("index_location") != "end" or config.get("index_codecs") != [
            {"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]:
        raise ValueError("Unsupported index codec/location")
    inner_bytes = math.prod(inner)
    entries = math.prod(o // i for o, i in zip(outer, inner))
    if inner_bytes > 262144 or entries > 10000:
        raise ValueError("Metadata exceeds decoder expansion/index limits")
    return {"shape": shape, "outer": outer, "inner": inner, "inner_bytes": inner_bytes,
            "grid": tuple(o // i for o, i in zip(outer, inner)), "entries": entries,
            "index_bytes": entries * 16 + 4}


def validate_shard_index(blob, metadata, *, maximum_encoded_bytes,
                         expected_sha256=None, expected_byte_length=None):
    """Validate complete bytes, CRC and every absolute payload interval."""
    layout = validate_metadata(metadata)
    if type(maximum_encoded_bytes) is not int or maximum_encoded_bytes <= 0:
        raise ValueError("A positive encoded byte cap is required")
    if not isinstance(blob, bytes) or len(blob) > maximum_encoded_bytes:
        raise ValueError("Whole object exceeds byte cap or is not bytes")
    if expected_byte_length is not None and len(blob) != expected_byte_length:
        raise ValueError("Whole object length mismatch")
    digest = hashlib.sha256(blob).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Whole object SHA256 mismatch")
    index_bytes = layout["index_bytes"]
    if len(blob) < index_bytes:
        raise ValueError("Truncated shard index")
    data_end = len(blob) - index_bytes
    index = blob[data_end:-4]
    stored_crc, = struct.unpack("<I", blob[-4:])
    if google_crc32c.value(index) != stored_crc:
        raise ValueError("Shard index CRC32C mismatch")
    entries = list(struct.iter_unpack("<QQ", index))
    intervals = []
    for flat, (offset, length) in enumerate(entries):
        if offset == UINT64_MAX or length == UINT64_MAX:
            if (offset, length) != (UINT64_MAX, UINT64_MAX):
                raise ValueError("Half-missing index entry")
            continue
        # struct values are Python integers: addition cannot wrap uint64.
        if length <= 0 or length > layout["inner_bytes"] + 16 or offset + length > data_end:
            raise ValueError("Payload length/offset outside bounded data region")
        intervals.append((offset, offset + length, flat))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping or aliased payload intervals")
    return layout, entries, {"byte_length": len(blob), "sha256": digest,
                             "metadata_canonical_json_sha256": hashlib.sha256(json.dumps(
                                 metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                             "index_byte_length": index_bytes, "index_crc32c": stored_crc,
                             "index_sha256": hashlib.sha256(index).hexdigest(),
                             "present_inner_chunks": len(intervals)}


def decode_shard_roi(blob, metadata, bbox_xywh, *, maximum_encoded_bytes,
                     maximum_expanded_bytes, expected_sha256=None, expected_byte_length=None):
    """Return (uint8 ROI, encoded-payload-present bool ROI, receipt).

    The bbox is local array XYWH. No coordinate offset, depth or support policy
    is applied. Index intervals/caps precede decoding; each header is checked
    before its own bounded decompression.
    """
    layout, entries, receipt = validate_shard_index(
        blob, metadata, maximum_encoded_bytes=maximum_encoded_bytes,
        expected_sha256=expected_sha256, expected_byte_length=expected_byte_length)
    if (not isinstance(bbox_xywh, (list, tuple)) or len(bbox_xywh) != 4
            or any(type(v) is not int for v in bbox_xywh)):
        raise ValueError("BBox must contain four integers")
    x, y, width, height = bbox_xywh
    if (min(x, y) < 0 or min(width, height) <= 0 or x + width > layout["shape"][1]
            or y + height > layout["shape"][0]):
        raise ValueError("BBox outside declared array")
    iy, ix = layout["inner"]
    coords = [(cy, cx) for cy in range(y // iy, (y + height - 1) // iy + 1)
              for cx in range(x // ix, (x + width - 1) // ix + 1)]
    expanded = 2 * width * height + len(coords) * layout["inner_bytes"]
    if type(maximum_expanded_bytes) is not int or expanded > maximum_expanded_bytes:
        raise ValueError("ROI/inner chunk expansion exceeds byte cap")
    output = np.zeros((height, width), dtype=np.uint8)
    present = np.zeros((height, width), dtype=np.bool_)
    codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE, blocksize=0)
    records = []
    for cy, cx in coords:
        offset, length = entries[cy * layout["grid"][1] + cx]
        record = {"inner_yx": [cy, cx], "canonical_fill": offset == UINT64_MAX}
        if offset != UINT64_MAX:
            payload = blob[offset:offset + length]
            if len(payload) < 16:
                raise ValueError("Truncated Blosc header")
            _, _, flags, typesize, nbytes, blocksize, cbytes = struct.unpack("<BBBBIII", payload[:16])
            if (nbytes != layout["inner_bytes"] or cbytes != len(payload) or typesize != 1
                    or blocksize <= 0 or blocksize > nbytes or flags & 1
                    or (not flags & 2 and not flags & 4)
                    or cbuffer_complib(payload).lower() != "zstd"):
                raise ValueError("Blosc header differs from bounded codec contract")
            raw = bytearray(layout["inner_bytes"])
            codec.decode(payload, out=raw)
            chunk = np.frombuffer(raw, dtype=np.uint8).reshape(layout["inner"])
            y0, x0 = max(y, cy * iy), max(x, cx * ix)
            y1, x1 = min(y + height, (cy + 1) * iy), min(x + width, (cx + 1) * ix)
            output[y0-y:y1-y, x0-x:x1-x] = chunk[y0-cy*iy:y1-cy*iy, x0-cx*ix:x1-cx*ix]
            present[y0-y:y1-y, x0-x:x1-x] = True
            record.update({"absolute_offset": offset, "encoded_byte_length": length,
                           "encoded_sha256": hashlib.sha256(payload).hexdigest(),
                           "decoded_byte_length": len(raw), "decoded_sha256": hashlib.sha256(raw).hexdigest()})
        records.append(record)
    receipt.update({"bbox_xywh": list(bbox_xywh), "shape_yx": list(output.shape), "dtype": "uint8",
                    "expansion_budget_charge_bytes": expanded, "selected_inner_chunks": records,
                    "encoded_payload_present_pixels": int(present.sum()),
                    "canonical_fill_pixels": int(present.size - present.sum()),
                    "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                    "payload_presence_is_participation_or_geometry": False,
                    "coordinate_or_depth_transform_applied": False})
    return output, present, receipt
