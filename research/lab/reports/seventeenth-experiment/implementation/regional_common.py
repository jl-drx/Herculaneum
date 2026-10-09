"""Small offline-only primitives for the frozen seventeenth diagnostic.

Importing this module does not read any experiment input. All file access is
explicit, bounded, and checked against retained SHA-256 identities.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path))


def checked_bytes(pin, cap):
    path = Path(pin["path"])
    size = pin.get("size_bytes", pin.get("bytes"))
    require(type(size) is int and 0 <= size <= cap, "Input exceeds declared byte cap")
    require(path.is_file() and path.stat().st_size == size, "Input size differs")
    with path.open("rb") as handle:
        data = handle.read(cap + 1)
    require(len(data) == size, "Input changed while reading")
    require(hashlib.sha256(data).hexdigest() == pin["sha256"], "Input SHA-256 differs")
    return data


def checked_json(pin, cap=32 * 1024 * 1024):
    return json.loads(checked_bytes(pin, cap))


def write_json(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def verify(pin):
    actual = record(pin["path"])
    require(actual["size_bytes"] == pin.get("size_bytes", pin.get("bytes")), "Input size differs")
    require(actual["sha256"] == pin["sha256"], "Input SHA-256 differs")


def decode_batch(body, batch, contract, case):
    """Decode only explicitly retained TIFF rows, preserving uint16 samples."""
    x, y, width, height = case["bbox_source_xywh"]
    require(case["native_bbox_yxhw"] == [y, x, height, width], "XY/YX crop disagreement")
    fields = [x, y, width, height, contract["width"], contract["height"],
              contract["strip_offset"], contract["strip_bytes"], contract["object_total_bytes"],
              batch["native_row_start"], batch["row_count"]]
    require(all(type(v) is int for v in fields), "Integer geometry required")
    require(0 <= x and 0 <= y and width > 0 and height > 0
            and x + width <= contract["width"] and y + height <= contract["height"],
            "Crop escapes source canvas")
    require(contract["dtype"] in ["<u2", ">u2"], "Unsupported sample dtype")
    require(contract["strip_offset"] >= 0 and contract["strip_bytes"] >= 0
            and contract["strip_offset"] + contract["strip_bytes"] <= contract["object_total_bytes"],
            "Invalid strip interval")
    require(batch["native_x_start"] == x and batch["native_crop_width"] == width,
            "Batch horizontal crop differs")
    start, count = batch["native_row_start"], batch["row_count"]
    require(count > 0 and y <= start < start + count <= y + height, "Batch escapes crop")
    rows = batch["rows"]
    require(len(rows) == count and all(type(r["native_y"]) is int for r in rows),
            "Batch row count/type differs")
    require(sorted(r["native_y"] for r in rows) == list(range(start, start + count)),
            "Missing or duplicate row")
    intervals = []
    output = np.empty((count, width), dtype=np.uint16)
    for row in rows:
        a, size = row["payload_offset"], row["payload_bytes"]
        require(type(a) is int and type(size) is int and a >= 0 and size == 2 * width
                and a + size <= len(body), "Payload interval/size differs")
        intervals.append((a, a + size))
        source_start = contract["strip_offset"] + 2 * (row["native_y"] * contract["width"] + x)
        source_stop = source_start + 2 * width
        require(row["range_inclusive"] == [source_start, source_stop - 1], "Row source range differs")
        require(contract["strip_offset"] <= source_start < source_stop
                <= contract["strip_offset"] + contract["strip_bytes"], "Row escapes strip")
        payload = memoryview(body)[a:a + size]
        require(hashlib.sha256(payload).hexdigest() == row["payload_sha256"], "Row payload SHA-256 differs")
        output[row["native_y"] - start] = np.frombuffer(payload, dtype=contract["dtype"])
    intervals.sort()
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "Overlapping payloads")
    require(output.dtype.isnative and output.flags.c_contiguous, "Native contiguous crop required")
    return output


def pool_triplet(planes):
    require(isinstance(planes, np.ndarray) and planes.ndim == 3 and planes.shape[0] == 3,
            "Three source planes required")
    require(planes.dtype.kind == "u" and planes.dtype.itemsize in (2, 4), "Unsigned source required")
    _, height, width = planes.shape
    require(height > 0 and width > 0 and height % 3 == width % 3 == 0, "Factor-three geometry required")
    require(np.all(planes <= 65535), "Source exceeds uint16 range")
    sums = planes.reshape(3, height // 3, 3, width // 3, 3).sum(axis=(0, 2, 4), dtype=np.uint32)
    return (sums.astype(np.float64) / np.float64(6939)).astype(np.float32)


def coordinate_roundtrip(case, shape=(384, 384)):
    """Exhaustive integer lattice round trip; no image registration/search."""
    x0, y0, width, height = case["bbox_source_xywh"]
    require(case["native_bbox_yxhw"] == [y0, x0, height, width], "Crop coordinate systems differ")
    require((height, width) == (3 * shape[0], 3 * shape[1]), "Native/pooled shape differs")
    yy, xx = np.indices(shape, dtype=np.int32)
    native_y, native_x = y0 + 1 + 3 * yy, x0 + 1 + 3 * xx
    require(np.array_equal((native_y - y0 - 1) // 3, yy)
            and np.array_equal((native_x - x0 - 1) // 3, xx), "Coordinate round trip failed")
    require(np.all((native_y >= y0) & (native_y < y0 + height))
            and np.all((native_x >= x0) & (native_x < x0 + width)), "Centers escape crop")
    expected_mapping = dict(y=f"{y0}+1+3*y_coarse", x=f"{x0}+1+3*x_coarse")
    supplied_mapping = case.get("source_pixel_center_mapping")
    if supplied_mapping is not None:
        require(supplied_mapping == expected_mapping, "Released label/source center mapping differs")
    return dict(status="PASS", coarse_centers_checked=int(yy.size),
                first_source_center_yx=[int(native_y[0, 0]), int(native_x[0, 0])],
                last_source_center_yx=[int(native_y[-1, -1]), int(native_x[-1, -1])],
                mapping_y=f"{y0}+1+3*y_coarse", mapping_x=f"{x0}+1+3*x_coarse",
                released_mapping_compared=supplied_mapping is not None,
                source_plane_centers=list(range(8, 57, 3)),
                optical_registration_residual_measured=False)
