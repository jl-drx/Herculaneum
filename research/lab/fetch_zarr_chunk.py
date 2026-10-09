"""Read one bounded public Zarr v2 chunk and retain its provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from numcodecs import get_codec
import requests


def fetch(url, limit):
    with requests.get(url, stream=True, timeout=(10, 45)) as response:
        if response.status_code == 404:
            # Retain a bounded diagnosis before the streaming response closes.
            # HTTPError.response.content cannot be read after that closure.
            body = bytearray()
            for part in response.iter_content(4096):
                body.extend(part)
                if len(body) > 64 * 1024:
                    raise ValueError("Unexpectedly large missing-object error response")
            error = requests.HTTPError(f"HTTP 404 for public object: {url}", response=response)
            error.zarr_missing_key = b"<Code>NoSuchKey</Code>" in body
            raise error
        response.raise_for_status()
        if not response.url.startswith("https://"):
            raise ValueError("HTTPS is required")
        chunks, size = [], 0
        for block in response.iter_content(64 * 1024):
            size += len(block)
            if size > limit:
                raise ValueError(f"Response exceeds {limit} byte budget")
            chunks.append(block)
        data = b"".join(chunks)
        return data, {"url": url, "final_url": response.url,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                      "etag": response.headers.get("ETag"),
                      "last_modified": response.headers.get("Last-Modified"),
                      "content_encoding": response.headers.get("Content-Encoding")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("array_url", help="HTTPS URL of a single pyramid level")
    parser.add_argument("--index", nargs=3, type=int, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory to preserve existing results")
    base = args.array_url.rstrip("/")
    if not base.startswith("https://"):
        parser.error("An HTTPS array URL is required")
    raw_metadata, metadata_source = fetch(base + "/.zarray", 512 * 1024)
    metadata = json.loads(raw_metadata)
    if metadata["zarr_format"] != 2 or len(metadata["shape"]) != 3 or metadata.get("filters"):
        raise ValueError("This helper supports 3D Zarr v2 arrays without filters")
    dtype = np.dtype(metadata["dtype"])
    chunk_shape = metadata["chunks"]
    decoded_size = int(np.prod(chunk_shape)) * dtype.itemsize
    if decoded_size > 16 * 1024 * 1024:
        raise ValueError("Decoded chunk exceeds the 16 MiB memory budget")
    start = [i * c for i, c in zip(args.index, chunk_shape)]
    if any(i < 0 or s >= total for i, s, total in zip(args.index, start, metadata["shape"])):
        raise ValueError("Chunk index lies outside the declared array")
    separator = metadata.get("dimension_separator", ".")
    key = separator.join(map(str, args.index))
    encoded, chunk_source = fetch(base + "/" + key, 20 * 1024 * 1024)
    decoded = get_codec(metadata["compressor"]).decode(encoded) if metadata.get("compressor") else encoded
    if len(decoded) != decoded_size:
        raise ValueError(f"Decoded size {len(decoded)} does not match declared chunk size {decoded_size}")
    full = np.frombuffer(decoded, dtype=dtype).reshape(chunk_shape, order=metadata["order"])
    valid_shape = [min(c, n - s) for c, n, s in zip(chunk_shape, metadata["shape"], start)]
    array = full[tuple(slice(0, n) for n in valid_shape)]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "source.zarray.json").write_bytes(raw_metadata)
    (args.output_dir / "source.chunk").write_bytes(encoded)
    np.save(args.output_dir / "array.npy", array, allow_pickle=False)
    result = {"metadata_source": metadata_source, "chunk_source": chunk_source,
              "chunk_index_zyx": args.index, "array_start_zyx": start,
              "valid_shape_zyx": valid_shape, "dtype": str(dtype),
              "nonzero_fraction": float(np.count_nonzero(array) / array.size),
              "value_range": [int(array.min()), int(array.max())],
              "role": "Observed CT intensity or surface-volume pixels; no ink inference"}
    (args.output_dir / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
