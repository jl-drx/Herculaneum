# Local indexed-shard decoder

`indexed_shard.py` is a pure local decoder. It opens no files or sockets and imports no model code. Root has authorized only synthetic decoder fixtures at this stage. The scientific acquisition, transform, depth and evaluation contracts remain separate and unfrozen.

```python
roi, encoded_payload_present, receipt = decode_shard_roi(
    complete_encoded_object_bytes,
    parsed_array_metadata,
    [x, y, width, height],
    maximum_encoded_bytes=frozen_object_cap,
    maximum_expanded_bytes=frozen_expansion_cap,
    expected_byte_length=frozen_inventory_length,
    expected_sha256=retained_object_sha256,  # after acquisition receipt exists
)
```

The supported format is a single outer YX shard, uint8 with fill 0, bytes + Blosc zstd/bitshuffle, and a little-endian CRC32C-protected end index. Metadata and all payload intervals are checked before selected inner chunks are decoded into bounded outputs. The expansion charge conservatively includes every selected inner chunk's full decoded size plus the ROI and storage-presence arrays. No scientific offset or depth transform is applied.

`encoded_payload_present` describes physical storage only. A valid omitted ink chunk supplies the declared zero fill, which may be an annotated negative where a separate participation mask permits it. Omitted supervision/validation chunks supply zero membership. Failed acquisition, malformed index or failed decode raises; those failures never become canonical fill or annotation background. This boolean must not be used as an acquisition-observed, geometry-valid or participation mask.

Synthetic fixtures are run locally with:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/reader-preparation/check_indexed_shard.py
```

The synthetic receipt records exact code identity and local Zarr parity/corruption checks. It establishes decoder mechanics only, not the correctness of public label transfer or downstream metrics. `READER_DESIGN.md` and `reader-source-receipts.json` describe retained source evidence and the unresolved source contract.

`check_indexed_shard_compressed.py` is the additive actual-zstd/bitshuffle fixture. The first suite passed 41 cases and the compressed suite passed eight; their separate receipts remain preserved. The independent reviewer also exercised actual compression, a Castagnoli known vector and nonmonotonic absolute payload offsets. No real label arrays, CT arrays or models were used.
