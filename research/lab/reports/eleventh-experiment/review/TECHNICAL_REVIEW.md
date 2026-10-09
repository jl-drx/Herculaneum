# Eleventh-study technical review

## Bounded indexed-shard decoder

**PASS for the supported storage format.** `reader-preparation/indexed_shard.py`, SHA256 `1cc42ed68ad0e9e23fc9741364655b5aba85fe1c50936e768355543a1a8ea6e1`, rejects unsupported metadata, checks the trailing CRC32C index, validates every absolute payload interval without uint64 wraparound, rejects overlap and half-missing sentinels, and checks each selected Blosc header before bounded output-buffer decompression. Canonical absent chunks remain fill zero with a separate encoded-payload-presence diagnostic; that diagnostic is explicitly not participation or geometric validity.

The independent synthetic run passed 44 cases. It re-executed the owner's 41 cases without rewriting their receipt, and added the known Castagnoli check vector, a separately generated genuine compressed Zstd/bitshuffle fixture spanning partial edge chunks, and reversed physical payload ordering to test nonmonotonic absolute offsets. The synthetic arrays were generated with installed Zarr and compared both with their known construction and standard Zarr reads. Receipt: [reader-independent-receipt.json](reader-independent-receipt.json), SHA256 `c1a0d9f717d3a82634384ffc81d0195b404703746deec85e3b384e1eda2dde01`.

This establishes the narrow decoder contract. It does not establish source authenticity from CRC alone, transport correctness, label semantics, coordinate registration, geometry validity, annotation depth, or scientific independence. No candidate shard was acquired or decoded by this reviewer.

## PHerc0009B source contract

Read-only review of `surface-contract/CONTRACT.md` found no material correction to its distinctions: intended transfer offsets are baked into target-canvas output, while the actual dirty publisher execution remains unverified; missing mapping validity is not replaced with S/V participation; renderer plane 54 is not promoted to annotation depth; and categorical-decimation versus CT-mean kernel phase is not presented as a measured alignment residual or an authorized correction.

## Optical mechanical-quality observations

Read-only inspection of the quality report, executed acquisition code, request ledger, and TIFF structural observations supports the reported raster dimensions and arithmetic. All three PNGs report a 9791-by-6064 canvas. Ink and mask are palette-index images, so index 0/1 counts require a recorded PLTE/tRNS check before describing their displayed polarity; even confirmed black/white polarity would not by itself establish exhaustive supervised negatives. Equal dimensions do not establish registration accuracy.

The TIFF tags establish one uncompressed 16-bit strip starting at byte 8 with length 118745248, ending exactly at IFD offset 118745256. Consequently the original 16-byte header probe included **eight bytes of the CT strip**, rather than merely potentially doing so. They were retained as zero bytes and not decoded as CT pixels. The original plan/code and this scope deviation must remain in the record. The subsequent IFD wrapper checks the actual object bounds and requests only its count/table. No additional structural blocker was found.

The original transport's retained-body accounting can understate consumed bytes if a low-level read raises after consuming a partial fragment. A future transport should charge failed reads conservatively and distinguish that charge from retained bytes. The observed seven requests succeeded, so that failure path does not alter their recorded 12189137-byte total.

This reviewer performed no network requests, candidate-array reads, image decoding, model construction, inference, training, or paid operations during these checks. The only executed data checks were tiny synthetic reader fixtures.
