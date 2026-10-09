# Bounded data access and GPU storage audit

Read-only audit, 8 October 2026. No repository edits, deletion, upload or GPU execution. The selected Git checkout is clean and has no tracked files; the prepared lab lives outside it.

## Current behavior

- S3 access is bounded HTTPS chunk acquisition. `fetch_zarr_chunk.py` reads 64 KiB HTTP blocks but buffers the complete object, retaining raw bytes, metadata, decoded arrays and SHA-256 provenance. Inference reads local memory-mapped `.npy` arrays; it does not stream scans during inference.
- `prepare_larger_region.py` downloads 144 chunks with four threads into an in-memory list before writing. The 1536-square acquisition cap is **69.75 MiB**; actual bytes were **72,122,368**, with two explicit S3 `NoSuchKey` objects reconstructed as declared Zarr fill. Other failures stop the acquisition. No shared cache, transient retry or per-chunk resume exists.
- Models and input tensors explicitly stay on **CPU**, FP32, batch one. `setup.sh` syncs the CPU-only PyTorch lock from the CPU wheel index; running it on a GPU image would replace its CUDA PyTorch installation. Outputs are saved after inference, without resumable tile state.
- Current storage is about **1.2 GiB** for the lab and **1.5 GiB** for its environment. Within the lab: checkpoints 264 MiB, larger acquisition 172 MiB, geometry planning maps 345 MiB. Acquisition retains both encoded chunks and assembled arrays. About 28 GiB remains on this workspace. R2 persistence and RunPod execution are unimplemented in the inspected helpers.

## Required worker changes

The assistant should submit a frozen job specification; the GPU worker should fetch bounded S3 data directly into fast local storage and return compact manifests/results. Use a separate CUDA image and dependency lock, explicit device placement and bounded batching/prefetch. Validate a fixed ROI against the CPU baseline and record actual device, driver, CUDA, precision and numerical differences.

Write and verify each chunk before marking it reusable. A cache key needs source prefix, volume/segment, level, chunk coordinate and snapshot identity. Add bounded retries and a resume ledger; cap disk, decoded RAM and concurrency. Preserve source metadata, retrieval times and content hashes. ETags are not universal checksums; resumed jobs must not silently mix changed source snapshots or treat arbitrary failures as fill.

Checkpoint tile/shard progress during GPU jobs, retaining Hann weighted sums **and weights**, evaluated masks and geometry/source masks. Upload verified artifacts to immutable R2 run prefixes and write a completion manifest last; object stores do not provide atomic directory promotion. Existing provenance embeds absolute paths: preserve archived bytes and provide a resolver or identical container layout, with a new execution manifest.

## Storage responsibilities

**GitHub:** reviewed code/tests, CPU/GPU locks, container definitions, job schemas, small frozen configurations/selections, source/model hashes, citations and compact reports. Promote code deliberately from the external lab. Exclude secrets, scans, environments, weights and bulk score arrays.

**R2:** durable private scores, masks, necessary coordinate maps, provenance/logs/reviews, resumable state and selected retained inputs; deduplicate by content hash where useful. Keep S3 authoritative rather than mirroring entire scrolls. Dense planning maps are reproducible from small TIFXYZ sources and need an explicit retention choice.

**Worker local:** bounded cache, decoded ROIs, prefetch and temporary outputs. Inject R2/RunPod credentials through scoped secret bindings. No present evidence was deleted.

Inspected: acquisition helpers, `run_ink_large.py`, `setup.sh`, dependency locks, and retained acquisition/run provenance.
