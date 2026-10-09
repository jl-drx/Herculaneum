# Verified local capabilities

Verified 8 October 2026 in the restored cloud environment: Python 3.12.14, four CPU threads, CPU PyTorch 2.12.0, no GPU execution.

- `bash scripts/setup_cpu.sh` completed the hash-pinned environment sync, editable project installation, dependency compatibility check and **60 passing tests**. No tests skipped. The tests cover local HTTP acquisition, cache limits/pins, corruption and interruption, local/fake-R2 artifact publication, CLI integration and GPU budget declarations.
- `python scripts/verify_sources.py` verified four unchanged adapters, 30 pinned upstream modules and four retained licenses.
- A live public S3 surface chunk, 507,904 bytes, matched the retained SHA-256 and ETag. A second CLI fetch reused the verified local object.
- `scripts/smoke_baseline.py` strictly loaded one retained official checkpoint and scored one genuine 128 by 128 patch through the relocated adapter. All 16,384 evaluated pixels had finite scores in [0,1]. Altered input provenance was rejected before output creation. Original input/checkpoint hashes stayed unchanged.
- The CLI published and independently verified the four real inference artifacts in a local immutable store.
- Astra reviewed cache/artifact semantics and the budget validator. Its schema/currency and malformed-input findings were fixed and covered by regression tests.

Large outputs and detailed evidence remain in the lab snapshot, outside Git. The local verification artifacts are under `/workspace/herculaneum-lab/reports/infrastructure-smoke-20261008` and `/workspace/herculaneum-lab/artifacts-infrastructure`.

The session-token increment passed **62 tests with no skips**, including 30 artifact tests. Real botocore signing checks show that `R2_SESSION_TOKEN` enters the signed headers and changes Authorization at a fixed time; a permanent pair ignores an unrelated `AWS_SESSION_TOKEN` when the R2 token is unset or empty. These checks use dummy values behind a network tripwire.

Separately, the authorized local setup task reported ten live R2 artifact checks against the original implementation, followed by six scope/signing checks for a bucket-only temporary credential triplet. The latter include denial on another bucket and denial without a session token. They are separate evidence from the offline tests; live verification of the updated `from_env` path and actual Runpod worker secret delivery remain pending. No verified CUDA image, GPU inference/training, provider shutdown or cloud billing test is claimed. The native reference specification remains blocked on its explicit scientific and operational prerequisites. No paid GPU job, new reading or translation occurred.
