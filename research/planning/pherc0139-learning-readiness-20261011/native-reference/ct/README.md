# Native CT inputs and support are ready

11 October 2026. All 536 declared native CT chunks were acquired and verified in 54 seconds: **245,891,072 bytes (234.5 MiB), no failures or retries**. Every object has its original response ETag, length, Last-Modified, SHA256 and retrieval timestamp. Both reserved and observed body-byte totals remain below the frozen 256 MiB cap. First-read identities are current-object evidence, not historical whole-volume hashes.

The frozen mask-only cohort is unchanged:109 w035 cores, 54 w039 cores and 102 w044 cores. **Every pixel of every 256-square context has at least one nonzero value across its full28-plane native depth window.** All 2,825,834 supervised core pixels therefore retain this native array support; no supervised positive or negative pixel is lost to all-zero CT. This does not establish correct surface placement, sheet identity or ink accuracy. The separate mesh audit owns geometric/physical-overlap qualification.

`support/RESULTS.json` and the three CSVs preserve all 265 core results. `support/masks/` stores only small compressed core/context validity masks and origins. The source chunks are retained once under `chunks/`; no duplicated full patch tensors are stored.

The acquisition plan and implementation were frozen in `FREEZE.json` before requests. `BODY_BUDGET.json` reserves an entire expected body before every attempt, conservatively accounting for interrupted attempts. The public acquisition had no interrupted/failed attempts; no claim of a live crash/retry test is made. Completed chunks support verified local reuse by exact length/SHA256 receipt. Three synthetic construction checks cover asymmetric chunk assembly, nonzero context origin and dependence on the final native plane 27.

Offline replay independently reduces each source chunk across depth first, constructs global XY validity mosaics, and compares every saved core/context mask against those mosaics. All 536 source hashes,265 context coverages, origins and validity masks agree. This differs from the original implementation, which first reconstructs28×256×256 contexts and then reduces depth. Results are in `OFFLINE_REPLAY.json`; run again with:

```bash
python replay_ct.py --output /absolute/path/to/a/new/ct-replay-directory
```

The source chunk archive must be restored under this `ct/` directory before replay. No network request, model or paid resource is involved in replay. No model was trained or inferred during this intake. GPU spend: £0.
