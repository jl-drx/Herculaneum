# Native annotation coverage: w035, w039 and w044

11 October 2026. **A useful native-frame annotation-agreement reference is available at modest data cost. It is not independently authenticated ink truth or a pristine held-out benchmark.** The annotation stage fetched 65,648 bytes of current published supervision/annotation shards, with no CT, model inference or training. All six arrays passed source identity checks, shard-index CRC32C/range checks, and full-array agreement between official Zarr and explicit index/numcodecs decoding.

The grid was frozen from supervision masks before any label shard was fetched. It retained every 128-square core on a global 128-pixel grid with at least 1,024 supervised pixels and a complete 256-square canvas context. No core was selected for positive ink. These are canvas-contained contexts; native CT/material validity has not yet been read or qualified.

| Native winding | Eligible cores | Supervised pixels in cores | Ink within core supervision | Non-ink within core supervision | Cores with both classes |
|---|---:|---:|---:|---:|---:|
|w035|109|1,066,610|333,871|732,739|75|
|w039|54|571,603|182,380|389,223|36|
|w044|102|1,187,621|450,775|736,846|86|
|Total|265|2,825,834|967,026|1,858,808|197|

All 265 cores remain in the record, including 67 without supervised ink and one without supervised non-ink. `*-eligible-core-class-counts.csv` gives every result. Adjacent contexts overlap; cores/pixels are not independent experimental replicates.

## The supervision mask matters

Every array contains only 0 and 255. Stored ink labels outside the supervision mask are substantial in w035:

| Winding | Full-array supervised pixels | Stored ink pixels | Ink inside supervision | Ink outside supervision |
|---|---:|---:|---:|---:|
|w035|1,071,121|396,164|334,035|62,129|
|w039|576,220|184,948|182,429|2,519|
|w044|1,190,307|455,469|451,297|4,172|

The outside-mask labels are reported and excluded from supervised class counts. Do not interpret an unlabelled/unsupervised pixel as a negative, or simply score against every nonzero label. For w035, 15.68% of its stored ink lies outside declared supervision. This does not establish why the released arrays differ; it establishes the support semantics a future endpoint must respect. The three arrays' stored total-ink and supervision counts agree with current release metadata.

## Prospective CT cost

The union of native 128-square chunks needed for all eligible 256-square contexts is:

| Winding | Unique chunks | Bytes, all 28 planes |
|---|---:|---:|
|w035|231|105,971,712|
|w039|125|57,344,000|
|w044|180|82,575,360|
|Total|536|245,891,072 (234.5 MiB)|

No CT was fetched. Exact chunk coordinates are retained in `MASK_COVERAGE.json`. This is the complete deterministic intake budget, not a request to train on every winding. Future train/development/evaluation roles and capacity/exposure controls must be declared separately. Keep alternate surfaces and physical aliases in the same winding group; w044 is the aligned-training alias `pherc0139-w028`. Known upstream model/training exposure prevents calling these newly discovered pristine test data.

## What this unlocks—and what remains

These native annotations provide a potential endpoint separate from the fine-scan teacher whose supervision operators are under comparison. They require no new fine78-to-native registration to define their declared native raster coordinates. They can support a carefully named **current native annotation-agreement** experiment after CT support/context qualification and a frozen physical split.

The unavailable private native-rerun annotation-transfer history still limits physical alignment claims. There is no per-pixel independent mapping-validity certificate. The current public arrays are pinned current objects, not verified historical override bytes. Their source-plane14 storage does not establish physical ink depth. This result does not rehabilitate the failed w040 fine/native correspondence gate, prove sealed-scroll transfer, admit training, or justify GPU spending.

The practical next qualification is prospective native CT support and an exposure-aware split, using the preserved full core list and declared annotation support. The current target-operator comparison can proceed independently of that future training design.

## Evidence and replay

`PROTOCOL.md`, `FREEZE.json`, `SOURCE_PLAN.json` and `MASK_STAGE_COMPLETE.json` record the declared protocol/source/code pins and guarded mask-before-label execution; the independent review states the timing limits. `REQUEST_LEDGER.json` contains each exact URL, current ETag, length, SHA256 and retrieval time. `MASK_DECODE_CHECKS.json` and `LABEL_DECODE_CHECKS.json` preserve decoded hashes and decoder agreement; `SYNTHETIC_CHECKS.json` records six passing fixture/rejection checks. Sources and admission caveats are under `provenance/`.

Only the six tiny encoded shards are stored; offline decoders reconstruct the full arrays. No prediction panels or manual image interpretation were performed. Install the pinned packages in `requirements.txt`, then run:

```bash
python replay.py --output /absolute/path/to/a/new/replay-directory
```

The offline replay validates frozen inputs and source hashes, both decoders, every grid/core count and every prospective CT chunk. `intake.py` is the original acquisition implementation and is retained unchanged after `FREEZE.json`; ordinary review should use `replay.py`, which has no acquisition path.

A separately frozen CT follow-up is recorded under `ct/`; its receipts and report distinguish native input acquisition from the annotation-only stage above.
