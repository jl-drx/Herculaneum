# Phase28: one-pass chunk accumulation verified; Plinius depth study prepared

The scheduler now processes the retained geometry with **70chunk reads/decodes instead of200**, a65%reduction in decoding work, and **five batches instead of13**. Every sample value and validity flag matched the existing method exactly on the native-layout synthetic fixture. The remaining scientific priority is a documented reference for the fixed depth-uncertainty comparison; no new model performance claim follows from this implementation result.

## Qualification result

A pre-freeze development run passed24tests. The separately source-frozen qualification ran once and passed the same24tests with0failures/errors/skips, then completed the sparse native-layout comparison. Total frozen run4.935seconds; peak processRSS131354624bytes (~125.27MiB). No package installation, model execution, candidate image access or network call was used. All prior27phases and the existing checkout remain unchanged.

The new chunk_accumulator.py reuses the frozen phase26 contribution planner and phase27 strict local Zarr reader. It groups all positive interpolation-corner contributions in canonical ZYX chunk order, keeps original point identifiers, and releases each returned chunk before the next loader call. An explicit expected/processed counter tracks complete support. Missing and nonfinite source flags remain separate from geometric invalidity; zero remains valid. Output is returned only after every expected contribution is processed. Altered plans, corruption and interrupted execution stop the invocation. There is no persisted accumulator or resume API; any later recovery requires a fresh complete invocation after diagnosis.

Tests cover analytic/scalar multilinear interpolation, unchanged sampler agreement, chunk faces/corners/edges, batch-size invariance, invalid geometry, missing/nonfinite/zero-weight support, valid zero, duplicate points, physical edge-byte budgets, plan mutation, dtype/endianness/shape mismatch, terminal interruptions including after the final chunk, corrupt encoded objects and weak-reference checks for one live returned chunk. No codec/cache regression suite was unnecessarily replayed; the composed corruption path was exercised.

## Native-layout comparison

The reference Zarr writer created only the70required chunks in a synthetic sparse array with declared shape[8855,2260,6300], chunks[128,128,128], uint16 and the published Blosc-Zstd configuration. Values are the known function z+2y+3x at voxel centers. Every written chunk was read back with installed Zarr and compared exactly:146800640synthetic voxels checked. No full native-shaped array was allocated or read. These values are not scan intensities.

Both algorithms used the same synthetic object manifest and unchanged local reader, with no cross-batch cache. The baseline used the literal13contiguous point ranges from phase27. All729retained PPMrecords remain present:558mapped/171sentinels, three hypothetical depth probes and2187outputslots,1674valid/513invalid,13392positive-weight contributions. The original candidate source frame and3.24um physical probe assumption remain unverified; this experiment reads no candidate voxels.

| Measurement | Existing point batches | New chunk schedule |
|---|---:|---:|
| Source reads and decodes |200|70|
| Batches |13|5(16+16+16+16+6chunks)|
| Decoded source work |800MiB|280MiB|
| Encoded synthetic bytes read |9474655|3315894|
| Measured local execution |0.490041s|0.218413s|

Maximum sample difference between paths was **0**, with identical validity. Maximum error against the analytic truth was7.275957614183426e-12, below the prospective1e-10absolute tolerance. All expected and processed contribution counts match. Root's separate scalar second-pass reconstruction verifies coordinates, support, chunk union, saved values, byte identities and accounting; no independent-agent review is claimed.

The new planner took1.050524seconds; planning plus optimized execution was1.268937seconds. The baseline timing used a previously saved plan. Therefore the measured execution reduction is **not an end-to-end one-shot speedup claim**. Reusing a fixed plan may amortize construction, but no repeated-run/network/GPU benchmark was performed. Synthetic compression is not a native scan-transfer estimate.

Plan NumPy buffers total434025bytes and reported accumulator buffers26244bytes; these are not total process-memory accounting. The scheduler held at most one returned4MiBchunk, verified using weak references. The unchanged reader can temporarily hold decoded/cropped copies and encoded bytes; the64MiB limit applies to physical source work per batch, not RSS. Total support is bounded to128unique chunks/512MiB and10000points by this prototype; this is not a full-scroll scalability demonstration.

## Scientific next step

PLINIUS_DEPTH_STUDY.md now defines one narrow H2 comparison: unchanged forward seed42 versus a fixed equal average of three nearby forward17plane windows. Five unique windows from a21effective-plane union cover the imposed shifts[-delta,0,+delta]. Delta is one validated effective rendered plane interval, with its physical value and pooling pending the reference contract. Candidate operational cost is three model evaluations per output, despite sharing five windows across the paired diagnostic.

This is an ordinary fixed averaging comparator for a research hypothesis, not an asserted new architecture. Seed42 already trained with depth jitter up to2. The earlier negative reversal result is respected: no direction reversal, enforced invariance, per-region best-depth selection, weight fitting or checkpoint sweep. Primary reporting is paired AP/clean performance by context and parent; false-positive comparisons require separate calibration and report achieved specificity. Shared planes are not independent samples.

REFERENCE_CONSTRUCTION_PLAN.md distinguishes existing maintainer records from a newly documented CT/render/optical/annotation workflow. DEPTH_STUDY_CONTRACT.json binds the baseline artifact and comparison algebra but leaves reference identity, physical delta, annotation/support/registration, evaluation panel, practical decision margins and resource bounds unresolved. It is explicitly **not executable for model scoring** yet. Unknown upstream exposure remains a qualified field, not a reason to mislabel development as held-out or supplied labels as independent optical truth.

The immediate next work is reference qualification/construction. Do not turn the successful scheduler into another broad infrastructure project or repeat unchanged provenance searches. The user was asked asynchronously for maintainer-request status; no reply is recorded at report creation. No external message was sent. If new evidence arrives, evaluate it against phase25's gap matrix. Otherwise prepare a bounded new-reference protocol for one exposed fragment.

All reserved candidates remain unscored; Frag6StageC stays closed. Reused Paris contexts remain development. No model promotion, optimizer step, transcription, translation, publication, prize submission or R2upload occurred. GPU spend£0;£10TOTAL pilot cap and verified worker credentials/current all-in pricing/pinned CUDA/independent shutdown prerequisites remain unchanged.

Results, design and registerv24 are sealed locally. Saved environment startup instructions are separate from publication, external backup or verified future restoration.

## Public-source follow-up after the initial design

The user confirmed the maintainer request has not been sent. A targeted public search then found a recent Frag1–Frag6 depth benchmark and specific Frag6 transform/stitching issues. Read [PUBLIC_EVIDENCE_ADDENDUM.md](PUBLIC_EVIDENCE_ADDENDUM.md) and the [source audit](../frag6-public-reference-recheck-20261009/RESULTS.md). Our next action is to audit/adapt that public benchmark before constructing a replacement or sending a broad request. This separate follow-up used20text/metadata requests and247358responsebodybytes; the frozen scheduler experiment remained offline. No candidate pixels or model outputs were acquired, but third-party published aggregate metrics were read.
