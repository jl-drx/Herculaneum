# Frag6 Stage B: retained-run verification

**PASS.** Independent parsing and numerical recomputation agree with the retained run. This review made no network request and read no native CT pixel, optical image, label or model. Only the already authorized, retained Stage B payload was decoded. All earlier sealed directories remain unchanged.

The run used one HTTP 206 multipart response with the exact pinned PPM ETag and object total. All 243 parts match the independently derived fixed schedule, with no missing, duplicate, overlapping, coalesced or unrequested part. The retained response is 64,747 bytes; its 34,992 useful bytes exactly reassemble the retained 729-record array. The full 222,640-byte reservation remains charged and there was no retry. The controller finished in 1.917 seconds; measured worker peak RSS was 19,791,872 bytes, below 256 MiB.

The independent check uses its own cursor parser, integer lattice derivation, double decoding, scaled vector norms and `math.fsum` reductions. It passed 6,669 scalar, framing and identity assertions. Diagnostic comparison tolerances were frozen before binary decoding: `5e-12` absolute plus `5e-12` relative. These reporting tolerances are separate from the unchanged `1e-5` unit-normal admission rule. No tolerance changed and no review failure occurred.

| Interpretation | Mapped records | Zero sentinels | Nonfinite mapped | Nonunit mapped | Admission |
|---|---:|---:|---:|---:|---|
| Little endian | 558 | 171 | 0 | 0 | Unique conditional pass |
| Big endian | 558 | 171 | 0 | 558 | Reject |

The maximum little-endian normal-length error is `2.22e-16`. Every mapped coordinate falls within the declared primary candidate's bounds. This establishes only invariant-consistent serialization and sparse compatibility; unit normals and bounds do not authenticate the producer, source UUID or energy scan.

Exactly 62 of the 81 fixed neighborhoods are completely mapped. The other 19 are wholly zero sentinel neighborhoods. This sparse lattice is not an estimate or certification of global mapped area.

Seven of the nine predeclared possible native-comparison sites are complete. The sites at raster `(y,x)=(1770,1240)` and `(1770,3102)` remain unmapped and were not replaced. The seven complete sites satisfy the minimum-six coverage condition only. They cannot supply the missing source-frame and render-provenance preconditions; **Stage C remains closed**.

Across the 62 complete neighborhoods:

| Diagnostic | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Raster-x tangent norm | 0.931054 | 0.963134 | 1.794237 |
| Raster-y tangent norm | 0.931111 | 0.962984 | 2.255288 |
| Local area scale | 0.868349 | 0.947390 | 2.539265 |
| Tangent-x / tangent-y cosine | −0.313301 | 0.027402 | 0.778602 |
| Stored normal / raster cross-product cosine | −0.999947 | −0.998743 | −0.964213 |

No complete neighborhood has zero tangent norm or zero local area. The maximum absolute stored-normal/tangent cosine is 0.158945 for raster x and 0.260222 for raster y, so local tangency is approximate. All stored normals oppose the cross product formed in raster x-right/y-down order. This observed coordinate orientation does not identify inward/outward anatomy, signed render offsets or ink depth, and does not justify flipping the stored normals or selecting a different depth window. No diagnostic threshold was fitted to these results.

The worker-result anchor is `586de6fe885c608fba92eda106b737ca2bfc71a9c5762793ce341e8541be25a1`; the geometry-result anchor is `584ad931220ebd6394937c40833af9e337bc727b266985322d9b4e49bc59db50`. `verification.json` contains the complete checks and independent summary; `independent-geometry.json` preserves all recomputed neighborhood diagnostics and the fixed-site schedule.

No CT reproduction, full-stack input contract, label agreement, model accuracy, tuning, transcription or paid computation is established or performed by this review.
