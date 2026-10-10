# A fixed Frag6 reference panel is now available

Four reference contexts have passed mask/label checks and been selected without model predictions. Their model crops and scoring masks are now bound before inference. This supports a **supplied-reference agreement** experiment; independent ink truth and model performance remain unestablished.

## Reference quality

All three public PNGs share a 6205×8853 canvas: grayscale mask, binary black/white palette labels, and grayscale optical pixels stored as RGBA. PNG CRCs and retained SHA256 identities were checked. Of 54,932,865 pixels, 36,912,011 have white-mask support: **4,273,526 supplied positives and 32,638,485 supplied negatives**. No positive lies outside the mask; neither mask nor labels contains ambiguous values. Binary encoding does not prove that unmarked areas are exhaustively confirmed negatives.

The frozen rule examined 187 nonoverlapping 512×512 grid tiles; 87 met coverage and class requirements. It selected the first eligible tile in each vertical quarter. All exclusions and unselected tiles remain in TILE_TABLE.json. Optical appearance and model scores did not affect selection.

| Tile origin (row, column) | Positives | Negatives | Excluded |
|---|---:|---:|---:|
| 544, 4128 | 66,452 | 195,003 | 689 |
| 2080, 3616 | 93,475 | 168,669 | 0 |
| 4640, 1056 | 52,044 | 210,100 | 0 |
| 6688, 1568 | 27,849 | 234,295 | 0 |

These are four annotated development contexts on **one PHerc51 parent**. Frag6 is now CT-, label- and optical-inspected; its model scores remain uncomputed. PHerc0009B, PHerc0500P2 and PHerc0841 remain untouched. Upstream model/teacher exposure is unresolved, and previously seen public aggregate benchmark scores prohibit a blanket blindness claim.

## Optical consistency and its limits

All four contour overlays approximately follow visible dark strokes, and supplied positives are darker on average in each tile. The optical image is blurred at native resolution while labels have sharper outlines. No measured registration residual, authenticated annotation author or exhaustive negative policy has been established. Whole-image verification confirms opaque optical support on every white-mask pixel and no color-conversion ambiguity.

See the [overview](qc-v1/optical-overview.png) and [optical review](OPTICAL_REVIEW.md). No offsets, polarity, warps or labels were adjusted. This consistency is not a CT model score or measured optical-to-CT registration.

The PNG names GIMP 2.10 and has a different embedded OriginalDocumentID from the previously inspected Photoshop PSD. This adds an editing-history clue without authenticating a direct derivation, author or chronology. No maintainer message has been sent.

## The next comparison is concrete

Each model crop is the central 384×384 region of a selected tile. Fixed 3×3 XY pooling gives a 128×128 input; scoring uses its central 64×64 core. A pooled label is known only when all nine source labels agree and all have valid mask support. Mixed blocks remain unknown. A separate scalar implementation agreed on all 65,536 pooled blocks.

| Context | Core negatives | Core positives | Core unknown | Positive fraction |
|---|---:|---:|---:|---:|
| r544-c4128 | 1,088 | 2,920 | 88 | 72.85% |
| r2080-c3616 | 549 | 3,485 | 62 | 86.39% |
| r4640-c1056 | 3,942 | 129 | 25 | 3.17% |
| r6688-c1568 | 3,652 | 388 | 56 | 9.60% |

Every core has both classes. Total scored support is 16,153 coarse pixels; 231 are unknown/excluded. Differing positive fractions make absolute APs difficult to compare between contexts: report prevalence alongside AP and retain paired differences. Pixels and portions of one stroke are not independent research subjects. Homogeneous-block scoring omits mixed boundaries and cannot demonstrate boundary precision.

[CPU_STUDY_SPEC.md](CPU_STUDY_SPEC.md) fixes historical layers 01–63, exact 3×3×3 pooling, five 17-plane windows, unchanged forward seed42, and equal averaging of three neighboring probability maps at shifts −1, 0 and +1. The maximum is **20 CPU forwards**. Source index spacing is explicit; anatomical displacement claims remain qualified. Both arms share rendering, support and normalization.

Before any scores, practical screening margins were fixed at mean robust AP gain ≥0.01 and mean clean AP loss ≤0.01. They select a candidate for replication, not statistical significance or model promotion. This is an averaging baseline, not a novel model claim.

The CT request manifest, each TIFF identity/layout and model-runtime freeze still need implementation and verification. The four crops across 63 layers contain 74,317,824 useful CT bytes; bounded multipart ranges avoid whole-layer downloads. Missing annotation-author history does not block this qualified comparison. No CT or model execution ran under this phase's PNG-only acquisition contract.

## Verification and resources

Four focused QC tests passed. A separate root implementation verified all three source hashes, whole-image histograms and joint support, all 187 tile counts, four-tile selection, four label arrays and eight optical histograms. The scalar pooled-label check also passed. No independent-agent review is claimed. Qualification covers these observed encodings; the generic helper needs additional handling before reuse on grayscale/RGB PNGs with tRNS transparency. The actual mask has no tRNS, and palette labels were correctly converted through RGBA.

Three HTTPS requests returned 15,113,665 body bytes, with 23,068,675 reserved. No redirects, retries or raw-scan downloads. QC took 9.215 seconds with 913,137,664 bytes peak RSS, within the 1 GiB cap; the separate verifier took 2.265 seconds with 370,106,368 bytes peak RSS. Plotting emitted nonfatal fontconfig-cache warnings; all figures completed. Future plotting should set both MPLCONFIGDIR and XDG_CACHE_HOME to fresh writable directories.

Repository tracked files and prior evidence are preserved. **GPU spend remains £0; total first-pilot cap £10.** No training, Frag6 inference, transcription, translation, R2 upload, publication or submission occurred.
