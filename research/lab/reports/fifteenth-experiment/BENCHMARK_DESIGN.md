# Plinius optical-reference benchmark: frozen roles and label-only construction

Frozen before new image/label payloads or decoding. This stage constructs support for a later reference comparison; it performs no CT acquisition, model prediction, calibration fit or training. Root owns scientific review. Source contract SHA-256: `53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890`.

## Roles and statistical units

| Physical fragment | Parent papyrus | Locked role | Number of selected contexts |
|---|---|---|---:|
| PHercParis1Fr34 / Frag3 | Paris1 | New acquisition-domain calibration | 2 |
| PHercParis2Fr47 / Frag1 | Paris2 | Reserved from our training/calibration; evaluation | 2 |
| PHercParis2Fr143 / Frag2 | Paris2 | Reserved from our training/calibration; evaluation | 2 |
| PHercParis1Fr39 / Frag4 | Paris1 | Existing seen diagnostics; unchanged | 0 new |

Keep each whole fragment and its aliases/scan versions within its declared role. Never use the two Paris2 fragments for gradients, teacher targets, normalization fitting, cutoff fitting, crop/depth/orientation search or model selection. Label inspection in this stage serves predefined support selection and quality checks, so the evaluation labels are not described as blinded. No local prior scientific acquisition/use of Frag1–3 was found; unknown upstream teacher/pretraining exposure still prevents an upstream-independent test claim.

The benchmark adds three physical fragments but only two parent papyri. Its two evaluation fragments share one evaluation parent, distinct from the calibration parent. Report per-context and per-fragment results and an equal-fragment Paris2 summary in a later study; do not count four evaluation crops as four independent specimens, or these two fragments as two independent parent papyri. Calibration parent Paris1 already contains the seen Frag4 diagnostic; disclose this history.

Frag5 has current label/mask/IR assets but shares the already used PHerc1667 parent. Frag6 offers another parent and also has those assets. Their inspected current directories contain 65 TIFFs but no surface meta.json. Neither is admitted here because the retained four-fragment publication does not establish their specific annotation/render-depth contract. Defer them explicitly, without declaring their assets absent or universally unusable. No replacement of a failed selected fragment with Frag5/6 in this stage.

## Source identity and label-only acquisition

Use only the six exact PNG URLs in SOURCE_CONTRACT.json: inklabels.png and mask.png from each original 54keV_exposed_surface directory. Their expected native YX canvases are Frag1 [8181,6330], Frag2 [14830,9506], Frag3 [7606,5249], taken from each current rendered surface meta.json. The publication-backed convention and matching current metadata support 54keV, nominal3.24µm, 65 planes and surface index32; exact historical renderer arguments, optical-registration residual and publication-to-current-file hash bridge remain unverified.

Sequential acquisition order: Frag1 labels/mask, Frag2 labels/mask, Frag3 labels/mask. Six HTTPS GET attempts maximum, 512KiB per response and 2MiB combined encoded bodies. Require verified TLS, HTTP200, identity encoding, PNG content, complete response and declared-length agreement when supplied. No redirects, retries, credentials or alternative URLs. Record every attempt including failures, bytes, source URL, HTTP identity fields and local SHA-256. An oversize/invalid/failed object stops dependent work; retain partial receipts without substitution. No registered IR, photographs, PSD/PPM, TIFF headers or CT bytes are requested.

Decode only after all six objects are complete and hashed under the frozen plan. Validate PNG signature/CRC, exact declared canvas and a maximum150,000,000 pixels per image before full decoding. Allow only binary-compatible modes1/L/P. For P, used background index0 must map to black and the allowed nonzero index to white; reject ambiguous/inverted palettes and transparency. Accept raw Boolean, uint8 {0,1}, or uint8 {0,255} binary encodings and canonicalize membership by nonzero, consistent with the published consumer. Do not infer or flip polarity from image appearance. Retain raw mode, used values and palette facts. Record positive labels outside surface support but exclude them from participation; do not turn unsupported pixels into negatives. Selected support must still meet all gates below.

## Exact deterministic selection

Reuse the eleventh Frag4 spatial/label contract, now parameterized only by fragment identity and declared canvas. No scans, model scores or optical photographs enter selection.

1. On the full native binary canvas, a pixel is known only if all surface-mask values in its centered33×33 square are true and all supplied labels in that square are identical. Outside-canvas values are zero. This is the existing conservative symmetric16-native-pixel exclusion, not a claim of reproducing the paper's FIND_EDGES procedure exactly or of measuring registration uncertainty.
2. Enumerate complete native1152×1152 contexts on a1152-pixel origin-aligned grid in row-major order. No padding or partial context.
3. A coarse384×384 label uses native positions(y+1+3i,x+1+3j), matching the center of a future3×3 CT mean. Retain the coarse sampled surface mask, known mask and canonical uint8 labels. Primary participation is known mask intersected with a32-coarse-pixel context border exclusion, leaving [32:352,32:352].
4. A context is eligible iff at least90% of its full native pixels are inside the surface, checked exactly as10*inside_count >=9*1152*1152, and its primary participation has at least512 supplied positives and8192 supplied negatives. Unknown pixels never supply targets. Retain counts for every candidate; no criterion relaxation.
5. Select the first eligible row-major context and the last eligible row-major context whose native y is at least2304 beyond the first. Both must exist. These contexts do not overlap in source-canvas coordinates; native3D disjointness remains unmeasured. If any fragment lacks either context, mark benchmark construction incomplete and do not replace its role, shrink masks, move the grid, lower support gates or choose from predictions.

Save four arrays per selected context: canonical labels, surface mask, known mask and primary mask, together with exact source XYWH/YXHW, role, parent group, class/support counts, source receipts and hashes. Unknown regions remain distinguishable from supplied negatives. This label-supported sample is intentionally selected for measurable class support, not a random estimate of whole-fragment prevalence or performance.

## Resources and verification

Before root runs the stage, freeze implementation, pinned input plan and narrow technical review. Source fixtures may use synthetic arrays only. Root performs the real six downloads and subsequent three selection workers, sequentially in Frag1/Frag2/Frag3 order; no scientific retry. Worker caps:120 seconds wall,360 CPU seconds, four threads,3GiB peak-RSS acceptance,12GiB address space. Combined selection controller cap420 seconds; download controller cap180 seconds. Total acquisition/selection outputs <=8MiB. No new package installation. Network is disabled in selection workers and analysis.

An independent checker, without importing the author selection functions, must reconstruct binary membership/palette facts, known support, all candidate counts, exact90% eligibility, first/last separated choices and all24 selected arrays. Use independent integral-image or equivalent arithmetic, with synthetic boundary/exclusion fixtures before real arrays. Verify role/parent exclusions, array identities and source pins. No models, Torch, checkpoints or network in independent verification. Current worker RSS/time and downloaded-byte totals remain measured evidence, not estimates substituted for execution.

## Later benchmark question, not executed here

The intended first comparison keeps seed42 and the two thirteenth states unchanged. Compare the already frozen original calibration cutoffs with each model's new Frag3-only cutoff, selected using the same conservative negative-only95%-specificity rule and maximum across its two contexts. Lock all cutoffs before predicting/scoring the four Paris2 contexts. No Paris2 value selects a cutoff or model. Report ranking separately from operating performance and retain failures. This tests calibration transfer between different parent papyri within the declared optical-fragment acquisition domain, rather than another parameter sweep on Frag4.

Future CT acquisition still requires per-object TIFF header/IFD, shape/dtype/byte-layout and HTTP Range/ETag validation, a byte/request/storage budget and a frozen reader plan. A possible retained adapter is51 source planes7:58, factor3 XYZ pooling into17×384×384 float32 at nominal9.72µm; it must be bound explicitly in that future plan and must not be selected from prediction performance. This stage does not assert CT input readiness or launch those450 prospective tile forwards. No fifteenth model result is produced by label-only construction.

GPU spend remains GBP0 and first paid pilot cap GBP10 TOTAL across retries/overheads. No new credentials, GPU, artifact upload, publication, text-reading claim, prize submission, outreach or terms acceptance. Legacy release restrictions and EduceLab citations remain attached. Plinius is a working research-series name for credited model adaptations; benchmark construction is not a novel-method or authenticated-ink claim.
