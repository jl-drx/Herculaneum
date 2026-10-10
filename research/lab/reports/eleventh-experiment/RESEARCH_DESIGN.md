# Eleventh study: transfer to an optical-derived fragment reference

Frozen scientific choices recorded before CT crop decoding or any new model prediction, 8 October 2026. Root owns scientific interpretation. The executable plans will bind the completed acquisition, exact model/code identities and resource limits before inference.

The question is whether the ninth and tenth continuation effects persist on another specimen, using the unchanged seed42 initializer, A_smooth, B_smooth, A_preserve and B_preserve. There is no new training, coefficient search, checkpoint selection or reference-based calibration. The previous PHerc0814 panel remains development evidence.

## Reference and claim

Use PHercParis1Fr39, legacy Frag4, at 54 keV and nominal 3.24 µm. This specimen is distinct from the PHercParis4 scroll named in the current models' explicit configurations. Frag4 is absent from the 29 configured representations audited for these initializers, and absent from our retained continuation/calibration work. Upstream initialization, teacher ancestry and historical pixel exposure are not fully known. This is a new specimen in our current evaluation, not a proven historically unseen test.

The EduceLab-Scrolls publication (arXiv2304.02084v4, A.5–A.10) describes manually warped 1000 nm IR and ink tracing under papyrology supervision. Its linked fragment release explicitly associates the surface stack, labels and surface mask. The historical ink-ID loader treats supported zero/nonzero labels as supplied nonink/ink. This supports a supplied-reference agreement diagnostic with stronger optical provenance than the reused scroll references. No independent reannotation, quantified registration residual, hidden-scroll reading or frontier-performance claim follows.

The root has acquired the full small label/mask/IR PNGs and inspected their formats. They share the declared 9791×6064 canvas. Labels and mask use palette index0=black/index1=white, with no transparent label class. All4,610,281 marked pixels lie within the44,404,741-pixel surface support. Matching dimensions and this containment do not measure registration accuracy. The original IR and registration PSD were not downloaded.

Modern PHerc0009B remains reserved but is not used here. Its intended canvas offset is already applied during transfer; the exact dirty publication is unreproduced, the advertised mapping-validity asset is missing from the complete inspected label prefix, and annotation depth is unresolved. Do not repair these gaps by shifting predictions or inventing validity.

## Selection and participation

The frozen label-only selection uses complete1152×1152 native-grid contexts, stride1152 from origin0,0. At coarse pixel(i,j), labels use native(3i+1,3j+1), the same center as a3×3 CT mean. A participating pixel requires constant supplied label and surface support over its entire33×33 native neighborhood. This symmetric16-native-pixel exclusion is our conservative variant, not an exact recreation of PIL FIND_EDGES followed by MaxFilter in the original benchmark. Exclude a32-coarse-pixel context border from primary metrics.

Eligibility requires at least90% native surface coverage,512 participating positives and8192 participating negatives. Choose the lexicographically first eligible(Y,X) context, then the last eligible context at least2304 native pixels farther in Y. No ranking by model score, IR appearance or amount of ink. Stop rather than relax eligibility if the second context is absent.

The executed label-only selection found25 eligible contexts among40. It chose native(Y,X,H,W)=(0,1152,1152,1152) and(8064,3456,1152,1152), with primary positive/negative counts3818/87626 and15252/62599. An independent integral-image implementation reconstructed all candidate counts and all eight output arrays. Two contexts from one fragment are not independent specimen replicates; canvas separation alone is not a native-volume separation proof.

## CT and input representation

The publication explicitly sets radius32 and normal step1; index32 is therefore the declared segmented surface. The stack metadata and region linkage are consistent with this convention. The actual2023 executable/arguments are not reproduced, and anatomical normal orientation relative to the modern detector is uncertain.

Acquire only rows needed for the two contexts from exact released TIFF layers07 through57, inclusive. Every TIFF must independently satisfy the fixed little-endian uint16, one uncompressed strip,6064×9791, minimum-black layout. Strong ETags bind metadata and subsequent ranges; all ranges, bytes and hashes are retained. No native scan, full TIFF plane, full PPM or registration PSD is needed. Because the TIFF uses one full-width strip, row bands contain unused columns: planned raw CT transfer is1,425,088,512 bytes, plus structural metadata. The stage cap is1.6GiB and260 attempts with no automatic retries or redirects.

Group3×3×3 native voxels using exact uint32 sums, divide in float64 by27×257, then cast once to float32. There is no uint8 quantization. The result is17×384×384 with nominal9.72µm spacing, depth centers8,11,...,56 and center32. This is our new preprocessing, not the historical ink-ID pipeline or the existing2.399µm/factor4 render recipe. Its spacing is approximately1.29% larger than9.596µm used in the previous pooled reference. The unchanged robust-MAD normalizer follows this conversion.

The primary input uses the source's forward order. A predeclared secondary diagnostic reverses depth for every model/context. It reuses the same CT data. Report both directions separately; never select the better direction per model or replace the primary result after inspection. No XY warp or fitted offset is allowed.

## Inference and comparisons

Run all five retained states on both contexts in both directions:20 context/model/direction runs, at most500128×128 tile forwards through the unchanged reference adapter. Preserve each raw tile logit, tile placement, blended probability map and evaluated mask. A logit reconstructed from a blended probability is not a raw model logit.

Report per-context and equal-context macro AUC/AP, plus the pre-existing model-specific primary and secondary thresholds with sensitivity and specificity. Never fit a threshold on Frag4 or assume an old95%-specificity calibration retains95% specificity here. Compare A_preserve−A_smooth and B_preserve−B_smooth, and every variant versus the unchanged baseline. Report all arms and directions regardless of outcome. The primary mask is fixed before inference; require all its pixels to have valid finite predictions rather than silently shrinking the domain after failure.

These are descriptive results. No pixel-IID significance test, confidence interval, pass/fail frontier claim or generalization guarantee is justified by two contexts from one historically uncertain specimen. New training is not an automatic next step if transfer is poor; input-domain and optimization explanations must remain separate.

## Operational limits

CPU only, four Torch threads,15-minute total inference wall cap,3GiB worker RSS. No optimization/backward calls, paid GPU, R2 write, external publication, submission, outreach or terms acceptance. Preserve the£10 total first-pilot budget, with£0 spent. Applicable legacy source restrictions and EduceLab citation requirements remain recorded; retained raw data is private and must not be added to GitHub.

The initial structural probe read16 TIFF bytes where8 would suffice. The next8 zero bytes are confirmed inside the CT strip; they were incidentally retrieved and not decoded. This limited deviation is retained verbatim, and subsequent structural reads use the exact8-byte header and validated IFD bounds. It is not described as zero CT bytes. All successful quality-stage request bodies are accounted for; future failure accounting conservatively charges in-flight reads.
