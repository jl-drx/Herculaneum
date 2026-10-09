# Draft: Frag6 annotation and surface-render provenance

**Status: prepared for review; not sent or posted.** Intended audience: Vesuvius Challenge data maintainers / EduceLab-Scrolls curators. The retained official data documentation lists [GitHub support](https://github.com/scrollprize/villa/issues) and [the community Discord](https://discord.gg/V4fJhvtaQn). It does not identify an individual custodian for this fragment. This draft contains no unpublished text, images, credentials or model results.

---

We are preparing a reproducible ink-detection evaluation using Frag6, also named PHerc51Cr4Fr8 / PHerc0051Cr04Fr08. We have kept this parent out of our training and model evaluation. Could you point us to existing records or the appropriate curator for its annotation and rendering history?

The exact package is:

https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/

The relevant assets are `inklabels.psd`, `inklabels.png`, `ir.png`, `mask.png`, `result.ppm`, and `surface_volume/00.tif` through `64.tif`. The PSD is 86,107,922 bytes, server ETag `"656fa2a5-521e712"`; its header is 6205 × 8853 pixels, grayscale, 8-bit. Its embedded XMP records Photoshop 25.2 editing on 1–5 December 2023 and conversion from TIFF. Its OriginalDocumentID is `xmp.did:a9afc308-3230-4011-bd20-fd8566cf1fbb`. We inspected metadata only, not its image or layer content, and have not inferred an annotation author from the software field.

We would especially appreciate existing records for:

1. **Annotation origin:** who produced/reviewed the labels, which optical image or other evidence was used, whether any predictions or reconstructed letter forms influenced them, and how the current PNG relates to the PSD.
2. **Support semantics:** which pixels are confidently uninked, unannotated/unknown, or invalid; the role of `mask.png`; and whether labels are intended to be exhaustive within any defined region.
3. **Optical registration:** the exact source/target image identities, transform or preparation steps, validity region and any measured registration residuals. If no residual assessment exists, that is useful to know.
4. **Surface rendering:** the source CT UUID and coordinate frame associated with `result.ppm`, renderer version/arguments, sampling/interpolation rule, plane-to-signed-offset mapping, spacing, normal orientation and any rescaling/cropping. The co-located 53 keV / 3.24 µm volume is a candidate; we have not treated its pathname as proof of the render recipe.

Logs, archived commands, manifests, hashes or a curator's description with its limits would all help. We do not need newly computed results or a guarantee that no model has ever seen this parent. If you know of historical Ink-ID use or alternate specimen names, pointers would help us investigate model-specific exposure separately. “Unknown” or “not retained” is an acceptable answer; we will record the uncertainty.

Our aim is to distinguish reproducible agreement with supplied annotations from optically anchored accuracy and from model-specific independent evaluation. Thank you for any pointers to the existing evidence.

---

## Handling a reply

Retain the reply or public URL with date, author/custodian role and exact artifact scope. A maintainer statement is attributable evidence, not a substitute for any claimed numeric registration measurement. Populate the corresponding rows of EVIDENCE_GAPS.json; do not fill unknown fields from defaults. Do not change candidate reservation or open inference automatically. Root should assess the usable input/annotation contract, then freeze the next acquisition/evaluation design before predictions.
