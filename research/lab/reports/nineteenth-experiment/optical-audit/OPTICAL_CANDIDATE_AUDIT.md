# Frag6 / PHerc51 candidate audit

**Decision: reserve Frag6 as a potentially new parent, but do not admit it as a fresh confirmatory evaluation set or start inference.** This audit materially improves source-volume identity and scan grouping. It does not authenticate the rendered depth origin, optical annotation lineage or complete model/teacher exposure needed for the proposed comparison. Frag5 is conclusively unsuitable for a *new-parent* claim: the official catalog explicitly identifies it as a fragment of PHerc1667 / Scroll 4, already used upstream and locally.

Only public text metadata was acquired: 18 attempts, 1,794,098 response-body bytes retained, no retries or automatic redirects, within the 24-attempt / 16 MiB limits. Seventeen responses have retained bodies, including one HTTP 404. The final material-descriptor request returned `application/octet-stream` and was rejected before reading its body; no further requests followed. No CT, label, optical image, mesh, PPM, scientific array or checkpoint was downloaded or decoded. No model, GPU, upload, credentials or outreach was used.

## Identity and newly resolved source geometry

The retained official full catalog, legacy package config and working-directory listing jointly identify `Frag6`, `PHerc51Cr4Fr8` and `PHerc0051Cr04Fr08` as this fragment. Reserve the whole PHerc51 / PHerc0051 parent group, all cradle/fragment aliases and all scan/resolution versions. This is one candidate parent, not four independent datasets. No cross-parent equivalence is inferred from numerical similarity to PHerc1451 or other numbers containing 51.

The new volume metadata establishes these native CT descriptors:

| Native volume UUID | Published volume name | Native voxel pitch | Native shape Z × Y × X |
|---|---|---:|---:|
| `20231121152933` | PHerc51Cr4Fr8, 53 keV, 3.24 um, merged | 3.24 µm | 8855 × 2260 × 6300 |
| `20231130112027` | PHerc51Cr4Fr8, 53 keV, 7.91 um, merged | 7.91 µm | 3683 × 1068 × 2724 |
| `20231201112849` | PHerc51Cr4Fr8, 88 keV, 3.24 um, merged | 3.24 µm | 8855 × 2260 × 6300 |
| `20231201120546` | PHerc51Cr4Fr8, 70 keV, 3.24 um, merged | 3.24 µm | 8855 × 2260 × 6300 |

All four descriptors declare `type=vol`, `min=0`, `max=65535`. Those intensity bounds do **not** independently establish TIFF bit depth, dtype, byte layout or actual image dimensions; those remain uninspected. A retained reconstruction log names the 53 keV / 3.24 µm merged source and confirms 8,855 slices, but contains no surface-render arguments.

The official package also publishes a `Transform3D` affine descriptor from `20231121152933` to `20231201120546` and lists analogous transforms to the other two volumes. This is useful evidence that repeated scans belong in one exclusion group. It is not an optical-to-CT registration transform, a validation of its residual error, or proof that current rendered planes used that transform. The raw matrix is retained rather than applied.

The modern catalog currently supplies a photo entry and a link back to the legacy package, with empty `scans`, `volumes` and `segments` maps for this sample. That does not mean its legacy CT is absent. The legacy `volumes_zarr` directory lists all four named scan/resolution versions; no Zarr arrays or chunks were acquired.

## What remains unresolved at the rendered surface

The exact prospective association is the co-located directory:

`https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/`

Its phase 15 retained listing has `inklabels.png`, `inklabels.psd`, `ir.png`, `mask.png`, `result.ppm` and `surface_volume/00.tif` through `64.tif`. Co-location identifies candidate assets; it does not establish their registration residual, annotation author/method, label completeness or human-only origin. No image header or content was inspected here. Thus canvas dimensions, encoding/polarity and validity semantics also remain unverified.

A direct metadata probe of `surface_volume/meta.json` returned HTTP 404. The package `renders/` and `paths/` directories expose no actual records beyond their empty listings. The inspected reconstruction metadata/log does not specify surface center, sampling interval, order of depth planes, normal orientation, PPM-to-volume selection or renderer version/arguments. These are bounded search results, not a claim that no such records exist anywhere.

Most importantly, **the native 3.24 µm voxel pitch is now authenticated, but the rendered stack's depth pitch is not.** A count of 65 files does not authenticate center plane 32 or an inward/outward convention. The earlier Paris benchmark's `[7,58)` / factor-three XYZ pooling recipe must not be inherited here merely because it produces 17 planes. PHerc51 has a distinct unresolved surface contract.

## Optical annotation and upstream exposure

The retained EduceLab arXiv v4 paper supports optical tracing and surface sampling for its documented four Paris fragment experiment. It describes a larger six-fragment dataset, but the two additional named entries are Paris Objet 59 and P.Herc. s.n. (Cass. 110), with a warning that some names are temporarily obfuscated. None of the inspected text establishes their equivalence to today's Frag6 / PHerc51. Therefore the paper's general method cannot be asserted as the exact provenance of this fragment's current `inklabels.png`, and its phrase “six fragments” cannot establish whether PHerc51 was used by the historical Ink-ID model.

The companion retained-source audit pins 15 text sources and finds no affirmative named Frag6 / PHerc51 training appearance in the inspected legacy records. This is **unknown historical exposure**, not evidence of complete absence. The phase 7 configurations list seed42, seed43 and the Russo pair's direct curated roots under PHerc0139, PHerc0814, PHerc1667 and Paris4. The separate phase 19 upstream audit owns the wider initializer/teacher conclusions. Inherited checkpoint initialization, annotation teachers and private historical runs must not be declared clean from a missing direct-root name.

Our retained phase 15 and phase 18 records treat Frag6 as metadata-only and deferred, with no acquired CT, label arrays, inference or performance-driven selection. It remains locally unscored on that documented history. Frag5's specific pixels may be new to us, but the official catalog directly says its parent is the already exposed PHerc1667, so it cannot replace Frag6 as fresh-parent confirmation.

## Restricted next use, fixed before any scientific payload

1. Keep all PHerc51 aliases, all four scans and future discovered same-parent fragments out of gradients, teacher targets, cutoff fitting, normalizer fitting and development selection while provenance is unresolved. Do not silently substitute another scan after a failure.
2. Permit source-contract completion and, only under a separately frozen acquisition plan, label/mask **quality-control construction**. Record this as inspected supplied labels, not a blinded or independently human-verified benchmark. Use deterministic support rules, source hashes and stop conditions; do not view predictions or optimize crops, depth or alignment against them.
3. Before CT accuracy evaluation, authenticate a surface-specific render recipe or independently verify its mapping to the declared native volume with a bounded, prespecified geometry check. Required outputs include canvas identity, plane-to-offset rule, interval, PPM association, interpolation, normal/order convention and the limitations of physical surface placement. Native source agreement alone does not prove ink depth or correct papyrus anatomy.
4. If those geometry checks pass but annotation/teacher ancestry remains unresolved, the maximum permitted scientific claim is a **prospectively locked comparison against supplied labels on a locally unscored parent with qualified upstream exposure**. It is not fresh independent confirmation or validated text. Keep a truly fresh confirmatory set separate for later claims.
5. No acquisition, inference or fit is authorized by this audit itself. It only records eligibility requirements for root review. The £10 total first GPU pilot cap remains unchanged; this audit used no paid resources.

## Rights and attribution

The pinned official data documentation identifies pre-2025 DLS Scrolls 1–4 and Fragments 1–6 as EduceLab-Scrolls and links the same legacy license. The retained license restricts redistribution and public hidden-text revelations/related code, requires EduceLab-Scrolls attribution in abstracts, the Parsons et al. paper citation, and an appropriate methods acknowledgement. Public accessibility is not unrestricted redistribution permission. This audit accepted no new terms and published no data or text findings. Exact license text and official source pins remain available for any later publication review.

`CANDIDATE_CONTRACT.json` records the machine-readable decision and source identities; `SOURCE_PINS.json`, `request-ledger.json` and the retained response bodies support every new source claim. The legacy exposure subreport and its own seal are under `exposure/`.
