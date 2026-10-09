# Structural landmark review: Frag6 / PHerc51, four development regions

**Result: no reliable shared non-ink landmark in any of the four regions.** There are zero accepted optical–CT correspondences and zero sufficiently localized possible pairs to retain. No transform, registration claim, physical depth, or anatomical direction follows from this review.

This is a separate visual review for subsequent review by the root agent. The regions are already development data. It is neither independent ground truth nor a blinded experiment. The reviewer did not inspect `REVIEW_PACKET.md`, model predictions, ink labels, or phase 33/34 conclusions.

## Evidence inspected

Read source identities from `../PACKET_PLAN.json` and `../packet-v1/COMPLETE.json`. Viewed `optical-and-cross-sections.png` and each region's complete 1–63 layer montage. Then viewed all four exact retained `optical_RGBA` arrays as lossless 384×384 PNGs and rendered fixed-scale optical/CT detail panels from the retained arrays. All four NPZ hashes and five packet image hashes match `COMPLETE.json`; the native optical exports match their arrays pixel for pixel.

Optical values are unchanged uint8 RGBA (display range 0–255); all 147,456 pixels in every crop have alpha 255. Each CT view is the retained 128×128 array of raw 3×3 spatial means, with grayscale fixed at 0–65535. Detail panels use layers 1, 10, 20, 30, and 40 as regularly spaced illustrations after inspection of all 63 layers, not as selected matches. No sharpening, local contrast normalization, warping, feature fitting, network access, GPU work, or model inference was used. The diffuse appearance is present in the retained optical pixels, not just in the resized packet overview; its origin is not established here.

## Region findings

| Region | Global crop origin (row, column) | Observation | Shared landmark |
|---|---:|---|---|
| `r544-c4128` | (608, 4192) | Optical has a broad bent dark zone but no resolved boundary, endpoint, or structural junction. CT has distinct curving and roughly orthogonal bands whose configurations change across layers. A broad tonal resemblance would not identify one of those structures. | None reliable |
| `r2080-c3616` | (2144, 3680) | Optical has a diffuse, mainly vertical dark zone and broad light patches. CT resolves near-vertical bands, bends, and transverse interruptions. The optical does not resolve the CT bends or interruptions, so no unique pair can be localized. | None reliable |
| `r4640-c1056` | (4704, 1120) | Optical consists of smooth light and dark patches. CT contains a conspicuous thin oblique discontinuity through the upper-middle portion, along with other bands and turns. No corresponding thin line or identifiable endpoint is visible in optical. This is a CT-only observation. | None reliable |
| `r6688-c1568` | (6752, 1632) | Optical has diffuse light patches and a dark upper/central zone. CT resolves several bounded bands, turns, and gaps. Their recognizable boundary geometry is absent from the optical view, and no optical patch can be confidently classified as the same non-ink structure. | None reliable |

The CT descriptions are visual observations, not authenticated tear/fold diagnoses or candidate correspondences. Dark/light optical patches have not been assigned an ink or structural identity. No optical-to-CT coordinate pair or layer interval is reported because none was defensibly identifiable. A broad patch can overlap multiple CT structures and layers; positional plausibility alone does not establish identity.

## Coordinate conventions and limits

Local raw coordinates are zero-based row/column indices 0–383. Add the origin above to obtain the global raster coordinate. Display pixel `(y, x)` in a CT mean image covers raw rows `3y..3y+2` and columns `3x..3x+2`. Historical CT layer labels are 1–63. Raster orientation is retained; no physical surface direction is inferred. Cross-section row193 and column193 show CT structure, but lack an independently identified optical counterpart.

This negative finding applies to the supplied four optical crops and retained CT representations. It does not prove that no shared feature exists elsewhere, that optical and CT are misregistered, or that finer source data could not resolve one. These views do not support a landmark-based alignment or layer choice. A future match would need a non-ink feature with distinctive topology or endpoints resolved in both images; the current diffuse optical patches do not meet that condition.

Artifacts: `findings.json` contains machine-readable results; `*-optical-native.png` contains lossless optical exports; `*-detail.png` contains fixed-scale comparison panels; `render_details.py` reproduces those panels using cache directories inside this review folder; `render-provenance.json` and `validation.json` document sources and checks. Only this newly created review directory was written by this review.
