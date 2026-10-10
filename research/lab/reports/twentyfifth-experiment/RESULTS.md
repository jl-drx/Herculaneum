# Frag6 provenance follow-up: editing history found; evaluation contract still incomplete

The new metadata audit recovered a concrete annotation-document history for Frag6, but it does **not yet establish a usable independent evaluation reference**. The PSD records Photoshop editing in December 2023 and a TIFF-to-PSD conversion. It names no author in the inspected XMP or EXIF first-directory fields and supplies no optical-registration or surface-render recipe there. Frag6 remains reserved and unscored; Stage C remains closed.

This is a new source inspection, not a repeat of the earlier catalog or sparse-geometry checks. The completed phase24 integrity check passed over all 18,909 retained records before this work. No earlier experiment was rerun or modified.

## What the new source establishes

The exact [published PSD](https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/inklabels.psd) is 86,107,922 bytes with strong server ETag `"656fa2a5-521e712"`. Every requested range was bound to that identity. We did not download or hash the full PSD; the ETag is an HTTP object-version check, not a claim of a complete-file cryptographic digest. Each retained range has its own SHA-256.

| Observation | Evidence and limit |
|---|---|
| PSD storage | Version 1; one channel; 6205 × 8853 pixels; 8-bit grayscale; no color-mode data. These are header fields, not decoded pixel measurements. |
| Embedded creator tool | `Adobe Photoshop 25.2 (Macintosh)`. This identifies software, not a human annotator. |
| Embedded creation date | `2023-12-01T15:37:48-05:00`. Self-reported document metadata, not independently authenticated chronology. |
| Embedded last modification | `2023-12-05T12:20:03-05:00`. EXIF also contains `2023:12:05 12:20:03`, without an EXIF timezone. |
| Embedded history | Created; converted from `image/tiff` to `application/vnd.adobe.photoshop`; saved on 4 and 5 December 2023. The originating TIFF is not identified. |
| Author and annotation method | No `dc:creator` field in the inspected XMP; no Artist or XPAuthor in EXIF's first directory; no IPTC resource in the enumerated PSD resource section. This does not prove that an author is absent from every possible layer, private record or subordinate metadata field. |
| Internal encoding discrepancy | EXIF BitsPerSample is 16, whereas the current PSD header depth is 8. Treat embedded EXIF as potentially inherited metadata; it cannot override the current file header or identify the source TIFF. |
| Canvas agreement | PSD header dimensions match the previously inspected PPM/TIFF canvas. Equal dimensions do not demonstrate pixel registration, correct anatomy or annotation support. |

The complete extracted fields and namespace-qualified history are retained in [METADATA_VERIFICATION.json](METADATA_VERIFICATION.json). Document IDs give a curator a more precise record to locate; they are not evidence of the document's relationship to `inklabels.png` or `ir.png`.

## Bounded execution and verification

The first frozen reader performed one HEAD and 55 conditional GETs, returning 15,359 body bytes. It enumerated all 26 PSD image-resource records, read the 15,013-byte XMP resource and skipped the other resource payloads. The XMP includes substantial padding. No IPTC resource was present.

The newly discovered EXIF resource justified a separately frozen addendum: five conditional GETs, 235 body bytes, limited to its TIFF header, first directory and Software/DateTime strings. The subordinate EXIF and next-IFD pointers were retained as metadata and not followed. Thumbnail, layer/mask and composite-image payloads were not requested. No image decoding or inspection of candidate label/optical/CT pixels occurred.

**Total: 61 HTTP attempts, 15,594 returned body bytes, 15,654 reserved body bytes, zero retries.** Counts are application-observed response bodies, not total wire traffic. All live requests succeeded. The original PSD protocol and completed run stayed frozen when the EXIF follow-up was added; this was not a retry or a post-result model change.

The final PSD reader passed 28 synthetic checks covering forbidden image payloads, padding, corrupt extents, duplicate/oversized metadata, full-file/redirect refusal, changed ETags, framing errors, truncation and excess bytes. An earlier 28-check preparation run also passed before an artifact-size guard was added; both receipts remain. The EXIF reader passed nine additional checks, including thumbnail/subordinate-directory poison cases and string-overlap rejection.

A separate offline audit of the retained ranges passed 1,847 assertions. It rebuilt the resource framing, checked exact read extents and hashes, verified that skipped payloads were untouched, compared HTTP identities, parsed XMP and checked the EXIF discrepancy. This was a root-authored second pass, **not an independent-agent review**. It made no network request and executed no model. [Frozen PSD protocol](PROTOCOL.md), [EXIF addendum](EXIF_ADDENDUM.md), source/fixture releases and both request ledgers retain the exact boundaries.

## Research decision

The editing history narrows the question we can put to the curator. It does not authenticate the historical source image, human-only labeling, confidently negative support, optical warp/residual, source-volume association or render depth rule. Sparse geometric compatibility from phase21 remains useful, but supplies none of those missing historical facts. Do not infer plane 32 as the ink surface, reverse normals or reuse the Paris pooling recipe merely from matching canvases.

The next step is to obtain the specific existing records named in [the prepared maintainer request](MAINTAINER_REQUEST_DRAFT.md). It is ready for review and forwarding through the support channels listed by the official documentation; **nothing has been sent or posted**. [EVIDENCE_GAPS.json](EVIDENCE_GAPS.json) maps each unresolved requirement to the exact evidence needed and separates dataset provenance from model-specific training/teacher exposure.

We are not demanding an impossible guarantee about every historical model. A sufficiently documented source/annotation contract can support a prospectively fixed **supplied-reference agreement** study while explicitly qualifying unknown upstream ancestry. An optically anchored accuracy claim additionally needs attributable optical labels, support and registration assessment. A model-specific independence claim needs its own artifact-bound exposure evidence; absence from a model configuration does not establish that claim.

If historical records were not retained, the alternative is to construct a new auditable render and optical-registration/annotation workflow from the public source assets, with expert review and a prospective evaluation partition. That is a separate research work package, not an automatic relabeling of this historical stack. No such acquisition or inference is released by this report.

The CPU research environment remains usable. The current blocker to independent-reference admission is scientific provenance, not cloud storage, credentials or GPU capacity. We made no model calls, gradient updates, training runs, candidate predictions, credential operations, uploads, outreach, publication or prize submission. GPU spend remains **£0**, and the **£10 total first-pilot cap** remains unchanged. Local evidence retention and saved startup instructions do not establish external backup, publication or fresh-task restoration.
