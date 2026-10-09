# Public records found: revise the reference plan before contacting maintainers

The missing information cannot be described as unavailable publicly. A targeted issue/repository search found relevant recent work that our prior retained audits had not incorporated. The public scans, optical images and supplied labels were already known; the new findings add an implemented fragment-depth benchmark and specific Frag6 data-quality reports. They do not yet close every artifact-specific provenance gap.

## Public benchmark worth auditing and reusing

[kartoun/vesuvius-fragment-ink-depth](https://github.com/kartoun/vesuvius-fragment-ink-depth) was linked in the [villa#192 discussion](https://github.com/ScrollPrize/villa/issues/192#issuecomment-6016079974), with an October6follow-up reporting a scorer and leave-one-fragment-out results on Frag1–Frag6. We fetched the repository tree and three specific text blobs, verified their Git blobSHA1 and retained SHA256 identities. No repository code, model or scientific data was executed/read.

Pinned tree identity191126e0e677b4cf17131a1f7bbaefb1fa8f92e1 (a tree SHA, not asserted as a commit SHA):

| Inspected text | Git blob SHA1 |
|---|---|
| README.md |0d127c1c8a8e71aa479a09341c6cd7ae7ee2afb8|
| download_fragments.py |06344a7392fa51c8791538fad97194469d5d8360|
| step3_build_labels.py |d13cec2e6a88e306438b1de132a7c50b67b77a94|

Useful contributions:

- Exact Frag6 source URLs match the legacy53keV/3.24um surface_processing directory we retained. This is a concrete reusable data bridge, not a newly discovered raw dataset.
- The label builder is inspectable: it finds a smoothed papyrus-to-air intensity drop in layer range10:56, constructs a surface band[-4,+5]layers and defines ink/not-ink/ignore. It ignores a3pixel stroke-edge band and unreliable ink columns.
- The README describes a3Dscorer separating2Dink agreement, depth relative to the detected surface and “surface share” (clear-papyrus versus ink in-band response). This is directly relevant to avoiding a model that merely detects papyrus surfaces. The scorer source itself was not fetched in this bounded pass and remains to be audited.
- The discussion contains negative results and corrected interpretations: model-based occlusion depths can move substantially when the model changes; a nearby intensity interface does not by itself establish ink position. Reported performance belongs to the authors and was not reproduced here.

Limits and concrete checks before adopting it:

1. The inspected downloader reuses existing ir.png/inklabels.png/mask.png and surface TIFFs. It does not supply the original Frag6 annotation author/review trail, optical warp/residuals or historical source-volume/render recipe. Assertions that assets are aligned and IR-traced are contributor claims, not those missing original records.
2. The builder defines a reproducible new label policy; it classifies most unmarked in-fragment pixels as not-ink. That does not prove the original tracing exhaustively marked all ink. Its gradient-derived depth is a surface proxy, not an independent volumetric ink measurement.
3. Its reported six fragment folds are not six independent parent groups under our conservative ledger: Frag1/2 share Paris2 and Frag3/4 share Paris1. We must audit full training/support splits rather than inherit a new-parent claim from leave-one-fragment-out naming.
4. The README acknowledges that Frag4 has no aligned IR image in its downloader and contains a stale “one fragment so far” limit alongside later six-fragment results. Inspect implementation/outputs and per-fragment evidence instead of treating narrative labels as complete validation.
5. The downloader retrieves full TIFF stacks (it estimates6.8GB forFrag6) and validates transfers by length, with retries. We will not run it as-is within our bounded streaming research setup. Its data-license comment is not a replacement for the retained official terms.

Reading the public README exposed us to third-party aggregate Frag6 performance. We still read no candidate pixels or model output maps and made no local candidate scores. Future claims must not call us blind to all published results; our pre-existing candidate selection and reservations remain recorded.

## Frag6 alignment and stitching reports

[Issue129: Fragment6volume alignment/transforms](https://github.com/ScrollPrize/villa/issues/129) supplies four correspondence points and proposed3Daffines between the three3.24um scans. The proposed53-to70keV matrix differs from the transform retained in our earlier audit. Closure comments attribute the discrepancy to imperfect DLS stitching; closure is not evidence that published transforms were replaced. These are cross-scan transforms, not optical registration, and four fitting points do not provide held-out residual validation.

[Issue130: discontinuity near layer7975](https://github.com/ScrollPrize/villa/issues/130) reports the defect in the53and70keV volumes, with88keV apparently unaffected. A [maintainer comment](https://github.com/ScrollPrize/villa/issues/130#issuecomment-3619669180) identifies it as a stitching artifact. We inspected text only, not linked images. This is relevant to source/region selection and must enter a prospective quality audit; it does not authorize switching scans or selecting favorable outputs after evaluation.

## What changes now

**Audit the existing public benchmark before building a replacement or sending a broad maintainer request.** Next inspect its scorer, shared preprocessing and fold construction at pinned identities; compare against our parent/support ledger and reproduce only bounded permitted controls. Decide which components can be reused as supplied-reference diagnostics. The fixed Plinius depth-study proposal remains an unexecuted candidate; these new sources must inform a prospective update before scoring, not a retrospective explanation of results.

A later maintainer request can be narrower: original Frag6 render/source mapping, annotation/negative-support semantics and optical registration records that are still absent from the inspected public material. No assertion is made that those records do not exist elsewhere. The user confirmed the request is NOT SENT. We sent no message and posted nothing.

## Scope and accounting

20public text/metadata requests;247358returned body bytes;10158100reserved body bytes. Seventeen responses200; the first three GitHub searches422because an explicit issue qualifier was required, retained with the diagnosis. Three corrected issue-only queries then ran. No automatic retry or redirect followed. Four exact web searches returned no-results pages; those results cannot establish absence. The searches do not exhaust all code, pull requests, papers, Discord or private records.

All20responses and derived text blobs are retained with identities. No CT, optical/label image, PPM or weights body was requested; no paid resource or credential was used. This separate user-requested audit follows phase28's frozen zero-network scheduler qualification and does not change its execution accounting. Public findings are linked back through an additive phase28report addendum and retention manifest.
