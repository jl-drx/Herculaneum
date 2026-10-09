# Herculaneum / Vesuvius Challenge research snapshot

Retrieved 2026-10-07 (Europe/London date). This is a review snapshot, not an award guarantee. No submissions, messages, registrations, or terms acceptances were made.

## Evidence and recency

Direct HTTPS to `scrollprize.org` was denied by the environment proxy (CONNECT 403); the official website source was retrieved read-only using the existing GitHub HTTPS proxy. This avoids relying on stale search results. Official `ScrollPrize/villa` checkout `/tmp/herculaneum-villa-research` is pinned to `e0bbb8b40a2db58b1d71864f286eb85717e59e64`, commit dated 2026-10-06 (`3D DCT based compression backend for VC3D (#1704)`). Raw GitHub assets work. Official `ScrollPrize/open-data` HEAD is `f4d8ec2b895b10cf8535bd6504375b552ff5484a`; checkout `/tmp/herculaneum-open-data-research`.

- Current prizes: https://scrollprize.org/prizes ; retrieved source https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/34_prizes.md
- Current open problems, explicitly last updated July 10 2026: https://scrollprize.org/2026_open_problems ; https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/37_2026_open_problems.md
- Current FAQ: https://scrollprize.org/faq ; https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/09_faq.md
- Exact eligible scroll/volume pairs: https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/src/data/prizeEligibility.json
- Official data documentation: https://github.com/ScrollPrize/open-data/blob/f4d8ec2b895b10cf8535bd6504375b552ff5484a/README.md

## Current award options (2026-10-07 snapshot)

| Track | Award | Target / date |
| --- | --- | --- |
| 2027 Grand Prize | $800,000 first; $100,000 second; $50,000 third; $50,000 fourth | Fully unroll and make readable one eligible scan; deadline June 25 2027 11:59pm Pacific. |
| First Letters | $50,000 per eligible scroll, max 10 scroll awards | Ten visible, legible letters in one 4 cm² area; same June 25 2027 deadline. |
| PHerc. Paris 4 Title | $50,000 | Produce a readable image of title from any scan, including 2.4 µm; same June 25 2027 deadline. |
| Progress Prizes | $20,000 guaranteed best submission monthly; other awards typically $250–$20,000 | Open source advances that measurably help; source explicitly says next deadline October 31 2026 11:59pm Pacific. |

The official source currently points toward First Letters and Progress Prizes as useful starting points. We should begin with a bounded reproducible ink-recovery experiment. Provisional discovery target: PHerc0800 volume 20250521135224, with live segment inventory and quality to be confirmed before commitment (details below). We should not claim that reproducing published letters wins a new-text prize.

The Grand Prize is an eventual research target, not a credible first setup deliverable: all recto surface unrolled (outer disconnected patches below 10% may be skipped), at least 70% of preserved characters in each counted column legible letter by letter without papyrological interpolation, VC3D-integrated, automated except at most eight documented hours of input, no higher-resolution scan data of target volume, no training/prediction overlap, fixed seeds, public tracking and stagewise pseudo-label/checkpoint releases.

First Letters has 22 listed scroll-volume pairs; Grand Prize 13. A name alone is insufficient: eligibility binds the scan ID. Example present in both lists: `PHerc0125` volume `20250821151825`; other paired options include PHerc0191, PHerc0211, PHerc0257, PHerc0268, PHerc0358, PHerc0800, PHerc0813, PHerc0826, PHerc1203, PHerc1218, PHerc1545. A final target needs current segment availability and inventory review, not arbitrary selection.

## What is already known

The June 25 2026 announcement says PHerc.1667 (community Scroll 4) has been read across the preserved part end to end: about 1.4m and twenty-two partial Greek columns, papyrologist-reviewed. It describes an ethical treatise in a Stoic context and names Aristocreon; it does not establish a confident author/title for that text. PHerc.139 title/author was recovered as Philodemus, On Gods, Book 8. PHerc.Paris4 high-resolution direct-3D ink matches the previously published 2023 reading. Therefore claiming an entire scroll has never yet been read would now be outdated.

- Announcement and published translation examples: https://scrollprize.org/firstscroll ; https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/36_firstscroll.md
- Full transcription / methods preprint: https://arxiv.org/abs/2606.29085 (linked by official source; not independently fetched here).

Translation follows image-supported diplomatic transcription. Preserve uncertain letters, gaps, alternate readings and provenance. Language models can help interpret a documented reading but should never be the source of invented glyphs or text. Official prize rules demand false-positive mitigation and held-out validation; prediction regions must not overlap training regions. Known recovered text is a benchmark, not independent validation if a released model was trained on it.

## Tools and practical first workflow

- Maintained code: https://github.com/ScrollPrize/villa (MIT top-level; check component/model licenses). The older standalone `ScrollPrize/vesuvius` repo is archived and points to villa.
- Current `villa/vesuvius/pyproject.toml` requires Python `>=3.14,<3.15`; full ML stack includes torch `>=2.12,<3`. Do not present a Python 3.12 setup of the archived library as the current full stack.
- For light CPU data inspection, current official direct access example supports Python `>=3.9`: https://github.com/ScrollPrize/open-data/blob/f4d8ec2b895b10cf8535bd6504375b552ff5484a/examples/zarr_volume_access.py ; dependencies zarr, fsspec, s3fs, numpy, matplotlib, tifffile, imagecodecs. Use small chunk crops and coarse pyramid levels, headless plotting, read-only anonymous access. This is preferable to building the whole monorepo now.
- Interactive surface geometry / segmentation: VC3D https://github.com/ScrollPrize/villa/tree/main/volume-cartographer and https://scrollprize.org/tutorial_VC3D . TIFXYZ consists of float32 X/Y/Z TIFF arrays + metadata and possibly mask; supports pixel-to-source coordinates. Can extend later with supported container/source workflow when resources/GUI/GPU available.
- Recommended baseline for the provisional 8.640 µm target: https://huggingface.co/scrollprize/ink_9um , documented in current https://scrollprize.org/tutorial5 and maintained `vesuvius.ink_detection.inference.infer` (details below). The separately released https://huggingface.co/scrollprize/ink_canonical_2um with https://github.com/ScrollPrize/villa/tree/main/ink-detection/optimized_inference targets 2.4 µm data; it is an additional option for compatible scan benchmarks, not the default for PHerc0800.
- Direct 3D ink (Paris4-specific): https://huggingface.co/scrollprize/ink_3d_dino_guided . This is not evidence of an already-general cross-scroll detector.
- Other current paths: spiral-fitting, lasagna, and dinovol; thaumato-anakalyptor is deprecated. Open problems include topology-preserving segmentation, conservative fiber tracing, more accurate surface labels, diagnostics, and 3D ink localization.

Recommended first deliverable: a provenance-aware CPU explorer with fixed artifact URLs/checksums and bounded Zarr reads, reference-image inspection, then a held-out ink baseline with reproducible metrics and overlays. GPU inference, unwrapping, and ancient-Greek scholarly review are later prerequisites for original discoveries. A diagnostic tool can independently fit the Progress Prize path if it shows actionable measured improvements.

## Data, licenses, network

Official modern data are in `s3://vesuvius-challenge-open-data/`, browsable at https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html ; start with https://scrollprize.org/data_browser . OME-Zarr permits bounded partial reads. Current labels are curated July 2026 datasets: https://huggingface.co/buckets/scrollprize/datasets/tree/ink and `/tree/surfaces`; all ink labels total roughly 804GB, so never bulk download for onboarding.

Current data portal https://scrollprize.org/data and official open-data README state CC-BY-NC 4.0 unless specific assets say otherwise. Cite the newer Vesuvius Challenge CT dataset for current scans; cite EduceLab-Scrolls / Parsons et al. 2023 for pre-2025 DLS Scrolls1–4 and Fragments1–6. `vesuvius.accept_terms` includes additional legacy terms (individual acceptance, no redistribution without written approval, hidden discoveries/public associated code restrictions). No terms acceptance was performed here. Verify per-asset terms before distributing derived data.

Current new-text prize rules require unpublished discoveries remain private until official prize announcement, permissively licensed public code to accept prizes, and organizer review (technical + papyrological). Grand Prize adds CC-BY-NC4.0 releases for training datasets and pseudo-labeling stages. They may change rules; recheck near submission. No need to publish or contact organizers during setup.

Observed connectivity: scrollprize.org and direct S3 `vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com` blocked by proxy CONNECT403; GitHub HTTPS Git and raw.githubusercontent.com work. Parent is managing additive draft network settings. Necessary domains for subsequent data/model work include scrollprize.org, dl.ash2txt.org (legacy licenses/data), vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com, huggingface.co and the specific Hugging Face download/CDN host(s) observed on actual requests; do not replace unknown allowlists.

## Available tiny official reference assets

Both images below were fetched over verified HTTPS from the pinned official website source. They are public presentation images, **not raw CT data or independent ground truth**, and can verify image I/O / display/provenance but cannot support a new-discovery or ink-accuracy claim.

1. `/tmp/herculaneum-official-ink-label.webp`: texture + ink-label comparison shown in the official curated datasets documentation, 512×312 pixels, 12,016 bytes. URL https://raw.githubusercontent.com/ScrollPrize/villa/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/static/img/firstletters/ink-label.webp . SHA256 `75156a23694f1789521066610ed13a75357f15e065ef175db435f06e1e1b50ae`.
2. `/tmp/herculaneum-official-banner-full.webp`: official June25 2026 published PHerc1667 recovered surface, 12000×924 pixels, 1,008,024 bytes. URL https://raw.githubusercontent.com/ScrollPrize/villa/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/static/img/firstscroll/banner-full.webp . SHA256 `5e47ec5613608287ea6d8a67a4eec6ec3bb4967f72e9ae5f9013648470d210cd`. Caption/source https://scrollprize.org/firstscroll . It contains already published recovered text, not new work by us.

## Additional target inventory and 9 µm baseline

The official bundled data-browser index is at https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/static/data_browser/index.json . Its `updated` field is `2026-06-15` (an older fallback snapshot, even though it includes later asset metadata), so confirm live availability before committing a target. It is saved as `/tmp/herculaneum-official-data-browser-index.json`; the current prize-pair file is saved as `/tmp/herculaneum-official-prize-eligibility.json`.

| First Letters eligible sample | Eligible volume | Voxel size / X-ray energy | Bundled inventory |
| --- | --- | --- | --- |
| **PHerc0800 (provisional choice)** | **20250521135224** | **8.640 µm / 116 keV** | **6 recorded segments; surface + lasagna predictions.** |
| PHerc0125 | 20250821151825 | 9.362 µm / 113 keV | 0 segments; surface + lasagna predictions. |
| PHerc0268 | 20251110183117 | 8.640 µm / 116 keV | 0 segments; surface + lasagna predictions. |
| PHerc1203 | 20250820131727 | 9.362 µm / 113 keV | 0 segments; surface + lasagna predictions. Also a separate 2.403 µm scan whose volume is not this eligible pair. |
| PHerc0175A | 20250521115057 | 8.640 µm / 116 keV | 0 segments; surface prediction. |

The provisional PHerc0800 choice is based on its exact eligible scan and existing segmentation count, not a claim of easier ink. The six recorded segment IDs/quality were not exposed in the bundled index and need live inventory inspection. All 22 First Letters eligible scan pairs currently match either 8.640µm/116keV or 9.362µm/113keV in that index; no 2.4µm eligible pair appears in it.

The current ink tutorial documents an appropriate **9 µm pretrained cross-scroll baseline**: https://scrollprize.org/tutorial5 (section “Ink detection at 9 µm: pretrained cross-scroll models”), source https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/07_tutorial5.md . Model repository https://huggingface.co/scrollprize/ink_9um , `hybrid_3d2d-seed42/step-075000.pth` and seed43 counterpart; small local 3D stem + 2D U-Net, trained on PHerc0139, PHerc1667, PHercParis4, PHerc0814 at roughly 9µm isotropic. PHerc0800 is not in that stated training-scroll list. Seven trajectory checkpoints and two seeds enable stability checks; released `ink_9um` label set has 24 pooled2.4µm segments and 5 native9.362µm segments. Actual held-out-region/model-training provenance still needs verification. Do not mistake seed agreement for independent proof of true letters. The official worked example PHerc0139/w035 is explicitly in the models’ own training set: useful for checking the pipeline, unsuitable for claiming held-out accuracy. The tutorial notes depth-offset sensitivity, direction checks, and compressed prediction range (label-smoothed targets 0.25 background, 0.75 ink); record any display rescaling separately from probability outputs. It estimates full training at roughly five hours on one H100, while inference batch32 assumes a large GPU and can be reduced to4 or1. These are official examples, not measured local hardware capability.

### Exact future bounded-read endpoints

- Live lighter inventory, URL verified from official `useAtlasData.js`: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.min.json
- Full authoritative inventory from `atlasDataAccess.json`: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json
- PHerc0800 Zarr root recorded in the bundled inventory: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/
- Expected OME-Zarr root metadata: append `.zattrs` (and `.zgroup`); expected level0 array metadata: append `0/.zarray`. These exact object existences are not yet network-verified.
- Inspect metadata before requesting a chunk: official open-data README says normally 8-bit intensities in 128³ chunks (~2MiB uncompressed each). Read at most one central chunk on a coarse pyramid level, bounded by the actual metadata, rather than full slices or the entire volume. A 64³ crop may still fetch one or multiple 128³ source chunks depending on alignment; choose chunk-aligned coordinates. Compression/wire bytes unknown until metadata/read.
