# Study49: fixed PHerc0800 rendering and four-model qualification pass

The original-mesh PHerc0800 transfer is feasible on the existing CPU environment.
A fixed128x128 interior patch passed the frozen rendering comparisons, and all
four models passed strict loading, finite-output and batch-consistency checks.
This qualifies the tested patch and input conventions, not an entire target map,
ink accuracy, reading side or letters. The separate VM's phase48 geometry result
was still pending; this study does not prejudge it.

## Actual observations

| Check | Result |
| --- | --- |
| New31-layer render versus retained published31-layer crop | 506,139/507,904 exact (99.6525%); all within1 uint8 level |
| New66-layer render versus independent raw-CT sampler | 1,078,256/1,081,344 exact (99.7144%); all within1 level |
| Independent31-layer sampler versus published crop | 99.7076% exact; all within1 level |
| Wrong reversed published ordering control | Mean discrepancy22.45 levels, maximum197 |
| Native source | Seven observed128³ chunks,14MiB; one verified retained chunk, six newly fetched (12MiB); no missing-fill chunks |
| Model qualification | dnative, seed42, seed43, Hecate all PASS;20 tile forwards total |
| Largest single-versus-batch probability difference | 4.77e-7, below the frozen1e-4 tolerance |
| Read-only saved-output replay | PASS; three rendering comparisons, seven raw chunks, four model output sets |

The first process stopped before rendering because the remote cache defaulted to
read-only `/home/agent/.VC3D`. Supplying the documented `viewer/remote_cache_dir`
in the study's VC3D.ini fixed it. The original failure, execution logs and script
are retained; no geometry, source whitelist, comparison tolerance or executable
changed. Actual pinned-renderer execution took approximately4.94s for31 planes
(including metadata setup), then0.27s for66 planes with a warm source cache.
This tiny-patch timing is not a full-field render forecast.

## Transfer contracts clarified

The public PHerc0800 export records `flip-normals=true`; the PHerc343 example did
not. Applying the PHerc0800 recorded convention reproduced the fixed crop.
This is evidence for source ordering, not a papyrological reading-side decision.

For nominal depth d0, the released recipe's Villa models consume native planes
25..41 (reversed41..25), physically centred+0.5 voxel. Hecate samples16 planes at
9.6um spacing around native index32.5, physically centred on the mesh. Preserve
this half-voxel difference and report it; do not silently align the two contracts.
Hecate uses the full2580-to2322 global resampling grid. Its selected64x64 input
has fully observed contributors; changing all unknown placeholders from0 to255
at the tested source plane leaves that input crop unchanged. Those placeholders
are explicitly marked unknown, not recorded as observed full-field CT.

## Planning the larger field

The scoring field remains x512:2048,y512:2048 on segment20251028222030, explicitly
reusing earlier inspected tissue. No untouched-test designation or AP is possible.
The all-grid, ROI-contributing counts for both padding phases are1,301 tiles per
Villa model and4,050 Hecate tiles. Actual nonzero occupancy may reduce Villa work.
The full-canvas alternative would require3,320 per Villa model and10,878 Hecate
tiles. Cropping the canvas before model tiling/resizing is not equivalent.

Measured fixed-tile batch4 rates forecast **2.60 CPU hours** of model work for
one nominal depth, one direction, both phases and all four models. Hecate accounts
for2.45h (93.9%). Full-canvas inference would be6.97h. These are throughput
forecasts, not completion deadlines: CPU contention, data preparation, I/O and
batch choices still matter. A31-depth, two-direction sweep is about161.5 model
hours under the same unfiltered ROI forecast and is not scheduled.

A conservative66-layer source envelope spans384 raw chunks, **768MiB**; that
larger set was not fetched. Dense66-layer arrays would be419.0MiB plain and
458.5MiB padded. Hecate16-layer arrays are82.3 and90.0MiB; its largest two float
accumulators total45.0MiB. Model and preparation working memory is additional.
The qualification's measured peak process RSS is retained in the JSON, not a
validated bound on a larger process.

Within the scoring field,2,190,160/2,359,296 pixels (92.8%) have valid original
mesh interpolation/normal support. Requiring **every positive-weight contributing
patch context** in both grids and both phases to be known leaves1,875,648 pixels
(**79.50%**). These are geometry-only planning masks; complete source access and
field rendering remain unverified. Unsupported geometry stays unknown even if a
renderer would emit zero. A full-field run must freeze that output admission rule.

## Decision

Advance to an original-mesh, centre-depth development run after freezing actual
source acquisition, output support and batch/checkpoint scheduling. Retain the
full66-layer slab. Compare both input directions prospectively, since reading
side is unverified; keep both padding phases and individual raw maps separate.
Do not transfer PHerc343's two-mesh ownership rule or fit new target contrast.
Use fixed0..255 display and unknown masks. No blanket M7 snapping or depth search
has been introduced. See NEXT_STUDY_DRAFT.md for the bounded next execution.

No full target map was produced in49. No new training, GPU resource, R2 write,
letter, transcription or translation is claimed. GPU spend remains£0.
