PHerc343 saved-map independent review
2026-10-09 | Snapshot f85c6ab579da054d3ca94c3b7e4b58699f7c6a3a

BOTTOM LINE
The six saved maps are intact and contain recurring spatial structures, but
padding/tile phase materially changes the seed42/43 outputs. The prescribed
ownership merge introduces measurable edge discontinuities. These results
support further reproducibility review; they do not establish ink truth,
transcription accuracy, or completion of the four-model reproduction.

VERIFICATION
The supplied ZIP SHA256 matches the GitHub artifact digest reported by the
parent: b1410446f5eabaa0942837252c3e86cf6b807e7845f629c29e734e7eb0d7f5cb.
Both compressed archives match their independently retrieved Git LFS hashes.
All six TIFF byte lengths, encoded SHA256 hashes, decoded SHA256 hashes,
uint8 types, dimensions and PASS receipts were checked. The reference PNG
hash and the packed valid-mask hash on the published crop match published
pins. The full mask contains 10,678,375 valid pixels; the published crop
contains 8,887,441. Archive members were checked for regular-file paths and
traversal before extraction. No new model execution or checkpoint loading.

PADDING SENSITIVITY: PUBLISHED CROP, VALID PIXELS ONLY
Model      Phase MAE (uint8)   Phase Pearson r   Absolute-error p95 / p99
dnative          6.28               0.946                  18 / 28
seed42          10.37               0.776                  39 / 63
seed43           9.75               0.785                  38 / 63
The mean changes correspond to 4.9%, 8.1%, and 7.6% of the fixed display
window width of 128 levels. This is a scale comparison, not a probability
error or an accuracy estimate. Maximum changes are 99, 126, and 141 levels.

OWNERSHIP SEAMS
At 2,244 valid adjacent-pixel edges across the full native field where the
prescribed ownership changes, mean absolute gradients are:
Model        Plain phase    Padded phase    Prescribed merged map
dnative          1.00           0.93                  6.39
seed42           0.93           0.94                  7.44
seed43           0.89           0.88                  8.98
Of these full-field edges, 2,159 are fully within the published crop. The
gradient table is explicitly full-field; near/away phase MAEs below use the
published crop, so they describe slightly different spatial populations.
The comparison uses the same physical edges for each map. Switching between
phase outputs adds a discontinuity at these locations. This does not show
that any particular candidate is caused by a seam. Across the published
crop, phase MAE within 20 pixels of an ownership boundary versus elsewhere
is 6.32 vs 6.28 (dnative), 7.25 vs 10.40 (seed42), and 9.07 vs 9.76 (seed43).
Every tested edge has both endpoints valid; specimen/background edges are
excluded. These are ownership-switch boundaries, not a diagnosis of generic
tile seams. Boundary endpoints occupy 0.048% of valid published-area pixels;
the 20-pixel neighborhood occupies 0.951%. Only box 5 intersects the boundary
(3.76% of its valid pixels within 20 pixels); the other 17 boxes do not.
The inspected box-5 overlay places the boundary at its lower-right margin,
away from the main highlighted reference structures. There is no evidence
here that this boundary explains those candidate shapes.
Thus phase sensitivity is not confined to the ownership seam neighborhood.
This is descriptive and spatially confounded; it is not a randomized test.

MODEL AGREEMENT: PRESCRIBED MERGES, VALID PUBLISHED CROP
Pair                  Pearson r     Mean absolute uint8 difference
dnative / seed42          0.583                  23.11
dnative / seed43          0.552                  24.49
seed42 / seed43           0.530                  15.10
Raw levels can differ in calibration between models. Correlation and shared
features are not independent evidence of ink, particularly for shared model
families and training data.

ALL 18 PRE-EXISTING REGIONS
Every supplied box was evaluated and displayed; none was excluded based on
appearance. The contact sheets use one fixed display transformation for
all maps, preserve aspect ratios, and show the original published composite
alongside each model and maximum phase difference. Visual observations are
descriptive, not pass/fail judgments or letter readings:
1: Several bright components recur, with variable connecting structure.
2: The lower bright component recurs; its size and the upper response vary.
3: Components vary strongly between seeds; seed-pair correlation is 0.004.
4: A lower spanning structure recurs; connectivity and upper components vary.
5: A lower-left component recurs; upper spanning structure varies.
6: Upper/lower components are present with different relative strengths.
7: A vertical feature is stronger in dnative/seed43 than in seed42.
8: Lateral components recur; the connecting response differs markedly.
9: An open-centered shape is visible but its width/connectivity vary.
10: Seed43 differs substantially from seed42 (pair correlation 0.194).
11: Upper and lower components change shape and relative prominence.
12: Multiple components recur but separation and intensity differ.
13: A lower-left bright response recurs; surrounding response differs.
14: Curved/right-side responses recur with different continuity.
15: Seeds place response differently (pair correlation 0.073).
16: A broad left/top structure recurs, with internal gaps differing.
17: Seed43 is weaker and less continuous than the other displayed maps.
18: Several bright components recur, with variable bridges and spacing.
All per-region values are in regions.csv and metrics.json. For example,
dnative phase correlations range from 0.910 to 0.978 across the 18 boxes;
seed42 ranges from 0.547 to 0.893; seed43 ranges from 0.629 to 0.920.
The examples above are not a newly selected evaluation set: all boxes are
previously exposed reproduction references, and some boxes overlap.

METHOD AND LIMITS
Padded maps are cropped [40:2600,200:6920] into native 2560x6720 coordinates.
Each node source is repeated 20x20; IDs 1 or 4 select padded values, other
IDs select plain values. The display crop [105:2560,380:6021] is reversed
on both axes. Each map is shown as clip((uint8-64)/128,0,1); no local contrast
fitting. Blue-gray denotes invalid geometry. Delta and model-range images
use a fixed 0..128 scale and clip above 128. PNG thumbnails are resampled
for viewing; numerical metrics use full-resolution source arrays.
The fixed plan was written before any pixel inspection. Per-region bounds
use floor minima and ceil exclusive maxima, intersecting the valid mask.
Correlations include all valid pixels, not only bright structures.
The published PNG is an annotated, stylized four-model composite, not raw
ink labels or calibrated probabilities. Hecate maps and raw published
reference maps are absent. CT layer34 was not supplied. No full ensemble,
AP, raw-map equality, transcription, or ink authenticity is claimed.
The retained input verification says the aligned plain/padded interiors are
identical physical CT inputs. The phase comparison tests padded sampling and
tile-grid sensitivity, not new tissue or a physical acquisition shift. This
review cannot independently recheck absent CT volumes. Greater dnative phase
stability is not evidence that dnative is more accurate.
Prior source rendering PASS is a retained receipt; this review independently
checks the supplied mask and saved outputs, not the absent CT volume.

FILES
overview.png: published context, three merges, phase differences, model range.
regions_1.png through regions_3.png: all 18 boxes, six per sheet.
region5_ownership_boundary.png: only box intersecting an ownership boundary;
red marks valid ownership-boundary pixels, cyan is a published annotation.
regions.csv: compact per-model region metrics.
metrics.json: full metrics, provenance, verified receipts and frozen plan.
analyze.py and ANALYSIS_PLAN.json: reproducible calculation and frozen plan.

INDEPENDENT ARITHMETIC CHECK
A second read-only audit independently recomputed all six file/decoded
hashes, archive hashes, valid-mask hash, full-field and crop phase metrics,
all18 region phase MAEs, ownership-edge means and pairwise model statistics.
These checks agreed. The audit also checked orientation/crop/ownership
against the source script and sidecar. This is an implementation check,
not an independent scientific validation dataset.
