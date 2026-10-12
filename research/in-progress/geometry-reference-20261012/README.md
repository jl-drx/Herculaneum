# CT reference qualification, 12 October 2026

The two relatively traceable candidates from the prior all-twelve review were
checked through serial CT sections. **One parent supplies plausible local
continuation controls; the two-parent reference gate is not yet met.** No native
surface field or ink map was opened, and no new inference or paid resource ran.

## What was actually inspected

The cohort was fixed in [PLAN.json](PLAN.json): PHerc0800-24 and PHerc0814-06,
thirteen Y sections each at 72, 76, …, 120, fixed uint8 display 0–255. These are
selected development cases, not a random sample or confirmation material. The
lead assistant viewed all 26 sections, both coordinate views, both path overlays
with their unmodified CT beside them, and six orthogonal cross-cuts. There was
no second reviewer or expert adjudication.

Four manually proposed paths contain 18 anchors, all within the original
128³ core. Source hashes, local/global ZYX coordinates and path lengths are
recorded in `path-review/COORDINATES.json`. The polylines mark candidate reviews;
their interpolation is not automatically a valid surface label.

| Candidate | Observed result | Permitted use |
|---|---|---|
| PHerc0800-24, A | A broad curved lamina is readily followed over the proposed short path; the orthogonal sections support local continuation. | Provisional CT continuation/transport control, pending separate identity review. |
| PHerc0800-24, B | A thinner neighbouring lamina remains locally separate over the proposed path. | Provisional local separation control; recto face remains unspecified. |
| PHerc0814-06, A | Several approximate anchors and intervening lines fall in dark gaps rather than following a single visible ridge. | Failed first annotation attempt; retain it, do not score a model against it. |
| PHerc0814-06, B | The middle anchor crosses a gap and the surrounding texture complicates continuation. | Requires retracing and review; not a qualified positive control. |

The PHerc0800 candidate paths span about 0.43–0.46 mm using catalogue spacing.
They do not qualify long-range tracing. Low CT intensity alone is not a rejection
rule; the gap observations use the raw images and serial context. No intensity
threshold was fitted and no path was snapped to a brighter response.

## Reference semantics

Three identities must stay separate: a local visible CT lamina, a local surface
branch/instance, and a complete physical papyrus sheet. Adjacent winding surfaces
can ultimately belong to the same rolled sheet. Neither these images nor the
legacy instance IDs establish that global relationship. The earlier reference
diagnostic measures preservation of **local annotated instance IDs**, not a
count of globally distinct papyrus sheets.

An interior ridge is also not a recto-face annotation. Before scoring recto
placement, the intended face/side and uncertainty band need explicit annotation.
Before a tracer comparison, specify fixed seed regions, withheld checking
anchors and a supported path corridor, with uncertain joins and boundary exits
marked unknown. A larger detector response must never choose the reference.

The present paths are useful review candidates, not independent ground truth.
In particular, the failed PHerc0814 attempt must not turn into an apparent model
failure or be silently replaced by an easier response-selected region. The
remaining ten contexts stay in the development cohort, with the previous
crowded/curved descriptions and no invented same-sheet labels.

## Next geometry experiment

First correct and separately review the short paths, with predictions hidden.
Freeze local branch identity and face semantics separately. Once that reference
is usable, compare m7/F0 on the same CT, context, TTA and tracer; then compare
tracers on the same field. Keep the as-published pipelines as a separate practical
comparison. Reference ambiguity does not prevent runtime preparation, but does
prevent a claim that a model improved correct-sheet recovery.

For a true tracing result, extend to a preselected contiguous path through
multiple contexts and include supported contacts, difficult joins and tears.
These short, relatively open candidates cannot establish the programme's
connected-area improvement criterion by themselves. Preserve the original G1
success/stop rules rather than lowering them because the reference is hard.

## Reproduction and source scope

Restore the existing 135.5 MiB [input bundle](../frontier-intake-20261012/PORTABILITY.md).
No additional raw CT is needed. With the retained `pherc343` CPU environment:

```bash
python research/in-progress/geometry-reference-20261012/prepare_review.py \
  --plan research/in-progress/geometry-reference-20261012/PLAN.json \
  --arrays /path/to/restored/native-contexts --output /new/ct-review
python research/in-progress/geometry-reference-20261012/check_paths.py \
  --plan research/in-progress/geometry-reference-20261012/PLAN.json \
  --anchors research/in-progress/geometry-reference-20261012/CANDIDATE_PATHS.json \
  --arrays /path/to/restored/native-contexts --output /new/path-review
```

All outputs require new directories. Coordinate verification is distinct from
scientific adjudication. The active native0139 producer, protected evaluation,
and PHerc0175/PHerc1203 reserve payloads are outside these scripts. Source CT is
CC BY-NC 4.0 under the catalogue; keep this research packet private.
