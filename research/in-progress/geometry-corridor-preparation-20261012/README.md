# PHerc0814 CT corridor follow-up

12 October 2026. Enlarging the failed reference corridor supports two **short
provisional local ridge paths**. It does not qualify an independently adjudicated
surface reference. The two-parent G1 gate is still unmet; the original rejected
annotations are preserved, and no model is scored against them.

The lead assistant reviewed all thirteen fixed serial sections Y=72,76,…,120
from PHerc0814-06, using the same source CT and fixed 0–255 display. Each view
enlarges X=24:104, Z=72:120 with nearest-neighbour scaling. The path overlay
and three orthogonal cuts were also viewed. Source hash, coordinates and
display rules are recorded in [PLAN_AND_RECEIPT.json](PLAN_AND_RECEIPT.json),
[PATH_PLAN.json](PATH_PLAN.json) and `path-review/COORDINATES.json`.

| Candidate | Local ZYX anchors, Y=96 | Length | Status |
|---|---|---|---|
| 0814-C-short | (94,96,32), (93,96,40), (92,96,48), (92,96,56) | 0.225854 mm | Plausible local ridge; approximate centres |
| 0814-D-short | (105,96,32), (103,96,40), (101,96,48), (102,96,56) | 0.229881 mm | Plausible neighbouring ridge; approximate centres |

These new IDs preserve the distinction from the prior failed A/B paths. All
anchors lie in the original 128³ core. Serial and orthogonal context support
local separation, but continuation around X=64–80 remains ambiguous. Do not
extend either polyline through those gaps or assign interpolated voxels as truth.
Source intensities are descriptive and were not fitted as a selection threshold.

[REVIEW.json](REVIEW.json) records the limited decision: no second independent
or expert adjudication, no whole physical-sheet identity, no recto-face label,
no confirmed tear and no accepted geometry-accuracy reference. No surface-field
or ink-model prediction was read. This work used already retained CT, with no
new inference, raw scan download or paid compute.

The next geometry need is independent local identity/face adjudication and a
supported contiguous path, including difficult joins and tears, before the G1
detector/tracer comparison. Repeated brightness-based retracing of this same
short corridor will not supply that evidence. The 0.23 mm candidates can aid
review or runtime preparation; they cannot meet a connected-area improvement
criterion by themselves. PHerc0800's earlier provisional controls are unchanged.

The raw 192³ uint8 CT context and all review figures are included in the shared
[native-coverage preparation bundle](../native-coverage-preparation-20261012/PORTABILITY.md).
The catalogue source is CC BY-NC 4.0; retain the source attribution and keep the
packet private. Reproducing coordinates or images is distinct from validating
the physical reference.
