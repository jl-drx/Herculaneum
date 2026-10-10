# Stage B — fixed small-region centre-depth comparison

Frozen after successful candidate geometry/source/render admission and before any
candidate detector output. This short in-chat CPU test supplements the planned
larger development field; it does not replace that field or create a pristine test.

Scoring core: segment20251028225813, x1024:1152,y280:408 (128x128, about1.1mm on a
side). Original mesh, qualified66-plane inputs, explicit8.64um, nominal d0.
Run dnative, seed42, seed43 and Hecate; both input orders and both fixed padding
phases, giving16 core maps. No ensemble, selected contrast, depth/origin search,
adaptation, training or GPU. Every map is kept and shown on the same0..255 scale.

Use the pinned original Villa dataset, probability conversion and accumulation
code, with the exact12 contributing global patches per phase. Hecate uses the
original full-grid predict function at batch1, serving zero placeholder inputs
for noncontributing patches to avoid unnecessary forwards. Such placeholders are
unknown and never used by any positively contributing scored input; all global
Hann denominators remain unchanged. Exact36 contributing Hecate patches per phase
are observed. No local resize/reset. Native preparation is global2300x2260 or
padded2340x2460; Hecate preparation is globally resampled to2070x2034 or2106x2214.
Both model families use fp32 CPU, four threads, batch1. This fixes batch membership
and isolates patches; byte equivalence to batch4/GPU inference is not claimed.

Preserve author depth contracts (Villa25..41; Hecate centre32.5 and9.6um spacing)
and uint8 conversion (Villa truncation, Hecate nearest rounding before native-grid
float probability resize). Read-side orientation remains unknown. Keep raw grids,
aligned float scores and fixed-scale display PNGs separately.

Save every singleton forward's logits and normalized-input/support hashes with
an immutable receipt. On restart, verified logits replay through the unchanged
original accumulation code; no saved batch contributes twice. Inject one controlled
stop after the second committed native forward, resume it, then replay all16 maps
from saved logits without new inference. Require byte/float equality of saved
outputs, allfinite probabilities and full model-context/source support. No partial
map is complete; completion receipt last. Stop on identity or replay mismatch.
Expected forwards:144 Villa plus144 Hecate =288, unless raw-empty filtering skips
some per original code. At study49 single-tile rates, roughly6–8 CPU minutes.

Report all within-model phase and direction correlations/absolute differences,
plus model response ranges, descriptively. Review the whole128-square region;
there are no selected letter boxes. These metrics establish response stability,
not ink accuracy. No AP, uncertainty confidence interval, authenticated reading or
independent specimen claim. Seal all outcomes, including a null visual result.
