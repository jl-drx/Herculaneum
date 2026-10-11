# Fixed CT panel review

Reviewed all six saved panels: each site's comparison and full-depth-context
image, using the supplied 0–255 display and no contrast adjustment, crop
selection, additional slices, or new data. This review used image inspection
only; it did not fit a shift or change a query. The reviewed file hashes are
in `VISUAL_REVIEW_INPUTS.json`.

**Assessment:** recognizable local structural correspondence exists at all
three sites. It is substantially stronger than generic similarity between
two parallel fibrous textures. The panels nevertheless do not establish
the same individual papyrus lamina or face at pixel precision, and do not
justify overriding the numerical admission failures for the current
point-to-point teacher transfer. Identifying one ink-bearing face is not a
prerequisite for studying paired observations of the same local 3D material.

This is an independent visual assessment of the fixed outputs, **not a blind
review**. Root disclosed that site1 failed the uniqueness margin and that
sites2/3 failed local-shift consistency before this inspection. This reviewer
also contributed the geometry contract and coordinate pullback. The
observations below should be treated as qualitative corroboration, not an
independent anatomical annotation or statistical validation set.

## Site1

The native and geometry-mapped fine images share several spatially specific
structures. At native layer10 the paired dark recesses around core
X120–160,Y50–105, the transverse darker band around Y115–125, and the
separate dark regions near the right margin recur in both acquisitions.
At layer17, the isolated material protrusion around X40,Y85 and the two
larger material bodies around X35–60 and X120–160 below Y115 have the same
relative arrangement in both. These jointly support correspondence of this
local material arrangement. They do not identify ink or uniquely label a
particular microscopic sheet.

The fitted ZYX shift `[0,-1,0]` is visually small at the displayed scale.
Fine and native retain differing edge sharpness, brightness and internal
fiber texture after the shift. Small bright inclusions are not all equally
conspicuous; I would not use their apparent intensity alone to adjudicate
alignment.

In the full normal context, both acquisitions show the broad undulating
material envelope and its lower boundary: across X, the gap below the
central/right material band and the protrusions near X40 and X125 are
traceable. The fine view resolves several thin adjacent bright/dark bands
where the native view merges them into a thicker feature. Across Y, curved
parallel bands turn through the central region and additional material
enters near the lower depth boundary. I cannot follow one individually
identified lamina across the entire displayed interval in both scans.
Repeated internal bands and coarsened edge profiles remain a credible source of
nearby matching alternatives. The visually recognizable material envelope
does not resolve the failed numerical uniqueness margin.

## Site2

This site has particularly clear large-scale landmarks: at layer10 a broad
dark channel occupies approximately X50–115, bounded by a narrow vertically
extended material strip near X120–145 and structured material to the right.
The channel's widening/narrowing and the separate material island around
X75,Y180 recur in the other acquisition. Across normal depth, the X-section
contains a distinctive sequence of a left crest, a valley around X65–90,
a crest around X135–150, and another valley around X165–180. Their sequence
is traceable in both acquisitions.

The unshifted fine panel does not match the native layer13 occupancy well:
the upper-left/central material and dark channel arrangement differ visibly.
The fitted `[2,1,1]` shift brings that arrangement much closer. This is
consistent with a remaining normal-depth displacement after geometry
mapping. It does not, on its own, prove that a global translation correctly
aligns all local structures or that the same particular lamina is selected.
The shift was estimated from the data; its visual benefit is not an
independent test of the fitted region.

The full-depth context contains a relatively separated central material
band, with a substantial gap to other material below it in parts of the
field. That separation is helpful for recognizing the corresponding broad
bundle. The fine image still resolves internal interfaces and additional
adjacent structures that are not individually separable in native CT.
There is no authenticated single-face marker visible across the complete
normal context. The magenta strip in the fitted comparison sections is
explicitly missing shifted support at the end of the retained mapped
depth interval, not an anatomical gap and not evidence of noncorrespondence.

## Site3

At layer10, a wide upper dark cavity and a transverse material band around
Y95–165 recur in both scans; the band's irregular lower boundary and the
dark areas beneath it have similar ordering. At layer13, the narrow bright
curving feature in the upper-left, the dark central/right recess around
X125–195,Y75–110, and the lower transverse dark interval around Y195 recur.
At layer17, the large upper-right cavity and the lower irregular material
bodies are also recognizable. These are useful structural associations
across depth, beyond merely matching average brightness.

The `[0,2,2]` fit shift produces only modest visible changes in the overview
panels. Native and fine retain differences in the width and connectivity of
smaller bright material and dark gaps. The finer image's greater internal
contrast and sharper boundaries make it difficult to declare exact edge
agreement from these displays alone.

Both full-context sections show substantial material above and below the
central band. Across X, cavities and angled internal interfaces are clearer
in fine CT; across Y, a lower folded material band rises near the center while the
central material band remains corrugated. Native CT captures their broad
envelopes but blends multiple nearby interfaces. The visible correspondences
support the same local folded material arrangement; they do not demonstrate
a unique one-to-one assignment of every interface. Contributions from
different internal structures remain unresolved; these images do not
establish that an adjacent wrap was selected.

## What the panels can and cannot decide

All three sites contain useful structural detail for studying acquisition
mismatch. The fine and native observations clearly preserve common
nontrivial spatial organization. These examples make the pairing worth
investigating; they do not establish that all apparent fine detail is
recoverable from coarse CT.

The full-depth fine panels hold UV fixed at the mapping for native layer13.
Their displayed nominal depth axes therefore do **not** define exactly
the same physical rays as native CT through the whole interval. Apparent
full-context differences can mix residual registration, unequal sampling,
different normal fields and acquisition/reconstruction effects. They should
not be read as exact overlay failures or as measurements of lamina spacing.

The fixed panels provide two central orthogonal sections through complete
retained source depths; they do not inspect every ray in the entire 3D crop.
Consequently, observing a distinctive large cavity or fold at the center
cannot certify all peripheral support or establish absence of wrap
contamination elsewhere. There is no visual basis here for declaring an
entire crop registered to an individual ink-bearing face.

Retain the current numerical failures and their original thresholds. The
appropriate present conclusion is **traceable structural correspondence,
with the precision of local CT correspondence still unresolved**. Do not
accept a visually preferable fitted image as pixelwise supervision solely
because the main bundle looks familiar, and do not reinterpret an ambiguous
or absent feature as a negative ink label. No ink inference or ink claim is
made by this review.

## Physical interpretation and the next useful comparison

Bright/dark bands within these CT sections are **not automatically different
papyrus wraps or individual sheets**. They can include internal fiber
structure, interfaces within one material bundle, reconstruction response,
and partial-volume effects. This review has not assigned the visible bands
to named laminae, counted sheets, or established a wrong-wrap match. Its
positive finding is correspondence of the same local CT material/structural
volume; its unresolved question is the precision and ambiguity of mapping
specific positions within that volume.

Requiring certification of an individual ink-bearing face before any paired
volume research would impose a stronger condition than these initial
acquisition experiments require. The present strict failures block treating
the current point mapping as precise teacher-transfer supervision. They do
not demonstrate that same-region, multi-depth paired learning is infeasible.

A useful next test would quantify the consequences of registration
uncertainty **on these same fixed supports**, before training on this mapping.
Preserve the complete score landscapes and accepted geometric support;
enumerate a prospectively specified uncertainty neighborhood around the
competing structurally plausible shifts rather than selecting one visually
preferred alignment. Report how stable structural correspondences and
cross-acquisition differences remain over that neighborhood, region by
region, including competing solutions. If the scientific contrast survives
this uncertainty, a paired-volume study could proceed with explicit spatial
uncertainty. If the contrast changes sign or disappears, registration remains
the more immediate limitation. This is a proposal, not a new analysis or a
retroactive relaxation of the frozen admission gate.
