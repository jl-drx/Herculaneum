# Frag6 reference support and optical inspection — phase 31

Frozen before reading mask, ink-label or optical pixels on 9 October 2026.

## Question and permitted interpretation

Can the supplied Frag6 annotation define a finite panel for a supplied-reference agreement experiment? Inspect support, encoding, class coverage and optical overlays; do not infer independent truth, exhaustive negatives, author identity or registration accuracy from co-location. Phase30's local historical CT correspondence remains supported; its original six-recipe failure and later successful historical-kernel explanation both remain unchanged. No CT/model acquisition or execution in this phase.

## Acquisition and resource limits

Acquire exactly mask.png, inklabels.png and ir.png from the retained official PHerc0051Cr04Fr08_53keV_3.24um/surface_processing directory. One full HTTPS request each, strong ETag and exact bounded Content-Length, verified TLS, no redirects/retries, immutable retained bytes with SHA256. Caps: mask and labels 1 MiB each, IR 20 MiB; maximum reserved response bytes 24 MiB, three requests, 45 seconds/request, 180 seconds total acquisition. A full PNG is necessary to decompress its pixels; no raw volume is downloaded. Require valid PNG framing/CRCs and the expected 6205×8853 canvas before decoding. Stop on unsupported encoding rather than guessing. Up to 1 GiB RSS, 240 seconds for QC, 96 MiB phase artifacts excluding the cumulative retention manifest. Retain failures and exclusions. Use public EduceLab-Scrolls data for local research under retained source terms; no redistribution/publication is performed.

## Explicit supplied-label convention

Allow 8-bit grayscale, binary, grayscale RGB, and grayscale palette representations, with only fully opaque pixels eligible. White (255) means supplied positive; black (0) means supplied negative by convention. Any intermediate intensity, chromatic pixel or nonopaque pixel is unknown. White mask is valid support, black is excluded, all other mask values are unknown/excluded. Report all exact value counts, all labels outside support and both-class coverage. This convention does not establish that unmarked pixels are truly ink-free. Do not threshold away unknowns. The declared convention is rejected if no known positive/negative support remains.

## Prospective panel rule

Enumerate full, nonoverlapping 512×512 tiles starting at (row=32,column=32), step 512, with a 32-pixel outer canvas margin. Record every tile and exclusion. Tile eligible if at least 95% pixels have white mask and binary labels, and at least 256 supplied positives and 256 supplied negatives lie on that common support. Divide the full canvas into four equal vertical bands using tile-center row; in each band select the first eligible tile in row-major order. No replacement from another band if empty; at most four tiles. Selection uses annotation/support only, never optical contrast, CT intensity or model output. The grid margins are not a proven model receptive-field halo. These are annotated development contexts from one conservative PHerc51 parent, not independent parents or a new unexposed test set.

## Optical inspection and measurements

Create a whole-canvas IR overview with labels and chosen tile boundaries, and native-resolution selected-tile optical/label pairs and contour overlays. Inspect all selected tiles. Report optical intensity histograms/means on positives and negatives as descriptive diagnostics, not a registration acceptance test or ink metric. Do not optimize offsets, invert label polarity, fit a warp, hand-edit labels or select favorable visual regions. A visual overlay can suggest local consistency or a visible problem; without disjoint attributable anatomical landmarks and a measured fit/check residual, optical registration remains unqualified. No transcription/translation is attempted.

## Review and next decision

Recompute whole-image counts by an independent Pillow histogram/row-chunk formulation and reselect tiles from the retained table; verify selected label arrays and source hashes. Check ambiguous/unknown handling with focused synthetic fixtures. Preserve all previous evidence and untouched PHerc0009B, PHerc0500P2 and PHerc0841 reserves. GPU spend £0, total first-pilot cap £10. If panel support passes, bind exact selected objects/coordinates and state the remaining input/depth contract needed for the fixed seed42 versus equal-three-window comparison. Independent optical truth remains a separate claim. Do not require unavailable annotation history merely to describe explicitly qualified supplied-reference agreement.
