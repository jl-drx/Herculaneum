# Frozen central control plan

The geometry-selected primary context begins at native `(512,512)` and spans 1536 × 1536 pixels. Its fixed central crop begins at relative `(256,256)`, native `(768,768)`, and spans 1024 × 1024 pixels. All 31 source depths are retained without resampling or intensity transformation. The crop and its Boolean geometry mask are frozen with parent acquisition hashes and an exact copy of the parent provenance.

[The fixed plan](fixed-controls-plan.json), SHA-256 `39c5a2f5cda2051a6c4d173f182f7d62607dd1249739ee3bf50311df05142b75`, specifies four sequential seed-42 conditions: center forward `7:24`, minus 2 forward `5:22`, plus 2 forward `9:26`, and center reverse `7:24`. The center-forward condition checks context consistency with the full-surface run. The other conditions diagnose sensitivity to depth selection and orientation.

The earlier budget proposal described three controls. Adding the predefined center-forward consistency check brings the control budget to **900 scheduled tiles** and the combined primary/control budget to **1958 tiles**. At 0.16–0.20 seconds per scored tile, the all-scored projection is approximately **5.2–6.5 minutes of inference**, plus model load and output overhead. Each process uses four CPU threads; processes run sequentially.

Both origins share the same 64-pixel tile phase. Crop `[64:960,64:960]` corresponds to primary `[320:1216,320:1216]` and native `[832:1728,832:1728]`. Within this area, seed-42 center-forward should reproduce the primary seed-42 result exactly because patch inputs, skip decisions, Hann weights and accumulation order coincide. This agreement verifies context consistency, not ink accuracy.

All paired sensitivity summaries use one shared mask: valid geometry, evaluated finite scores from every compared condition, observed support in every relevant 17-depth window, a 64-pixel crop-edge margin, and Euclidean distance ≥64 from invalid geometry in the original global mask. Observed support means at least one nonzero intensity among selected depths at each XY location; it does not establish all-layer coverage or ink.

A second, stricter mask excludes the entire footprint of every scored contributing tile containing any invalid geometry or XY location missing selected-depth support, then intersects these masks across conditions. Hann weights are floored above zero, so tile corners also contribute. Report both domain sizes; midpoint counts, map correlation and agreement remain descriptive behavior measures without independent accuracy or sample-size claims.
