# Fixed follow-on CPU comparison — supplied Frag6 reference

This specification is set after annotation/optical QC and before any Frag6 model scores. It narrows the first comparison to a descriptive historical-render index-shift experiment on one PHerc51 parent. It is not an anatomical surface-displacement validation or an independent optical-accuracy test. All four phase31 panel tiles stay fixed, including unfavorable future outcomes. No new CT/model execution is authorized by the completed phase31 QC acquisition contract; the next acquisition must bind per-object headers/ETags and frozen bounded transport before it runs.

## Actual inputs and scoring support

For each selected 512×512 tile, take the central 384×384 source raster: origin (tile_y+64,tile_x+64). Use supplied historical surface_volume layers 01.tif through 63.tif inclusive, with their explicit observed layout, unsigned16 values and historical interpolation. Do not switch either arm to corrected trilinear rendering. Three consecutive Z planes and 3×3 XY pixels are averaged using the retained exact production formula: uint32 sum of 27 values / float64(27×257), cast once to float32. No intermediate uint8 quantization. This creates 21×128×128 values with raw-plane group centers 2,5,...,62. The nominal depth step is three source-index steps, equivalent to 9.72 µm only under the documented 3.24 µm/unit-normal interpretation; do not claim authenticated anatomical displacement or measured global XY metric distortion.

Derive each 128×128 label from the exact aligned 3×3 source block. All nine source pixels must be mask-valid and all nine labels must agree: all positive→1, all negative→0, otherwise unknown→2. Score only the central 64×64 output pixels (indices32:96 on each axis). This fixed border excludes output near the model patch edge but is not a proof that the entire receptive field avoids internal padding. No padding or invented data in the CT crop itself. Keep full128×128 label arrays and separate64×64 scoring arrays. Do not replace a tile or shift the core if it lacks either class. Report undefined metrics and stop any promotion interpretation if class/support eligibility fails. This homogeneous-block convention discards mixed boundaries; report its lost coverage rather than claiming boundary performance.

## Model and fixed arms

Use unchanged forward seed42 checkpoint e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab, embedded step74999, with qualified FP32 eval adapter and robust-MAD normalization computed separately on each full17×128×128 patch. No mask-driven normalization, reverse depths, augmentation, model fitting, checkpoint selection or threshold search. Load only the retained verified model/source through the existing qualified loader.

Compute five unique windows with starts0,1,2,3,4 on the21-plane array, each17planes. Exactly20 single-patch forwards maximum across four contexts. For evaluation displacement d∈{-1,0,+1}, baseline is p[d+2]; candidate is (p[d+1]+p[d+2]+p[d+3])/3 in probability space. Both arms use identical support. The baseline at d=0 uses raw planes7..57 with the established group centers8..56. A single128×128 patch needs no cross-patch blending. Seed42 already trained with depth jitter; equal-window averaging is a baseline test, not a novel model claim. Window responses are not a3Dink probability volume.

## Outcomes fixed before scoring

Use the phase29 exact tie-aware AP/AUROC implementation. Compute paired candidate−baseline AP per context and displacement; average the three displacements within each context, then average four contexts within their single parent. Also report d=0 clean AP/AUROC, individual outcomes, score/range sanity and exact common support. No pixel confidence interval, significance claim or threshold optimization. Label-selection and unknown teacher ancestry prohibit an independent generalization claim.

For this feasibility experiment, require mean robust AP gain≥0.01 and mean clean AP loss≤0.01 before calling the averaging candidate promising for replication. These practical screening margins are fixed now without observed model scores, not a powered statistical guarantee. Retain all results even if thresholds fail. One parent cannot justify model promotion.

## Bounded implementation target

Next phase's exact source/request manifest must stay within: 63 TIFF objects, 800 requests total, 192 MiB reserved response bodies including framing and failures, no retries or redirects, 600 seconds acquisition, 1 GiB phase disk including raw payloads/results, 2 GiB RSS. Use validated multipart ranges to acquire only the required four384×384 crops (74,317,824 useful CT bytes); no full-layer download fallback. Metadata must establish each object's shape, codec, byte order, strip offset/length and identity before its pixels. Abort if multipart behavior/objects differ; retain the failed attempt and revise explicitly instead of silently expanding the cap.

CPU inference maximum20forwards, four threads, 600 seconds, 4 GiB RSS, FP32/no training. Final source and model input hashes, all output arrays and exact metric records must be retained. Resource/readback failure means no scientific result. This specification does not itself implement or test the next transport or model job. GPU spend remains£0, first paid pilot total cap£10.
