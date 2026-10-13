PHerc 343: where does plain-versus-padded disagreement occur?
2026-10-10 | Six previously saved maps; descriptive CPU-only analysis

ANSWER
Disagreement is spread across the field, with modest pooled modulation at
verified grid positions. It is not concentrated in a narrow neighborhood of
tile-start lines. This is separate from the already-reviewed ownership-switch
boundaries. Overlapping Hann blending means a tile-start line is not itself a
hard stitched seam.

Published valid crop: 8,887,441 pixels.
                    dnative    seed 42    seed 43
All-pixel MAE         6.28      10.37      9.75
Within 4px plain line  6.43      10.76     10.15
At least 13px away     6.14       9.97      9.35
Near-line error share 26.92%    27.27%    27.36%
Near-line area share  26.29% for all three models.

The padded-grid comparison is similar: near/far MAEs are 6.46/6.09,
10.71/10.04 and 10.02/9.49. These compare separately against each phase's
potential start lines; the two grid neighborhoods overlap.

Excluding the outer 128 native pixels does not change the conclusion.
Interior means are 6.30,10.39,9.79; plain-grid near/far are 6.45/6.15,
10.78/9.97,10.20/9.38. The 64 fixed phase bins span 5.79..6.81,
9.29..11.42 and 8.93..10.85. Their pattern is visible in the figure but most
error remains throughout the grid phase, not only at the largest bins.

All 18 pre-existing regions were included individually. Their non-double-
counted union contains 1,936,467 valid pixels. Its means are 7.22,14.67,13.79;
plain-grid near/far are 7.35/7.11,15.15/14.06,14.62/12.84.
Local behavior varies: near/far ratios exceed 1 in only 12/18,11/18 and 13/18
boxes for dnative/seed 42/seed 43; ranges are 0.83..1.22,0.90..1.32 and 0.90..1.54.
Do not generalize pooled modest modulation to every individual candidate.

INTERPRETATION AND NEXT TEST
This supports broad sensitivity to changed inference context/phase, with a
small pooled grid association. It does not identify which implementation or
model mechanism causes the change, or show which map is more accurate.
A global amplitude offset is small (padded-minus-plain means -0.05,-0.41,
-0.59), but local amplitude effects are substantial. In the paired-output
mean 128..192 band, plain near/far errors are 9.43/8.80,27.74/23.85 and 28.20/24.26.
The dominant 64..128 band also retains a smaller near-line increase. This is
an output-conditioned descriptive comparison, not a texture-controlled test.
Ranking/candidate persistence is analyzed separately; these metrics do not
measure rank stability or ink correctness. CT texture is unavailable here.

After completing the unchanged reproduction, the simplest controlled follow-up
would be a prospectively fixed extra grid-origin test over the same CT field
(or fixed source-coordinate region with a sufficient context halo), keeping
checkpoint, physical depth, preprocessing and blend weights unchanged. Compare
aligned interior maps in source coordinates and grid coordinates to see whether
the modulation follows the grid. Freeze offsets and regions before seeing
outputs. That would test grid locking more directly, still without establishing
ink truth. No new inference was performed or commissioned here.

GRID VERIFICATION
Villa source pinned to ScrollPrize/villa commit
0fb38c45b488e192566ea83ae36369cb6db92652. infer.py SHA256 matches the retained
runtime identity: 979aed8e1bf047bea8e30bd0bbe55538556289c72b5914bbbbecf355e5940389.
Qualified configuration logs show 128 x 128 for all three models. The actual
runner requests overlap 0.5 and Hann, deriving stride 64. Source starts at 0,
steps by 64, and appends length-128 when necessary. Native plain starts are
64k; padded starts are y=64k-40,x=64k-200. The appended final padded starts
align to the original last starts but break regular phase. Hence the fixed
outer 128-pixel exclusion for phase plots; full-crop distance calculations use
the finite exact schedules including those appended starts.
Hann is a nonperiodic outer product, normalized and floored at 0.001; weighted
probabilities are divided by accumulated weights, scaled and truncated to
uint 8. Model probabilities are restored to input-patch size if necessary.
Only potential geometric tile starts can be reconstructed: occupancy-selected
tile coverage cannot independently be rechecked without the original CT.
Native padding was recorded as zero rows/columns with identical interior CT.

Sources:
https://github.com/jl-drx/Herculaneum/blob/f85c6ab579da054d3ca94c3b7e4b58699f7c6a3a/research/in-progress/pherc343-reproduction/run_fixed_depth.py
https://github.com/jl-drx/Herculaneum/blob/f85c6ab579da054d3ca94c3b7e4b58699f7c6a3a/research/in-progress/pherc343-reproduction/work/prepare-qualify-v1.log
https://github.com/ScrollPrize/villa/blob/0fb38c45b488e192566ea83ae36369cb6db92652/vesuvius/src/vesuvius/ink_detection/inference/infer.py

REPRODUCIBILITY AND LIMITS
PLAN.json was frozen before new metrics; a technical clarification adds a
33..128 distance overflow bin for final tile tails. It does not change near/
far definitions. Figure shows the first four bins; overflow is in metrics.json.
All 18 regions, counts, signed changes, amplitude strata and 64 phase means are
in metrics.json. The source files are preserved for inspection, not execution.
No new model execution, training, paid compute, repository or original-run
changes. Analysis used at most two CPU threads. Spatially correlated pixels,
preexposed overlapping boxes, occupancy and papyrus texture prevent causal or
independent-sample interpretation. No p-values or ink-accuracy claims.
