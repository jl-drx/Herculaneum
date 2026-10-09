# Plinius preservation allocation: mechanism succeeds, development criteria fail

The candidate reduced prediction drift on the retired-reference part of the PHerc0139 training schedule while preserving manual fit within the frozen tolerance. All five mechanism criteria passed. That training benefit did not deliver the required development improvement: nine of fourteen practical criteria passed and five failed. **Retain unchanged seed42; do not promote either adapter or scale this recipe to a paid GPU run.**

The result narrows the explanation for earlier failures: adding protection to this particular nonmanual training support is insufficient under the tested recipe. It does not establish that preservation, depth features or adaptation generally cannot work, nor that geometry caused the regional losses.

## Frozen comparison

Both arms started with the same fresh, zero-output 5,296-parameter explicit-depth adapter and the same frozen 34,546,498-parameter seed42 backbone. They used the exact same 32 batches, augmentations, 17 source planes, manual targets, AdamW settings and CPU environment. Decoder input gradients were active. No rejected adapter was reused.

The only objective change reallocated one unit of output-preservation weight:

| Arm | Objective |
|---|---|
| Control, `manual_double` | Manual BCE + 2 × manual KL |
| Candidate, `manual_retired` | Manual BCE + manual KL + retired-reference KL |

KL uses seed42 probabilities, with a separate mean within each group. It does not use retired ink labels or pseudo-labels as targets. The total group coefficient is two in both arms; this is not a uniform union mean and does not leave the manual KL coefficient unchanged. Both arms' own objective and both common objective formulas were retained; unlike own objectives are not a direct common risk comparison.

The support audit found 54,701 unique scheduled manual pixels, 43,515 retired-reference pixels and 4,184 acquired pixels in neither group. Scheduled exposure counts were 618,208, 409,448 and 20,920 respectively. Supplied supervision and validation do not intersect, so relaxing the validation exclusion alone adds no supervision. Unknown pixels were excluded. These are supplied participation roles intersected with acquired support, not authenticated tissue or historical renderer-validity masks. Historical identity checks can read arrays containing retired label bytes; those labels never enter this objective or selection.

The [protocol](PRESERVATION_ALLOCATION_PROTOCOL.md), final execution plan, implementation and independent numerical tolerances were fixed before training. The source-geometry work is a separate audit and cannot change this experiment's masks or decisions.

## Verified results

| Final shared training quantity | Control | Candidate |
|---|---:|---:|
| Manual smoothed BCE | 0.593781507 | 0.593795691 |
| Manual KL | 0.001493309 | 0.000902448 |
| Retired-reference KL | 0.001205592 | 0.000561864 |
| Mean absolute probability drift, manual | 0.010075611 | 0.007516071 |
| Mean absolute probability drift, retired | 0.009569571 | 0.006543241 |

The candidate reduced retired-reference KL by 0.000643728, exceeding the required 0.0001 reduction. Its manual BCE exceeded the control by just 0.000014184, within the 0.001 allowance. Each arm improved its own objective by the required amount, and the candidate improved manual BCE from initialization. Lower manual and retired KL are observed results of this fixed run; they do not prove a general advantage of the allocation.

| Reused development aggregate | Seed42 AUROC / AP | Control AUROC / AP | Candidate AUROC / AP |
|---|---:|---:|---:|
| Paris2, equal weight per fragment | 0.657642 / 0.384286 | 0.668216 / 0.366364 | 0.640727 / 0.364451 |
| Paris1, equal weight per context | 0.784788 / 0.240483 | 0.795803 / 0.229569 | 0.799213 / 0.239333 |

The candidate loses 0.016915 AUROC and 0.019835 AP against seed42 on Paris2, failing both required +0.005 gains. These ranking metrics do not depend on the chosen decision cutoff. The control improves aggregate AUROC but loses AP; it is not a successful substitute selected after seeing results.

| Candidate minus seed42 | AUROC | AP |
|---|---:|---:|
| Paris2 Fr47, context 1 | -0.021066 | -0.006896 |
| Paris2 Fr47, context 2 | -0.086212 | -0.128053 |
| Paris2 Fr143, context 1 | +0.056445 | +0.023575 |
| Paris2 Fr143, context 2 | -0.016828 | +0.032034 |
| Paris1 Fr34, context 1 | +0.001830 | -0.000756 |
| Paris1 Fr34, context 2 | +0.027021 | -0.001544 |

The three failed regional retention criteria are Fr47 context 1 AUROC and Fr47 context 2 AUROC/AP, each below the -0.02 floor. Fr143 improves on aggregate while Fr47 worsens, so the mean does not support uniform transfer. All six outcomes, both final adapters and unchanged seed42 are retained. No threshold, crop, checkpoint, depth or hyperparameter was selected from these outcomes.

![Training and all six development regions](figures-v2/preservation-allocation.png)

## Verification and limits

Training, development and independently reconstructed report arithmetic all passed on their first attempts under the frozen tolerances. There were 64 total optimizer updates, four backbone loads, no scientific worker retries, no threshold fits and no checkpoint search. Actual original/teacher and zero-adapter predictions were bitwise equal on all 32 initial batches per arm. Frozen-backbone state checks and full saved optimizer traces passed. First-step tiny output-gradient differences from finite-precision KL arithmetic were correctly treated as diagnostics; parameter updates remained bitwise identical at that step. KL was computed in FP64; the saved output-gradient arrays are FP32.

Training and development controllers took 56.94 and 53.84 seconds. Their overlapping call counts sum to 364 full-model, 364 fusion, 1,112 original 2D-network and 748 adapter forwards, with 64 backward/decoder-input-backward/update calls each. The independent checker reconstructed losses, output derivatives, optimizer arithmetic, blending and metrics from retained evidence; it did not independently rerun convolution/autograd or decode checkpoint tensors. A layout-only figure rerender is retained and did not affect scientific outputs.

All six Paris contexts remain reused **development**, not independent confirmation. This is one matched initialization and 32 updates per arm, not a multi-seed or training-to-convergence study. Paris2 and its aliases stayed outside gradient training, teachers and cutoff fitting. The fixed cutoff was 0.7558947205543518. Upstream annotation and teacher ancestry remain unresolved. No statistical population claim, recovered text or translation follows from these scores.

The [complete metrics](allocation-analysis/metrics.json) contain all frozen criteria, common objectives and per-context confusion counts. CPU research cost no paid GPU spend. The £10 total first-pilot cap remains intact, including failures, retries and attributable overheads.

## Research decision

Pause objective and architecture sweeps on these same six regions. Preserve this negative development result alongside the positive training mechanism result. Complete source and geometry verification, then design the next study around a distinct falsifiable question and an independently defensible evaluation set. A new native model comparison would first need verified weights, alias consistency, strict loading and its own input/preprocessing contract; a header match alone is insufficient.

The companion phase report records geometry and source findings, including their unresolved historical-provenance limits. None authorizes altering this completed experiment or renting a GPU automatically.
