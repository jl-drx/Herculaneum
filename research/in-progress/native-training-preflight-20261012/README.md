# Training-loader exclusion preflight

12 October 2026. A supporting CPU check while the frozen five-model comparison
runs. No model checkpoint was loaded, no optimizer update occurred and no real
reference or new scan was opened. The running comparison was not altered.

**The retained flat loader can preserve exclusions, but the data contract matters.**
The first synthetic audit changed 473,088 excluded voxels. With repeated 2D masks
and zero label-coverage filtering, all 16 training patches retained exactly the
same membership, normalized inputs, effective targets, loss and probe gradients.
Raw target values differed only where ignored. The 25 validation patch locations
also stayed fixed; their labels were not used for training.

Two deliberate negative controls expose boundaries of that result:

- At a nonzero `min_labeled_coverage=0.1`, discovery used unmasked label bounding
  boxes. Changing excluded labels introduced 16 training patches. The released
  Reader config uses zero, so this is a configuration hazard, not evidence of
  leakage in that released training run.
- When supervision varied with depth, flat `amax` target projection admitted
  positive values from an excluded plane into 16 supervised 2D targets and
  changed gradients. A depth-varying mask cannot be treated as a repeated flat
  mask without defining how labels are projected.

The actual pinned Reader label builder writes both labels and supervision only
on the central plane, leaving all other planes zero. The emitted positive labels
are a subset of its supervision (teacher >=128; supervision also includes teacher
<=64). It therefore avoids the constructed off-plane condition by canonical data
construction. The audit does **not** establish corruption of Reader, its weights,
or the current comparison's references.

## Follow-up that passed

The second check exercised both repeated-2D and central-plane-only representations
with the actual default augmentation graph and real-layer jitter. Across 24 fixed
RNG seeds per representation, all five offsets −2 through +2 occurred. It checked
48 cases / 96 paired comparisons: perturbing excluded values on the representation's
label planes, and perturbing all excluded voxels followed by canonicalization.
Inputs, effective targets, losses and two-parameter probe gradients matched exactly.

The canonicalization is a **training-only view**, applied before augmentation and
depth projection:

```python
training_labels = np.where((supervision > 0) & (validation == 0), stored_labels, 0)
```

The original labels and validation references remain immutable. Patch selection
must also use admitted support, or canonicalized labels if a label-dependent
selection rule is deliberately part of the study. Existing zero-threshold
discovery passed without that additional label filter. This does not authorize
discarding uncertain supervision or converting unknown pixels to negative labels:
the separate supervision/ignore mask must still exclude those zeros from loss.

This closes a bounded synthetic preflight. The final training cohort still needs
real-array/provenance admission, physical-family exclusions, an explicit matched
draw schedule, all actual preprocessing, and the same counterfactual check through
the final GPU loader. No scientific learning improvement follows from this audit.

## Original-source loss weight needs its own control

[A separate loss-accounting probe](LOSS_ACCOUNTING.json) found that identical
original patch draws do not, by themselves, preserve the original-source loss
weight. The retained BCE divides by the number of supervised pixels across the
whole batch. With 64 original supervised pixels, changing only the added patch
from 64 to 256 supervised pixels reduced the original BCE gradient to **0.40×**:
the denominator changed from 128 to 320. Combined BCE+Dice original gradients
changed too; the 60% reduction refers specifically to the BCE contribution.

With fixed stream coefficients and separate stream-normalized objectives,
`0.5 * original_loss + 0.5 * added_loss`, original logit gradients were bitwise
identical across that counterfactual. Added-stream gradients still changed, as
expected. This used the actual retained smoothed BCE+Dice implementation and
synthetic float32 logits; there was no network or optimization. It is a control
for the proposed experiment, not a new loss-function contribution or an observed
training gain.

The future coverage pair must therefore freeze both source draws **and** objective
accounting. Apply the same fixed stream objective in both arms, including every
loss/deep-supervision term. Choose its coefficient with the actual cohort before
training; 0.5 was this probe's balanced demonstration, not a selected final value.
Do not claim equal supervision amount from equal patch counts. A broader corpus
can still change class balance, annotation amount and quality, and batch-dependent
normalization requires its own control. Only the specific loss-dilution mechanism
is removed by this test.

## Reproduce

Use the existing `pherc343` CPU environment and the separately installed versions
in `requirements-extra.txt`. They were installed into a dedicated `extra-deps`
directory, not into the live comparison environment. The initial import failures
and harness correction are retained in `ENVIRONMENT_NOTES.json`; no failed attempt
was counted as a passed check. Source pins are in `EXCLUSION_CHECK.json`; the full
retained runtime is additionally bound by the common-comparison GO manifest.

```bash
PYTHONPATH=/path/to/extra-deps /workspace/.venvs/pherc343/bin/python \
  research/in-progress/native-training-preflight-20261012/check_exclusions.py \
  --repo /workspace/Herculaneum --output /new/EXCLUSION_CHECK.json
PYTHONPATH=/path/to/extra-deps /workspace/.venvs/pherc343/bin/python \
  research/in-progress/native-training-preflight-20261012/check_training_contract.py \
  --repo /workspace/Herculaneum --output /new/AUGMENTATION_CONTRACT.json
```

Both checks require new output paths. Public code provenance: retained Vesuvius
runtime; Reader source at
[`02c7b46987813c78f56dd99f38c5a84b65a9cd2a`](https://github.com/DomRusso2/reader-v2/blob/02c7b46987813c78f56dd99f38c5a84b65a9cd2a/scripts/v2_run.py).

The loss-accounting check needs only the base `pherc343` environment:

```bash
/workspace/.venvs/pherc343/bin/python \
  research/in-progress/native-training-preflight-20261012/check_loss_accounting.py \
  --repo /workspace/Herculaneum --output /new/LOSS_ACCOUNTING.json
```
