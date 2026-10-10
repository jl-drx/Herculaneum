# Independent first-step output verification

**PASS.** NumPy reconstruction verified all six independently reset branches and 1,584 parameter-summary/probe rows. The ready plan is `d57d1a9d17bc11d8dd6630de58deb73338674e91530dc522ff28184a7afc599d`; the root run index is `afae91cf22d96ab305e6f74505a9fc0848fd33bfc482c6e0a6494bacc5f2ee4c`.

The saved normalized input, masks and hard targets reproduce the exact historical input fingerprint. Both objectives have identical inputs and archived initial FP32 logit bit patterns. Per-branch source/state/gradient identifiers, fresh-step receipts, operation counts, runtime and combined resource acceptance agree with the frozen plan. All retained output files were hash-checked; checkpoints and raw gradients were not read.

Independent FP64 reconstruction of the saved FP32 logits agrees with the recorded losses to at most **1.020662e−7**, and with the stable decomposition terms to at most **2.775558e−17**. These are changes on the identical first batch, not final-model or generalization results:

| Objective | Step learning rate | Smoothed-base change | Full-objective change |
| --- | ---: | ---: | ---: |
| A | 1e−6 | −0.006744742 | −0.005870369 |
| A | 1e−5 | −0.018163432 | +0.024002314 |
| A | 1e−4 | +0.014596069 | +0.118164937 |
| B | 1e−6 | −0.007344414 | −0.006495624 |
| B | 1e−5 | −0.021086381 | +0.026476169 |
| B | 1e−4 | +0.020079710 | +0.145220191 |

Every recorded parameter-gradient/update dot product is nonpositive; aggregate dots are strictly negative. Reported per-parameter norms, changed/unchanged counts and error maxima aggregate consistently. Each retained parameter-coordinate probe satisfies the analytical fresh AdamW formula and the predeclared FP32 allowance; the maximum independently recalculated probe allowance ratio is **0.163918**, below 1.

This parameter check rechecks summaries and one retained coordinate per gradient-bearing tensor. It **does not independently replay every parameter update**, because complete gradients and updated parameter states were not retained. The reviewed worker performed its all-coordinate identity/reset checks during root execution. No full-gradient reconstruction or checkpoint reloading is implied by this receipt.

The first reviewer run completed all numeric assertions but failed while serializing its final receipt: a NumPy float32 ratio was not converted to a Python float. `verify_step_outputs_v1.py` and `output-verification/failure.json` preserve that failure. The corrected checker changes only that final scalar conversion and the output directory; all scientific assertions and tolerances are unchanged. The corrected read-only verification passed. There was no experiment retry, new prediction or model call.

Final receipt: [output-verification-corrected/verification.json](output-verification-corrected/verification.json), SHA256 `3e4d4c78da97cd4266075de495cf75aba44492dba110a16329a6585bb169a65e`. Reviewer operations were NumPy and standard-library reads of retained outputs, with zero network, model, checkpoint or raw-gradient calls.
