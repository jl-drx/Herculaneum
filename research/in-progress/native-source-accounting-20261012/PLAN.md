# Source loss accounting before the coverage-pair freeze

This offline preparation uses the already qualified PHerc0139 population and
first complete PHerc0814 case. No model, optimizer, new scan, evaluation reserve,
new label threshold or paid runtime is involved. The eight-source intake and all
historical results remain unchanged. This is not another learning experiment.

The earlier preflight established that separate original/added denominators
preserve the original loss component. The remaining design question is whether
matching segment draw quotas also matches segment weights inside the added
stream. A pooled masked BCE assigns a segment the fraction of batch supervision
pixels it supplies, whereas retained per-image Dice follows image counts.
This distinction was not quantified by the original occurrence ledger.

1. Verify the saved tables against the preceding qualification receipts. Report
   every original winding, and both fixed support sets in the complete added case.
   Compute large-batch source coefficients for equal source draws; label them as
   limits, not expected finite-batch gradients.
2. Enumerate every cross-winding pair, one uniform patch from each source, to
   compute the actual finite-pair BCE coefficient. Also enumerate all ordered
   two-patch draws with replacement for the complete added case's A and B pools
   to distinguish mean patch class share from batch-normalized class share.
   These are deterministic diagnostic samplers, not the future training schedule.
3. Propose an explicit weighted mean of separately normalized segment losses
   for the added stream. Keep the original stream definition unchanged. Require
   every configured source within each optimizer accumulation window and use
   post-augmentation support denominators. Missing sources may not silently
   renormalize the others. Actual coefficients and batch sizes remain unfrozen.
4. Exercise the proposed arithmetic using the first four qualified added patches
   in saved CSV order, split into two *probe groups*, not two physical sources.
   Alter only the second group's supervision mask, then check the first group's
   conditional logit gradients. Include pooled-loss and naive-microbatch negative
   controls. No parameter update or prediction accuracy is measured. Compare
   full versus split accumulation to relative gradient tolerance 1e-6.

This removes a potential source-normalization confound, not within-source class,
quality, support, or geometry differences. The full population/class ledger is
still required. If the remaining sources invalidate the coverage contrast, stop
or rename that design rather than treating this check as admission.
