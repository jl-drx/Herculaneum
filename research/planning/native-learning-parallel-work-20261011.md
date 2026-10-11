# Useful work during the native CPU comparison

11 October 2026. Planning note following the user's GPU price comparison.
This does not amend the frozen CPU study or authorize a paid launch.

## Current evidence

At the 10:39 UTC check, the worker was alive with5,120 committed updates in the
first arm, w035-seed17. Training AP was0.937880 and BCE0.243773,62.8% below the
training-prevalence constant's entropy. All139 admitted training cores had
appeared. This first arm now meets the descriptive training-fit heuristics;
training loss is still improving. It is not a stopped endpoint, cross-winding
transfer result or authenticated ink finding. Held-out model predictions remain
locked until all9 endpoints finish. Private backups were current, with no
reported worker or backup error.

The experiment has already moved beyond the earlier question of whether a tiny
optimization budget can fit these training annotations. The remaining question
is how consistently the trained models transfer to excluded physical tissue.
Keep the current comparison intact while doing preparation that does not use its
held-out outputs or compete with the four-thread training worker.

## Recommended parallel work, in order

1. **Define the next finite paired-acquisition training packet.** Use retained
   source metadata to inventory candidate windings such as w040/w041/w043, their
   physical overlap, native/fine acquisition and rendering contracts, teacher
   ancestry, common supervision support and compressed download sizes. Select
   proposed blocks from geometry/validity before viewing teacher response.
   Deliver an explicit admission table and bounded acquisition plan, not another
   training run on the four inspected w040 supports. Preserve their failed
   correspondence screens. New teacher transfers must declare uncertainty and
   shared support; the four local alternatives do not validate entire windings.
2. **Prepare references that can distinguish learning from teacher imitation.**
   Inventory the author-designated w024 fine59 alternate reference and w053 blank
   control, then establish source frame, provenance and complete model-context
   exclusions before admitting either. Their names do not establish independent
   ink truth or unused tissue. The current w035/w039/w044 endpoints will be
   reused development benchmarks after this study, not a new untouched test.
   Keep native annotation agreement, alternate-teacher agreement and blank
   response separate. Reserve additional physical support prospectively where
   feasible; do not manufacture independence with random neighboring patches.
3. **Prepare a GPU execution qualification packet.** Pin a CUDA runtime and a
   prospective workload, preserve full28-depth/context/sampling semantics, and
   specify a short timing/memory/restart test on training or synthetic inputs.
   Measure complete step throughput, host preparation/transfer, validation and
   checkpoint overhead, peak VRAM and end-to-end setup/export duration. Qualify
   float32 first; mixed precision or a changed batch is a separate declared
   choice. A device change need not reproduce CPU floating-point trajectories
   bit for bit, so do not silently mix CPU/GPU arms in the current frozen study.

These are bounded data/design and engineering tasks. Existing novelty reviews
already cover generic distillation and Hecate: another broad literature sweep
or a new uncertainty architecture is lower priority until a specific gap needs
resolving. For the user's/Sam's review, the valuable artifact is the proposed
physical split/reference table and matched supervision contrast, not early
held-out model maps. No spare-VM assignment is necessary for these short tasks.

If the current study shows consistent transfer, the ordinary fine-observation
soft-teaching comparison has a credible native learner beneath it. If it fits
but does not transfer, broader coverage/reference mismatch becomes the immediate
research priority. The preparation above is useful in either case. The completed
soft-target controls do not presently justify adding a geometry-uncertainty
architecture as the main intervention.

## GPU selection using the user's quotes

Prices below were supplied by the user and were **not** independently checked
against current Runpod availability or billing. USD compute-only arithmetic:

| GPU | VRAM | Per hour | Four hours | Ten hours |
|---|---:|---:|---:|---:|
| RTX A5000 |24 GB|$0.27|$1.08|$2.70|
| RTX3090 |24 GB|$0.50|$2.00|$5.00|

For a fixed workload and equal non-compute overhead,3090 throughput must exceed
500/270 = **1.85185 times** A5000 throughput to have a lower GPU compute bill.
That is a break-even calculation, not a measured speed comparison. Availability,
host CPU, transfer/setup time and reliability may change the practical choice.
The **A5000 is the first benchmark candidate** at these quotes. Both are likely
to have ample capacity for the present122,513-parameter CNN at batch4, but peak
activation/workspace memory still needs measurement. No H100 requirement follows
from this workload; larger future models need their own sizing.

Recommend choosing future hardware by scientific readiness, measured time saved
and cost per completed experiment. A favorable CPU AP delta should not become a
universal prerequisite for an affordable GPU timing pilot. This recommendation
does not launch a resource or override the current CPU protocol.

The existing **£10 total first-pilot cap** remains: failed attempts, retries,
attributable storage/transfer, tax and exchange-rate margin all count. These
USD examples are not all-in GBP quotes. Before a launch, use a frozen workload,
qualified/pinned CUDA environment, current account-specific pricing, verified
private checkpoint export and an independent shutdown mechanism. The prior R2
worker-secret import was not completed; do not assume it now works. A verified
alternative private export path is acceptable, rather than requiring R2 by name.
No new cloud credentials, GPU resources, Actions jobs or external messages were
requested or created for this planning note.

Sources: [current study](../in-progress/pherc0139-native-learning-20261011/README.md),
[training inventory](pherc0139-learning-readiness-20261011/design/TRAINING_INVENTORY.md),
[data readiness](pherc0139-learning-readiness-20261011/design/DATA_READINESS.md),
[earlier matched-learning review](pherc0139-learning-readiness-20261011/design/DESIGN_REVIEW.md).
Those earlier reviews are historical recommendations; the later completed intake
and controls supersede their statements that native labels/CT were not acquired.
