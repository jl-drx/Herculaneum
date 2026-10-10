# Remaining worker qualification — not yet executed

Input and schedule feasibility passed. The reduced schedule also reconstructed
all 16 d0 maps exactly from retained logits. These results do not qualify a
shifted-depth worker that has not yet been implemented and frozen.

Before production, test every combination of the two nonzero offsets, two
padding phases, two input orders and four models: 32 qualification cases.
For each family/phase, choose the first and last retained original cohort with
four members, by original batch index, before looking at shifted predictions.
Use the first cohort's four distinct tiles for batch-versus-singleton comparison.
Require maximum probability difference ≤1e−5, unchanged preprocessing identities
and finite outputs. Retain raw qualification arrays.

Use those same two complete cohorts for direct original-framework versus cached
output equality, followed by a read-only exact replay. Preserve original cohort
IDs and the full global accumulation/interpolation conventions. Do not regroup
or use local crop coordinates. Inject a stop after the second committed batch
in the first case, then resume from both committed batches without another
model forward. Read-only tamper checks must reject changed depth, order, model,
cohort or input identity, as well as missing receipt, missing logits and changed
logit bytes. Perform tampering only on copied fixtures.

The initial qualification budget is at most 288 new model calls and 768 tile
evaluations: per case, one batch-of-four call plus four singleton calls, then
two direct and two cached cohort calls. Replays and rejection checks require
zero fresh model calls. If any gate fails, preserve evidence and diagnose it;
do not automatically retry with weakened tolerance or run production. Any
scientifically material adjustment needs a separately recorded protocol version.

On pass, freeze the production worker, all prepared inputs, model/runtime pins,
depth positions, masks and exact cohort lists in an execution GO. The new depth
must be in both the job key and cache binding. Do not reuse d0 logits for shifted
inputs. Production is exactly 3,096 committed batches across 32 shifted maps;
this is distinct from the qualification budget.

Use the same deterministic CPU environment, fp32, four threads, maximum batch4.
Save each raw logit and receipt atomically, with receipt last. Replay through
fresh accumulation before each map COMPLETE receipt, compare only on inherited
support, and record whole-run completion separately from process liveness.

Prepare private checkpoint storage and a restore check before long execution.
Retained study51 inputs/runtime can be references to their pinned private
archives; new depth preparations, script identities and committed scores need
coverage. No worker or backup watcher has been launched by the feasibility
stage. Runtime estimate for production is roughly 3h09m from study51 cohort
timings, plus qualification, preparation/replay fixed costs and backup overhead.
It is not a deadline or a measured runtime for the new worker.
