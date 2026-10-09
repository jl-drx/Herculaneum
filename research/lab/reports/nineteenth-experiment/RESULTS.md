# Plinius: evaluation provenance and feature-fusion readiness

**The deeper fusion adapter is mechanically ready for a bounded developmental training experiment. No independent confirmatory dataset is ready yet.** Source audits found specific geometry and publication gaps rather than resolving them through assumptions. The actual seed42 integration check passed on synthetic input; this is not a new ink-detection performance result. The unchanged seed42 remains our reference.

The next execution should test the matched fusion branches on the retained developmental protocol while source-contract work continues for Frag6. Development can proceed with that limited claim. A fresh, independently annotated confirmation set remains necessary for stronger generalization and ink-accuracy claims. [NEXT_STUDY_DESIGN.md](NEXT_STUDY_DESIGN.md) specifies the comparison and its remaining scientific-runner checks.

## What the source audit established

| Candidate | Newly established facts | Current role |
|---|---|---|
| PHerc0009B | Complete inspected public label inventory lacks the advertised mapping-validity array and linked provenance. Mesh-relative center is plane 54 of 109. Older paper R50 model explicitly reports training on this parent's aliases. | Metadata-only reserve; exact transfer, participation and annotation-depth contract still required. |
| PHerc0500P2 | Same publication gaps. The mesh lies between planes 58 and 59 of 118. Acquisition metadata records 111 keV despite literal 110-keV filenames. Older paper R50 training exposure is explicit. | Metadata-only reserve; useful future supplied-reference agreement candidate after geometry closure. |
| PHerc51 / Frag6 | Four native scan identities, voxel pitches and dimensions plus a scan-to-scan transform are resolved. This is one parent across all scans. | Preferred locally unscored candidate for source-contract completion; rendered-stack geometry and specific annotation lineage remain unresolved. |
| PHerc1667 / Frag5 | Official catalog corroborates the same parent as Scroll 4 / PHerc1667. | Already exposed parent; cannot replace Frag6 for a new-parent claim. |

For the two scroll candidates, each exact public label prefix has a complete 39-object inventory. Missing mapping-validity and linked provenance records are an observation about those prefixes at the recorded time, not about every possible location. The named transfer code has unpublished changes: its public base converter writes 65-plane labels at plane 32, while the published candidate arrays are 2D. That base code cannot authenticate the current artifacts' annotation depth. Source-canvas offsets are inputs to transfer; do not apply them again to target labels or optimize shifts against predictions.

Frag6's documented native pitches include 3.24 and 7.91 µm. Native voxel pitch does not by itself authenticate its rendered depth interval, center or order. A listing of 65 TIFFs is insufficient to borrow the Paris pooling recipe. Surface-specific metadata was not found in the bounded searches; no image, label, CT chunk, mesh or PPM was acquired. Exact candidate requirements and aliases are retained in [DATA_ROLES.json](DATA_ROLES.json), the [scroll audit](scroll-audit/RESULTS.md) and the [optical audit](optical-audit/OPTICAL_CANDIDATE_AUDIT.md).

The upstream audit also corrects an overly broad earlier uncertainty. Seed42's actual embedded configuration has **no configured pretrained backbone**, and all three candidates are absent from its 29 declared training representations. Original annotation and teacher ancestry remains unresolved. The paper's ResNet3D-50 iteration-0 model explicitly reports PHerc0009B and PHerc0500P2 training. Production canonical ResNet-152 is a different model whose released metadata omits training roots. No inspected artifact proves that either exact model generated seed42's original labels. These facts establish neither clean upstream independence nor confirmed seed42 contamination. The [model-specific evidence table](exposure-audit/AUDIT.md) keeps those channels separate.

The audits made **50 public text/metadata requests**, retaining **3,631,835 response-body bytes**. Failed requests and the one content-type rejection remain recorded. No new scientific payload was acquired or scored. Existing attribution and use restrictions remain applicable; no new terms, redistribution or publication occurred.

## The implemented integration

The original model computes `U-Net(depth_fusion(X))`. The new wrapper computes `U-Net(depth_fusion(X) + adapter(features(X)))`. The original depth stem and entire 2D U-Net stay frozen and in evaluation mode. The U-Net remains in the autograd path so its input gradients can train the adapter.

Both raw-neighbor and explicit-depth branches have **5,296 trainable parameters** and identical initial tensors. The branch uses a 3→8 local 3D convolution, depth mean/max reduction, and a zero-initialized 16→32 projection. The candidate's features are center intensity, axial difference and nominal depth coordinate; the control uses previous/center/next intensity. The released model already has 3D filters and learned depth attention, so no claim is made that this introduces depth awareness for the first time.

This is a matched within-study representation comparison. Against phase 18, both insertion location and parameter count change; a later cross-phase improvement would not isolate either one. The feature bundle also does not separately identify the contribution of coordinates versus differences. [Implementation and exact contracts](fusion-prototype/README.md) explain the integration and limitations.

## Executed checks

The author ran 45 synthetic checks, then four targeted checks after two defensive guards were added. The earlier tested source and receipt were preserved. An independent reviewer ran 42 feature/arithmetic/autograd checks and reviewed the pinned real-backbone smoke before root execution.

The actual-checkpoint smoke used the strict retained seed42 loader and one deterministic Gaussian input of shape 1×1×17×128×128. Each arm received one synthetic SGD update against a fixed spatial MSE target, followed by a second backward pass. This target contains no scan labels and is not the proposed scientific training objective. No resulting adapter weights were saved.

| Actual seed42 integration requirement | Result |
|---|---|
| Identical initial arm states and 5,296 parameters each | Pass |
| Exact original fused features and logits at zero initialization | Pass, both arms |
| Nonzero projection gradient through frozen U-Net | Pass, both arms |
| Initially zero hidden-layer gradient, then nonzero after projection update | Pass, both arms |
| Frozen-model parameters, buffers and evaluation mode preserved | Pass, both arms |
| Original parameters receive no gradients | Pass |
| Fixed call schedule and no retries | Pass |

The controller took **5.10 seconds**, with peak sampled worker RSS **704,802,816 bytes**, within its 300-second / 4 GiB limits. Hook counts were one original full-model call, two original depth-fusion calls, five original 2D-network calls and four adapter calls, with four backwards and two synthetic SGD steps. The full-model count contains its component calls and must not be added to them as extra work. The checkpoint was loaded once. These timings measure this mechanics check, not a full scientific training workload or scans.

[Worker evidence](actual-backbone-smoke/worker-result.json) and [controller evidence](actual-backbone-smoke/controller-result.json) retain exact input/state hashes, observations and resources. The independent technical review is under [fusion-review](fusion-review/); its subsequent receipt review passed 78 consistency checks without rerunning the model. The actual seed42 has no named buffers; the independent toy fixture separately exercised buffer preservation. The prototype author's sealed README correctly says that author did not execute the actual checkpoint; root executed it afterward as documented here.

No new ink accuracy, transcription, translation or model promotion follows from this phase. GPU spend remains **£0**, and the first paid-pilot **£10 total cap** persists. A paid launch still needs a useful frozen experiment, pinned CUDA environment, current pricing, independent shutdown and verified bucket-scoped worker delivery. Local retention and a saved startup draft do not constitute external backup or verified fresh-task restoration.
