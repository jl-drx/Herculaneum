# PHerc0139: soft-target controls and a usable native learning cohort

Completed 11 October 2026. This supporting research follows the four-support
supervision study at commit cfb5775. It is not a new numbered training experiment.
No neural inference or training ran, and no paid resource was used.

**The useful next learning step is now a full-depth native-CT baseline on three
folds with training/evaluation context separation.** Its input chunks, current native-frame
annotations, conservative context exclusions and portable replay are prepared.
Fine-to-native teacher alignment does not block this annotation-agreement test.
The four nearby teacher supports were not promoted into a learning benchmark.

## Soft-target control: complete, no predictive benefit claimed

The [64-case comparison](soft-controls/README.md) holds target scale and support
fixed while comparing ordinary soft targets, local CT-candidate averaging, a
global smoothing kernel and sampling at the local mean coordinate. Local
averaging differs from the centroid control by only0.051–0.271 source units in
the three informative primary supports; it stays below0.329 across both
operators/all eight planes. Local versus global averaging differs by roughly
0.8–1.25 source units in the primary cases, with differing kernel means/spreads.

This reduces the priority of an uncertainty architecture on this evidence.
It does not establish that soft targets improve a model, that centroid placement
is physically correct, or that the differences cannot affect learning. Ordinary
continuous teacher scores remain the first paired-supervision baseline when
broader transferred training support is qualified. Generic paired distillation
and BCE against averaged targets are not new architectural contributions.

All64 cases and8,000 scalar summaries passed an independent source replay.
Support7's32 out-of-crop pixels are excluded equally from every arm. The negative
support and a harmless nonprimary floating-point threshold discrepancy remain
in the evidence. All previous strict correspondence failures remain unchanged.

## A broader native-frame cohort, acquired without selecting ink responses

The [reference source audit](reference-admission/REPORT.md) establishes that the
current w035/w039/w044 annotation arrays are declared in the native113keV /
9.362µm render frame. Their historical transfer/cleanup and annotation ancestry
remain imperfectly documented. Use agreement with released annotations; do not
call them authenticated ink truth or a pristine test for pretrained models.

The [native intake](native-reference/README.md) read all three supervision masks
first and froze a128-pixel grid with complete256-square contexts, requiring at
least1024 supervised core pixels. Only afterward were labels read. Every retained
core remains, including67 without supervised ink and one without supervised
nonink. The mask/code/source identities and guarded execution order are recorded;
these are local execution records, not independent timestamp attestations.

| Winding | Cores | Supervised core pixels | Supervised ink pixels |
|---|---:|---:|---:|
|w035|109|1,066,610|333,871|
|w039|54|571,603|182,380|
|w044|102|1,187,621|450,775|
|Total|265|2,825,834|967,026|

The six compressed mask/label objects total65,648bytes. Independent decoding and
coverage calculations reproduce every core ID and class count. Stored ink also
exists outside supervision (62,129/2,519/4,172 pixels by winding); those pixels
are excluded, not counted as adjudicated positives or negatives.

The exact536 native CT chunks are downloaded and hash-verified:245,891,072bytes
(234.5MiB), with zero failed requests or retries. All28 planes are retained.
Every selected context pixel has at least one nonzero native CT value, so all
2,825,834 supervised pixels retain array support. That test does not identify
an ink plane, verify label alignment or certify every depth sample as material.

## Full-depth physical-context separation

The [geometry audit](native-reference/geometry/README.md) uses all256×256 context
pixels and all28 depth positions, with conservative source-voxel bounds and both
retained arithmetic conventions. It found46 possible context-overlap pairs,
all between w035 and w039. Box intersection is not proof of shared actual voxels;
excluding such training contexts provides a conservative split under the pinned
renderer/mesh contract. No context or depth window was shrunk to pass the check.

| Held-out winding | Evaluation cores retained | Training contexts retained | Training supervised pixels |
|---|---:|---:|---:|
|w035|109/109|139/156|1,578,067|
|w039|54/54|196/211|2,112,553|
|w044|102/102|163/163|1,638,213|

All265 contexts pass the declared geometry screen. Independent reconstruction
verified every context and all22,512 cross-winding box comparisons. These folds are portions of
one scroll, not independent specimens or proof of no broader spatial dependence.
Physical aliases must remain grouped; released models already saw these native
windings. The proposed learner therefore starts from scratch.

## Next experiment and limits

[NEXT_STUDY.md](NEXT_STUDY.md) defines the next scientific question: train a
modest complete native-input learner to a measured optimization endpoint and
compare it with constant/simple native-input baselines on the excluded winding.
Use identical masks, full depth and the frozen context exclusions. Report
training fit and each fold, with annotation-agreement AP/AUC rather than ink
accuracy. The exact model, stopping rule, paired initializations and advancement
margin still need a frozen training protocol before model predictions are seen.
No run is currently processing from this packet.

A consistent gain after adequate fit supports the next matched paired-supervision
study. Failure to fit would leave representation/optimization unresolved;
training fit without transfer would instead point toward coverage or reference
mismatch. Neither result could prove ink absence. Do not revive the rejected
depth/fusion adapters or repeatedly inspected Paris tests.

All data/code remain private on the existing research branch, with separate
analysis and native-CT LFS archives. No Actions workflow, new PR, Hecate call,
spare-VM assignment, Runpod resource or R2 credential operation was needed.
The formal cumulative seal still ends at study50; studies51/52 and these
supporting packets retain their separate receipts. See PORTABILITY.md.
