# Plinius: explicit-depth residual pilot

**The candidate did not pass the predeclared feature screen.** Explicit depth features produced a smaller, more conservative correction than the matched raw-slice control, but the gains were too small and Paris2 ranking remained slightly below the unchanged seed42 baseline. Keep seed42 as the reference. Preserve both final adapters as research results; this pilot does not justify GPU scaling of this output-residual design.

The complete experiment and independent numerical checks ran on CPU. Training took 15.34 seconds and final development inference 14.37 seconds of controller wall time. GPU spend remains £0 and the £10 TOTAL first-pilot cap is unchanged. No new data acquisition, new backbone execution, threshold fitting or external publication occurred.

![Training behavior and all six regional comparisons](figures/depth-residual-pilot.png)

The experiment isolates a feature bundle, not a new pretrained backbone. The credited scrollprize/ink_9um seed42 model already uses 3D convolutions and learned depth fusion. Its previously verified, hash-bound logits were reused unchanged. Both new branches have 801 trainable parameters: a 3D convolution, LeakyReLU, mean/max depth reduction and a 2D output projection. The projection starts at exact zero, preserving the initial baseline predictions.

The raw control sees previous/center/next normalized slices, with clamped boundary indices. The candidate sees the center slice, an axial difference and the fixed coordinate (j−8)/8. Interior differences are centered; endpoints use a full one-sided difference. Coordinates are nominal stack indices, not authenticated anatomical depth or micrometers. The same architecture, initialization, manual PHerc0139 data, 32-batch schedule, optimizer and 32-update budget were used for both arms. Features were formed after the unchanged XY augmentation. No shift, reversal, new depth window, crop or learning-rate sweep was tried.

The source/data audit found no ready retained fresh confirmatory set. All six Paris1/Paris2 regions below are reused DEVELOPMENT evidence. Paris2 remains excluded from gradients, teacher targets and cutoff fitting. Two PHerc0139 regions supplied the training objective; historical PHerc0139 pseudo arrays were read only to reproduce the old combined batch fingerprints and never contributed targets or gradients. Full upstream model exposure remains incompletely known. Candidate fresh parents remain metadata-only, with unresolved provenance requirements.

| Fixed 32-update training result | Raw control | Explicit depth |
|---|---:|---:|
| Initial base/full objective | 0.597077174 | 0.597077174 |
| Final smoothed manual BCE | 0.596114274 | 0.596571647 |
| Final preservation KL | 0.000636029 | 0.000092387 |
| Final full objective | 0.596750303 | 0.596664034 |
| Full-objective improvement | 0.000326871 | 0.000413141 |
| Mean absolute probability drift on manual support | 1.2201 percentage points | 0.4258 percentage points |

The full objective is smoothed manual BCE plus unit-weight KL from the unchanged initializer. Each reported risk averages the same 32 scheduled batch losses, preserving repeated and overlapping exposures. It is empirical training risk, not an independent population estimate. The candidate's better full objective comes from a smaller preservation penalty: its manual BCE improvement is smaller than the raw control's. This supports a preservation interpretation and does not establish a better learned ink signal.

The complete trajectories are retained at updates 1, 8, 16 and 32. The raw control's full fixed risk initially rises before falling; the candidate also has a nonmonotone trajectory. Only each final 32-update adapter was saved for development inference. No favorable intermediate checkpoint or early-stop outcome was selected.

Both final adapters were evaluated on every fixed context after independent training verification, even though the candidate's training advantage was already below the required margin. The following are equal-context-within-fragment and then equal-fragment Paris2 metrics. AP means average precision.

| Paris2 DEVELOPMENT mean | Unchanged seed42 | Raw control | Explicit depth |
|---|---:|---:|---:|
| AUROC | 0.657642 | 0.655292 | 0.657107 |
| AP | 0.384286 | 0.382845 | 0.384051 |
| Sensitivity at unchanged cutoff | 2.5981% | 2.2372% | 2.5017% |
| Specificity at unchanged cutoff | 98.5120% | 98.8792% | 98.5477% |

The common cutoff is the literal original seed42 value 0.7558947205543518. These operating metrics describe drift at that existing cutoff; they are not a new calibration or a fresh transfer claim. The candidate loses about 0.000535 AUROC and 0.000234 AP against the unchanged baseline on Paris2. The raw control loses more. Relative superiority over the raw branch therefore does not amount to beating the original model.

| Region | Seed42 AUROC / AP | Raw control AUROC / AP | Explicit depth AUROC / AP |
|---|---:|---:|---:|
| Frag1 context 1 | 0.446052 / 0.185973 | 0.446573 / 0.186240 | 0.446595 / 0.186152 |
| Frag1 context 2 | 0.903678 / 0.500096 | 0.901764 / 0.496990 | 0.903572 / 0.499924 |
| Frag2 context 1 | 0.501298 / 0.269839 | 0.502613 / 0.270379 | 0.501358 / 0.269847 |
| Frag2 context 2 | 0.779541 / 0.581235 | 0.770217 / 0.577771 | 0.776904 / 0.580282 |
| Frag3 context 1 | 0.870064 / 0.298078 | 0.870532 / 0.299285 | 0.870583 / 0.299106 |
| Frag3 context 2 | 0.699512 / 0.182887 | 0.698991 / 0.181745 | 0.700384 / 0.183658 |

Frag1/2 are two fragments from one Paris2 parent; Frag3 is Paris1. The two weak CT regions remain weak. The candidate slightly improves both Paris1 contexts versus seed42, but these changes are small and both contexts belong to an already inspected parent. No independence, confidence interval or significance claim follows from the large pixel counts.

All 17 practical screen components were reported. The candidate passed its own base/full improvement requirements and all 12 per-context AUROC/AP retention bounds. It failed the three comparative requirements below. Independent reconstruction agreed with every decision and found no numerical boundary ambiguity.

| Failed predeclared component | Observed improvement | Required minimum |
|---|---:|---:|
| Full training risk versus raw control | 0.000086270 | 0.000100000 |
| Paris2-development AUROC versus raw control | 0.001815731 | 0.005000000 |
| Paris2-development AP versus raw control | 0.001206541 | 0.005000000 |

These are pragmatic development margins, not biological or statistical significance thresholds. Missing them is evidence against advancing this specific small-head recipe under this budget. It does not rule out depth-aware learning with other architectures, broader training data or different budgets. Those would be separate experiments, not post-hoc repairs to this result.

Independent review reconstructed the exact historical training inputs and new feature tensors; zero-initialization identity; all loss and manual-logit-gradient calculations; clipping and 801-parameter AdamW traces; current-batch/fixed-risk consistency; final tile composition before sigmoid; score blending and coverage; all 18 context/model rank and cutoff summaries; and the 17-component decision. The saved gradients originate from Torch autograd: their logit-level derivative and optimizer arithmetic were independently checked, while convolutional autograd and execution history were source-reviewed rather than replayed. Cached backbone outputs establish no new backbone execution.

The first independent training checker stopped on an output-schema expectation: its expected initialization-proof dictionary omitted one additional producer proof field. The original checker and STOP receipt remain retained; version 2 adds the explicit field and checks cached-plus-zero identity without changing scientific outputs or tolerances. No training or development run failed or was repeated. Synthetic preflight revisions are also retained.

Execution totals are 384 training adapter forwards, 64 backwards/updates, and 300 final development adapter forwards. New backbone loads/forwards/updates and threshold fits are all zero. Training peak RSS was 744,738,816 bytes, below the 3 GiB cap. Each final adapter checkpoint is 5,721 bytes; the existing backbone is referenced, not duplicated. Source, protocol, configuration and inputs are pinned. The unchanged application checkout remains at `49fabf3e890d5b880f391b27d581f1e7a1ebdbcc`, with its pre-existing untracked `.codex/` preserved.

The next useful research move is to finish a defensible fresh-parent data/exposure contract and then test depth features inside multiscale feature fusion, with an otherwise identical raw-input control. This output-only branch had little capacity to alter the established representation. The current result supplies a clean negative control and a reproducible low-cost training/inference path; it does not supply a reason to rent a larger GPU yet. [NEXT_STUDY_DRAFT.md](NEXT_STUDY_DRAFT.md) records the proposed direction and the remaining paid-pilot prerequisites.

Machine-readable [metrics](analysis/metrics.json), [training results](implementation/run/training-results.json), [development results](implementation/run/development-results.json), and [independent publication check](independent-review/publication-review/comparison-v1/comparison.json) retain the complete evidence. EduceLab/source attribution and existing use restrictions continue to apply. No model promotion, authentic reading, transcription, translation, upload or prize submission occurred. Local retention and a saved startup draft are not external backup or proof of fresh-task restoration.
