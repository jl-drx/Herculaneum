# Assessment of the proposed research direction

9 October 2026. This is an assessment and public-metadata audit, not a paired-scan learning result.

I support making acquisition learnability the main next research question, with privileged-observation training as a hypothesis to test after a conventional paired baseline. This is worthwhile regardless of whether phase41 attributes a small development benefit to PHerc1667 supervision. Phase40 already replicated improved retention on a second crop but still failed the material-gain screen. Another crop or preservation-coefficient sweep on the old panel has less expected value than measuring where the signal survives.

The linked **Develop a fresh research agenda** thread was read through the supported thread tool. Its file-change output supplies most of the memo plus the later Paris4 pairing/split amendment; the initial 21,508-character change was truncated at20,000 characters. [AGENDA_THREAD_EXTRACT.json](AGENDA_THREAD_EXTRACT.json) records that limitation. The separate thread reports a successful93-file replay; this turn does not claim to have repeated that replay or transferred the full memo verbatim.

## What is now publicly available

The [current public catalogue](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.min.json) supplies a concrete pair:

| Item | Public evidence | Remaining qualification |
| --- | --- | --- |
| Old Paris4 CT | Volume20230205180739,7.91µm,54keV; [level0 array metadata](https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr/0/.zarray) was retrieved | Local ink signal and reconstruction characteristics have not been inspected here |
| Modern Paris4 CT | Volume20260411134726,2.4µm,78keV; [level0 metadata](https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0/.zarray) was retrieved | This is a different acquisition, not merely a smaller voxel grid |
| Coordinate transforms | Catalogue contains directed3×4 transformations and derivation paths between these volume IDs | Algebra is consistent; physical residuals and local deformation are unmeasured |
| Same named surface | Segment20231016151002 has public tifxyz files and metadata in both old and modern coordinates | Shared segment identity does not establish pixel correspondence or a correct local wrap |

Both native arrays use128³ uncompressed uint8 chunks:2MiB per chunk. This is compatible with bounded streaming. Nineteen metadata GETs delivered104,487 bytes, plus one HEAD; no CT, label or coordinate-image payload was fetched. The first catalogue parse encountered gzip bytes, retained them, then decoded locally without another download. Five literature/model/documentation GETs delivered550,031 bytes separately.

The [published schema](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.schema.json) specifies XYZ full-resolution voxel coordinates for transforms and meshes, versus ZYX array order. The published transformation includes a reflection; implementations must retain it, not substitute a positive scale-only map. Forward/inverse composition residuals are below2e-12, and mesh bounding boxes lie within the stated volumes. These checks establish metadata consistency only. They provide no held-out landmark error. [PAIR_FEASIBILITY.json](research/PAIR_FEASIBILITY.json) records exact IDs, matrices and limitations.

This improves on the memo's availability status: a machine-readable registration lead exists. We should test it rather than first requesting a transform from maintainers. The remaining question is whether it is accurate enough locally for the intended supervision.

## Where I agree, and where I would qualify the proposal

The [2026 unwrapping paper](https://arxiv.org/html/2606.29085v1), Figure3 and the Paris4 methods, explicitly compare2.4µm and7.91µm observations and describe directly visible deposits. But the dense volumetric labels primarily combine DINO embeddings, model-based support and subsequent self-distillation. They are useful teacher supervision; they are not automatically independent test truth. A reviewed subset of visible deposits and confidently non-ink substrate is needed. Low detector score alone cannot certify a negative, and uncertain material stays unknown.

The [topography study](https://www.nature.com/articles/s41598-026-58467-1), Methods and Figure3, separately trains matched-resolution models and tests a native model on degraded inputs. Its conclusions justify that distinction here. They do not supply a universal CT resolution threshold: this was optical profilometry on14 samples, with heterogeneous cross-papyrus transfer and synthetic averaging that does not reproduce real CT contrast. Even successful fine learning followed by failed coarse learning is evidence about a tested learner and degradation, not proof that all useful information was destroyed.

The appropriate next comparison is therefore native, synthetic coarse and real coarse observations on the same physical support, with matched-quality training as well as cross-quality testing. Synthetic degradation is necessary but insufficient. Energy, phase retrieval, blur, noise and specimen movement can all differ. Preserve physical field of view and depth coverage; do not mistake an upsampled coarse tensor for native information. The ordinary learner should have controlled capacity and an adequate training-fit check before a negative scientific interpretation.

Paris4 is not an upstream-unseen parent for seed42: the retained release audit lists Paris4 training representations. Freshly chosen regions cannot erase that exposure. Use a cleanly initialized conventional learner for the primary within-parent experiment, or limit a pretrained comparison explicitly to prospective adaptation agreement. Treat the canonical2µm ResNet152 as a separately qualified reference, not the paper's ResNet3D50 or an interchangeable9µm initializer. Its [model card](https://huggingface.co/scrollprize/ink_canonical_2um) confirms that distinction.

[Generalized distillation](https://arxiv.org/abs/1511.03643) fits the training-only-observation proposal. It is an established framework, so the potential contribution is reliable physical pairing, uncertainty handling and demonstrated coarse-input gains. Compare equal-capacity coarse students with ordinary supervision, a coarse-observation teacher and a fine-observation teacher; keep reviewed labels and update exposure fixed. Include a spatially mispaired teacher control. Teachers cannot turn information absent from the deployed input into uniquely recoverable ink, and model-generated test truth would make the claim circular.

Geometry remains conditional. The [official tutorial](https://scrollprize.org/tutorial5) already provides native3D sampling with a target-wrap channel. A new method should beat both a suitable existing representation and a capacity-matched geometry-free model, with uniform averaging as a further control. Only measured placement uncertainty should motivate it; the old texture-depth proxy remains insufficient.

Finally, phase41 and the memo's blank-slot control answer different questions. Phase41 keeps new-patch KL and removes only new-patch BCE, conditional on annotation-selected support. The memo's B removes both. Phase41 closes the approved supervised-gradient question; it cannot prove that the entire coverage benefit is independent of reduced PHerc0139 exposure. An additional blank-slot run is optional later and should not delay the acquisition comparison.

## Decision

Prioritize a bounded, physically split Paris4 learnability/degradation comparison. Qualify the existing transform using non-ink structure and held-out landmarks, then run one ordinary learner at each admissible observation quality. Freeze training, selection and locked comparison blocks before candidate predictions, including full context and depth guard bands. Keep the existing eight regions as an explicitly reused regression panel. PHerc0009B,PHerc0500P2 and PHerc0841 remain unaccessed as scientific payloads in this turn.

If real pairing fails its local registration criterion, report a native-versus-synthetic result with that limitation and stop the real-pair arm. Do not spend successive experiments asking the same provenance question. If even disjoint native learning is weak despite successful training fit, repair the reference or choose a better documented material pair before distillation. If native and matched synthetic work but real coarse fails, focus on acquisition mismatch. If coarse matched learning works, a privileged teacher becomes a justified next test.

No paired model was trained or promoted here. GPU work remains deferred. Dataset-specific licensing and attribution remain separate for the modern and legacy acquisitions.
