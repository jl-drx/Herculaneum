# Native PHerc1667 reference experiment

[The frozen specification](../experiments/pherc1667-reference.json) prepares a bounded **iteration-0 versus iteration-5 visibility comparison on native approximately 2.4 µm PHerc1667 data**. It is a scientific plan, with `ready_for_gpu: false`. No reference weights have been downloaded and no inference, training or transcription has been run for this plan.

The comparison tests whether the released adaptation produces more coherent, source-supported stroke structure under one audited recipe. A separate `l_2` crop checks adaptation/training behavior. It stays outside the reference conclusions. The released family is the paper's ablation, rather than verified final weights from the full three-column reading pipeline.

## What is pinned

The retained official model manifests, cards and configurations identify:

| Model | HF revision | Safetensors bytes | Expected SHA-256 |
| --- | --- | ---: | --- |
| [Iteration 0](https://huggingface.co/scrollprize/PHerc.1667-iteration-0/tree/06c306449b39df42c745608ceade243498d24243) | `06c306449b39df42c745608ceade243498d24243` | 333,531,996 | `62c43b98f55c169ae874c684a59953b5ce1cb45140129107f861501cce3edaef` |
| [Iteration 5](https://huggingface.co/scrollprize/PHerc.1667-iteration-5/tree/6288ba60115ce827018926bcb0ea8e5696f33d3b) | `6288ba60115ce827018926bcb0ea8e5696f33d3b` | 333,531,996 | `9920d4338e51d38b241cbf48392610be0519e4e19c2e9d987ccfc2e78bd3301b` |

These are expected checksums from HF LFS metadata, not a claim that local weights were verified. The JSON pins separate card/config hashes and the shared model-wrapper SHA-256 `2264e9882f4a9b695f10e74d0dcd031ce58fc88f253d14bee8f89d328851ec16`. Configuration-module content still needs review. Execution must use pinned reviewed sources and a strict state match.

Both configurations describe a ResNet3D-50 encoder, per-stage depth maximum and a 2D U-Net decoder. Input is `[1,1,62,256,256]`; output is `[1,1,64,64]` logits. Iteration 0 reports cross-segment training without `l_2` labels; iteration 5 reports `l_2_inklabels5.png`. Exact physical mappings of the initial training aliases remain unverified.

The paper is [arXiv:2606.29085v1](https://arxiv.org/html/2606.29085v1), particularly Supplementary Table 2 and Extended Data Figure 6. Its reproducibility code reference is Villa commit `e583fb67468f483fadd73d52e06f0ab0fe5ba813`. Retained source files and hashes reside outside this checkout under `/workspace/herculaneum-lab/data/methods-review/`; the specification records the public URLs so these local evidence paths are not required infrastructure.

## Resolve inputs and preprocessing first

The exact public source URL, scan/segment identity, native layer quantization, geometry mask and original `l_5`/`l_2` render mapping are **null**. An available PHerc1667/w013 control was pooled to 9.596 µm for a later model; it cannot replace this native reference. A current 109-layer public stack cannot be assumed to match the paper's 65-position sampling without explicit depth correspondence.

Intensity instructions conflict: the paper specifies clipping raw layers to `[0,200]` followed by `Normalize(mean=0,std=1)`; the cards suggest approximately `[0,1]`; their tiling examples feed raw floats; the wrapper header says z-score normalization. The wrapper performs no input normalization itself. The exact divisor, dtype and centering are therefore unresolved. The paper's referenced `data.py` was not identified in the retained code-tree snapshot. Resolve the original transforms before running, without choosing a recipe because it yields brighter predictions.

The paper names layers 1–62. Their source indices and direction also remain unresolved: `1:63` would be appropriate only after confirming zero-indexed source correspondence. The iteration-0 card explicitly reverses `l_2`; it does not establish every `l_5`/checkpoint pairing. The original 9 µm MAD/depth-pooling/Hann adapter is incompatible with these requirements.

The cards document 256-pixel tiles, stride 128, skipping every tile containing a zero fragment-mask pixel, sigmoid before bilinear ×4 and uniform overlap averaging. Pin interpolation semantics and runtime versions before implementation. Preserve quarter-resolution logits, unquantized scores, evaluation/support masks and original coordinates; the cards' 180° rotation and 16× downsampling are presentation conventions.

## Frozen selection and bounded workload

The specification fixes three disjoint 1024-square `l_5` contexts, one in each third of the geometry-valid native-Y extent. Enumerate origins on the 128-pixel native grid. Require at least 95% geometry validity, 36 of 49 fully valid tiles, verified selected-depth source coverage and a physical-support overlap audit. Rank by fully valid tile count, geometry coverage, distance to the stratum center and then source Y/X. Detector maps, released prediction previews and textual identities cannot choose regions. Freeze the resulting coordinates and acquisition hashes before inference.

Choose one separate `l_2` sanity crop from verified label support before model outputs. Its exact label bytes, alignment and deterministic selection criterion need an acquisition record. Record selection failure instead of relaxing criteria after viewing predictions.

Each 1024-square context schedules 49 tiles per checkpoint. Three reference contexts total **294 checkpoint tiles**; the separate sanity control adds **98**, for a maximum **392**. Models and contexts run sequentially, with at most one GPU job active. Source payload downloads are capped at 512 MiB and total acquisition at 1.5 GiB, including the expected 667,063,992 bytes of weights. Source metadata must prove that bounded chunk requests fit those limits; downloading full native layer TIFFs is not an assumed workaround.

## Evidence and leakage limits

Supplementary Table 2 describes `l_5` as **inference only, with no labels**. Consequently this plan measures visibility and source correspondence. It cannot report precision, recall, F1 or held-out accuracy. The allowed separation is exclusion from `l_2` adaptation supervision; broader pretraining independence requires physical alias mapping. Multiple render IDs, disjoint UV boxes and repeated runs do not establish independent physical samples.

Compare only common geometry-valid, observed and evaluated support, at least 128 pixels from crop edges. Exclude pixels influenced by any scored tile with invalid geometry or missing required depth support. Report domain sizes, raw score distributions, paired differences, scored/skipped tiles and runtime. Use identical source/display scales and blinded iteration identities for stroke-continuity and native-CT placement review. Record ambiguity and unsuccessful results without supplying letters from language expectations.

Native reference behavior does not establish transfer to PHerc0800's eligible 8.640 µm scan. Neither enlargement nor interpolation supplies missing native information. Target adaptation requires its own credible supervision and physically separate validation region. This plan makes no prize claim.

## GPU launch gate

The approved **£10 ceiling is aggregate across all pilot GPU jobs**, including known spending and active commitments. It is not a fresh £10 allowance for each reference run. This specification grants no additional spend and cannot launch itself.

A separate [operational job contract](../experiments/gpu-pilot.template.json) must record frozen input/config hashes, an audited adapter and image digest, a bounded provider quote/currency conversion and billed runtime within the remaining aggregate allowance, and independently verified shutdown. Those operational fields remain null or false here. Hardware throughput and peak memory must be measured; the paper's reported H100 run does not establish this pilot's fit, price or timing.

Resolve the explicit blockers, version and hash the completed acquisition/runtime contract, then validate packaging before scheduling a paid job. The existing 9 µm CPU benchmark can check packaging only; it cannot resolve or validate native June preprocessing.
