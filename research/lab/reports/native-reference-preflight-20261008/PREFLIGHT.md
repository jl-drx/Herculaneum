# Native PHerc1667 reference: source preflight

Reviewed 8 October 2026 against the retained `pherc1667-native-reference-v1` plan. **The two pinned configuration sources are verified and reviewed. The native-reference job still cannot run as a faithful ablation comparison:** original source identity/depth mapping, exact preprocessing lineage, labels, executable runtime and operational contract remain incomplete. No weights, scans or annotations were downloaded and no model inference was performed.

## Configuration and model source checks completed

Both iterations' `configuration_inkdetection.py` files are identical: **1,742 bytes**, Git blob **`24ac6d12668bfb5cf4b448c8f61723e97264ee25`**, SHA-256 **`e9d8efd18c1ee7d9cda59eda0aa1c2b6e9eab31f3e715f3e351fa0fd0f0c0990`**. The Git blob calculation includes the canonical `blob <length>\0` prefix and matches each pinned Hugging Face manifest, rather than merely matching the two downloaded files to each other. Their configurations and model sources also match retained SHA-256 pins and HF blob identities. All four Python source files parse successfully; they were inspected without importing a floating remote revision.

The common architecture is ResNet3D-50, one image channel, four feature widths `[256,512,1024,2048]`, stagewise depth maximum and a 2D decoder. Configuration metadata declares **62 × 256 × 256** input and `decoder_upscale=1`. The model forwards the supplied tensor directly: it contains **no clipping, intensity division, image-statistics centering or z-score transform**. Its header's “z-score normalised” phrase is documentation, not executable preprocessing. The forward method checks dimensionality but does not enforce the declared depth, spatial size or normalized intensity range; a future adapter must enforce these.

The model returns raw **64 × 64 logits** for the declared input. Apply sigmoid once. The card then upsamples probabilities fourfold using bilinear interpolation and uniformly averages accepted overlapping tiles. The card's mask gate rejects any tile with any zero fragment-mask pixel, and leaves uncovered output zero. Its full-segment example transfers raw floating values directly to the model, contradicting the card's separate normalized-input guidance; copying that example alone does not reproduce the paper's preprocessing.

## Normalization: evidence now distinguishes the pipelines

| Retained executable source | Defined behavior | What it establishes |
| --- | --- | --- |
| Paper-pinned `train_resnet3d_lib/data_ops.py` and `config.py` | TIFF grayscale reading, clipping to `[0,200]`, then `A.Normalize(mean=0,std=1)` and tensor conversion. Zarr path explicitly converts uint16 using `value >> 8` to uint8 before clipping. | A concrete training-source candidate consistent with the paper's reported transform; quantization and exact ablation linkage still require verification. |
| Reference Albumentations **1.3.1**, pinned commit `e3b47b3a127f92541cfeb16abbb44a6f8bf79cc8` | `Normalize` defaults to `max_pixel_value=255`; formula `(image - mean×255)/(std×255)`, producing float32. | With mean 0 and std 1, the above candidate becomes **`clip_uint8_0_200.astype(float32) / 255`**, rather than per-image z-score normalization. This library source is a semantic reference, not proof of the original installed version. |
| Paper-pinned `optimized_inference/inference.py` | Clip to 200, then explicit `A.ToFloat(max_value=200)`. It also defaults to Hann blending, optional looser tile-mask filtering and autocast. | A distinct **`/200`** recipe with different scheduling/blending. Its defaults cannot silently replace the ablation recipe. |
| Paper-pinned `infer_resnet3d_vesuvius.py` | Uses the clipping Zarr wrapper then explicit `/255`, but MONAI Gaussian-window inference and a sigmoid after logits stitching. | Additional support for `/255` in an official source path, with a materially different output-combination procedure. |

For a clipped value of 200, these candidates give approximately **0.784314** (`/255`) versus **1.0** (`/200`). [The arithmetic probe](runtime-probe.json) records this without running a detector or selecting the path with the most attractive output. Source evidence favors `/255` for the paper-reported `Normalize(mean=0,std=1)` path. It does **not** prove the exact ablation runtime, source-layer quantization or per-run invocation, so the full normalization blocker remains partially open.

The paper-pinned `ink-detection/metadata.json` is a different example: batch 24, 30 epochs and source aliases including `l2_0_crop` under **PHerc139**. It contains neither the PHerc1667 `l_2` nor `l_5` entries needed here. It must not be substituted for the original ablation's configuration merely because its filename appears generic.

## Depth and interpolation findings

The current source uses half-open layer ranges and center-crops ranges longer than 62. Thus `[1,63]` selects names **1…62**; `[1,108]` selects **23…84**. It then applies the recorded per-segment reversal. That establishes index semantics but does not map the paper's names onto a current 109-plane public render. Iteration 0's card explicitly documents `reverse_layers=True` for `l_2`; neither that sentence nor the shared architecture proves the `l_5` direction or every iteration-5 pairing.

On installed **torch 2.12.0+cpu**, omitting `align_corners` for bilinear interpolation is exactly equivalent to `align_corners=False` in the recorded synthetic probe, and the inspected installed implementation sets that default explicitly. The comparison with `True` differs, so the choice matters. This closes the local default check and supports explicitly pinning `False` in the eventual adapter; it does not pin the original June runtime or a future GPU image.

The existing CPU environment has torch 2.12.0+cpu, torchvision 0.27.0+cpu and safetensors 0.8.0. **Transformers and Albumentations are not installed.** No existing environment, lock file or tracked plan was changed. Reviewed source is not a strict weight-load check: the two expected safetensors payloads remain unacquired and unverified.

## Remaining plan gates

| Plan blocker | Result of this audit |
| --- | --- |
| `native-source-identity` | Open: exact original `l_2`/`l_5` assets, geometry and render correspondence remain unresolved. |
| `source-depth-convention` | Partially resolved: current index algorithm and the explicit iteration-0/`l_2` reversal established; source/checkpoint pair mapping remains open. |
| `normalization` | Partially resolved: executable `/255` versus `/200` paths distinguished; original run linkage, quantization and runtime remain open. |
| `configuration-source-and-runtime` | Configuration/model source identity and static review complete; local interpolation default checked. Native runtime, reviewed strict-load adapter, weights, lock and image digest remain open. |
| `frozen-acquisition` | Open: no geometry/coverage-based crop coordinates, physical-overlap audit, payload/mask hashes or bounded source-object plan acquired. |
| `sanity-label-lineage` | Open: exact `l_2_inklabels5.png` bytes/version/alignment remain unresolved. |
| `operational-budget-and-shutdown` | Outside this source audit and still open; no account, GPU, provider or billing actions performed. |

The next useful gate is original native input/render and per-run metadata correspondence, followed by a versioned adapter contract using an explicit source-supported transform. Do not replace missing lineage by choosing whichever normalization, direction or source crop looks most like writing.

## Retained evidence

[status.json](status.json) records the blocker assessment and source hashes. The immutable [initial source manifest](../../data/native-reference-preflight-20261008/source-manifest.json) and [supplemental manifest](../../data/native-reference-preflight-20261008/supplemental-source-manifest.json) retain **31** verified local/public source records. Successful source downloads total **516,459 bytes**, within the **20 MiB** source-only cap. The manifests include the unchanged plan snapshot, pinned model/configuration sources, exact paper-code files and explicitly qualified reference-library sources. [runtime-probe.json](runtime-probe.json) and [runtime-source-record.json](runtime-source-record.json) retain the local CPU check. Original lab evidence and tracked repository files were preserved.
