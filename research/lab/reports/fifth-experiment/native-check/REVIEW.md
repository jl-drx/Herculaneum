This package is frozen for review before native downloads. Only the `plan` command and a small synthetic interpolation check have run. No native CT, model arrays, network requests, or old-file edits were performed.

The two supplied detector-selected peaks are source XY (1278,504) and (1100,628), on PHerc0800 segment 20251028220955, target volume 20250521135224. Each peak has its four fixed cardinal neighbors at ±1 source pixel. All 31 archived layers are checked at offsets −15…15 native voxels. The acquired fourth-experiment crop is source XYWH [1024,384,384,384]; its source canvas is 2100×2100.

The plan retains each point's archived base XYZ, both signs of its unit normal, source mesh U/V tangents, and a separately labeled orthogonal visualization frame. Every one of the 310 sample positions retains all eight actual trilinear corner coordinates. The complete support requires only native chunk ZYX [96,50,26] and [97,51,27]: 4,194,304 encoded/decoded bytes. Neither appears in the existing verified same-volume cache. The cap remains eight distinct chunks / 16 MiB plus 256 KiB metadata; figures cannot add chunks. The two groups use nearest normal grids [64,25] and [55,31], respectively, so adjacent profiles are dependent.

Renderer commit 530584554c031c982ba3aa243e1ea39c4bef9818 is pinned, including retained C++ source hashes. The helper preserves the original archived half-pixel tile/halo float32 base arithmetic, nearest-grid normal, recorded normal flip and renormalization, centered depth order, float32/FMA sampling, and clipped/truncated uint8 conversion. No normal, offset, or transform is fitted. The maximum archived-base versus retained-map difference is 0.0009765625 native voxel, below the predeclared <0.01-voxel norm gate.

Before reading CT, `run` verifies the supplied frozen-plan SHA and all frozen file hashes, including its own code. It makes one bounded metadata request and demands the frozen metadata byte SHA and schema. Each native object gets one attempt with no redirects, no automatic retry, and no missing-fill substitution. Raw errors are retained and S3 NoSuchKey is distinguished from generic HTTP failures. Any metadata/convention mismatch, missing observed support, budget violation, or intensity difference >1 stops expansion pending diagnosis. The result reports exact and ≤1 intensity counts separately. A passing check establishes local source-render correspondence; it cannot establish intended papyrus wrap placement, ink, letters, or physical independence.

The offline synthetic check passed: an affine field sampled across all eight neighboring chunks produced the expected float32 value 4.0; removing one corner's chunk was rejected. All 310 frozen samples have eight unique corners and the exact two-chunk union above. Importing the new helper performs no work; the original top-level helpers were not imported.

Frozen plan SHA256: `50c8fa7cea430111b61a4c66dfd87965dee8bd4e53d06323151a83c592ce9a67`.

Code SHA256: `4d74a492e05c38f7a5074b9b8b6b6d35d5f69d041bb539310386971f6b4357b5`.

Executed offline command:

```sh
/workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/fifth-experiment/native-check/check_native_ct.py plan
```

After parent review/authorization, the bounded run command is:

```sh
/workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/fifth-experiment/native-check/check_native_ct.py run --plan-sha256 50c8fa7cea430111b61a4c66dfd87965dee8bd4e53d06323151a83c592ce9a67
```

The raw cubes will be retained under `data/experiments/pherc0800-fifth-native-check/`, with exact paths already frozen in the plan. `result.json` and `manifest.json` will appear beside this note; the manifest hashes inputs and outputs without hashing itself. Local native cross-sections may use only those retained cubes; any pixel missing even one interpolation corner is unknown. Letter, transcription, and translation remain null.
