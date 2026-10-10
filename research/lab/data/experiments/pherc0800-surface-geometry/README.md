# PHerc0800 surface geometry pilot

Small source-data experiment for eligible volume `20250521135224`, segment `20251029010146`. No ink model, transcription, or translation has been run.

The exact fetched sources, HTTP headers/ETags, byte counts and SHA256 digests are recorded in `manifest.json`. Upstream data license: CC BY-NC 4.0. This is a mutable public data service; our local hashes identify this retrieved snapshot.

`surface.level3.chunk` contains the entire coarse overview, uint8 shape 31×123×123. Its live array metadata declares that same chunk shape and its HTTP payload has exactly 468,999 bytes: no 128×128 padding was present or clipped. `overview.npy` decodes those bytes. `overview.png` displays depth index15 and depth maximum with fixed grayscale0–255 and a labeled future ROI; the maximum is a processing choice, not an ink prediction.

`mesh.x.tif`, `mesh.y.tif`, and `mesh.z.tif` are 49×49 float32 coordinates in the eligible native CT volume. The valid points exactly reproduce the source bbox; 1,337/2,401 vertices have finite non-sentinel coordinates. Source `mesh.meta.json` scale is [0.05,0.05] in X/Y order and nominal canvas is980×980.

## Mapping the forthcoming native ROI

For rows/columns256:512, native render pixel centers map to source mesh grid `(y+0.5)*scale_y`, `(x+0.5)*scale_x`. This derives from the maintained C++ renderer's centered canvas origin and `QuadSurface` center/generation transform; cited URLs and pinned commit are in `geometry-validation.json`. It uses multiplication, not division by scale. The source September2026 export's exact code commit/offsets are not included in its metadata, so this convention must still be checked against source CT before declaring correspondence pixel-perfect.

Bilinear interpolation requires all four finite/non-sentinel vertices. The entire256×256 ROI satisfies that criterion. `native-roi-ct-coordinates.xyz.npy` stores this pixel-to-CT mapping; `native-roi-valid-mask.npy` preserves the validity mask. The geometric ROI center is at y/x383.5 (halfway between center pixels); the actual integer sample384 is recorded separately.

The computed center CT XYZ is `[3271.30297,6228.44184,12163.52930]`; base-surface bbox is `[[3180.01157,6099.63103,12037.33227],[3357.48169,6366.66763,12285.90972]]`. A conservative16voxel margin gives `[[3164,6083,12021],[3374,6383,12302]]`; it covers a centered±15voxel depth window plus rounding, but archived depth orientation/zero offset has not yet been confirmed. Native planar ROI nominally spans2.21184mm per side; flattening distortion has not been quantified.

This pilot verifies real surface-data I/O, coordinate integrity and an auditable bounded region. It makes no claim that any pixel contains ink or that any letter can be read.
