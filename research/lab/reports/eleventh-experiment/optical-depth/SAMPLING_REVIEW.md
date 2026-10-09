# Legacy Frag4 depth convention

The primary dataset paper supplies the declared Frag4 rendering convention: **65 planes, zero offset at layer 32, and nominal3.24 µm steps along the supplied PPM normal**. Algorithm 3 samples `p+(k−32)n` using trilinear interpolation, with layers increasing from negative to positive normal offset. This conclusion comes from the published protocol, not inference from the65-plane count or current renderer defaults. The exact historical executable/arguments and byte-level native rendering have not been reproduced.

The immutable input metadata is [legacy-surface-meta.json](../optical-reference/legacy-surface-meta.json:1). The [region descriptor](../optical-reference/legacy-surface-render.json:4) links the surface volume, a flattened surface-volume PPM, mask and ink labels, and sets `invert_normals:false` at line 7. This flag describes use of that PPM; it does not establish the orientation of the original native-CT sampling normals. The [package configuration](../optical-reference/legacy-volpkg-config.json:1) declares material thickness150 µm and version7. None is an invocation record.

## Dataset-declared protocol

The retained complete [arXiv 2304.02084v4 paper](../optical-reference/annotation-provenance/educelab-arxiv-v4-complete.html:247), dated 20 May 2024, covers P.Herc.Paris. 1 fr. 39 at 54 keV in its dataset table (lines 1024–1030) and identifies its 54 keV surface in Appendix A's dataset table (line 2345). Appendix A.1 states all fragment scans have reconstructed voxel size approximately3.24 µm at line 2260. Appendix A.5 explicitly specifies an interpolated **normalized** PPM normal at line 2754.

Appendix A.6, paragraph 7, states: “For the surface volumes in this work, the sampling radius was set to `r=32` (creating a surface volume of65 channels) and the interval `δ=1` was chosen to preserve the original image resolution” (line 2917). Algorithm 3 loops `z_k∈[−r,r]` (line 2883), samples at `p_hat=p+z_kδn` (line 2890), and stores the value in `S[z_k+r,y,x]` (line 2904); paragraph 6 specifies trilinear interpolation at line 2812. Thus for zero-based layer`k`, the declared signed normal distance is `(k−32)*3.24 µm`, spanning approximately−103.68to+103.68 µm. This normal direction is local to the PPM; it is not a globally positive native-CT Z direction or an authenticated papyrus side.

The paper names `vc_layers_from_ppm` in Algorithm 3's caption (line 2820). Its protocol supplies the intended settings for this dataset, while the legacy metadata's shorter `vc_layers` name is not an exact executable/version pin. No historical command line is needed to describe this **published** convention; reproducing the actual artifact's native rendering would require additional original-to-output mapping evidence. [paper-depth-extract.json](paper-depth-extract.json) preserves the relevant text/formula excerpts and source hash.

## Verified official code convention

Retained official Vesuvius community documentation links the EduceLab Volume Cartographer repository. This lookup pinned its current `develop` revision to **7a3ebcd20cc9844dd055e8c59c505b7edcd33795**. These sources establish a code convention, not the implementation or settings used for the2023 publication. In this revision [apps-CMakeLists.txt](apps-CMakeLists.txt:123) names the executable `vc_layers_from_ppm`, whereas the retained publication uses `vc_layers`.

[LayerFromPPM.cpp](LayerFromPPM.cpp:70) defaults the interval to1.0 and direction to0/bidirectional. An omitted radius is computed as `materialThickness /2 /nativeVolumeVoxelSize` at line 199. Parsed radius, interval and direction are passed to the line generator at lines 233–236. Optional transformations can first change the input PPM at lines 209–229.

[LineGenerator.cpp](LineGenerator.cpp:14) constructs offsets in increasing order. For radius `r=abs(radius)`, interval `i` and bidirectional direction, the range begins at`−r` and contains `floor(2r/i)+1` offsets, with `offset[k]=−r+k*i` (lines 18–49). Positive-only direction begins at0; negative-only direction begins at`−r` and ends at or below0. [NeighborhoodGenerator.hpp](NeighborhoodGenerator.hpp:25) defines direction values−1,0,+1. Sampling is `nativePosition + originalPPMNormal *offset` at [LineGenerator.cpp:78](LineGenerator.cpp:78). [LayerTexture.cpp](LayerTexture.cpp:95) reads that normal directly from PPM components3–5 and assigns the resulting neighborhood to layers in its original order at lines 100–103. The eager path uses the same formula at lines 63–70. Neither path flips or renormalizes those input normals. Thus ascending layers move along the supplied normal, whose native-CT sign and unit length remain unverified here.

With radius 32, interval 1 and bidirectional sampling, these sources give65 offsets`−32,…,+32`, matching the paper-declared unit-normal convention. This is a code consistency check against the published protocol, not a recovered historical invocation.

## Midpoint PPM is a separate operation

[LayerFromPPM.cpp](LayerFromPPM.cpp:286) optionally generates a new PPM for the output layer volume. Its assigned Z coordinate is `(texture.size−1)/2` at line 298, and each mapping is`(x,y,z,0,0,+1)` unless `negative-normal` is present (lines 299–309). For65 layers this would place the flat output PPM at layer 32. The code assigns this midpoint independently of the radius, interval and sampling direction. Consequently, even establishing that this generated PPM was used would establish the flat reference-plane convention; it would not prove that layer 32 sampled zero offset from the original native surface. The `negative-normal` option concerns this **new output PPM**, not reversal of the original sampling normal.

## What remains unresolved

Using the current defaults and the retained package values150 µm/3.24 µm gives radius 23.148148…,47 planes and offsets beginning at−23.148148…. The middle plane 23 has offset−0.148148… rather than exactly0. These defaults differ from the paper's explicit radius 32 and must not replace the declared dataset settings. There is no conflict between a configurable renderer default and a publication that specifies its chosen radius.

Plane count alone admits other radii, intervals and one-sided directions; this initially unresolved metadata-only ambiguity is resolved for the **declared dataset protocol** by the primary paper. The remaining uncertainty concerns exact execution revision/arguments, native PPM values and orientation, and actual native-to-surface reproduction. Original `PHercParis1Fr39_54keV.ppm` and flattened `PHercParis1Fr39_54keV_surface_volume.ppm` are distinct artifacts; the latter must not be used as a native-CT map.

A bounded diagnostic may use zero plane32 and3.24 µm steps explicitly as the **dataset-declared** convention. It must distinguish that contract from direct native rendering verification and refrain from assigning a native-CT normal sign or papyrus-side interpretation. No original PPM or array was examined here. Ink annotation depth and optical registration are not authenticated by this renderer review; the primary paper's separate label-alignment evidence is owned by the annotation audit.

## Evidence and bounds

The [request ledger](request-ledger.json) retains **11 HTTPS attempts, all HTTP200,152316 response-body bytes**. Every attempt had TLS verification, no redirects and no retries; per-response cap 256 KiB, total cap 1 MiB and maximum 12 attempts. GitHub API blob envelopes were preserved and their decoded source lengths and Git blob SHA1 identities verified before source files were saved. These counts concern observed response bodies, not total wire traffic. No arrays, images, checkpoints, model calls or paid resources were used; earlier files remain untouched. No further network lookup is planned in this package.

Machine-readable claims and SHA256 input/source pins are in [sampling-contract.json](sampling-contract.json). [evidence-manifest.json](evidence-manifest.json) covers all package files except itself.
