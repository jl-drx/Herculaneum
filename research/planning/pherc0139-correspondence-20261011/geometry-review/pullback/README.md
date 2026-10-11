# Geometry-only representative pullback

All **4,439,040** queries (three sites ×20×272×272) satisfy the frozen
numerical/support checks. Maximum accepted residual is below0.01 native
voxel. This is a coordinate inversion result, **not evidence of cross-scan
registration, matched CT point-spread functions, or ink accuracy**.

`PLAN.json` was saved before the solve; SHA256:
`a631e6bc8d0c5943927bdf7fbed1d3f2dc71e215ecf6439a1e4edd3120258626`.
The code hash and every coordinate input hash are in that plan. No CT,
labels, or predictions were read. Runtime was approximately27seconds using
one CPU thread. All sites are the frozen w040 training-tissue locations;
none was replaced or selected by a model response.

The representative fine field averages the base coordinate and flipped
normal of all16 group0 XY samples in a group2 footprint. It maps both through
the published affine and interpolates the resulting group2 base/normal
fields bilinearly. Mean normals are **not renormalized**. The representative
ray is `B(u,v)+N(u,v)*(k-54)`. This linear geometry representation does not
assert that a mean of16 CT values equals CT sampled at their mean position.

Source-derived UV and nominal physical depth initialize the inverse. Newton
uses exact bilinear derivatives including depth-dependent normal-field
derivatives. It rejects incomplete geometric support, wrong normal sign,
positive/folded Jacobians, minimum singular value below1e-6, condition above
1000, movement beyond±2 fine-group2 pixels or±8 fine planes, and residual
above0.01nativevoxel after10updates. All queries passed in at most4updates;
zero invalid/support/basin/conditioning/nonconvergence failures occurred.

Each `site*/queries_uvk.npy` has shape **(20,272,272,3)**, float64, final axis
`[u,v,k]` in **global fine group2 X, global fine group2 Y, original fine
depth-plane index**. First dimension corresponds to native depth4…23.
Second/third dimensions follow global native Y/X increasing from the site's
272square halo origin in `RESULTS.json`. `initial_uvk.npy`, `support.npy`,
`failure_codes.npy`, `residual_native_voxels.npy`, and `iterations.npy` retain
all attempted samples and their status. No unsupported query should be used
as a valid CT sample merely because its numeric coordinates are finite.

| Site | Native halo XY origin | Fine group2 XY interpolation footprint, inclusive | Fine depth interpolation footprint, inclusive |
|---|---|---|---|
| site1 | 2296,1784 | X2238…2504, Y1740…2006 | 14…96 |
| site2 | 3832,3064 | X3733…3998, Y2984…3249 | 14…94 |
| site3 | 5496,4344 | X5351…5616, Y4228…4493 | 15…95 |

These bounds include trilinear interpolation neighbors, **before** any
additional Gaussian Z context. Root adds the planned radius5 context and
freezes actual CT object requests separately. Each per-site receipt contains
unrounded coordinate bounds, residual quantiles, initialization displacement,
output hashes, central basis and basis variation.

`LOCAL_BASES.json` contains native-XYZ bases, columns `[normal,dy,dx]`, at
the eight fixed checking-support centers. Geometry uses the same staged
renderer sampling and ±1native-raster-pixel position derivatives. The maximum
normal angles relative to each core-center basis are30.74°,13.00°,22.69°.
Use the corresponding local basis for each support's physical shift norm
and normal/tangential decomposition. A single core basis hides substantial
curvature. These bases are defined at the mesh surface; off-surface ray
derivatives can also vary with depth.

The meaningful synthetic checks cover inversion with a varying normal field
(residual1.4e-5voxels), invalid-support rejection, singular-Jacobian rejection,
and trust-basin rejection. Their results are retained in
`SYNTHETIC_CHECKS.json`. Production arithmetic stages raster coordinate and
normal operations in FP32; the representative-field averaging and inverse
are float64. Renderer fused-operation address ambiguity is rejected during
the axis mapping.

Replay `pullback.py` with NumPy/tifffile; it verifies frozen input and code
hashes before solving. `local_bases.py` independently recreates the fixed
support bases. Absolute retained-catalog paths in the scripts need remapping
when restoring the packet elsewhere; the corresponding catalog hash must
remain unchanged.
