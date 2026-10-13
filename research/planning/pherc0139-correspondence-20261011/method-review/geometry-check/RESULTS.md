# Independent geometry-only check

All six downloaded coordinate TIFF hashes/lengths passed their receipt checks. The two meshes actually have XYZ array shapes1244×1552×3 and320×399×3. No CT, prediction or label data were read. The fixed sites were the5×5 normalized canvas fractions0.1,0.3,0.5,0.7,0.9, with native pixel indices rounded down to the corresponding output pixel. All25 were retained; site(u0.5,v0.1) had invalid coordinate support and remains unknown.

The exact pinned transformation caller measures axis spacing before/after the affine, resamples the grid using half-pixel bilinear interpolation and resets its scale to the original value. Both transformed grids therefore share source coordinates according to

```
q_fine = (q_native + 0.5) * (fine_mesh_count/native_mesh_count) - 0.5
```

This is an analytic composition of the two source-grid resamplings, not a fit to output coordinates. The unknown common source-grid dimensions cancel. Interpolating the two resampled meshes does not generally commute on curved geometry; float quantization also remains.

Composing that equation with the renderers' canvas coordinate equation gives an extra28.897fine pixels inX and28.875inY compared with naive half-pixel resizing of the complete rendered images. These are approximately7.429 and7.428native pixels respectively. A shared canvas shape after image resizing would therefore conceal a material coordinate mismatch.

| Diagnostic across24 valid fixed sites | Median | Maximum |
|---|---:|---:|
| Source-derived coordinate residual, native voxels |0.05667|0.35942|
| Naive full-canvas resize residual, native voxels |10.50591|12.41274|
| Affine-transformed fine ray versus native normal, degrees |1.23682|4.37692|
| Ray-only discrepancy at native depth13.5, native voxels |0.29407|1.03359|
| Base plus ray discrepancy at native depth+13.5, native voxels |0.29700|1.08292|

For comparison, the covector-transformed geometric normal differs from the native cached normal by median1.24045°, maximum4.43418°. This also includes different physical stencil widths and nearest-mesh-vertex normal sampling. The fine and native grids are not identical geometric surfaces off their central mesh merely because their center points approximately correspond.

The largest base residual, at(u0.7,v0.9), is0.35942native voxel, almost entirely normal (signed component−0.35907; tangent norm0.01598). This is consistent with curved-grid reinterpolation being relevant; this diagnostic does not establish its exclusive cause. It is not an observed physical displacement between CT acquisitions.

The independent code uses float64 bilinear point interpolation and stagedFP32 centered-difference normal computation. It reproduces the stated mathematical coordinate contract, **not every renderer rounding/FMA operation**. The normal cache is sampled at the renderer's nearest mesh vertex and flipped. Input/source hashes, every point, every residual vector and the five checked depth positions−13.5,−6,0,6,13.5 are retained in `RESULTS.json`.

## Consequence for the proposed surface comparison

Simple image resizing plus a scalar depth ratio is not sufficient. A geometry-only3D pullback is justified before CT comparison. The root's proposed inverse of the mean base/normal fields over each fine4×4 pooling footprint is a defensible *representative-coordinate* correction: it must be described as centroid/footprint handling, not equality of individual CT samples or matching point-spread functions. Fine group2 uses two integer-rounded reductions, so the observed pooled intensity is not exactly an unrounded16-sample average either.

For that inverse, residual alone is insufficient. Require a bounded local basin, nonsingular/consistently oriented Jacobian, sensible conditioning, complete source stencil, and rejection of branch changes/nonconvergence. Preserve any excluded locations. Its validation residual quantifies coordinate-solver consistency, not tissue registration.

The proposed native depth template10:18 plus shifts±6 requires indices4:24 before filtering. Any structural3Dfilter or subvoxel interpolation adds context; acquire/compute that context explicitly. Similarly±6XY plus a radius2filter consumes an8-pixel halo before an additional interpolation footprint. Do not reflect-pad missing material and call it acquired support. A fixed fine-Z Gaussian sigma1.5 is a specified structural surrogate, not calibrated native scanner physics.

The corrected geometry can support the planned fixed-site CT experiment. Physical same-sheet correspondence remains an independent question, and a low inverse-coordinate residual cannot answer it.

## Replay

```
/workspace/.venvs/herculaneum/bin/python validate_geometry.py \
  --base /workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011 \
  --output /tmp/w040-independent-geometry-replay.json
```

The script needs NumPy and tifffile. It performs no network requests. The original reports, active study52, and root-owned geometry files were unchanged by this check.

## Independent scalar validation of the representative inverse

`check_pullback_scalar.py` reads the coordinate TIFFs directly and evaluates each
fine pooled representative from16 scalar group0 positions/normals, without
importing the author's evaluator. It checked81 fixed queries: XYarray indices
8,136,263 and depth indices0,9,19 at all three sites. Every saved output hash
and coordinate-TIFF receipt used by this check passed. All81 queries passed;
maximum independently evaluated residual was0.007727695native voxel, and the
largest difference from the saved residual was1.153e−12. Full records and failures
(empty) are in `PULLBACK_SCALAR_RESULTS.json`.

This is an independent finite subset check of the representative geometry
calculation, supplementing the author's4,439,040-query screen. It does not
independently recompute every query or validate physical CT registration.
