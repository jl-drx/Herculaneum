# Meshes

## The two source meshes (`outputs/w047_…`, `outputs/w048_…`)

I made them with the villa spiral chain (commit `0fb38c45`), on the band z 9500–11000, with the public m7 surface
prediction, lasagna normals and spiral tracks of PHerc. 343 and my own scroll axis (`inputs/umbilicus_PHerc0343.json`).
The mesh files are the unchanged pipeline output; only the folder names are mine.

1. **Distance field:** villa `make_surf_sdt.py` on the m7 prediction at level 1, threshold 128, over z 9500–11000 ± 64,
   masked by the CT occupancy at level 4 (dilated by 1).
2. **Spiral fit:** villa `fit_spiral`, sense clockwise, no patches, with the tracks, the lasagna normals (group 2,
   `lasagna_scale` 4), the distance field and my axis, and these overrides:
   `{"z_begin": 9500, "z_end": 11000, "input_disable_patches": true, "loss_weight_shell_outer": 0,
   "loss_weight_shell_patch_radius": 0, "dense_spacing_mode": "phase", "optimizer_num_training_steps": 30000}`.
   One line is patched so it runs without an `outer_shell/` folder: `fit_spiral_shell_optional.patch`.
3. **Snap:** `snap_meshes.py` (defaults, level 0) moves every node along its normal onto the nearest predicted sheet.
4. **Flattening:** `python lasagna/fit.py configs/flatten_fast_nofilter.json <name>_flatten_input.json --device cuda`
   on each snapped winding. The config is copied in full into `meta.json`, under `fit_config`.

The two windings used here are w047 and w048 of that fit.

## The segment (`outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz`)

```
python mesh/build_concat.py --w047 outputs/w047_R5B2_z9500-11000.tifxyz --w048 outputs/w048_R5B2_z9500-11000.tifxyz \
    --letter-boxes inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json --out-dir work/mesh
```

How the join works is in the header of `build_concat.py`. The letter boxes are only used to keep the transition strip
out of the letters; they are never drawn on the image.

## Render

```
mesh/render_66_layers.sh <VC3D AppImage> <CT zarr> <mesh>.tifxyz <out>.zarr
```

Usage and the exact VC3D build are in its header and in `ink/Dockerfile`.
