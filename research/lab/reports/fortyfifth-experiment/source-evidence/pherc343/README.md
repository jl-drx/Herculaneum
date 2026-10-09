# PHerc. 343: First Letters

Code and results of my First Letters submission for PHerc. 343 (Vesuvius Challenge).

- **Scan:** `20250509220833-8.640um-1.2m-116keV` (ESRF, 8.64 µm, 116 keV)
- **Volume:** `20250521140437-8.640um-1.2m-116keV-masked.zarr` (`s3://vesuvius-challenge-open-data/PHerc0343/volumes/`)
- **By:** Erwin Nieuwlaar

## Results (`outputs/`)

| file | what |
|---|---|
| `concat_w047-w048_R5B2_z9500-11000.png` | the image: ink in yellow over a dimmed CT layer, 1 cm scale bar |
| `concat_w047-w048_R5B2_z9500-11000.json` | its sidecar: every input with its sha256, the command, the area and all numbers |
| `concat_w047-w048_R5B2_z9500-11000.tifxyz` | the segment: one mesh across the spiral seam θ 360\|0 |
| `w047_R5B2_z9500-11000.tifxyz`, `w048_R5B2_z9500-11000.tifxyz` | the two spiral-fit meshes the segment is made of |
| `SHA256SUMS` | checksums (`sha256sum -c SHA256SUMS`) |

A tifxyz folder holds `x.tif`, `y.tif`, `z.tif` (3D position of every grid node in level-0 voxels; -1 = no node) and
`meta.json`. One grid step is 20 voxels. `model.pt` in the two source meshes is lasagna's flattening state; rendering
doesn't need it.

## How it is made

1. **Meshes.** I ran the villa spiral chain (commit `0fb38c45`) on the band z 9500–11000 with my own scroll axis,
   snapped the windings onto the m7 surface prediction and flattened them with lasagna. Details: `mesh/README.md`.
2. **One segment.** `mesh/build_concat.py` joins w047 and w048 at θ 360|0. Both meshes keep their own nodes; a
   transition strip on the w047 side takes up the small offset between the two grids, so no cell at the seam is
   stretched or skewed.
3. **Render.** VC3D `41701aa` renders the segment into 66 layers, one voxel apart along the normal
   (`mesh/render_66_layers.sh`). Depth d = layer − 33, + is away from the scroll axis.
4. **Ink.** Four 9 µm models: dense_native (my fine-tune), ink_9um seeds 42 and 43, and Hecate 9.6 µm. None of them
   was trained on this PHerc. 343 volume. Each runs at every depth d = −15..+15 (`ink/scripts/run_depth_sweep.sh`).
   The models work on tiles, so I ran them twice: on the render and on a zero-padded copy
   (`ink/scripts/pad_render.py`), and `ink/scripts/merge_tile_phase.py` takes each half of the segment from the run
   where it sits at the tile phase of its own mesh.
5. **Image.** `ink/scripts/ensemble_maps.py` sets one contrast stretch per model over all 31 depths and averages the
   four models' z-scores; `ink/scripts/make_submission_image.py` makes the image at d = +1.

## Reproduce

`reproduce.sh` has every command, in order. You need Linux, one NVIDIA GPU with 24 GB, the VC3D `41701aa` AppImage
and the CT (a local copy of level 0, or streamed from the open-data bucket). The ink environment is in
`ink/Dockerfile` (a draft I haven't built as a container; the same steps by hand work). `ink/scripts/fetch_checkpoints.sh`
downloads the four models and checks their sha256 (`ink/models.tsv`).

What I checked (7 October 2026, from this repository):
- `build_concat.py` gives the segment in `outputs/` bit for bit;
- the render of it is identical to the one I used;
- merging the maps of the two ink runs gives the maps I used, byte for byte, and from those the image in `outputs/` is
  bit-identical;
- one ink map recomputed on a local RTX 3090 matches the map from the rented RTX 4090/5090 machines at r 0.99999
  (no pixel off by more than 1/255); that is the usual difference between GPU types.

## Models

- dense_native, step 16 000: huggingface.co/Nieuwlaar/ink9um-dense-native
- ink_9um, seeds 42 and 43, step 75 000: huggingface.co/scrollprize/ink_9um
- Hecate 9.6 µm: huggingface.co/scrollprize/hecate

## Licence

Code: MIT (`LICENSE`). The meshes and images are derived from the scan data and stay under the organisers' licence,
CC BY-NC 4.0. VC3D (GPL-3.0) is not included.
