#!/bin/bash
# reproduce.sh - from the two source meshes to the submission image, step by step.
#
#   VC3D=/path/to/VC3D-41701aa.AppImage CT=/path/to/20250521140437-8.640um-1.2m-116keV-masked.zarr \
#   CKPT_DIR=/path/to/ckpt PY=/path/to/python bash reproduce.sh [work_dir]
#
# Needs: Linux, one NVIDIA GPU (24 GB), villa 0fb38c45 installed in $PY (see ink/Dockerfile), the four checkpoints
# in $CKPT_DIR (ink/scripts/fetch_checkpoints.sh ink/models.tsv $CKPT_DIR), and tifffile, zarr, numpy, scipy,
# opencv. The ink sweep is the slow part: 2 x 248 maps on 2560 x 6720 px (roughly 6 hours on one RTX 4090).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-$HERE/work}
N=concat_w047-w048_R5B2_z9500-11000
S=$HERE/ink/scripts
: "${VC3D:?set VC3D}" "${CT:?set CT}" "${CKPT_DIR:?set CKPT_DIR}" "${PY:?set PY}"
export CKPT_DIR PY
mkdir -p "$WORK"

# 1. the segment from the two source meshes (CPU, seconds); bit-identical to outputs/$N.tifxyz
"$PY" "$HERE/mesh/build_concat.py" --w047 "$HERE/outputs/w047_R5B2_z9500-11000.tifxyz" \
    --w048 "$HERE/outputs/w048_R5B2_z9500-11000.tifxyz" \
    --letter-boxes "$HERE/inputs/${N}_letter_boxes.json" --out-dir "$WORK/mesh"

# 2. 66-layer render (VC3D 41701aa) and its zero-padded copy for the second ink run
bash "$HERE/mesh/render_66_layers.sh" "$VC3D" "$CT" "$WORK/mesh/$N.tifxyz" "$WORK/${N}_66.zarr"
"$PY" "$S/pad_render.py" "$WORK/${N}_66.zarr" "$WORK/${N}_66_padded.zarr" --pad-rows 40 --pad-cols 200

# 3. ink: four models, d = -15..+15, both renders (GPU)
bash "$S/run_depth_sweep.sh" "$WORK/${N}_66.zarr" "$WORK/maps_plain" -15 15
bash "$S/run_depth_sweep.sh" "$WORK/${N}_66_padded.zarr" "$WORK/maps_padded" -15 15

# 4. merge: w047 pixels from the padded run, the rest from the plain run
"$PY" "$S/merge_tile_phase.py" --plain "$WORK/maps_plain" --plain-name "${N}_66" \
    --padded "$WORK/maps_padded" --padded-name "${N}_66_padded" \
    --node-source "$WORK/mesh/${N}_node_source.npz" --out "$WORK/maps" --out-name "$N"

# 5. contrast stretch per model over all 31 depths, then the image at d = +1
mkdir -p "$WORK/images"
(cd "$WORK/images" && "$PY" "$S/ensemble_maps.py" --surface "$WORK/${N}_66.zarr" --name "$N" --maps "$WORK/maps" \
    --out . --stat-depths -15..15 --write-depths 1)
(cd "$WORK/mesh" && sha256sum $N.tifxyz/x.tif $N.tifxyz/y.tif $N.tifxyz/z.tif $N.tifxyz/meta.json) > "$WORK/mesh_sums"
M=$WORK/maps
"$PY" "$S/make_submission_image.py" --mesh "$WORK/mesh/$N.tifxyz" --umbilicus "$HERE/inputs/umbilicus_PHerc0343.json" \
    --ct "$WORK/${N}_66.zarr" --overlay-style dimmed-ct --caption compact --geometry-sums "$WORK/mesh_sums" \
    --area 380,105,6021,2560 --max-cm2 12 --ct-layer d+1 \
    --ink "$M/${N}__dnative__L66s26_reverse.tif" "$M/${N}__hecate__L66zc33.50_reverse.png" \
          "$M/${N}__hybrid_3d2d-seed42__L66s26_reverse.tif" "$M/${N}__hybrid_3d2d-seed43__L66s26_reverse.tif" \
    --ensemble --ink-stats "$WORK/images/${N}__stats.json" \
    --rows "$HERE/inputs/${N}_rows.json" --letters "$HERE/inputs/${N}_letter_boxes.json" --letter-markers off \
    --sizes-csv "$WORK/image/${N}_letter_sizes.csv" --out-dir "$WORK/image" --force
echo "compare: sha256sum $WORK/image/$N.png $HERE/outputs/$N.png"
