#!/bin/bash
# render_66_layers.sh - render a flattened tifxyz mesh into a 66-layer surface volume with VC3D.
#
# Usage:  render_66_layers.sh VC3D_BIN CT_ZARR MESH.tifxyz OUT.zarr [THREADS] [CACHE_GB]
#
#   VC3D_BIN  the VC3D AppImage (or its extracted squashfs-root/AppRun). The renders of this
#             submission were made with the VC3D AppImage built from ScrollPrize/villa at 41701aa
#             (2026-09-24); the earlier 28-layer renders on the rented machine used d285029 and
#             33b0b91, and all three builds give bit-identical layers.
#   CT_ZARR   the PHerc0343 volume 20250521140437-8.640um-1.2m-116keV-masked.zarr, either a local
#             copy of resolution level 0, or an empty cache directory plus
#             --remote-url https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr
#             (add that flag below to stream the CT from the open-data bucket instead).
#
# Layer k of 66 samples the CT at (k - 32.5) voxels along the mesh normal, so the mesh surface lies
# between layers 32 and 33; depth d = k - 33. No --flip-normals: + is away from the scroll axis.
# Layer k of the 66-layer render is exactly layer k - 19 of the standard 28-layer render.
set -euo pipefail
BIN=$1; CT=$2; MESH=$3; OUT=$4; THREADS=${5:-4}; CACHE_GB=${6:-4}
env QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=$THREADS \
  "$BIN" vc_render_tifxyz --volume "$CT" --group-idx 0 --scale 1 --segmentation "$MESH" \
  --num-slices 66 --slice-step 1 --cache-gb "$CACHE_GB" --zarr-output "$OUT"
