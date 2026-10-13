# Spare-VM task: independently challenge PHerc0800 support admission

This is a CPU-only geometry task for a Linux VM with 9 cores, 9GB RAM and 30GB
disk. No GPU, CT intensity download, detector, model weights, cloud credentials or
paid resource is needed. Do not run archived producers against their original
absolute paths. The reference implementation is evidence, not a portable runner.

## First verify the package

After downloading and unzipping the private Actions artifact:

```bash
sha256sum -c SHA256SUMS
python3 verify_packet.py
```

Record the Actions run URL and downloaded ZIP SHA256. Both commands must pass.
The manifest identifies every source by original path, byte count and SHA256.
The public source meshes were already retrieved and verified; no new download
is needed. The source arrays are small coordinate TIFFs, not CT intensity images.

## Independent assignment

Read frozen/PROTOCOL.md and frozen/GO.json first. Write your own implementation
before reading reference-implementation/ or expected/ wherever practical. Return
both successful and failing checks, source/code hashes and limitations. Do not
change the original masks, declare a reading, or treat inspected regions as a
new held-out benchmark.

A. Reconstruct and test the frozen decision from the raw meshes.

- Six mesh folders contain original x/y/z float32 TIFFs, metadata, declared
  surface dimensions and source-render metadata for the five larger meshes.
- Regenerate previous base centres using the historical definition:
  gx=(x+0.5)*scale_x, gy=(y+0.5)*scale_y; bilinear float64 interpolation cast once
  to float32; all four vertices must be positive and finite, inside the grid.
  Crop exactly the three previous_surface_extents from GO.json. Expected union
  count before deduplication is3,260,832. There is no intensity array involved.
- Candidate bases/normals use the archived tile/halo float32 arithmetic described
  in the protocol. The16 archived-point receipts allow a check independent of
  our new outputs. Regenerated base-map error may be at most0.01 voxel relative
  to the float64-then-float32 historical map; sample receipt tolerances are
  0.002 voxel for bases and1e-6 for normal components.
- Apply the51.1-voxel nearest-old-centre bound, integer66-layer corner envelope,
  all seven prior CT boxes and radius128 context rule. The exact expressions and
  units are frozen in the protocol. Do not confuse conservative exclusion with
  proven physical overlap. Level3 voxel support remains a nominal scaling bound,
  not a rederived multiscale reconstruction kernel.
- Regenerate component restrictions from inputs/phase47-components/*.npz:
  choose label[floor((y+10)/20),floor((x+10)/20)] with indices clamped to the saved
  label extent. Evaluate full-resolution support; coarse labels only restrict
  which pre-existing component supplies the scored core. Reproduce the ranking
  and largest-square selection with deterministic y/x ties.
- Compare all stage counts, component counts, squares and the final decision
  with expected/ only after saving your result. Audit the assumptions as well as
  the arithmetic. The expected point_support masks are not provided; derive them
  from source geometry to avoid merely replaying our outputs.

B. Challenge the conservatism, as a separate diagnostic.

Use the same reconstructed per-pixel point_support without changing the51.1
bound, normal requirements, depth support or old CT boxes. Compute exact context
support for a **single native128px/stride64 grid** at origin(0,0), including the
usual final start at length-128. A scored pixel passes only if every in-canvas
patch covering it has all128x128 source pixels supported. The all-origin radius128
result must be a subset of this exact context. Report maximum square and area
for every component, including failures; do not select a different origin or
relax exposure rules. This single-grid diagnostic is not admission of the full
four-model/plain+padded recipe, whose Hecate rescaling and padding contracts need
separate exact footprints. If it produces a128px core, preserve that candidate
and the exact contributing patch list for review. If it does not, report that.

## Practical execution and return

Use Python3.12 and the included requirements.lock (NumPy, SciPy and tifffile
only). A virtualenv and hash-locked pip installation are sufficient. Use4 worker
threads at most; process one mesh at a time and keep previous centres in one
float64 KD-tree. Expected working RAM is a few GB and source input size is under
2MiB; avoid materializing every66-layer sample for the entire canvas. Endpoint
AABBs suffice for the conservative slab test. Stop rather than use swap if your
implementation exceeds7GB resident memory. No inference should run.

Return a ZIP with REPORT.md, your code, environment/version record, JSON/CSV
counts, exact candidate coordinates and any counterexample fixtures. Include
SHA256SUMS and distinguish exact numerical agreement, bounded numerical error,
assumption disagreement and unknown support. No need to return generated dense
base arrays; compressed masks and fixed-site discrepancies are more useful.

You can review expected/SUPPORT.png after the numerical implementation is saved.
The current result is a coverage constraint, not evidence of absent ink. Another
VM is asked to test it independently before it drives surface acquisition or a
claim that new evaluation coverage is unavailable.
