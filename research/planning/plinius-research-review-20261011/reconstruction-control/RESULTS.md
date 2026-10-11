# Same-scan CT reconstruction control: results

Nine preselected 128³ uint8 chunks were acquired successfully: **18 MiB total**,
three fixed locations × three reconstructions of scan20250720065842. No model
was run. The frozen [plan](PLAN.md), exact [requests](REQUEST_PLAN.json),
[hash receipts](REQUEST_LEDGER.json) and [metrics](RESULTS.json) are retained.

**The usual release and alternate reconstructions differ materially at every
site.** The two siblings agree strongly with one another. Their shared anatomy
is visible in all three fixed orthogonal panel sets; these are not empty cubes.
However, the prescribed phase-correlation estimator favours +1Z at site2 for
usual-versus-sibling comparisons. Accordingly, this check does not admit all
three usual/sibling sites as exact voxel-aligned pairs. It supplies no correction
and triggers no adaptive acquisition or registration.

| Site | Pair | Full-cube Pearson | Common-nonzero Pearson | Nonzero-edge gradient correlation | FFT shift applied to second volume, ZYX |
|---|---|---:|---:|---:|---|
| site1 | usual / pag0 | 0.8316 | 0.7927 | 0.8106 | [0, 0, 0] |
| site1 | usual / pag50 | 0.7768 | 0.7204 | 0.8345 | [0, 0, 0] |
| site1 | pag0 / pag50 | 0.9865 | 0.9821 | 0.9852 | [0, 0, 0] |
| site2 | usual / pag0 | 0.7496 | 0.7408 | 0.6587 | [1, 0, 0] |
| site2 | usual / pag50 | 0.7279 | 0.7171 | 0.7073 | [1, 0, 0] |
| site2 | pag0 / pag50 | 0.9877 | 0.9838 | 0.9828 | [0, 0, 0] |
| site3 | usual / pag0 | 0.7905 | 0.8043 | 0.6273 | [0, 0, 0] |
| site3 | usual / pag50 | 0.7338 | 0.7541 | 0.6582 | [0, 0, 0] |
| site3 | pag0 / pag50 | 0.9857 | 0.9826 | 0.9816 | [0, 0, 0] |

The nonzero peak is a result of this fixed, integer, periodic phase-correlation
estimator. Different reconstruction filters, intensity clipping, shared noise
and tapering can affect it. It is **not proof of a physical one-voxel displacement**,
a surface-normal offset, or justification to shift a rendered mesh. The results
have not been recomputed after any fitted shift. Other zero peaks do not prove
subvoxel tissue alignment either.

## What the images add

[Site1](site1-orthogonal.png), [site2](site2-orthogonal.png) and
[site3](site3-orthogonal.png) show all three reconstructions with fixed0–255
mapping and fixed orthogonal midplanes. Across them, the usual release presents
smoother, wider bright structures; pag0/pag50 show narrower structural edges and
more fine-scale variation. This is visual description of exported CT, not a
measurement of ink contrast or proof that finer-looking detail is physical.
Main-versus-sibling mean absolute differences are28.1–48.3 uint8 levels; sibling
pairs differ7.2–11.0. Differing export windows prevent interpreting those raw
brightness differences as information gains.

Only76.6–87.6% of sibling voxels are nonzero, versus99.55–99.92% in the usual
release. Zeros are intensity codes, not authenticated absence-of-tissue labels.
A benchmark that equates nonzero CT with valid tissue would therefore change
its evaluation population when switching these reconstructions. The common
nonzero scores are reported alongside full-cube scores to expose that effect;
there is no fitted threshold or score-based site exclusion.

**Metric naming qualification:** the frozen JSON key `saturated255_fraction`
means exactly the fraction with code255. It is not a complete saturation or
clipping measure. Pag50's observed maximum is254; its zero255 count does not
show that it is unclipped. The original metric/code are preserved unchanged.

## What this supports

This is a useful small reconstruction/export comparison using the same measured
acquisition. It demonstrates a concrete input-distribution difference that a
fine-versus-native study would otherwise mix with voxel size, energy and geometry.
The local identity failure at site2 must remain visible. A later surface/model
comparison would need its own fixed same-sheet render and local correspondence
checks; these raw cubes are not model-ready surface inputs.

It does not choose a better reconstruction, establish ink signal, identify a
writing surface, or validate PHerc0139 fine/native registration. All sites were
chosen from existing registration landmarks and are convenience locations,
not independent validation landmarks or a representative sample of the scroll.
Sibling correlations also benefit from shared measured rays and reconstruction
noise. Neither these three sites nor a metadata identity transform establishes
global voxel alignment.

The publisher reports additional differences in detector-distortion handling
and uint8 export windows. This is a reconstruction/export-pipeline contrast,
not a pure Paganin-parameter intervention. The existing depth experiment and
its model inputs remain unchanged.

## Verification and next decision

Five semantic checks cover phase-correlation sign/axis order, flat input,
Spearman ties and Pearson degeneracy/self-correlation. All passed before
acquisition. All five frozen source/config hashes remained unchanged afterward.
An [independent review](REVIEW.md) recomputes the saved numerical comparisons
and checks all source objects; its scope and any limitations are explicit.
The main reviewer inspected all27 fixed slice views and retained the zero-lag
failure without expansion.

Close this control after the declared nine comparisons. **The main next research
step remains a CT-only local-correspondence qualification on known training
winding w040, followed by a matched-physical-tissue learning comparison if it
passes.** Reconstruction choice is now a concrete supporting factor; it should
not become another open-ended branch or be optimized on PHerc0800 brightness.

Data attribution: Vesuvius Challenge open data, PHerc0139, catalogue-stated
[CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/). Exact public URLs,
volume identities and source bytes are bound by the request ledger. These are
private research copies and derivative diagnostic figures.
