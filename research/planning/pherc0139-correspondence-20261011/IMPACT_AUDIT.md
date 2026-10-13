# Scope of the new PHerc0139 coordinate finding

Bounded read-only source audit, 11 October 2026. This text is an addendum
outside the frozen correspondence bundle. No CT, model, label, or prediction
arrays were fetched or evaluated, and no previous report or result was
modified.

**No direct application of the new fine-to-native raster correction was
found in our earlier PHerc0139 training path.** Those experiments used a
fine-scan-derived coarse representation and separately published aligned
targets. They did not construct real-native PHerc0139 supervision by
resizing a fine teacher map to the native113 raster. This conclusion concerns
the inspected computational contracts; it is not a retrospective certificate
of upstream annotation accuracy.

## Verified contracts

| Work | Verified dependency | Impact of this finding |
|---|---|---|
| Study21 PHerc0139 training and geometry support | Curated `pherc0139-w016` is public **w029**, segment `20250108000004-w029_2025010827`, fine78 volume `20260102150214`; source level2 XY, source depth13:97 pooled into21 planes, then model planes2:19. The geometry check uses that fine transformed mesh and its own renderer. | It does not cross between fine78 and native113 raster parameterizations. The new **w040** fine/native correction should not be applied to these arrays or their geometry mask. |
| Studies37–41 optimization, coverage, mixture and supervision controls | Study37 reuses the frozen study13 plan and worker. The two PHerc0139 contexts have the same fine78 source URL and supplied masks above. Study38 reconstructs PHerc1667/w013 from its own fine109-plane source and supplied21-plane labels; study40 repeats this on another w013 crop; study41 reuses the mixed batches with a changed objective. Study39 combines retained predictions. | No identified direct dependency on the newly rejected fine/native whole-raster resize. Conditional training/transfer results remain results for their recorded inputs and references. This finding does not explain their small AP gains or authorize a new training-input shift. |
| Study22 dense-native qualification | Strict checkpoint identity and historical/current CPU model-factory parity on nine **synthetic** cases. No real PHerc0139 images were scored there. | Checkpoint bytes, loading, preprocessing and execution qualification remain valid within their original scope. Qualification never certified training-label alignment or ink accuracy. |
| Studies42–45 supporting work | Study42 used Paris4 raw-volume blocks and its separately recorded published volume transform. Studies43–45 profilometry work used PHerc250/500P2/248. Study45's checkpoint panel was six Paris regions and two PHerc0814 regions; the target geometry lattice was PHerc0800. | None uses the w040 fine/native raster correction as an input contract. The numerical7.43-pixel correction is not transferable to another specimen, mesh pair, or label package without its own source derivation. This short audit does not independently revalidate every Paris4 geometry convention. |
| Qualified PHerc343 reproduction and current PHerc0800 four-model/depth recipe | Native CT rendered from each target mesh; model-specific tiling/depth processing, including Hecate's physical9.6µm resampling. No PHerc0139 fine teacher is resized onto those target rasters during inference. | Native target inputs and the released inference recipe are unchanged. Hecate's input resize is a different operation from cross-acquisition teacher registration. The finding does not invalidate observed published-image agreement or execution reproducibility. |

The first rows were traced through the actual frozen study13
`implementation/final-plan/training-plan.json`, study37 `prepare_plan.py` and
`study_common.py`, study38/40 `prepare_data.py`, and study41 `data_loader.py`.
The PHerc0139 provenance explicitly names fine78 volume `20260102150214`,
public w029, level2, the source slices and pooling rule. Its cached
inklabels/supervision/validation `.zattrs` declare
`aligned-public-level2-zmean4-21slice-v1` and the same source volume/level.
There is no native113 volume `20250728140407` in these two adaptation-input
records.

The supplied label metadata also names an unavailable historical
`/scratch/elian/mask-align/open-data-training-manifest-v1.json`, an
`aligned-public-21slice-v2` rebuild, and a relaxed/cleaned mapping-validity
policy. Therefore “published aligned targets” means their **declared
coordinate contract**, not independently authenticated physical alignment.
An upstream transfer error remains possible; this audit neither demonstrates
it nor rules it out. That limitation was already present in our reports.

## A genuinely relevant upstream contract, not yet a demonstrated label error

Nieuwlaar's pinned `training/scripts/native_common.py`, function
`teacher_on_grid` (lines97–104), loads a fine78 TIFF and calls
`cv2.resize(t, (w,h), interpolation=cv2.INTER_AREA)`. The pinned
`build_native_dense_labels.py` calls this helper and then merges official
manual native labels where present on w040/w041. The pinned
`native-eval/native_bench.py`, `load_teacher` (lines92–102), independently
uses the same complete-image resize rule for its teacher reference.

Those are precisely the type of cross-acquisition mapping for which the new
source-frame finding matters. The w040 coordinate analysis shows that a
simple whole-raster resize is not the source-derived correspondence between
the inspected public fine and native geometry. It does **not yet show** the
actual displacement/error of every historical teacher pixel, the effect of
manual overrides, or the impact on the released checkpoint's learned weights.
The exact teacher-file shapes, padding/cropping conventions, intermediate
asset versions and generated label coordinates must be connected to the
verified public geometry before making such a claim. No blanket upstream
bug or invalid-checkpoint conclusion is justified here.

Hecate's documented fine/native training and alignment choices likewise
create a possible ancestry question. The inspected card does not provide a
segment-level asset/mapping manifest sufficient to test this correction
against its historical training. Its inference implementation qualification
and its training-supervision provenance are separate questions.

## Consequence and smallest next dependency

Keep earlier frozen results and current inference unchanged. Add this
finding to the interpretation of future **real-native PHerc0139 reference
agreement**, where raw fine-map resizing can otherwise be mistaken for
verified registration. The earlier native benchmark sources remain useful
for reproducing the published calculation, with that explicit limitation.

For a bounded impact test, first inspect the exact historical w040 teacher
and native-label asset metadata/header contracts and any retained generation
manifest, without training. Establish whether they use the same full raster
origins and versions as the geometry tested here; distinguish manual-target
support from transferred-teacher support. Only then can a fixed coordinate
replay measure which transferred targets change. Broader model-training
ancestry or upstream optical-label accuracy remains unknown until its own
provenance is available.

Primary retained references:

- [Study21](../../lab/reports/twentyfirst-experiment/RESULTS.md) and its
  `pherc0139-geometry/TEXT_FINDINGS.md` distinguish fine-render geometry
  support from annotation alignment.
- [Study22](../../lab/reports/twentysecond-experiment/RESULTS.md) states the
  synthetic-checkpoint qualification scope.
- [Study37](../../lab/reports/thirtyseventh-experiment/RESULTS.md),
  [study38](../../lab/reports/thirtyeighth-experiment/RESULTS.md),
  [study40](../../lab/reports/fortieth-experiment/RESULTS.md), and
  [study41](../../lab/reports/fortyfirst-experiment/RESULTS.md) retain the
  precise input/control limitations.
- [Study45](../../lab/reports/fortyfifth-experiment/RESULTS.md) identifies the
  released-checkpoint panel, profilometry subjects and native target audit.
- [Pinned native training helper](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/training/scripts/native_common.py)
  and [pinned native benchmark](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/native-eval/native_bench.py)
  provide the relevant cross-acquisition resize contract. Local helper
  SHA256: `34f5bb233b13675ec95a5982c0061096d02d87ed5207cc63f49c7e841f19d94d`;
  benchmark SHA256:
  `bcd7f65d58b7698f41a7bd1e13d502e98ec8353918131d267678b51559a06120`.
