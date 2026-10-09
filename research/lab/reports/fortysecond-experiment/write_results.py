"""Summarize the frozen admission outputs without performing another experiment."""
from common import *
import numpy as np
a=json.loads((HERE/'alignment/RESULTS.json').read_text());v=json.loads((HERE/'ADMISSION_VERIFICATION.json').read_text());review=json.loads((HERE/'REFERENCE_REVIEW.json').read_text());table=[]
for r in a['rows']:
 if r['fit'] is None:table.append(f"| {r['id']} | — | — | — | — | Coverage rejected |")
 else:
  t=np.array([x['tangential_residual_coarse_voxels'] for x in r['checks']])*7.91
  table.append(f"| {r['id']} | {r['public_mapping_fit_NCC']:.3f} | {r['fit']['NCC']:.3f} | {np.median(t):.2f} | {max(t):.2f} | Screen failed |")
text='''# Phase42: paired acquisitions available; learning panel not admitted

The bounded Paris4 acquisition succeeded. Local correspondence and reference qualification did not. Stop ink-learning on this six-center panel under the frozen protocol. No model was trained or called, no GPU resource was used, and seed42 remains the baseline. This result does not reject paired-acquisition learning or establish that ink is absent or unrecoverable.

## What was tested

Paris4 segment 20231016151002 has old 7.91 µm/54 keV and modern 2.4 µm/78 keV CT. Six geometry-selected centers, each more than 1 mm apart, were fixed before intensity or label inspection. Their twelve native 256³ blocks have nonoverlapping same-volume footprints. We retained the published XYZ transformation, including reflection and calibrated scale; raw arrays use ZYX. Old/new mesh samples differ by about 11.79–37.30 µm under a normalized-grid comparison, which diagnoses mesh parametrization only—not independent physical alignment error.

Geometry used 18 requests and 3,672,730 delivered bytes. CT used 96 unique native 128³ chunks, 192 MiB, with no network retry. Total delivered scientific payload was 204,999,322 bytes, below the frozen 32 MiB geometry/reference and 256 MiB CT caps. Public label/prediction bodies were not fetched. The admission decision was recorded {:.2f} minutes after acquisition GO, inside the 45-minute bound.

A fixed translation-only correlation diagnostic fitted one union of structural supports and checked four disjoint local supports. One fixed blur and coarse resampling produced the synthetic structural template. The frozen screen required unambiguous interior correlation peaks, local checking NCC >= 0.5, competitor margin >= 0.05 and tangential shift agreement within 0.5 coarse voxel median/1 voxel maximum. Independent physical landmarks and wrap identity would still be needed even if this numerical screen passed.

| Region | Public-map fit NCC | Best fit NCC | Checking-shift disagreement median (µm) | Maximum (µm) | Result |
| --- | ---: | ---: | ---: | ---: | --- |
{}

These disagreement values are differences between correlation maxima on different supports, not measured physical registration errors. The supports may contain repeated substrate or weak structure; no expert-identified landmark set was established. Translation improves the fit objective, but all four computed regions fail the frozen checking screen. Even pair05's high correlations have alternative local peaks. No scan pair is admitted for ink-scale real-coarse learning.

Pairs 03/06 fail because the mapped template lacks our required fine-block blur context. Pair03 extends outside the acquired block; pair06 falls short of the six-voxel guard. This is a limitation of our fixed block placement, not evidence that the public volume lacks data. The original worker stopped at pair03. An explicitly frozen continuation retained pair01/02 score cubes, classified coverage failures, and computed pair04/05 once. It did not repeat the initial searches, expand acquisition, move centers or relax thresholds. See [interruption record](ALIGNMENT_INTERRUPTION.md).

## Reference and visual findings

The root assistant inspected all six multi-plane figures. Both scans show related large folds/layers and compact bright structures; modern CT resolves finer laminae and fibers. That is useful paired structural information. It neither identifies those bright structures as ink nor establishes an ink-quality advantage. Some nominal planes intersect an edge or gap; no automatic recentering is justified.

No independent ink-positive or defensible non-ink-papyrus pixels were established. Ambiguous material remains unknown. This is a limited assistant review of selected planes, not an exhaustive expert volume annotation. It cannot establish ink absence. Air is not negative ink on a writing surface, nor is quiet-looking papyrus a verified negative. Public transferred dense Paris4 labels have incomplete ancestry and cannot alone supply independent test truth for descendant methods. Native/synthetic learning also stops because synthetic registration does not solve the missing-reference problem.

All six regions are now disclosed as reference-review exposed, including the two nominal comparison regions. No split is promoted to an untouched test. Paris4 already occurs in seed42's upstream representations; a future within-parent learnability comparison needs clean initialization and verified physical splits.

## Verification and limits

The independent audit implementation verified 77 frozen pins, all 18 geometry responses and all 96 CT bodies; compared 201,326,592 assembled voxels byte for byte against raw chunks; checked all 20 score cubes and independently reconstructed peak, residual and admission arithmetic. It also compared 100 selected NCC entries against direct Pearson calculations, maximum error {:.3g}. Both coverage rejections reproduce. The audit does not establish physical truth or external review. Six pre-run fixtures had passed across ranged TIFF decoding/chunk assembly and structural-correlation contracts.

The full raw chunks, arrays, score cubes, view pixels, masks and original figures remain locally retained. GitHub receives selected source, receipts, reports and compact scientific figures; it is not a bulk CT/checkpoint backup. No new R2 objects, Runpod resources, model checkpoints, transcription or translation were produced. GPU spend remains £0 against the £10 total first-pilot cap. CPU monetary spend is unrestricted.

## Next decision

Proceed reference-first: obtain a documented visible-ink coordinate/annotation packet and defensible non-ink support before selecting another panel. Then native-versus-synthetic matched-quality learning can test learnability without waiting for real-scan alignment. Real coarse learning remains conditional on independently checked correspondence and wrap uncertainty. The [next task draft](NEXT_STUDY_DRAFT.md) specifies a bounded route and the evidence that would change this stop decision; the [review packet](REVIEW_PACKET.md) exposes the present data and ambiguities. More blind crops or additional adapter updates would not resolve the missing reference.
'''.format(review['seconds_since_acquisition_GO']/60,'\n'.join(table),v['maximum_direct_Pearson_error'])
with (HERE/'RESULTS.md').open('x') as f:f.write(text)
print('Results written')
