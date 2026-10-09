"""Write reviewable findings from the verified, fixed family; no metric fitting."""
import csv
from diag_common import *
j=json.loads((HERE/'run-v1/COMPLETE.json').read_text());v=json.loads((HERE/'VERIFICATION.json').read_text());cs=j['contexts']
require(j['status']==v['status']=='PASS','Verified output required')
roles=['baseline','candidate','optical_darkness']
rows=[]
for c in cs:
    families=[('original',[dict(variant='zero',metrics=c['original'])]),('stable',[dict(variant='zero',metrics=c['stable'])])]
    families += [(name,[dict(variant=str(row['shift']),metrics=row['metrics']) for row in c[name]]) for name in ['common_shifts','moving_support_shifts']]
    families += [('bands',[dict(variant=row['band'],metrics=row['metrics']) for row in c['bands']])]
    for family,items in families:
        for item in items:
            for role,m in item['metrics'].items():
                rows.append(dict(context=c['id'],family=family,variant=item['variant'],predictor=role,**m['counts'],
                    prevalence=m['prevalence'],status=m['status'],AP=m['ap'],AUROC=m['auc']))
require(len(rows)==276,'Every metric record exported')
with (HERE/'all-metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Frag6 cached reference-sensitivity diagnostic', '',
'Small raster shifts and exclusion of uncertain label edges do not resolve the two weak regions within this fixed diagnostic family. The supplied optical image agrees closely with the supplied labels, while the CT model still ranks many clear pixels above ink pixels in those regions. Keep the unchanged seed42 baseline; phase32’s negative averaging result and primary scores are preserved.', '',
'This is post hoc analysis of four previously scored development contexts from PHerc51 (Frag6), one parent. Optical imagery and annotation may share provenance. Their agreement is not independent ink truth, benchmark qualification or recovered text. No model was trained or called in this phase.', '',
'## Original support', '',
'AP is average precision; its level depends on prevalence. All predictors in each row use the same known support. The optical column is a descriptive consistency check, not a competing CT model.', '',
'| Context | Ink / clear | Ink prevalence | Baseline AP | Averaging AP | Optical AP | Baseline AUROC | Optical AUROC |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for c in cs:
    b=c['original']['baseline'];ca=c['original']['candidate'];o=c['original']['optical_darkness'];n=b['counts']
    lines.append(f"| {c['id']} | {n['scored_positive']} / {n['scored_negative']} | {b['prevalence']:.4f} | {b['ap']:.5f} | {ca['ap']:.5f} | {o['ap']:.5f} | {b['auc']:.5f} | {o['auc']:.5f} |")
lines += ['', '## All nine label shifts', '',
'Every dy, dx ∈ {−1, 0, +1} was evaluated. One coarse pixel is three original raster pixels, not a measured physical registration error. The original 128×128 label canvas supplies the shifted 64×64 core without padding or interpolation.', '',
'The common known-support intersection removes 683 of the original 16,153 known pixels. Only 20 of the remaining 15,470 pixels can change class across the nine shifts. Two contexts have no changeable labels at all. Therefore their exactly constant scores are uninformative about alignment; the common-support test removes most of the boundary sensitivity it was intended to inspect.', '',
'| Context | Common known | Removed | Potentially changeable | Common-support baseline AP range | Own-support baseline AP range |',
'|---|---:|---:|---:|---:|---:|']
for c in cs:
    s=c['support'];a=c['common_shift_ranges']['baseline'];b=c['moving_support_shift_ranges']['baseline']
    lines.append(f"| {c['id']} | {s['common_known']} | {s['removed_for_common']} | {s['potentially_changeable_common']} | {a['min_AP']:.5f}–{a['max_AP']:.5f} | {b['min_AP']:.5f}–{b['max_AP']:.5f} |")
lines += ['',
'The secondary test uses each shift’s own known support. It changes coverage and class prevalence as well as labels. Baseline AP remains 0.08392–0.14475 and 0.13533–0.17917 in the two weak regions. These ranges do not identify a true warp, and no preferred shift or corrected reference is selected.', '',
'[All common-support shifts](figures-v2/common-shift-ap.png) · [All own-support shifts](figures-v2/moving_support-shift-ap.png)', '',
'## Fixed boundary-distance bands', '',
'Distance is measured to the nearest different class or unknown pixel in the conservative supplied mask. Bands are ≤2, >2 to ≤4, and >4 coarse pixels. These are mask distances, not independently authenticated ink edges. Every band retains both classes here.', '',
'| Context | Band | Ink / clear | Ink prevalence | Baseline AP | Averaging AP | Optical AP | Baseline AUROC |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for c in cs:
    for row in c['bands']:
        b=row['metrics']['baseline'];ca=row['metrics']['candidate'];o=row['metrics']['optical_darkness'];n=b['counts']
        lines.append(f"| {c['id']} | {row['band']} | {n['scored_positive']} / {n['scored_negative']} | {b['prevalence']:.5f} | {b['ap']:.5f} | {ca['ap']:.5f} | {o['ap']:.5f} | {b['auc']:.5f} |")
lines += ['',
'On the >4-pixel interior, baseline AP is 0.02851 and 0.08882 in the weak regions, versus optical AP 1.0 on the identical masks. Only 37 and 167 ink pixels remain, at prevalences 0.00959 and 0.04744. AUROC is 0.84685 and 0.76832: there is ranking information, despite poor AP. The lower AP compared with the original support cannot be attributed to erosion itself because prevalence and sampled pixels changed. Perfect optical ordering in these small interiors does not establish perfect annotation or independent accuracy.', '',
'## Visual and numerical review', '',
'[All four optical / label / CT / band panels](figures-v2/all-context-panels.png) show the same fixed probability scale. The third region’s prediction extends across clear areas along its right side; the fourth has broadly low probabilities including its annotated ink. These descriptive observations do not identify the cause. All four contexts and every shift are retained, with no region selected for the headline.', '',
'All 276 metric records are defined; none were omitted. Separate scalar indexing verified 147,456 translated labels, scalar sums verified 16,384 optical core values, and direct lattice comparisons verified all 16,384 boundary-band assignments without a distance transform. A separate ranking formulation reconciled all 552 AP/AUROC values with maximum difference 4.44×10⁻¹⁶. All eight original CT predictor/context records and the sealed phase32 primary result are unchanged. Six mechanics tests passed with no failed, errored or skipped tests. This is root numerical and visual review, not a separate agent review.', '',
'Execution took 1.432 seconds with peak RSS 274,337,792 bytes; verification took 1.458 seconds with peak RSS 251,703,296 bytes. Both fit their individual 120-second / 1-GiB limits. No network requests, model forwards, optimizer updates, R2 writes or GPU usage were added. Plot layout was revised in figures-v2; the first render and script are retained. Repository tracked files and the pre-existing untracked .codex directory were preserved.', '',
'Full precision, all predictors, counts, prevalence and both metrics: [all-metrics.csv](all-metrics.csv). Per-shift support changes and arrays: [run-v1/COMPLETE.json](run-v1/COMPLETE.json). [Verification receipt](VERIFICATION.json).', '',
'## Decision and next step', '',
'This narrow test leaves small rigid shifts and uncertain boundaries insufficient as an explanation under the supplied reference. It does not exclude larger or nonrigid misregistration, incomplete labels, historical renderer effects, depth mismatch or limitations of the model’s representation. It does not establish that CT lacks ink signal.', '',
'Run one fixed, cached CT observability audit next: examine depth-local intensity and local spatial contrast through every retained layer and its corresponding depth-triplet mean. Report all profiles without choosing a best depth or sign. Use the outcome to define one representation hypothesis and a prospectively selected, disjoint development replication, rather than another checkpoint sweep on these four regions. The [next study draft](NEXT_STUDY_DRAFT.md) is not yet executed or frozen.', '',
'GPU spend remains £0; the £10 total pilot cap remains in force. A paid workload still requires its frozen experiment, pinned CUDA environment, current all-in pricing, independent shutdown and verified worker delivery of scoped R2 credentials. No maintainer request, publication, submission, transcription or translation was made. Independent reference/exposure qualification remains unresolved.', '']
(HERE/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps(dict(status='PASS',report=pin(HERE/'RESULTS.md'),exported_metric_records=len(rows))))
