"""Export the entire verified family and a qualified scientific interpretation."""
import csv
from audit_common import *
run=json.loads((HERE/'run-v1/COMPLETE.json').read_text());v=json.loads((HERE/'VERIFICATION.json').read_text());require(run['status']==v['status']=='PASS','Verification required')
cs=[json.loads(verify(r['result']).read_text()) for r in run['contexts']]
metrics=[];summaries=[];distributions=[]
for c in cs:
    for r in c['metrics']:metrics.append(dict(context=c['id'],**{k:x for k,x in r.items() if k!='counts'},**r['counts']))
    summaries += [dict(context=c['id'],**r) for r in c['summary']]
    for r in c['distributions']:
        for cls,s in r['classes'].items():distributions.append(dict(context=c['id'],**{k:x for k,x in r.items() if k!='classes'},label_class=cls,**s))
for filename,rows in [('all-metrics.csv',metrics),('all-profile-summaries.csv',summaries),('all-class-distributions.csv',distributions)]:
    with (HERE/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
require(len(metrics)==2688 and len(summaries)==64 and len(distributions)==2688,'Complete exports')
def summary(c,scale,feature,polarity,support):
    return next(s for s in c['summary'] if (s['scale'],s['feature'],s['polarity'],s['support'])==(scale,feature,polarity,support))
lines=['# Fixed cached CT contrast audit', '',
'The fixed local-contrast feature does not expose a consistent shared depth signal in these four supplied-reference regions. On original support, neither polarity exceeds AUROC 0.6 at any of the 63 single layers or 21 depth triplets in any context. Median AUROC stays near 0.5. Intensity profiles vary more with depth and region, with changing polarity and no selected winning layer.', '',
'This is an inconclusive diagnostic for choosing a learned representation. The scalar contrast feature also performs weakly in the two regions where the retained CT backbone ranks labels well; therefore its failure cannot establish absent CT ink or dismiss learned spatial/depth structure. Follow the protocol’s inconclusive branch: prepare a concrete geometry/reference review before another model or depth sweep. Retain unchanged seed42 and the original negative averaging decision.', '',
'## Fixed experiment', '',
'All four previously scored PHerc51/Frag6 contexts were included, one parent. Each retained 63×384×384 uint16 crop produced 63 spatial 3×3 means and 21 nonoverlapping spatial/depth 3×3×3 means. The two features were intensity and intensity minus a fixed 9×9 coarse-pixel mean. Exact integer numerators were used to preserve ties, with real margins around the central 64×64 scoring area. Both signs and the original/interior supports were fixed before execution. No new labels, fitted thresholds, selected depths, model forwards, training or acquisition.', '',
'Historical layer numbers and retained TIFF intensities are not authenticated physical ink depths or calibrated attenuation. Phase30’s local reproduction of the historical interpolation behavior does not establish a globally correct optical warp or anatomical normal. Unknown teacher/model ancestry and annotation provenance remain unresolved.', '',
'## Original support: every context and both feature scales', '',
'Table values are medians across all depths, never the best depth. “+ / −” means the feature and its fixed reversed ordering. AP is prevalence-sensitive; AUROC 0.5 is the no-ranking reference. These dependent depth profiles do not supply confidence intervals or independent samples.', '',
'| Context | Feature | Ink prevalence | Single median AUROC + / − | Triplet median AUROC + / − | Single median AP + / − | Triplet median AP + / − |',
'|---|---|---:|---:|---:|---:|---:|']
for c in cs:
    for feat in ['intensity','contrast']:
        a,b,d,e=[summary(c,scale,feat,pol,'original') for scale,pol in [('single',1),('single',-1),('triplet',1),('triplet',-1)]]
        lines.append(f"| {c['id']} | {feat} | {a['prevalence']:.5f} | {a['auc_median']:.5f} / {b['auc_median']:.5f} | {d['auc_median']:.5f} / {e['auc_median']:.5f} | {a['ap_median']:.5f} / {b['ap_median']:.5f} | {d['ap_median']:.5f} / {e['ap_median']:.5f} |")
lines += ['',
'For local contrast, all 672 original-support feature/sign/depth/context combinations have AUROC ≤0.6. This descriptive cutoff was frozen; it is not a statistical test or model acceptance threshold. Triplet curves broadly follow the single-layer curves in the complete plots; this visual observation does not prove that averaging preserves every useful learned feature.', '',
'[Original AUROC profiles](figures/original-auc-profiles.png) · [Original AP profiles](figures/original-ap-profiles.png)', '',
'## Fixed interior support', '',
'The >4-pixel interior masks are exactly those from phase33. Some isolated local-contrast depths exceed AUROC 0.6 in the two sparse regions, but the first two contexts do not; there is no common consistent contrast profile across all four. All isolated responses are retained, without choosing a depth or sign. Only 37 and 167 ink pixels remain in the two sparse interiors; adjacent pixels are dependent.', '',
'| Context | Scale | Ink prevalence | Contrast median AUROC + / − | Depths AUROC >0.6, + / − | Contrast median AP + / − |',
'|---|---|---:|---:|---:|---:|']
for c in cs:
    for scale in ['single','triplet']:
        a,b=[summary(c,scale,'contrast',pol,'interior') for pol in [1,-1]]
        lines.append(f"| {c['id']} | {scale} | {a['prevalence']:.5f} | {a['auc_median']:.5f} / {b['auc_median']:.5f} | {a['AUROC_gt_point6']}/{a['total_depths']} / {b['AUROC_gt_point6']}/{b['total_depths']} | {a['ap_median']:.5f} / {b['ap_median']:.5f} |")
lines += ['',
'AP comparisons between original and interior support also change prevalence and sampled pixels. They cannot isolate an effect of boundary removal. The full summary export includes intensity and contrast on both supports, each sign and scale, interquartile ranges and both descriptive AUROC counts.', '',
'[Interior AUROC profiles](figures/interior-auc-profiles.png) · [Interior AP profiles](figures/interior-ap-profiles.png)', '',
'## Class distributions and visual review', '',
'The supplied ink and clear classes have substantially overlapping local-contrast distributions. In the later retained layers the plotted intensity distributions converge and contrast spreads narrow in all four regions. This is descriptive evidence of changing spatial structure in these arrays; it does not identify an ink-bearing surface, justify choosing a cutoff or authenticate which side of the papyrus is sampled.', '',
'[Original class distributions](figures/original-class-distributions.png) · [Interior class distributions](figures/interior-class-distributions.png). Curves show medians and shaded interquartile ranges, not confidence intervals. Feature axes are shared across contexts within each figure. All six figures were inspected by root; no separate agent review is claimed.', '',
'## Verification and resources', '',
'All 2,688 metric records are defined; none were dropped. Independent strided-slice sums and direct 81-offset box sums reproduced all 2,752,512 feature numerators exactly. There were 1,680 scalar spatial/depth pooling checks. All 1,376,256 prior pooled FP32 values were reproduced under the original scale contract. Signed-integer ranking, independent of the worker’s float encoding, reconciled all 5,376 AP/AUROC values with maximum difference 2.22×10⁻¹⁶. All 1,344 distribution records (10,752 summary values) and all 64 profile summaries were verified; maximum distribution difference was 1.46×10⁻¹¹ retained TIFF units.', '',
'Seven synthetic tests passed, none failed, errored or skipped. The initial shell launch was rejected because its new working directory did not yet exist; no command or scientific run started. The directory was created, then setup/tests succeeded. This is preserved in SETUP_RECEIPT.json. The scientific worker completed its sole attempt in 2.329 seconds, peak RSS 92,049,408 bytes; independent verification took 1.751 seconds, peak RSS 97,697,792 bytes. Both meet the 120-second/1-GiB limits. No model inference, optimizer updates, source requests, R2 uploads or paid resources occurred.', '',
'Raw metric records: [all-metrics.csv](all-metrics.csv). Every profile summary: [all-profile-summaries.csv](all-profile-summaries.csv). Class distributions: [all-class-distributions.csv](all-class-distributions.csv). [Verification](VERIFICATION.json). The frozen [protocol](PROTOCOL.md) and [GO](GO.json) identify exact sources and execution code.', '',
'## Decision', '',
'Keep the baseline and stop feature/depth sweeps on these four contexts. These scalar features do not establish a representation change worth scaling. Equally, they do not rule out useful learned texture or joint depth/spatial evidence. No model promotion, recovered text, translation or independent accuracy claim follows.', '',
'Next prepare a cached structural review packet that separates source-value reproduction, physical surface/depth interpretation and optical registration. Include all four regions, exact coordinates and the complete layer sequence or a prospectively fixed uniform montage, with a structured landmark form. Only genuinely shared non-ink landmarks or attributable curator evidence can strengthen the registration claim; if neither exists, keep it unresolved. A proposed correction would need held-out landmark checks. See the [next study draft](NEXT_STUDY_DRAFT.md). It has not been executed.', '',
'GPU spend remains £0 and the £10 total first-pilot cap is unchanged. No maintainer request has been sent; no new reply was supplied. The other parent reserves remain untouched. R2 worker delivery remains unverified; a paid launch still needs a useful frozen workload, pinned CUDA, current all-in pricing and independent shutdown.', '']
(HERE/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps(dict(status='PASS',report=pin(HERE/'RESULTS.md'),metric_rows=len(metrics),summary_rows=len(summaries),class_distribution_rows=len(distributions))))
