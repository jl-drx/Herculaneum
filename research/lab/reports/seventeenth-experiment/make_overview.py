"""Present already-computed fixed diagnostic metrics; no new model/data fitting."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/herculaneum-phase17-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
def record(p):
    p = Path(p).resolve()
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(relative, sha=None):
    p = HERE / relative
    ref = record(p)
    if sha is not None:
        assert ref['sha256'] == sha, relative
    inputs.append(ref)
    return json.loads(p.read_text())

inputs = []
ct = read('implementation/ct-run/ct-diagnostics.json', '6e3930f01b2310f479c92d206cadd514e5b48fc7497ed7518b8634a670629fbf')
ir = read('implementation/ir-run/ir-diagnostics.json', '4083f0b3100eb35a512655252b0d3c37a2c0239ce683daf26ee9ddd5aead2b27')
old = read('../sixteenth-experiment/inference/analysis/metrics.json', 'c911803a07ac44741eb9724b2e4b78dbfd8cb32f92848378960960f17d1ee175')
ppm = read('independent-review/ppm-reconstruction/independent-ppm-diagnostics.json')
assert ct['status'] == ir['status'] == ppm['status'] == 'PASS'
ir_cases = {c['id']: c for c in ir['cases']}
rows = []
for c in ct['cases']:
    ident = c['id']
    profiles = {p['plane_index']: p['supports'] for p in c['native_profiles']}
    model = old['per_context'][ident]['models']['frozen_seed42']
    geometries = [p['geometry'] for obj in ppm['objects'] if obj['variant'] == 'result'
                  for p in obj['points'] if p['case_id'] == ident]
    assert len(geometries) == 81
    geometry_summary = {}
    for field in ['du_norm', 'dv_norm', 'cross_product_area', 'signed_normal_dot', 'tangent_dot']:
        values = [g[field] for g in geometries if g.get(field) is not None]
        geometry_summary[field] = dict(count=len(values), minimum=min(values) if values else None,
                                      median=float(np.median(values)) if values else None,
                                      maximum=max(values) if values else None)
    rows.append(dict(id=ident, role=c['role'], baseline_CT_AUROC=model['auc'], baseline_CT_AP=model['ap'],
                     IR_darkness=ir_cases[ident]['darkness'], IR_local_darkness=ir_cases[ident]['local_darkness'],
                     surface_native_SD_by_fixed_plane={str(k): profiles[k]['surface']['std'] / 257.0 for k in [8,20,32,44,56]},
                     fixed_native32_positive_minus_negative_mean=(profiles[32]['primary_positive']['mean']-profiles[32]['primary_negative']['mean'])/257.0,
                     valid_result_PPM_centers=sum(g['valid_center'] for g in geometries),
                     sampled_result_PPM_geometry=geometry_summary))

out = HERE / 'overview'
out.mkdir(exist_ok=False)
summary = dict(status='descriptive presentation', inputs=inputs, cases=rows,
               interpretation='IR labels derive from registered IR; high optical agreement is not independent CT registration. PPM geometry conditional on frozen serialization invariants. No new prediction or fitted method.')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout='constrained')
labels = [c['id'].replace('frag','F').replace('-context',' / C') for c in rows]
y = np.arange(6)
for delta, key, name, color in [(-.21, 'baseline_CT_AUROC', 'Saved CT model', '#246a91'),
                               (0, 'IR_darkness', 'IR darkness (label-derived reference)', '#42865a'),
                               (.21, 'IR_local_darkness', 'Fixed local IR darkness', '#b57b2a')]:
    values = [r[key]['auroc'] if isinstance(r[key],dict) else r[key] for r in rows]
    axes[0].barh(y+delta, values, height=.19, label=name, color=color)
axes[0].set(yticks=y, yticklabels=labels, xlim=(0,1.02), xlabel='AUROC on unchanged primary support',
            title='Optical label agreement remains strong')
axes[0].invert_yaxis()
axes[0].axvline(.5, color='gray', linestyle=':', linewidth=1)
axes[0].legend(fontsize=8, loc='lower left')
axes[0].text(.02,-.19,'F1/F2: Paris2 evaluation; F3: Paris1 calibration.\nIR reference is not a CT model or independent ground truth.',transform=axes[0].transAxes,fontsize=8)
for c, label in zip(ct['cases'], labels):
    p = c['native_profiles']
    axes[1].plot([v['nominal_depth_um'] for v in p], [v['supports']['surface']['std']/257 for v in p], label=label)
axes[1].axvline(0,color='gray',linestyle=':',linewidth=1)
axes[1].set(xlim=(-81,81), ylim=(0,45), xlabel='Nominal depth: 3.24 × (plane − 32), µm',
            ylabel='Native surface-support population SD / 257', title='Shared reduction in texture at positive offsets')
axes[1].legend(fontsize=8, ncol=2)
axes[1].text(.02,-.19,'All 51 acquired depths shown; no depth selected or optimized.\nNominal zero is not proof of the microscopic ink surface.',transform=axes[1].transAxes,fontsize=8)
fig.suptitle('Plinius: regional alignment and depth audit',fontsize=14)
fig.savefig(out/'regional-overview.png',dpi=180)
fig.savefig(out/'regional-overview.svg')
plt.close(fig)
receipt = dict(status='PASS', source=record(__file__), inputs=inputs,
               outputs=[record(out/n) for n in ['summary.json','regional-overview.png','regional-overview.svg']],
               runtime=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
               model_calls=0,network_calls=0)
(out/'presentation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt=record(out/'presentation-receipt.json'))))
