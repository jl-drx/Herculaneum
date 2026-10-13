"""Verify uploaded audit against its replay and retained local source identities."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;A=P/'independent-audit';B=P/'audit-replay';R=P.parent
checks={}
for r in json.loads((A/'packet/INPUT_MANIFEST.json').read_text())['files']:
 f=Path(r['source_path']);b=f.read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],str(f)
checks['actual_retained_source_files_verified']=len(json.loads((A/'packet/INPUT_MANIFEST.json').read_text())['files'])
a=json.loads((A/'output-locked/independent-results.json').read_text());b=json.loads((B/'output-locked/independent-results.json').read_text())
ea=a.pop('environment');eb=b.pop('environment');assert a==b
maskarrays=0;points=0;context_pixels=0
for row in a['meshes']:
 s=row['segment'];x=np.load(A/'output-locked'/f'{s}-masks.npz');y=np.load(B/'output-locked'/f'{s}-masks.npz');assert x.files==y.files
 for k in x.files:assert np.array_equal(x[k],y[k]),(s,k);maskarrays+=1
 original=np.load(R/'fortyeighth-experiment/support-results'/f'{s}.npz')
 points+=int((x['point_support']!=original['point_support']).sum())
 assert np.array_equal(x['all_origin_context'],original['context_core'])
 context_pixels+=x['all_origin_context'].size
assert points==9
candidate=a['fixed_grid_selected'];assert candidate['segment']=='20251028225813' and candidate['component']==1 and candidate['fixed_grid_square']['xywh']==[1024,256,192,192]
checks.update(status='PASS',replayed_meshes=5,replayed_component_results=len(a['components_ranked']),all_non_environment_metrics_exact=True,replayed_mask_arrays=maskarrays,archived_independent_point_discrepancies=points,conservative_context_pixels_compared=context_pixels,all_conservative_contexts_identical=True,archived_point_receipts=len(a['receipts']),locked_versions_match=all(ea[k]==eb[k] for k in ['numpy','scipy','tifffile']),replay_seconds=eb['elapsed_seconds'],replay_peak_rss_kib=eb['max_rss_kib'])
(P/'AUDIT_VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
