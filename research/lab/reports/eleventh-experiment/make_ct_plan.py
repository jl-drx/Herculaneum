"""Create the concrete bounded CT plan after reader/review freeze, before live requests."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


inventory = ROOT / 'optical-reference/asset-inventory.json'
assets = json.loads(inventory.read_text())['assets']
layers = []
for index in range(7, 58):
    choices = [a for a in assets if a['name'] == f'{index:02d}.tif']
    assert len(choices) == 1
    layers.append(dict(index=index, url=choices[0]['url']))
selection = json.loads((ROOT / 'reference-selection/selection-result.json').read_text())
crops = []
for case in selection['selected']:
    y,x,h,w = case['native_bbox_yxhw']
    crops.append(dict(case_id=case['id'], bbox_source_xywh=[x,y,w,h]))
pins = ['RESEARCH_DESIGN.md','ct-acquisition/reader.py','reference-selection/selection-plan.json',
        'reference-selection/select_contexts.py','reference-selection/selection-result.json',
        'review/comparison/selection-verification.json','optical-depth/sampling-contract.json',
        'optical-reference/legacy-surface-meta.json','optical-reference/legacy-surface-render.json',
        'optical-reference/legacy-LICENSE.current.txt',
        'optical-reference/annotation-provenance/ANNOTATION_AND_DEPTH_ADDENDUM.md',
        'optical-reference/annotation-provenance/educelab-arxiv-v4-complete.html',
        'optical-acquisition/tiff-structural-observations.json','make_ct_plan.py']
plan = dict(schema_version=1,ready=True,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            input_pins=[record(ROOT/n) for n in pins],source_inventory=record(inventory),
            source_layers=layers,expected_tiff=dict(width=6064,height=9791),crops=crops,
            pooling=dict(factor_zyx=[3,3,3],formula='float32(sum_uint32_27/(27*257))',quantization='none'),
            limits=dict(maximum_http_attempts=260,maximum_charged_response_bytes=1717986918,
                        maximum_per_range_bytes=14680064,maximum_peak_rss_bytes=1073741824),
            expected_CT_row_band_bytes=1425088512,expected_http_attempts=255,
            scientific_role='Frozen Frag4 supplied-reference comparison; no training or pixel truth/novelty claim',
            plane_centers_source_indices=list(range(8,57,3)),declared_source_surface_index=32,
            nominal_spacing_um=9.72,nominal_plane_offsets_um=[(i-32)*3.24 for i in range(8,57,3)],
            outputs_are_private=True,paid_resources=False)
destination = ROOT / 'ct-acquisition/acquisition-plan.json'
with destination.open('x') as stream:
    json.dump(plan,stream,indent=2); stream.write('\n')
print(json.dumps(record(destination)))
