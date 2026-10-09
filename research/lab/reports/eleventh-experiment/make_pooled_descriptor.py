"""Bind completed, independently reconstructed CT inputs to frozen label masks."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha)


spec = importlib.util.spec_from_file_location('reference_schema', ROOT / 'comparison/prepare_reference.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
provenance_path = ROOT / 'ct-acquisition/run/provenance.json'
provenance = json.loads(provenance_path.read_text())
assert provenance['status'] == 'PASS' and provenance['http_attempts'] == 255
assert provenance['retained_body_bytes'] == provenance['charged_response_bytes']
assert 1425088512 < provenance['retained_body_bytes'] <= 1717986918
check_path = ROOT / 'review/comparison/retained-ct-verification.json'
check = json.loads(check_path.read_text())
assert check['status'] == 'PASS' and check['provenance'] == record(provenance_path)
selection_path = ROOT / 'reference-selection/selection-result.json'
selection = json.loads(selection_path.read_text())
cases = []
for case in selection['selected']:
    matched = [output for output in provenance['outputs'] if output['case_id'] == case['id']]
    assert len(matched) == 1
    output = matched[0]
    y,x,h,w = case['native_bbox_yxhw']
    assert output['bbox_source_xywh'] == [x,y,w,h] and output['shape_zyx'] == [17,384,384]
    assert check['outputs'][case['id']]['bitwise_float32_equal'] is True
    arrays = {'surface': output['file']}
    for key,name in [('labels','labels.npy'),('primary_mask','primary-mask.npy'),
                     ('known_mask','known-mask.npy'),('surface_mask','surface-mask.npy')]:
        original = next(r for r in case['files'] if Path(r['path']).name == name)
        assert original == record(original['path'])
        arrays[key] = original
    cases.append(dict(id=case['id'], native_bbox_yxhw=case['native_bbox_yxhw'], arrays=arrays))
descriptor = dict(status='pooled_reference_complete_no_inference', preprocessing=module.PREPROCESSING,
                  cases=cases, CT_acquisition=record(provenance_path), CT_ledger=provenance['request_ledger'],
                  CT_plan=provenance['plan'], independent_CT_review=record(check_path),
                  selection=record(selection_path), research_design=record(ROOT/'RESEARCH_DESIGN.md'),
                  descriptor_builder=record(__file__),
                  comparison_code_freeze=record(ROOT/'comparison/code-freeze.json'),
                  source_protocol=record(ROOT/'optical-reference/annotation-provenance/ANNOTATION_AND_DEPTH_ADDENDUM.md'),
                  reserved_specimen='PHercParis1Fr39/Frag4; no current training, teacher targets or calibration',
                  claim='Conditional agreement with supplied optical-derived labels; no complete historical holdout or registration accuracy claim')
runtime_path = ROOT / 'runtime-receipt.json'
runtime = json.loads(runtime_path.read_text())
descriptor['runtime'] = record(runtime_path)
descriptor['package_metadata'] = [item['metadata'] for item in runtime['packages'].values()]
descriptor['root_inference_review'] = record(ROOT / 'INFERENCE_REVIEW.md')
target = ROOT / 'pooled-reference-descriptor.json'
with target.open('x') as stream:
    json.dump(descriptor,stream,indent=2);stream.write('\n')
print(json.dumps(record(target)))
