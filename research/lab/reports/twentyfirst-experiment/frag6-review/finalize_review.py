"""Seal exact-source Stage A technical review without any live transport."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'frag6-stage-a'


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(record):
    assert pin(record['path'])==record,record['path']


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    path=SOURCE/'execution-plan.json'
    plan=json.loads(path.read_text())
    assert pin(path)['sha256']=='ae44baaed05f7faf85ee7fa592b88cc9d2931cf3a76667a914c2fa4b738bedeb'
    verify(plan['proposal'])
    proposal=json.loads(Path(plan['proposal']['path']).read_text())
    assert plan['objects']==proposal['present_evidence']['objects']
    assert plan['stage_rules']==proposal['stages'][0]
    assert plan['stage']=='A_headers' and plan['root_live_execution_go_required'] is True
    assert plan['scientific_payload_decoding_authorized'] is False
    for record in plan['code']:
        verify(record);ast.parse(Path(record['path']).read_text())
    verify(plan['synthetic_fixtures'])
    author=json.loads(Path(plan['synthetic_fixtures']['path']).read_text())
    assert author['status']=='PASS' and author['test_count']==102
    assert author['code']==plan['code'] and author['network_calls']==0
    receipts={}
    for name in ('parser-controls-v2.json','transport-controls-v2.json','complete-budget-controls-v1.json'):
        value=json.loads((HERE/name).read_text())
        assert value['status']=='PASS' and value['live_requests']==0
        for record in value['source_pins']:verify(record)
        verify(value['fixture_source'])
        receipts[name]=pin(HERE/name)
    assert json.loads((HERE/'complete-budget-controls-v1.json').read_text())['plan']==pin(path)
    go=dict(schema='herculaneum.frag6-stage-a-independent-technical-review.v1',status='GO',
            stage='A_headers',plan=pin(path),plan_sha256=pin(path)['sha256'],
            proposal=plan['proposal'],code=plan['code'],author_fixtures=plan['synthetic_fixtures'],
            independent_fixtures=receipts,review_report=pin(HERE/'TECHNICAL_REVIEW.md'),
            retained_reviewer_fixture_stop=pin(HERE/'transport-controls-v1-stop.json'),
            outstanding_source_issues=[],
            allow_live_requests=False,separate_root_live_execution_go_required=True,
            scope='Technically ready for exact frozen Stage A only; root owns live authorization and interpretation.',
            limits=plan['limits'],live_requests=0,scientific_arrays_read=0,
            numerical_ppm_records_decoded=0,image_pixels_decoded=0,model_calls=0,
            paid_resource_authorization=False,
            limitations=['Fake responses and source review do not establish actual server behavior or storage validity.',
                         'No process-crash/lost-success-response test or total-wire-byte claim.',
                         'No geometry/source-volume/normal/depth interpretation or Stage B/C authorization.'])
    save(HERE/'TECHNICAL_GO.json',go)
    records=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='retention-manifest.json']
    save(HERE/'retention-manifest.json',dict(schema='herculaneum.frag6-stage-a-review-retention.v1',
         records=records,record_count=len(records),total_bytes=sum(p['bytes'] for p in records),
         no_further_edits=True))
    print(json.dumps(dict(status='GO',technical_go=pin(HERE/'TECHNICAL_GO.json'),
                         retention=pin(HERE/'retention-manifest.json'))))


if __name__=='__main__':main()
