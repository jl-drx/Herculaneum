"""Bind a focused source review without inspecting scientific payload arrays."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'allocation-implementation'


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(record):
    assert pin(record['path'])==record,record['path']


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    path=SOURCE/'final-plan/experiment-plan.json'
    plan=json.loads(path.read_text())
    assert pin(path)['sha256']=='2e1cc8c46d5f7aaade31916cdc0304cd6cb8268e49d8aea73927c730dd535f74'
    assert plan['protocol']['sha256']=='498bf89ca0944ca26f596bdc8041951a82365667ccddf4eb88a8a1f0b33998b1'
    assert plan['arms']==['manual_double','manual_retired']
    for key in ('protocol','configuration','code_freeze','preservation_support','support_source_seal','independent_support_summary','independent_support_seal','prior_fusion_plan','historical_phase20_metrics'):
        verify(plan[key])
    for record in [*plan['code'].values(),*plan['source_code'].values()]:verify(record)
    assert set(plan['code'])=={p.name for p in SOURCE.glob('*.py')}
    for p in SOURCE.glob('*.py'):ast.parse(p.read_text())
    freeze=json.loads(Path(plan['code_freeze']['path']).read_text())
    assert freeze['no_further_edits'] and freeze['code']==plan['code']
    assert freeze['protocol']==plan['protocol'] and freeze['configuration']==plan['configuration']
    config=json.loads(Path(plan['configuration']['path']).read_text())
    assert all(plan[k]==config[k] for k in ('architecture','training','limits','gates'))
    assert config['architecture']['fixed_feature_mode']=='explicit_depth' and 'raw_control' not in config['architecture']
    assert config['training']['kl_coefficients']=={'manual_double':[2.,0.],'manual_retired':[1.,1.]}
    verify(freeze['fixtures'])
    fixture=json.loads(Path(freeze['fixtures']['path']).read_text())
    assert fixture['status']=='PASS' and fixture['check_count']==44 and fixture['source_code']==plan['code']
    own=json.loads((HERE/'allocation-controls-v1.json').read_text())
    assert own['status']=='PASS' and own['check_count']==36
    for record in own['source_pins']:
        verify(record);assert record==plan['code'][Path(record['path']).name]
    verify(own['fixture_source']);verify(own['synthetic_support_file'])
    go=dict(kind='phase21-allocation-independent-source-review-v1',status='GO',plan=pin(path),
            protocol=plan['protocol'],configuration=plan['configuration'],code_freeze=plan['code_freeze'],
            reviewed_code=plan['code'],reviewed_source_code=plan['source_code'],
            support_contract=plan['preservation_support'],support_seal=plan['support_source_seal'],
            independent_support=plan['independent_support_summary'],independent_support_seal=plan['independent_support_seal'],
            independent_fixtures=pin(HERE/'allocation-controls-v1.json'),author_fixtures=freeze['fixtures'],
            review_report=pin(HERE/'TECHNICAL_REVIEW.md'),outstanding_source_issues=[],
            resolved_preexecution_issue='Extended support descriptor canonicalized to strict file pin with separate shape/dtype/raw hash checks; synthetic acceptance and corruption rejection passed.',
            separate_root_execution_go_required=True,independent_training_pass_before_development_required=True,
            permitted_scope='Exact frozen CPU allocation experiment; both final arms and all six reused development contexts after independent training verification.',
            actual_model_forwards=0,checkpoint_reads=0,scientific_array_reads=0,optimizer_steps=0,network_calls=0,
            paid_resource_authorization=False,
            limitations=['Focused source review and finite synthetic controls, not convolution/autograd or training replay.',
                         'Retired supplied V role is not authenticated geometry support or fresh validation.',
                         'Different own objectives are not a common full-risk endpoint; saved numerical checks remain required.'])
    save(HERE/'TECHNICAL_GO.json',go)
    records=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='retention-manifest.json']
    save(HERE/'retention-manifest.json',dict(kind='phase21-allocation-technical-review-retention-v1',
        records=records,record_count=len(records),total_bytes=sum(r['size_bytes'] for r in records),no_further_edits=True))
    print(json.dumps(dict(status='GO',technical_go=pin(HERE/'TECHNICAL_GO.json'),retention=pin(HERE/'retention-manifest.json'))))


if __name__=='__main__':main()
