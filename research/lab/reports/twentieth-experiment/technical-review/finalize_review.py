"""Bind source-only technical review; never inspect scientific payload files."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
IMPLEMENTATION = PHASE / 'implementation'


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(pin):
    assert record(pin['path']) == pin, pin['path']


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    plan_path = IMPLEMENTATION / 'final-plan/experiment-plan.json'
    plan = json.loads(plan_path.read_text())
    assert record(plan_path)['sha256'] == '58bb5c64ad64db19995c87c84d1e6a1ce6585d380afeaa7ef347ac82b138d78a'
    for pin in [plan['protocol'], plan['configuration'], plan['code_freeze'], *plan['code'].values(), *plan['source_code'].values()]:
        verify(pin)
    freeze = json.loads(Path(plan['code_freeze']['path']).read_text())
    config = json.loads(Path(plan['configuration']['path']).read_text())
    assert freeze['no_further_edits'] and freeze['code'] == plan['code']
    assert freeze['configuration'] == plan['configuration'] and freeze['protocol'] == plan['protocol']
    assert all(plan[k] == config[k] for k in ('architecture', 'training', 'limits', 'gates'))
    assert set(plan['code']) == {p.name for p in IMPLEMENTATION.glob('*.py')}
    for p in IMPLEMENTATION.glob('*.py'):
        ast.parse(p.read_text())
    verify(freeze['fixtures'])
    authored = json.loads(Path(freeze['fixtures']['path']).read_text())
    assert authored['status'] == 'PASS' and authored['source_code'] == plan['code']
    own = json.loads((HERE / 'synthetic-controls-v2.json').read_text())
    assert own['status'] == 'PASS' and own['check_count'] == 44
    for name, pin in own['source_pins'].items():
        verify(pin)
        assert pin == plan['code'][name]
    verify(own['fixture_source'])
    go = dict(kind='phase20-independent-source-technical-review-v1', status='GO',
              plan=record(plan_path), protocol=plan['protocol'], configuration=plan['configuration'],
              code_freeze=plan['code_freeze'], reviewed_code=plan['code'],
              reviewed_source_code=plan['source_code'],
              review_report=record(HERE / 'TECHNICAL_REVIEW.md'),
              independent_synthetic_controls=record(HERE / 'synthetic-controls-v2.json'),
              retained_fixture_stop=record(HERE / 'synthetic-controls-v1-stop.json'),
              author_final_synthetic_controls=freeze['fixtures'],
              numerical_contract=record(PHASE / 'independent-review/NUMERICAL_CONTRACT.md'),
              outstanding_source_issues=[],
              requested_corrections=[dict(issue='Literal forked-RNG initialization contract',
                  resolution='Both stages use unchanged phase19 matched_adapter factory before source freeze', status='RESOLVED')],
              permitted_scope='Root-authorized bounded CPU experiment only; final development after independent training PASS',
              actual_checkpoint_reads=0, scientific_array_reads=0, actual_model_executions=0,
              reviewer_synthetic_fixture_attempts=2, reviewer_total_synthetic_optimizer_steps=4,
              network_calls=0, gpu_calls=0, paid_resource_authorization=False,
              limitations=['Finite synthetic controls and source review, not an independent convolution/autograd replay.',
                           'Actual outputs, numeric consistency and frozen execution remain subject to independent saved-output review.',
                           'All six contexts remain reused development; no scientific efficacy or confirmation claim.'])
    save(HERE / 'TECHNICAL_GO.json', go)
    pins = [record(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'retention-manifest.json']
    save(HERE / 'retention-manifest.json', dict(kind='phase20-technical-review-retention-v1', records=pins,
         record_count=len(pins), total_bytes=sum(p['size_bytes'] for p in pins), no_further_edits=True))
    print(json.dumps(dict(status='GO', technical_go=record(HERE / 'TECHNICAL_GO.json'),
        retention=record(HERE / 'retention-manifest.json'))))


if __name__ == '__main__':
    main()
