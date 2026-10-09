"""Seal the completed author audit; no scientific inputs are opened by this writer."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def record(path):
    p=Path(path).resolve(strict=True)
    with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),sha256=d,size_bytes=p.stat().st_size)


def write(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def main():
    contract_path=ROOT/'audit-v2/support-contract.json';v=json.loads(contract_path.read_text())
    write('SUPPORT_OPTIONS.json',dict(kind='phase21-preservation-role-support-options-v1',status='AUDIT_COMPLETE_NOT_A_TRAINING_PROTOCOL',
        source_contract=record(contract_path),source_schedule=next(p for p in v['source_pins'] if p['path'].endswith('/schedule.json')),
        manual_batch_mask=v['batch_arrays']['manual'],retired_validation_batch_mask=v['batch_arrays']['retired_validation'],
        diagnostic_unknown_batch_mask=v['batch_arrays']['annotation_unknown'],batch_source_index=v['batch_arrays']['source_index'],
        region_ids=[r['id'] for r in v['regions']],source_index_encoding=v['source_index_encoding'],
        all32_batches_and_all64_individual_patches_have_nonempty_manual_and_retired_membership=True,
        minimum_manual_batch_pixels=8372,maximum_manual_batch_pixels=30732,
        minimum_retired_validation_batch_pixels=2036,maximum_retired_validation_batch_pixels=24396,
        minimum_manual_patch_pixels=4008,maximum_manual_patch_pixels=15920,
        minimum_retired_validation_patch_pixels=464,maximum_retired_validation_patch_pixels=12376,
        supervision_validation_intersection_pixels=0,
        interpretation=dict(M='acquired & supervision & ~validation',V='acquired & validation; all V is outside supervision in these two contexts',
            U='acquired & ~supervision & ~validation; no target participation established',
            candidate_role_union='M | V, explicit released-reference participation, not geometric validity'),
        candidate_loss_definitions=dict(A='FP32 mean BCE_M(smoothed y=.25+.5*y) + 2*FP64 mean KL_M(seed42||student)',
            B='Same BCE_M + FP64 mean KL_M(seed42||student) + FP64 mean KL_V(seed42||student)'),
        candidate_loss_scope='Root proposed; needs a separate frozen execution protocol. Each mean pools both scheduled patches in a batch. No V ink target contributes.',
        reduction_interpretation='Sum of KL group coefficients 2 in both; B halves the manual KL derivative and reallocates one group coefficient to retired V. Not uniform union mean or unchanged manual strength.',
        broader_geometry_valid_support_ready=False,geometry_valid_support_mask=None,
        no_new_unlabeled_supervised_targets=True,scientific_execution_performed=False,
        fixtures=record(ROOT/'reallocation-fixtures-v1.json'),
        forward_compatibility='This audit read no inklabel arrays. A future runner may read whole historical label/pseudo arrays for exact fingerprint verification; then claim only that V labels do not participate in objective/teacher/selection, not that their bytes were unread.'))
    write('SOURCE_REVISION_RECEIPT.json',dict(
        support_v1=dict(original_path=str(ROOT/'audit_support.py'),preserved_source=record(ROOT/'audit_support.v1.py'),
            receipt=record(ROOT/'audit-v1/support-contract.json'),status='PASS',
            issue='An unused stale PLAN_SHA constant was present; the audit did record and hash the actual retained training plan.',
            correction='v2 binds the actual known plan SHA and checks it before reading mask arrays; all eight output arrays and counts agree with v1.'),
        support_v2=dict(source=record(ROOT/'audit_support.py'),receipt=record(contract_path),status='PASS'),
        objective_v1=dict(original_path=str(ROOT/'check_objectives.py'),preserved_source=record(ROOT/'check_objectives.v1.py'),
            stop_receipt=record(ROOT/'objective-fixtures-v1-stop.json'),
            issue='The synthetic oracle asserted exact zero on a mathematically zero stratified FP64 KL derivative; floating point summation left roundoff.',
            correction='v2 uses explicit 1e-15 identity derivative tolerance. Exact identity loss and exact unsupported gradient checks remain unchanged; objective kernel unchanged.'),
        objective_v2=dict(source=record(ROOT/'check_objectives.py'),receipt=record(ROOT/'objective-fixtures-v2.json'),status='PASS'),
        scientific_training_runs=0))
    old=json.loads((ROOT/'audit-v1/support-contract.json').read_text())
    assert old['totals']==v['totals']
    for key,pin in v['batch_arrays'].items():assert pin['raw_sha256']==old['batch_arrays'][key]['raw_sha256']
    entries=[record(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='retention-manifest.json' and '__pycache__' not in p.parts]
    write('retention-manifest.json',dict(status='SEALED',scope=str(ROOT),records=entries,record_count=len(entries),
        total_bytes=sum(p['size_bytes'] for p in entries),no_further_author_edits=True,
        source_reads='8 PHerc0139 masks per support audit; no CT/inklabel/teacher/prediction arrays, checkpoints, models, optimizers or network.',
        limitations='Geometry validity remains unavailable. Synthetic fixtures demonstrate mechanics only. Root owns any subsequent scientific design.'))
    print(json.dumps(dict(status='SEALED',manifest=record(ROOT/'retention-manifest.json'),contract=record(contract_path),options=record(ROOT/'SUPPORT_OPTIONS.json'))))

if __name__=='__main__':main()
