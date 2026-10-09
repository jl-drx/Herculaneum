"""Freeze the completed local context package before new acquisition; never fetch."""
from datetime import datetime, timezone
import json
from pathlib import Path

from acquire_context_v2 import CONTRACT, require, sha, validate_plan, validate_decision, write_json

LAB = Path('/workspace/herculaneum-lab')
HERE = Path(__file__).resolve().parent
ACQUISITION = LAB / 'reports/fifth-experiment/context-selection-v2/result/acquisition-plan.json'
ACQUISITION_SHA = 'e27d2fe30e14221cfd27a7507cd709a1d9600c6a081b777424dab1a859df1f3b'
DECISION = LAB / 'reports/fifth-experiment/context-decision.json'
DECISION_SHA = 'e202529d4958bd05eab60727cd7e57c349d68d43c69453ce81d8cd2c4ec53ae6'


def record(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, 'Freeze input differs: ' + path.name)
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def main():
    require(not (HERE/'inference-plan.json').exists() and not (HERE/'package-manifest.json').exists(),
            'Preserve frozen package; revisions require a new coordinated version')
    validate_plan(ACQUISITION, ACQUISITION_SHA)
    decision = validate_decision(DECISION, DECISION_SHA, ACQUISITION, ACQUISITION_SHA)
    old_plan = json.loads((LAB/'reports/fourth-experiment/inference-plan.json').read_text())
    verified = [record(r['path'], r['sha256']) for r in old_plan['verified_inputs']
                if Path(r['path']).name != 'run_screen.py']
    for path in [HERE/'acquire_context_v2.py', HERE/'run_context.py', HERE/'analyze_context.py',
                 HERE/'freeze_context.py', DECISION,
                 LAB/'reports/fourth-experiment/runs/execution.json',
                 LAB/'reports/fourth-experiment/runs/center/metadata.json']:
        verified.append(record(path))
    for role in ['technical_mapping', 'visual_review', 'recovery_manifest', 'geometry_review']:
        verified.append(record(decision[role]['path'], decision[role]['sha256']))
    plan = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        status='Frozen before expanded source acquisition or inference; post hoc context containing fourth field',
        acquisition_plan=record(ACQUISITION, ACQUISITION_SHA), context_decision=record(DECISION, DECISION_SHA),
        acquisition_contract=CONTRACT, conditions=old_plan['conditions'], verified_inputs=verified,
        device='cpu', threads=4, maximum_scheduled_tiles_per_model=121, maximum_scheduled_tiles=605,
        automatic_retries=0, per_condition_timeout_seconds=120,
        analysis=dict(crop_margin_pixels=128,
            common_domain='Geometry-valid AND observed HTTP200 source AND all5 outputs evaluated AND any nonzero selected17 profile in every condition',
            strict_domain='Exclude entire support of every scored128tile containing invalid, unobserved, or zero selected17 profile cells in any condition',
            score_midpoint=.5, score_display_limits=[.25,.75], source_display_limits=[0,255],
            repeatability='Exact full fourth384 source byte equality; central fourth256 center seed42/43 score raw-bit and evaluated/validity-mask equality at identical native tile phase',
            interpretation='Post hoc surrounding-context model diagnostics; no labels, independent accuracy, calibrated ink confidence or text'),
        physical_independence='Unresolved; larger context contains the fourth field',
        letter=None, transcription=None, translation=None)
    write_json(HERE/'inference-plan.json', plan)
    manifest = dict(frozen_at_utc=plan['frozen_at_utc'], inference_plan=record(HERE/'inference-plan.json'),
                    acquisition_plan=plan['acquisition_plan'], context_decision=plan['context_decision'],
                    verified_inputs=verified, no_further_edits_after_review=True,
                    original_fourth_files_preserved=True, network_requests_during_preparation=0)
    write_json(HERE/'package-manifest.json', manifest)
    print(json.dumps(dict(inference_plan_sha256=manifest['inference_plan']['sha256'],
                         package_manifest_sha256=sha(HERE/'package-manifest.json'),
                         code_sha256={Path(r['path']).name:r['sha256'] for r in verified if Path(r['path']).parent==HERE})))


if __name__ == '__main__':
    main()
