"""Freeze deterministic geometry/data-only contexts before model inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

LAB = Path('/workspace/herculaneum-lab')
REPORT = Path(__file__).resolve().parent
SURFACE = LAB / 'data/experiments/pherc0800-surface980/surface.npy'
SOURCE_PROVENANCE = SURFACE.parent / 'provenance.json'
MASK = LAB / 'data/experiments/pherc0800-surface-geometry/full-surface-valid-mask.npy'
GEOMETRY_SUMMARY = MASK.parent / 'full-surface-geometry-summary.json'
FORBIDDEN = [[256,256,256,256], [512,448,256,256]]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def overlap(a, b):
    x,y,w,h = a
    xx,yy,ww,hh = b
    return max(x,xx) < min(x+w,xx+ww) and max(y,yy) < min(y+h,yy+hh)


def main():
    plan_path = REPORT / 'independent-selection-plan.json'
    if plan_path.exists():
        raise SystemExit('Existing frozen plan is preserved; do not replace selections.')
    source_provenance = json.loads(SOURCE_PROVENANCE.read_text())
    geometry = json.loads(GEOMETRY_SUMMARY.read_text())
    surface_sha, mask_sha = sha(SURFACE), sha(MASK)
    assert surface_sha == source_provenance['array_sha256']
    expected_mask = next(entry for entry in geometry['derived_outputs'] if entry['file'] == MASK.name)
    assert mask_sha == expected_mask['sha256']
    surface = np.load(SURFACE, mmap_mode='r', allow_pickle=False)
    mask = np.load(MASK, mmap_mode='r', allow_pickle=False)
    assert surface.shape == (31,980,980) and surface.dtype == np.uint8
    assert mask.shape == (980,980) and mask.dtype == np.bool_
    observed = np.any(surface[7:24] != 0, axis=0)
    candidates, discarded_overlap, discarded_coverage = [], 0, 0
    for y in range(0, surface.shape[1]-256+1, 64):
        for x in range(0, surface.shape[2]-256+1, 64):
            bbox = [x,y,256,256]
            if any(overlap(bbox, old) for old in FORBIDDEN):
                discarded_overlap += 1
                continue
            valid = mask[y:y+256,x:x+256]
            count = int(valid.sum())
            fraction = count / 65536
            observed_count = int((valid & observed[y:y+256,x:x+256]).sum())
            observed_fraction = observed_count/count if count else 0.0
            if fraction < 0.5 or observed_fraction < 0.95:
                discarded_coverage += 1
                continue
            candidates.append({'bbox_source_xywh':bbox, 'geometry_valid_pixels':count,
                               'geometry_valid_fraction':fraction,
                               'observed_valid_pixels':observed_count,
                               'observed_fraction_among_geometry_valid':observed_fraction})
    candidates.sort(key=lambda c:(-c['geometry_valid_fraction'], -c['observed_fraction_among_geometry_valid'], c['bbox_source_xywh'][1], c['bbox_source_xywh'][0]))
    selected = []
    for candidate in candidates:
        if any(overlap(candidate['bbox_source_xywh'], earlier['bbox_source_xywh']) for earlier in selected):
            continue
        selected.append({**candidate, 'id':f'independent-{len(selected)+1:02d}'})
        if len(selected) == 3:
            break
    plan = {
        'schema_version':1,
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'three contexts selected' if len(selected)==3 else 'coverage/packing limitation; no acquisition or rule change',
        'inputs':{str(path):sha(path) for path in (SURFACE,SOURCE_PROVENANCE,MASK,GEOMETRY_SUMMARY)},
        'selection_code_sha256':sha(__file__),
        'criteria':{
            'input_model_scores_used':False,
            'context_shape_yx':[256,256],
            'origin_grid_stride_xy':[64,64],
            'origin_grid_start_xy':[0,0],
            'origin_grid_last_xy':[704,704],
            'excluded_bboxes_source_xywh':FORBIDDEN,
            'bbox_overlap':'positive-area intersection of half-open rectangles; touching edges permitted',
            'minimum_geometry_valid_fraction':0.5,
            'observed_data_definition':'any nonzero raw uint8 intensity among inference layers7:24; no intensity-ranking/contrast/detector feature used',
            'minimum_observed_fraction_among_geometry_valid':0.95,
            'ranking':'geometry-valid fraction descending; observed-data fraction descending; y ascending; x ascending',
            'packing':'greedily retain ranked contexts with no positive-area overlap until three contexts selected',
        },
        'candidate_count':len(candidates),
        'discarded_for_excluded_bbox_overlap':discarded_overlap,
        'discarded_for_coverage':discarded_coverage,
        'ranked_candidates':candidates,
        'selected':selected,
        'frozen_inference':{'checkpoint_repository_revision':'7109667e2607db1b90c37c8b09cb876ea7fe7bb1',
                            'checkpoints':['hybrid_3d2d-seed42/step-075000.pth','hybrid_3d2d-seed43/step-075000.pth'],
                            'layer_start_stop':[7,24], 'direction':'forward', 'tile_zyx':[17,128,128],
                            'stride_xy':[64,64], 'normalization':'upstream robustMAD1/99 per full patch',
                            'model_scores_retained':'unquantized float32 per seed with geometry masks',
                            'diagnostic_midpoint_threshold':0.5,
                            'summary_domain':'central128×128 AND geometry-valid AND both seeds evaluated AND any nonzero source data among selected17depthlayers'},
        'limitations':['All of this field was already model-scored in the first experiment; these are prediction-independent selections, not unseen or held-out validation.',
                       'A human reviewer previously saw first-experiment panels; the reproducible selection algorithm itself does not access predictions.',
                       'Disjoint render contexts are dependent observations from the same scroll/segment and shared training corpus.',
                       'The third context can be partially geometrically invalid; only common valid/evaluated/observed pixels enter its summaries.',
                       '0.5 is a fixed midpoint diagnostic, not an ink label, accuracy estimate or authenticity probability.'],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    if len(selected)!=3:
        print(plan_path)
        raise SystemExit('Insufficient disjoint eligible contexts; consult root before acquisition.')
    plan_sha = sha(plan_path)
    for selection in selected:
        x,y,w,h = selection['bbox_source_xywh']
        destination = LAB / 'data/experiments' / ('pherc0800-'+selection['id'])
        destination.mkdir(exist_ok=False)
        np.save(destination/'surface.npy', surface[:,y:y+h,x:x+w], allow_pickle=False)
        np.save(destination/'valid-mask.npy', mask[y:y+h,x:x+w], allow_pickle=False)
        provenance = {
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'sample_id':source_provenance['sample_id'], 'target_volume_id':source_provenance['target_volume_id'],
            'segment_id':source_provenance['segment_id'], 'pyramid_level':0,
            'bbox_source_xywh':selection['bbox_source_xywh'], 'shape_zyx':[31,h,w], 'dtype':'uint8',
            'voxel_spacing_zyx_um':source_provenance['voxel_spacing_zyx_um'],
            'array_sha256':sha(destination/'surface.npy'), 'valid_mask_sha256':sha(destination/'valid-mask.npy'),
            'full_surface_path':str(SURFACE), 'full_surface_sha256':surface_sha,
            'full_surface_provenance_path':str(SOURCE_PROVENANCE), 'full_surface_provenance_sha256':sha(SOURCE_PROVENANCE),
            'full_geometry_mask_path':str(MASK), 'full_geometry_mask_sha256':mask_sha,
            'selection_plan_path':str(plan_path), 'selection_plan_sha256':plan_sha,
            'selection':selection,
            'status':'Frozen prediction-independent crop from a previously scored source field; no inference performed by selection helper',
        }
        (destination/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    assert sha(SURFACE)==surface_sha and sha(MASK)==mask_sha
    print(plan_path)
    print('FROZEN_PLAN_SHA256',plan_sha)
    for selection in selected:
        print(selection['id'],selection['bbox_source_xywh'])


if __name__ == '__main__':
    main()
