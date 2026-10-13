"""Reproduce source-frame/provenance tables from retained metadata only."""
from pathlib import Path
import itertools
import json
import numpy as np
import select_regions as S
ROOT=Path(__file__).resolve().parent

def affine(cat,source,target):
    matches=[t for t in cat['volumes'][source]['properties']['transforms'] if t['to_volume_id']==target]
    assert len(matches)==1
    a=np.eye(4);a[:3]=np.array(matches[0]['transformation_matrix']);return a

def main():
    cat=S.read(ROOT/'CATALOG_EXTRACT.json');table=[]
    for w,segment in cat['segments'].items():
        for kind,volume in [('native','20250728140407'),('fine78','20260102150214'),('fine59','20260413113053')]:
            path=ROOT/'metadata'/f'{w}-{kind}-attrs.json'
            if not path.exists():continue
            attr=S.read(path);array=S.read(ROOT/'metadata'/f'{w}-{kind}-array.json')
            scales={r['path']:next(t['scale'] for t in r['coordinateTransformations'] if t['type']=='scale')
                    for r in attr['multiscales'][0]['datasets']}
            teacher=next((a for a in segment['data'] if a['type']=='ink-detection' and
                         (a.get('parameters') or {}).get('target_volume')==volume),None)
            params={} if teacher is None else teacher['creation_info']['provenance']['parameters']
            receipt=None if teacher is None else S.read(ROOT/'receipts'/f'{w}-{kind}-teacher.json')
            table.append(dict(winding=w,segment=segment['long_id'],kind=kind,volume_id=volume,
                source_group=attr['source_group'],shape_zyx=array['shape'],chunks_zyx=array['chunks'],
                surface_group_spacing_zyx_um=scales,compressor=array['compressor'],dtype=array['dtype'],
                teacher_model=None if teacher is None else teacher['parameters']['model_id'],
                teacher_model_type=params.get('model-type'),teacher_depth_half_open=None if not params else [int(params['start-layer']),int(params['end-layer'])],
                teacher_tile_stride=None if not params else [int(params['tile-size']),int(params['stride'])],
                teacher_force_reverse=params.get('force-reverse'),teacher_stored_bytes=None if receipt is None else receipt['content_length'],
                native_labels_present=any(a['type']=='ink-labels' and (a.get('parameters') or {}).get('target_volume')=='20250728140407' for a in segment['data'])))
    fine78_to_native=affine(cat,'20260102150214','20250728140407')
    fine59_to_fine78=affine(cat,'20260413113053','20260102150214')
    composite=fine78_to_native@fine59_to_fine78;inverse=np.linalg.inv(composite)
    coverage=[];shape=np.array(cat['volumes']['20260413113053']['properties']['shape'][::-1])
    for r in S.read(ROOT/'selection/RESULTS.json')['selected']:
        if r['winding']!='w024':continue
        box=np.array(r['bbox_conservative_xyz']);corners=np.array(list(itertools.product(*zip(box[0],box[1]))))
        mapped=corners@inverse[:3,:3].T+inverse[:3,3]
        coverage.append(dict(core_id=r['core_id'],global_affine_enclosure_fine59_full_resolution_xyz=[mapped.min(0).tolist(),mapped.max(0).tolist()],
                             entire_box_within_catalog_extent=bool((mapped.min(0)>=0).all() and (mapped.max(0)<shape).all())))
    plan=[]
    for w,kind in [('w040','fine78'),('w041','fine78'),('w043','fine78'),('w024','fine59')]:
        for axis in 'xyz':
            r=S.read(ROOT/'receipts'/f'{w}-{kind}-{axis}.json')
            plan.append(dict(id=r['id'],url=r['url'],bytes=r['content_length'],etag=r['etag'],head_receipt_sha256=S.sha(ROOT/'receipts'/f'{r["id"]}.json')))
    S.save(ROOT/'NEXT_FINE_GEOMETRY_PLAN.json',dict(status='NOT_EXECUTED',requests=plan,max_body_bytes=sum(r['bytes'] for r in plan),
        purpose='Fine XYZ only, to qualify renderer mapping/complete teacher contexts before a finite CT/target request plan',
        coordinate_frames='fine78 group0 meshes; fine59 full-resolution mesh with source_group1 renderer; never reuse group0 mapping unchanged',
        CT_teacher_annotation_bytes=0,max_workers=2,automatic_retries=0))
    result=dict(source_table=table,full_resolution_xyz_affines=dict(fine78_to_native=fine78_to_native[:3].tolist(),
        fine59_to_fine78=fine59_to_fine78[:3].tolist(),fine59_to_native=composite[:3].tolist()),
        fine59_group1_renderer=dict(source_group=1,render_scale=.5,base_coordinate_multiplier=.5,
            spacing_um=2.258,slices=116,source_code='sources/vc_render_tifxyz.cpp',
            limitation='Canvas mesh sampling also changes with render_scale; multiplying coordinates alone is insufficient. Exact mapping unimplemented/unqualified for this reference.'),
        w024_affine_extent_screen=coverage,
        w024_contexts_fully_inside_affine_catalog_extent=sum(r['entire_box_within_catalog_extent'] for r in coverage),
        limitation='Global affine enclosure is a planning screen, not local registration, zero-mask validity or teacher-context qualification',
        code_sha256=S.sha(Path(__file__)))
    S.save(ROOT/'SOURCE_CONTRACTS.json',result)
    print(json.dumps(dict(source_rows=len(table),w024_affine_extent_inside=result['w024_contexts_fully_inside_affine_catalog_extent'],
                         future_fine_coordinate_bytes=sum(r['bytes'] for r in plan))))

if __name__=='__main__':main()
