"""Freeze restricted/broad UV support before cohort-wide teacher decoding.

This is a mesh-based candidate reservation, not final CT/label admission or a
measurement of physical sheet area. No ink values are read.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import tifffile


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def block_arm(segment, bx, by):
    key=f'plinius-native-coverage-v1|{segment}|{bx}|{by}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % 4 == 0


def bins(box, spacing):
    low=np.floor(np.asarray(box[0])/spacing).astype(int);high=np.floor(np.asarray(box[1])/spacing).astype(int)
    return itertools.product(*(range(a,b+1) for a,b in zip(low,high)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=a.work
    out=r/'support-reservation-v2';out.mkdir(exist_ok=False)
    sources=json.loads((r/'MESH_SOURCES.json').read_text())['rows']
    rule=dict(status='FROZEN_BEFORE_SUPPORT_ENUMERATION',revision=2,patch_size=128,stride=96,block_size=512,
        narrow='SHA256 keyed parent-segment/block modulo 4 equals 0; keep a patch only if its entire 128-square lies inside the block',
        broad='All candidates passing the same geometry AND within-block screen, including narrow support',
        amendment='Version 1 applied the within-block screen only to A. Apply it to both arms before labels or training, preventing a block-boundary eligibility difference. Keep the same hash and all previous outputs.',
        geometry='Finite positive XYZ over the patch footprint plus two stored mesh nodes of halo; reject mesh-array boundary. Metadata spacing 0.05 stored nodes/native pixel.',
        context_box='Bounding box of that stored-node footprint padded by 11 native voxels for the declared 21-plane window. Conservative proxy, not independently verified render/ray support.',
        physical_proxy='Union of padded XYZ bounding-box grid cells at 64 and 128 native voxels; count only as input-neighbourhood coverage, not sheet area or exact aliases.',
        failed_screen='Keep counts; no response-based replacement',
        label_class_values_read=False,cohort_wide_teacher_payload_decoded_before_freeze=False,
        prior_exposure='Eight metadata-centred smoke locations were inspected; this is development training material, not blind or final evaluation.',
        original_stream='Unchanged PHerc0139; whole PHerc0841/0009 excluded; PHerc0175/1203 reserve unopened',
        sources_sha256=sha(r/'MESH_SOURCES.json'),geometry_audit_sha256=sha(r/'GEOMETRY_COVERAGE.json'),implementation_sha256=sha(Path(__file__)))
    (out/'RULE.json').write_text(json.dumps(rule,indent=2)+'\n')
    all_rows=[];summaries=[];unions={g:{'A':set(),'B':set()} for g in [64,128]}
    for source in sources:
        ident=source['id'];xyz=np.stack([tifffile.imread(r/'mesh-arrays'/f'{ident}-mesh-{axis}.raw') for axis in 'xyz'],-1)
        valid=(xyz>0).all(-1)&np.isfinite(xyz).all(-1)
        m=json.loads((r/'metadata'/f'{ident}-array.raw').read_text());d,h,w=m['shape']
        assert d==28 and (np.asarray(xyz.shape[:2])*20==[h,w]).all()
        rejected=boundary=block_boundary=total=0;uv_blocks=set();na=nb=0
        cells={g:{'A':set(),'B':set()} for g in unions}
        for y in range(0,h-127,96):
            for x in range(0,w-127,96):
                total+=1
                gy0=math.floor(y*.05)-2;gx0=math.floor(x*.05)-2;gy1=math.ceil((y+127)*.05)+3;gx1=math.ceil((x+127)*.05)+3
                if min(gy0,gx0)<0 or gy1>xyz.shape[0] or gx1>xyz.shape[1]:boundary+=1;continue
                if not valid[gy0:gy1,gx0:gx1].all():rejected+=1;continue
                footprint=xyz[gy0:gy1,gx0:gx1];low=footprint.min((0,1)).astype(np.float64)-11;high=footprint.max((0,1)).astype(np.float64)+11
                bx,by=x//512,y//512
                in_block=bx==(x+127)//512 and by==(y+127)//512
                if not in_block:block_boundary+=1;continue
                in_a=block_arm(source['segment'],bx,by)
                nb+=1;na+=int(in_a);uv_blocks.add((bx,by))
                all_rows.append(dict(id=f'{ident}-x{x:05}-y{y:05}',segment=source['segment'],x=x,y=y,
                    in_restricted_A=int(in_a),in_broad_B=1,block_x=bx,block_y=by,
                    **{f'{axis}_lo':float(v) for axis,v in zip('xyz',low)},
                    **{f'{axis}_hi':float(v) for axis,v in zip('xyz',high)}))
                for g in cells:
                    bb=set(bins([low,high],g));cells[g]['B'].update(bb)
                    if in_a:cells[g]['A'].update(bb)
        assert na>0 and nb>na
        for g in unions:
            for arm in ['A','B']:unions[g][arm].update(cells[g][arm])
        summaries.append(dict(id=ident,total_grid_patches=total,mesh_boundary_rejections=boundary,
            block_boundary_rejections_both_arms=block_boundary,
            invalid_mesh_footprint_rejections=rejected,restricted_candidates=na,broad_candidates=nb,
            broad_over_restricted_patch_ratio=nb/na,occupied_grid_cells={str(g):dict(A=len(v['A']),B=len(v['B']),ratio=len(v['B'])/len(v['A'])) for g,v in cells.items()}))
    csvpath=out/'CANDIDATES.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_rows)
    result=dict(status='MESH_SUPPORT_RESERVED_NOT_FINAL_TRAINING_ADMISSION',cases=summaries,
        restricted_candidates=sum(s['restricted_candidates'] for s in summaries),
        broad_candidates=sum(s['broad_candidates'] for s in summaries),
        pooled_context_grid_proxy={str(g):dict(A=len(v['A']),B=len(v['B']),ratio=len(v['B'])/len(v['A'])) for g,v in unions.items()},
        csv_sha256=sha(csvpath),rule_sha256=sha(out/'RULE.json'),
        interpretation='More UV patches need not yield proportionally more distinct CT input neighbourhoods. These bounding-box cell counts are conservative geometry proxies, not unique physical ink area.',
        pending=['Source chunk existence/validity across the final selected population','Teacher confidence/support class admission',
            'Exact normal/slab footprints if physical area is claimed','Matched schedule/source coefficients and GPU workload/budget'],
        new_model_forwards=0,paid_compute=False)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)


if __name__=='__main__':main()
