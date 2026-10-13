#!/usr/bin/env python3
"""Standalone offline source-target replay using saved derived query fields.

This mode verifies the cropped teacher and every case/query checksum; independently
resamples that teacher at saved UV coordinates and recomputes all reported scalar
summaries. It DOES NOT independently verify the geometry-derived coordinates or
CT-derived native-eligibility mask. Use independent_check.py with the original
correspondence and supervision packets for that stronger verification scope.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from independent_check import interpolate, metrics, fixtures
from compare_outputs import main as compare

EXPECTED_PROTOCOL='dd87ce2618f75564d072fbc08ab3a1689bc37d1fbc0128620f5452b2e69e3052'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(packet,output):
    output.mkdir(parents=True,exist_ok=False)
    assert sha(packet/'PROTOCOL.md')==EXPECTED_PROTOCOL,'Frozen protocol changed'
    metadata=json.loads((packet/'inputs/teacher/CROP.json').read_text());source=packet/'inputs/teacher/crop.npy';assert sha(source)==metadata['crop_sha256'],'Teacher crop mismatch'
    crop=np.load(source,allow_pickle=False);assert crop.dtype==np.uint8 and list(crop.shape)==metadata['crop_shape_yx']==[780,788];assert metadata['origin_fullfine_xy']==[9084,7088]
    c=json.loads((packet/'inputs/CT_CANDIDATES.json').read_text());kernels=[[tuple(v['shift_zyx']) for v in s['candidates']] for s in c['supports']];unique=sorted({s for k in kernels for s in k});assert len(unique)==24
    w=np.mean([np.array([1/len(k) if s in k else 0 for s in unique],float) for k in kernels],axis=0)
    data=json.loads((packet/'results/RESULTS.json').read_text());assert len(data['cases'])==64;seen=set();arrays={};rows=[]
    for r in data['cases']:
        query=packet/'results'/r['queries']['relative'];case=packet/'results'/r['arrays']['relative'];assert sha(query)==r['queries']['sha256'];assert sha(case)==r['arrays']['sha256'];q=np.load(query,allow_pickle=False)
        sid=r['support'];j=r['native_plane'];op=r['operator'];key=f's{sid}_j{j}_{op}';assert key not in seen;seen.add(key);si=[s['support'] for s in c['supports']].index(sid);assert j in range(10,18) and op in ('mean16_bilinear','point_bilinear')
        assert np.array_equal(q['shifts_zyx'],unique);coords=q['candidate_uv_group2'];valid=q['candidate_geometry_valid'];assert coords.shape==(24,32,32,2) and valid.shape==(24,32,32);li=[unique.index(s) for s in kernels[si]];assert np.array_equal(q['local_indices'],li)
        assert np.allclose(q['global_weights'],w,rtol=0,atol=1e-16);lw=np.array([1/len(kernels[si]) if s in kernels[si] else 0 for s in unique]);assert np.array_equal(q['local_weights'],lw)
        centroid=coords[li].mean(0);assert np.array_equal(centroid,q['centroid_uv_group2']);fi=unique.index((0,-1,0));assert np.array_equal(q['fit_uv_group2'],coords[fi]);assert np.array_equal(q['fit_geometry_valid'],valid[fi]);assert np.array_equal(q['centroid_geometry_valid'],valid[li].all(0)&np.isfinite(centroid).all(-1))
        raw,sv=interpolate(crop,coords,metadata['origin_fullfine_xy'],op);D,dv=interpolate(crop,centroid,metadata['origin_fullfine_xy'],op);A=raw[fi];B=raw[li].mean(0);C=np.mean([raw[[unique.index(s) for s in k]].mean(0) for k in kernels],axis=0)
        geom=valid.all(0)&q['centroid_geometry_valid']&np.isfinite(coords).all(axis=(0,3));complete=geom&sv.all(0)&dv&q['native_eligible'];common=complete&(raw>0).all(0)&(D>0);active=common&((raw>=48).any(0)|(D>=48))
        row=dict(key=key,common_pixels=int(common.sum()),active_union_pixels=int(active.sum()),pairs={})
        for label,x,y in [('B-C',B,C),('B-D',B,D)]:
            val=metrics(x,y,active)['mae'];row['pairs'][label]=dict(active_union_mae_below_one_quantization_interval=None if val is None else val<1/255)
        rows.append(row)
        for label,v in dict(A=A,B=B,C=C,D=D,transfers=raw,geometry_mask=geom,complete_mask=complete,common_mask=common,positive_union_mask=active,centroid_uv=centroid).items():arrays[key+'_'+label]=v
    assert len(seen)==64
    scope='Standalone replay from saved DERIVED UV fields and native eligibility; does not verify their source geometry/CT derivation'
    result=dict(status='COMPLETE',scope=scope,protocol_sha256=EXPECTED_PROTOCOL,fixtures=fixtures(),teacher_crop_sha256=sha(source),cases=rows)
    (output/'INDEPENDENT_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output/'independent_arrays.npz',**arrays)
    compare(packet,independent_dir=output,output_dir=output)
    (output/'SCOPE.md').write_text(scope+'\nFull geometry replay requires the original correspondence and supervision packets.\n')
    print(scope)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.packet.resolve(),a.output.resolve())
