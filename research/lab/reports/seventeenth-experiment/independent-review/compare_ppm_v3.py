"""Compare complete PPM validation, point records, geometry, deltas and support tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from reconstruct_ct import read_pin,read_array,digest,json_dump,PINS
from reconstruct_ppm import point_geometry,NEWPINS
from independent_ppm_v2 import decode_records

P=Path('/workspace/herculaneum-lab/reports/seventeenth-experiment')
counts=dict(integer=0,float=0,boolean=0,null=0,string=0);maxdiff=0.0

def same(a,b,path=''):
 global maxdiff
 if a is None or b is None:assert a is b,(path,a,b);counts['null']+=1
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),(path,set(a)^set(b))
  for k in a:same(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(aa,bb)in enumerate(zip(a,b,strict=True)):same(aa,bb,path+f'/{i}')
 elif type(a)is bool:assert type(b)is bool and a==b,(path,a,b);counts['boolean']+=1
 elif type(a)is int:assert type(b)is int and a==b,(path,a,b);counts['integer']+=1
 elif isinstance(a,float):
  assert math.isfinite(a)and math.isfinite(b)and math.isclose(a,b,abs_tol=1e-9,rel_tol=1e-10),(path,a,b)
  maxdiff=max(maxdiff,abs(a-b));counts['float']+=1
 else:assert a==b,(path,a,b);counts['string']+=1


def record_summary(row):
 xyz=row[:3];n=row[3:];norm=math.sqrt(math.fsum(float(v)*float(v)for v in n))
 return dict(mapped=not bool(np.all(row==0)),xyz=None if np.all(row==0) else xyz.tolist(),normal=n.tolist(),xyz_finite=bool(np.isfinite(xyz).all()),normal_finite=bool(np.isfinite(n).all()),all_components_finite=bool(np.isfinite(row).all()),zero_sentinel=bool(np.all(row==0)),zero_xyz=bool(np.all(xyz==0)),zero_normal=bool(np.all(n==0)),normal_norm=norm,normal_norm_zero=norm==0)


def expected_geometry(neighborhood):
 g=point_geometry(neighborhood);records=[[record_summary(neighborhood[y,x])for x in range(3)]for y in range(3)]
 return dict(center=records[1][1],sampled_records=records,valid_center=g['valid_center'],du=g['du'],dv=g['dv'],du_norm=g['du_norm'],dv_norm=g['dv_norm'],cross_product=g['cross_product'],cross_product_area=g['cross_product_area'],signed_normal_dot=g['signed_normal_dot'],tangent_dot=g['tangent_dot'],degenerate=dict(zero_du=g['du_norm']==0,zero_dv=g['dv_norm']==0,zero_cross_area=g['cross_product_area']==0,zero_normal=g['normal_norm']==0,du_undefined=g['du']is None,dv_undefined=g['dv']is None,normal_agreement_undefined=g['signed_normal_dot']is None,tangent_agreement_undefined=g['tangent_dot']is None))


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--sha',required=True);args=parser.parse_args()
 path=P/'implementation/ppm-run/ppm-diagnostics.json';assert digest(path.read_bytes())==args.sha
 author=json.loads(path.read_text());assert author['status']=='PASS'
 independentpath=P/'independent-review/ppm-reconstruction/independent-ppm-diagnostics.json';independent=json.loads(independentpath.read_text());byid={o['object_id']:o for o in independent['objects']}
 sample=json.loads((P/'acquisition/sample-run/result.json').read_text());assert digest((P/'acquisition/sample-run/result.json').read_bytes())==NEWPINS[P/'acquisition/sample-run/result.json']
 plan=json.loads(read_pin(sample['plan']));requests={r['request_id']:r for r in plan['requests']};bytes_by_obj={}
 for obj in sample['objects']:
  body=read_pin(obj['file']);parts={tuple(p['range_inclusive']):p for p in obj['parts']};pieces=[]
  for span in requests[obj['request_id']]['ranges']:
   part=parts[tuple(span)];data=body[part['payload_offset']:part['payload_offset']+part['payload_bytes']];assert digest(data)==part['payload_sha256'];pieces.append(data)
  bytes_by_obj.setdefault(obj['object_id'],[]).append((obj['case_id'],b''.join(pieces)))
 all_geometry={};zero_counts={}
 for obj in author['objects']:
  oid=obj['object_id'];ind=byid[oid];validation=obj['validation'];expected_order={'<':'little','>':'big',None:None}[ind['selected_endian']]
  raw=b''.join(raw for _,raw in sorted(bytes_by_obj[oid]));assert digest(raw)==ind['records_sha256']==validation['records_sha256']
  assert validation['selected_byte_order']==expected_order and validation['status']==('conditional'if expected_order else'unresolved')
  assert validation['record_count']==1458 and validation['byte_count']==69984
  for order,e in [('little','<'),('big','>')]:
   a=decode_records(raw,e);v=validation['candidates'][order];own=ind['candidate_checks'][e]
   assert v['passed']==own['pass_invariant']and v['record_count']==own['total_records']and v['zero_sentinel_count']==own['zero_sentinels']
   zero=np.all(a==0,axis=1);nonzero=~zero;finite=np.isfinite(a).all(axis=1)
   assert v['nonsentinel_count']==int(nonzero.sum())
   if obj['variant']=='result':
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):norms=np.sqrt(np.sum(a[:,3:]*a[:,3:],axis=1));errors=np.abs(norms-1)
    unit=np.isfinite(norms)&(errors<=1e-5);finite_errors=errors[nonzero&np.isfinite(errors)]
    assert v['nonfinite_nonsentinel_count']==int((nonzero&~finite).sum())
    assert v['nonunit_nonsentinel_count']==int((nonzero&~unit).sum())
    assert v['nonfinite_normal_norm_count']==int((nonzero&~np.isfinite(norms)).sum())
    same(v['finite_normal_norm_error_max'],float(finite_errors.max())if len(finite_errors)else None,oid+'/'+order+'/norm_error')
    assert v['normal_norm_absolute_tolerance']==1e-5
   else:
    points=ind['points'];xy=[]
    for point in points:
     cy,cx=point['native_center_yx'];xy.extend([[cx+dx,cy+dy]for dy in[-1,0,1]for dx in[-1,0,1]])
    xy=np.array(xy);xyok=np.all(a[:,:2]==xy,axis=1);const=(a[:,2]==32)&(a[:,3]==0)&(a[:,4]==0);sign=(a[:,5]==1)|(a[:,5]==-1)
    assert v['nonfinite_count']==int((nonzero&~finite).sum())and v['xy_mismatch_count']==int((nonzero&~xyok).sum())
    assert v['constant_component_mismatch_count']==int((nonzero&~const).sum())and v['invalid_sign_count']==int((nonzero&~sign).sum())
    positive=bool(nonzero.any()and np.all(a[nonzero,5]==1));negative=bool(nonzero.any()and np.all(a[nonzero,5]==-1))
    assert v['normal_sign_consistent']==(positive or negative)and v['normal_sign']==(1 if positive else -1 if negative else None)
  if expected_order is None:continue
  decoded=decode_records(raw,ind['selected_endian']).reshape(2,81,3,3,6)
  for case_number,case in enumerate(obj['cases']):
   points=[p for p in ind['points']if p['case_id']==case['id']];assert len(points)==case['point_count']==81 and case['sampled_record_count']==729
   for index,(point,ip)in enumerate(zip(case['points'],points,strict=True)):
    assert point['coarse_yx']==ip['coarse_yx']and point['native_center_yx']==ip['native_center_yx']
    expected=expected_geometry(decoded[case_number,index]);same(point['geometry'],expected,oid+'/'+case['id']+'/'+str(index))
    all_geometry[(case['id'],obj['variant'],tuple(point['coarse_yx']))]=expected
 for case in author['same_point_variant_comparisons']:
  for point in case['points']:
   first=all_geometry[(case['id'],'result',tuple(point['coarse_yx']))];second=all_geometry[(case['id'],'surface_volume',tuple(point['coarse_yx']))]
   delta=np.array(second['center']['xyz'])-np.array(first['center']['xyz'])if first['valid_center']and second['valid_center']else None
   expected=dict(comparison='second_minus_first_at_identical_scheduled_native_xy',center_xyz_delta=delta.tolist()if delta is not None else None,center_xyz_distance=math.hypot(*delta)if delta is not None else None)
   for key in ['du_norm','dv_norm','cross_product_area','signed_normal_dot','tangent_dot']:
    expected[key+'_delta']=second[key]-first[key]if second[key]is not None and first[key]is not None else None
   same(point['comparison'],expected,case['id']+'/variant_delta')
 desc=json.loads(next(p for p in PINS if p.name=='pooled-descriptor.json').read_text());cases={c['id']:c for c in desc['cases']}
 coverage=json.loads((P/'independent-review/ppm-support-coverage.json').read_text());sites={(s['case_id'],s['variant'],tuple(s['coarse_yx'])):s for s in coverage['all_centers']}
 for table in author['fixed_site_support_cross_tab']:
  caseid=table['id'];variant=table['variant'];labels=read_array(cases[caseid]['arrays']['labels']);groups={name:dict(center_count=0,mapped_center_count=0,zero_sentinel_center_count=0,unresolved_center_count=0)for name in table['support_cross_tab']};zeros=[]
  for key,site in sites.items():
   if key[:2]!=(caseid,variant):continue
   cy,cx=site['coarse_yx'];label=int(labels[cy,cx]);subset=dict(all=True,surface=site['surface'],outside_surface=not site['surface'],primary=site['primary'],outside_primary=not site['primary'],known=site['known'],primary_positive=site['primary']and label==1,primary_negative=site['primary']and label==0)
   for name,include in subset.items():
    if include:groups[name]['center_count']+=1;groups[name]['zero_sentinel_center_count'if site['zero_sentinel']else'mapped_center_count']+=1
   if site['zero_sentinel']:zeros.append(dict(coarse_yx=[cy,cx],surface=site['surface'],primary=site['primary'],known=site['known'],label=label))
  same(table['support_cross_tab'],groups,caseid+'/'+variant+'/coverage');same(table['zero_sentinel_sites'],zeros,caseid+'/'+variant+'/zero_sites')
 report=dict(status='PASS',geometry_points=len(all_geometry),sampled_records=8748,candidate_invariants_checked=12,cross_map_comparisons=486,support_tables=12,scalar_comparisons=counts,max_absolute_float_difference=maxdiff,tolerance=dict(atol=1e-9,rtol=1e-10),author_sha256=args.sha,independent_sha256=digest(independentpath.read_bytes()),source_sha256=digest(Path(__file__).read_bytes()),limitation='Conditional serialization and coordinate geometry only; no historical/physical registration authentication')
 json_dump(P/'independent-review/ppm-comparison-receipt.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
