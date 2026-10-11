#!/usr/bin/env python3
"""Independent offline replay; never import the study implementation."""
import argparse,json,pathlib,hashlib,zlib
from fractions import Fraction
import numpy as np

def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):pathlib.Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ratio(a,b):return float(a/b) if b else None
def mean(a):return float(np.mean(a)) if len(a) else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=pathlib.Path,required=True);ap.add_argument('--packet',type=pathlib.Path,required=True);ap.add_argument('--results',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True)
 report=json.load(open(a.results/'RESULTS.json'));planned=json.load(open(a.study/'REQUEST_PLAN.json'));receipt=json.load(open(a.study/'teacher/REQUEST_LEDGER.json'))
 assert receipt['complete'] and receipt['plan_sha256']==digest(a.study/'REQUEST_PLAN.json')
 tilebytes={};claims={r['index']:r for r in receipt['requests']}
 for r in planned['tiles']:
  p=a.study/'teacher'/r['name'];b=p.read_bytes();c=claims[r['index']];assert len(b)==r['bytes'] and digest(p)==c['compressed_sha256'];raw=zlib.decompress(b);assert len(raw)==1048576 and hashlib.sha256(raw).hexdigest()==c['decoded_sha256'];tilebytes[tuple(r['xy'])]=np.frombuffer(raw,np.uint8).reshape(1024,1024)
 assert len(tilebytes)==4
 def pixels(x,y):
  x,y=np.broadcast_arrays(np.asarray(x),np.asarray(y));v=np.empty(x.shape,np.uint8);covered=np.zeros(x.shape,bool)
  for (tx,ty),tile in tilebytes.items():
   inside=(x//1024==tx)&(y//1024==ty);v[inside]=tile[y[inside]%1024,x[inside]%1024];covered|=inside
  assert covered.all();return v
 folder=a.packet/'geometry-review/pullback/site1';q=np.load(folder/'queries_uvk.npy',mmap_mode='r');geom=np.load(folder/'support.npy',mmap_mode='r');candidates=json.load(open(a.study/'CT_CANDIDATES.json'))
 bysupport={s['support']:s for s in candidates['supports']};st=np.load(a.results/'transferred_teacher_float64_uint8_scale.npz');complete=np.load(a.results/'transferred_teacher_complete_support.npz')
 fresh={};errs=[];mismatch=0; threshold_details=[]; near_total=0
 for meta in report['records']:
  x0,y0,x1,y1=meta['native_xy_half_open'];yy,xx=np.mgrid[y0:y1,x0:x1]
  if meta['variant']=='naive_canvas':
   u=((xx+.5)*(31040/7980)-2)/4;v=((yy+.5)*(24880/6400)-2)/4
  else:
   dz,dy,dx=meta['shift_zyx'];zz=meta['native_plane']+dz-4
   chosen=q[zz,yy+dy-1784,xx+dx-2296];assert geom[zz,yy+dy-1784,xx+dx-2296].all();u=chosen[...,0];v=chosen[...,1]
  for op in ['mean16_bilinear','point_bilinear']:
   su,sv=(u,v) if op=='mean16_bilinear' else (4*u+1.5,4*v+1.5)
   ix=np.floor(su).astype(np.int64);iy=np.floor(sv).astype(np.int64);dx=su-ix;dy=sv-iy
   corners=[]
   for by,bx in [(0,0),(0,1),(1,0),(1,1)]:
    if op=='mean16_bilinear':
     # Integer pixel sum first, independent of crop reshaping/pooling implementation.
     accum=np.zeros(ix.shape,np.int64)
     for py in range(4):
      for px in range(4):accum+=pixels(4*(ix+bx)+px,4*(iy+by)+py)
     corners.append(accum.astype(np.float64)/16)
    else:corners.append(pixels(ix+bx,iy+by).astype(np.float64))
   # Nested 1D interpolation has different arithmetic ordering from root's four products.
   out=(corners[0]*(1-dx)+corners[1]*dx)*(1-dy)+(corners[2]*(1-dx)+corners[3]*dx)*dy
   key=op+'_'+meta['key'];assert complete[key].all();e=float(np.max(np.abs(out-st[key])));assert e<1e-9,(key,e)
   changed=int(np.count_nonzero((out>=48)!=(st[key]>=48)))
   near_total+=int(np.count_nonzero(np.abs(st[key]-48)<1e-9))
   for row,col in np.argwhere((out>=48)!=(st[key]>=48)):
    fx,fy=Fraction(float(dx[row,col])),Fraction(float(dy[row,col]));cs=[Fraction(float(c[row,col])) for c in corners]
    exact=cs[0]*(1-fx)*(1-fy)+cs[1]*fx*(1-fy)+cs[2]*(1-fx)*fy+cs[3]*fx*fy
    threshold_details.append({'key':key,'support_pixel_yx':[int(row),int(col)],'native_pixel_xy':[x0+int(col),y0+int(row)],'root_score':float(st[key][row,col]),'alternate_score':float(out[row,col]),'query_uv':[float(u[row,col]),float(v[row,col])],'corners':[float(c) for c in cs],'fractional_xy':[float(fx),float(fy)],'exact_rational_comparison_to48':0 if exact==48 else 1 if exact>48 else -1,'exact_rational_float':float(exact)})
   errs.append({'key':key,'maximum_abs_difference':e,'threshold_disagreements':changed});fresh[key]=out
 # Independently reconstruct max_z native eligibility from original retained raw chunks.
 cp=json.load(open(a.packet/'ct-study/REQUEST_PLAN.json'));cl=json.load(open(a.packet/'ct-study/REQUEST_LEDGER.json'));assert cl['complete'];seen={r['name']:r for r in cl['requests']};shape=tuple(cp['volumes']['native']['array']['chunks']);elig=np.zeros((272,272),bool);covered=np.zeros_like(elig)
 for r in cp['requests']:
  if r['site']!='site1' or r['acquisition']!='native':continue
  p=a.packet/'ct-study/chunks'/r['name'];assert digest(p)==seen[r['name']]['sha256'];arr=np.frombuffer(p.read_bytes(),np.uint8).reshape(shape);_,ry,rx=r['chunk_zyx'];tx,ty=128*rx,128*ry
  l,t=max(tx,2296),max(ty,1784);rr,bb=min(tx+128,2568),min(ty+128,2056)
  elig[t-1784:bb-1784,l-2296:rr-2296]=np.any(arr[:,t-ty:bb-ty,l-tx:rr-tx]!=0,axis=0);covered[t-1784:bb-1784,l-2296:rr-2296]=True
 assert covered.all() and np.array_equal(elig,np.load(a.results/'native_eligible_max28_gt0.npy'))
 # The metric replay uses the saved scores, separate from the direct-source replay above.
 metric_errors=[];envelopes=[];pair_count=0
 def check(where,key,got,want):
  if got is None or want is None:assert got is want,(where,key,got,want)
  elif isinstance(got,(bool,int,np.integer)):assert got==want,(where,key,got,want)
  elif isinstance(got,str):assert got==want,(where,key,got,want)
  else:
   err=abs(float(got)-float(want));assert err<1e-9,(where,key,got,want);metric_errors.append(err)
 for op,oreport in report['operators'].items():
  for saved in oreport['finite_set_summaries']:
   sid,j=saved['support'],saved['native_plane'];sup=bysupport[sid];x0,y0,x1,y1=sup['native_xy_half_open'];e=elig[y0-1784:y1-1784,x0-2296:x1-2296]
   names=[f'{op}_s{sid}_fit_j{j:02d}']+[f'{op}_s{sid}_candidate_{n:02d}_j{j:02d}' for n in range(len(sup['candidates']))]
   stack=np.stack([st[n] for n in names]);all_valid=np.logical_and.reduce([complete[n] for n in names])&e
   nz=stack>0;nsunion=all_valid&np.any(nz,axis=0);nscommon=all_valid&np.all(nz,axis=0)
   positive=(stack>=48)&nscommon;pu=np.any(positive,axis=0);pc=np.all(positive,axis=0);neg=nscommon&~pu;variable=pu&~pc
   point_ranges=stack.max(axis=0)-stack.min(axis=0);nschange=ratio(int(nsunion.sum()-nscommon.sum()),int(nsunion.sum()));varchange=ratio(int(variable.sum()),int(pu.sum()))
   envelope={'complete_across_finite_set_pixels':int(all_valid.sum()),'common_pseudo_supervision_pixels':int(nscommon.sum()),'positive_union_pixels':int(pu.sum()),'positive_common_pixels':int(pc.sum()),'negative_common_pixels':int(neg.sum()),'threshold_variable_envelope_pixels':int(variable.sum()),'threshold_variable_envelope_fraction_of_positive_union':varchange,'teacher_nonzero_union_pixels':int(nsunion.sum()),'teacher_nonzero_common_pixels':int(nscommon.sum()),'teacher_nonzero_variable_envelope_fraction_of_union':nschange,'score_range_mean_uint8':mean(point_ranges[nscommon]),'score_range_p95_uint8':float(np.percentile(point_ranges[nscommon],95)) if nscommon.any() else None,'score_range_positive_union_mean_uint8':mean(point_ranges[pu]),'score_range_positive_union_p95_uint8':float(np.percentile(point_ranges[pu],95)) if pu.any() else None,'adequate_positive_union_and_negative_common_counts':bool(pu.sum()>=100 and neg.sum()>=100),'operational_material_envelope':bool((varchange is not None and varchange>.05) or(nschange is not None and nschange>.01)),'positive_evidence':'present' if pu.any() else 'no-positive-evidence'}
   mads=[]
   for cr in oreport['comparisons']:
    if cr['support']!=sid or cr['native_plane']!=j:continue
    key=op+'_'+cr['key'];baseline='geometry_zero' if cr['variant']=='naive_canvas' else 'fit';basekey=f'{op}_s{sid}_{baseline}_j{j:02d}';s0,s1=st[basekey],st[key];v0,v1=complete[basekey],complete[key];ge=v0&v1&np.isfinite(s0)&np.isfinite(s1)&e
    n0,n1=(s0>0)&ge,(s1>0)&ge;dom=n0&n1
    cand=cr['candidate_index'] is not None
    if cand:dom &= nscommon
    p0,p1=(s0>=48)&dom,(s1>=48)&dom;union=p0|p1;intersection=p0&p1;xor=p0^p1;ncommon=dom&~union;maemask=pu if cand else union;delta=np.abs(s0-s1);mad=mean(delta[maemask]);supportchange=ratio(int((n0^n1).sum()),int((n0|n1).sum()));thresholdchange=ratio(int(xor.sum()),int(union.sum()))
    met={'complete_pixels':int(ge.sum()),'requested_pixels':1024,'geometry_or_source_support_xor_pixels':int((v0^v1).sum()),'metric_common_pseudo_supervision_pixels':int(dom.sum()),'complete_teacher_nonzero_baseline':int(n0.sum()),'complete_teacher_nonzero_candidate':int(n1.sum()),'teacher_nonzero_common_pixels':int((n0&n1).sum()),'teacher_nonzero_union_pixels':int((n0|n1).sum()),'teacher_nonzero_support_xor_fraction_of_union':supportchange,'positive_baseline_pixels':int(p0.sum()),'positive_candidate_pixels':int(p1.sum()),'positive_common_pixels':int(intersection.sum()),'positive_union_pixels':int(union.sum()),'common_negative_pixels':int(ncommon.sum()),'positive_xor_pixels':int(xor.sum()),'gained_positive_pixels':int((~p0&p1).sum()),'lost_positive_pixels':int((p0&~p1).sum()),'threshold_xor_fraction_of_positive_union':thresholdchange,'threshold_iou':ratio(int(intersection.sum()),int(union.sum())),'mean_absolute_difference_all_uint8':mean(delta[dom]),'mean_absolute_difference_geometrically_complete_supplement_uint8':mean(delta[ge]),'mean_absolute_difference_positive_union_uint8':mad,'mean_absolute_difference_positive_union_normalized':mad/255 if mad is not None else None,'positive_mae_domain_pixels':int(maemask.sum()),'positive_evidence':'present' if union.any() else 'no-positive-evidence','adequate_positive_union_and_negative_common_counts':bool(union.sum()>=100 and ncommon.sum()>=100),'operational_material':bool((supportchange is not None and supportchange>.01)or(thresholdchange is not None and thresholdchange>.05)or(mad is not None and mad>12.75))}
    for k,v in met.items():check(key,k,v,cr['metrics'][k])
    pair_count+=1
    if cand and mad is not None:mads.append(mad)
   worst=max(mads) if mads else None;envelope['worst_candidate_mean_absolute_difference_positive_union_uint8']=worst;envelope['operational_material']=bool(envelope['operational_material_envelope'] or(worst is not None and worst>12.75));envelope['routing']='inconclusive_counts' if not envelope['adequate_positive_union_and_negative_common_counts'] else 'conditional_material' if envelope['operational_material'] else 'within_declared_conditional_tolerance'
   for k,v in envelope.items():check(f'{op}-s{sid}-j{j}',k,v,saved[k])
   envelopes.append({'operator':op,'support':sid,'native_plane':j,**envelope})
 alternate_envelopes=[];routing_changes=[];alternate_metric_maxima={}
 for env in envelopes:
  op,sid,j=env['operator'],env['support'],env['native_plane'];sup=bysupport[sid];x0,y0,x1,y1=sup['native_xy_half_open'];e=elig[y0-1784:y1-1784,x0-2296:x1-2296]
  names=[f'{op}_s{sid}_fit_j{j:02d}']+[f'{op}_s{sid}_candidate_{n:02d}_j{j:02d}' for n in range(len(sup['candidates']))]
  stack=np.stack([fresh[n] for n in names]);ns=stack>0;common=e&np.all(ns,axis=0);nunion=e&np.any(ns,axis=0);pos=(stack>=48)&common;pu=np.any(pos,axis=0);pc=np.all(pos,axis=0);neg=common&~pu;var=pu&~pc;vfrac=ratio(int(var.sum()),int(pu.sum()));sfrac=ratio(int(nunion.sum()-common.sum()),int(nunion.sum()));dr=stack.max(axis=0)-stack.min(axis=0);maes=[mean(np.abs(v-stack[0])[pu]) for v in stack[1:]];maes=[v for v in maes if v is not None];worst=max(maes) if maes else None
  ae={'operator':op,'support':sid,'native_plane':j,'common_pseudo_supervision_pixels':int(common.sum()),'positive_union_pixels':int(pu.sum()),'positive_common_pixels':int(pc.sum()),'negative_common_pixels':int(neg.sum()),'threshold_variable_envelope_pixels':int(var.sum()),'threshold_variable_envelope_fraction_of_positive_union':vfrac,'teacher_nonzero_union_pixels':int(nunion.sum()),'teacher_nonzero_common_pixels':int(common.sum()),'teacher_nonzero_variable_envelope_fraction_of_union':sfrac,'score_range_mean_uint8':mean(dr[common]),'score_range_p95_uint8':float(np.percentile(dr[common],95)) if common.any() else None,'score_range_positive_union_mean_uint8':mean(dr[pu]),'score_range_positive_union_p95_uint8':float(np.percentile(dr[pu],95)) if pu.any() else None,'worst_candidate_mean_absolute_difference_positive_union_uint8':worst,'adequate_positive_union_and_negative_common_counts':bool(pu.sum()>=100 and neg.sum()>=100),'operational_material_envelope':bool((vfrac is not None and vfrac>.05)or(sfrac is not None and sfrac>.01))}
  ae['operational_material']=bool(ae['operational_material_envelope']or(worst is not None and worst>12.75));ae['routing']='inconclusive_counts' if not ae['adequate_positive_union_and_negative_common_counts'] else 'conditional_material' if ae['operational_material'] else 'within_declared_conditional_tolerance'
  if ae['routing']!=env['routing']:routing_changes.append({'operator':op,'support':sid,'native_plane':j,'root':env['routing'],'alternate':ae['routing']})
  ae['differences_from_root']={}
  for k,v in ae.copy().items():
   if k in env and isinstance(v,(int,float)) and not isinstance(v,bool) and env[k] is not None:
    delta=float(v-env[k]);ae['differences_from_root'][k]=delta;alternate_metric_maxima[k]=max(alternate_metric_maxima.get(k,0),abs(delta))
  alternate_envelopes.append(ae)
 dump(a.output/'ALTERNATE_ARITHMETIC_ENVELOPES.json',alternate_envelopes);dump(a.output/'THRESHOLD_BOUNDARY_DISCREPANCIES.json',threshold_details)
 summary={'status':'PASS_WITH_THRESHOLD_BOUNDARY_ARITHMETIC_CAVEAT' if threshold_details else 'PASS','alternate_arithmetic_routing_changes':routing_changes,'alternate_arithmetic_maximum_metric_changes':alternate_metric_maxima,'source_score_occurrences_within_1e_9_of48':near_total,'source_tiles':len(tilebytes),'independent_transfers':len(errs),'transferred_pixels':sum(st[n].size for n in st.files),'maximum_transfer_difference':max(e['maximum_abs_difference'] for e in errs),'threshold_disagreements':sum(e['threshold_disagreements'] for e in errs),'finite_set_summaries_checked':len(envelopes),'pairwise_comparisons_checked':pair_count,'maximum_metric_difference':max(metric_errors),'native_max28_eligibility_reassembled':True,'imports_study_implementation':False,'source_frozen_request_plan_sha256':digest(a.study/'REQUEST_PLAN.json'),'source_results_sha256':digest(a.results/'RESULTS.json'),'review_script_sha256':digest(__file__)}
 dump(a.output/'RESULTS.json',summary);dump(a.output/'TRANSFER_CHECKS.json',errs);dump(a.output/'ENVELOPE_REPLAY.json',envelopes);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
