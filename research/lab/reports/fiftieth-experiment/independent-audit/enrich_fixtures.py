from reconcile_archived_convention import *
p=Path(__file__).parent;d=json.loads((p/'reconciliation/metrics.json').read_text())
for seg in sorted(set(q['segment'] for q in d['discrepancy_fixtures'])):
 m,sc,shape=load(seg);ab,an,_,_=archived_geometry(m,sc,shape)
 for q in d['discrepancy_fixtures']:
  if q['segment']!=seg:continue
  x,y=q['xy'];xx=x//128*128;yy=y//128*128;b,n,_,_=tile(m,sc,xx,yy,min(128,shape[1]-xx),min(128,shape[0]-yy));ib=b[y-yy,x-xx];inn=n[y-yy,x-xx]
  ilo=np.floor(ib.astype(float)-32.5*np.abs(inn.astype(float))-.01).astype(int);ihi=np.floor(ib.astype(float)+32.5*np.abs(inn.astype(float))+.01).astype(int)+1
  e1=ab[y,x].astype(float)-np.float32(32.5)*an[y,x];e2=ab[y,x].astype(float)+np.float32(32.5)*an[y,x];alo=np.floor(np.minimum(e1,e2)-.01).astype(int);ahi=np.floor(np.maximum(e1,e2)+.01).astype(int)+1
  def hits(lo,hi):return [i for i,r in enumerate(GO['previous_CT_boxes']) if np.all(hi>=r['lo_xyz']) and np.all(lo<=r['hi_xyz_inclusive'])]
  q.update(independent_corner_AABB_inclusive=[ilo.tolist(),ihi.tolist()],archived_corner_AABB_inclusive=[alo.tolist(),ahi.tolist()],independent_old_box_indices_zero_based=hits(ilo,ihi),archived_old_box_indices_zero_based=hits(alo,ahi))
  q['cause']='nearest-old-centre threshold' if (q['independent_nearest_distance']>51.1)!=(q['archived_nearest_distance']>51.1) else 'integer CT-box boundary'
(p/'discrepancy-fixtures.json').write_text(json.dumps(d['discrepancy_fixtures'],indent=2));print({cause:sum(q['cause']==cause for q in d['discrepancy_fixtures']) for cause in set(q['cause'] for q in d['discrepancy_fixtures'])})
