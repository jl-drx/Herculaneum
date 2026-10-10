from audit import *
rows=[json.loads(l) for l in (OUT.parent/'run.log').read_text().splitlines() if l.startswith('{"segment"')]
components=[]
for row in rows:
 s=row['segment'];a=np.load(OUT/f'{s}-masks.npz');point=a['point_support'];allorigin=a['all_origin_context'];exact=a['fixed_grid_context'];h,w=point.shape
 labels=np.load(P/'inputs/phase47-components'/f'{s}.npz')['far_component_labels'];full=labels[np.minimum((np.arange(h)+10)//20,labels.shape[0]-1)[:,None],np.minimum((np.arange(w)+10)//20,labels.shape[1]-1)[None,:]]
 for lab in np.unique(labels):
  if not lab:continue
  cm=full==lab;aa=allorigin&cm;e=exact&cm;sq=largest(e)
  item={'segment':s,'component':int(lab),'sampled_size':int((labels==lab).sum()),'point_area':int((point&cm).sum()),'all_origin_area':int(aa.sum()),'all_origin_square':largest(aa),'fixed_grid_area':int(e.sum()),'fixed_grid_square':sq}
  if sq['side']>=128:
   xx,yy,sw,sh=sq['xywh'];item['contributing_patches_xywh']=[[px,py,128,128] for py in starts(h) for px in starts(w) if px<xx+sw and px+128>xx and py<yy+sh and py+128>yy]
   assert all(point[py:py+ph,px:px+pw].all() for px,py,pw,ph in item['contributing_patches_xywh'])
  components.append(item)
preferred=max(c['sampled_size'] for c in components if c['segment']=='20251028225813');components.sort(key=lambda r:(0 if r['segment']=='20251028225813' and r['sampled_size']==preferred else 1,-r['sampled_size'],r['segment'],r['component']))
receipts=[]
for file in sorted((P/'inputs/archived-point-checks').glob('*.json')):
 d=json.loads(file.read_text());s=d['segment_id'];m,sc,shape=load(s)
 for pt in d['points']:
  x,y=pt.get('surface_xy',pt.get('source_xy'));x0=x//128*128;y0=y//128*128;b,n,v,nv=tile(m,sc,x0,y0,min(128,shape[1]-x0),min(128,shape[0]-y0));i=(y-y0,x-x0)
  receipts.append({'segment':s,'xy':[x,y],'base_max_abs':float(np.max(np.abs(b[i]-pt['base_ct_xyz']))),'normal_max_abs':float(np.max(np.abs(n[i]-pt['normal_xyz_flipped']))),'valid':bool(v[i]&nv[i])})
d={'meshes':rows,'components_ranked':components,'receipts':receipts,'old_counts':{'20251029010146':497600,'20251028222030':2196160,'20251028220955':567072},'all_origin_selected':next((c for c in components if c['all_origin_square']['side']>=128),None),'fixed_grid_selected':next((c for c in components if c['fixed_grid_square']['side']>=128),None),'note':'Recovered deterministic summaries from saved independent full-resolution masks after list.sort key bug; geometry unchanged.'}
(OUT/'independent-results.json').write_text(json.dumps(d,indent=2));print(json.dumps(d['fixed_grid_selected'],indent=2));print('conservative selected',d['all_origin_selected'])
