from pathlib import Path
import numpy as np,zarr,json,csv,hashlib,datetime
R=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
load=lambda p:json.loads((R/p).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
readcsv=lambda p:list(csv.DictReader((R/p).open()))
coverage=load('MASK_COVERAGE.json');result=load('RESULTS.json');stage=load('MASK_STAGE_COMPLETE.json');freeze=load('FREEZE.json');ledger=load('REQUEST_LEDGER.json');plan=load('SOURCE_PLAN.json');ctp=load('ct/REQUEST_PLAN.json')
for rel,d in freeze['files'].items():assert sha(R/rel)==d,('freeze mismatch',rel)
for rel,d in stage['files'].items():assert sha(R/rel)==d,('mask stage mismatch',rel)
assert ledger['freeze_sha256']==sha(R/'FREEZE.json');assert result['mask_stage_sha256']==sha(R/'MASK_STAGE_COMPLETE.json')
assert [(x['role'],x['winding']) for x in ledger['requests']]==[(role,w) for role in ['supervision','inklabels'] for w in ['w035','w039','w044']]
expected_set=set();sites=[]
for winding in ['w035','w039','w044']:
 arrays={};decodes={}
 for role in ['supervision','inklabels']:
  source=next(x for x in plan['arrays'] if x['winding']==winding and x['role']==role)
  receipt=next(x for x in ledger['requests'] if x['winding']==winding and x['role']==role)
  store=R/'arrays'/winding/role;obj=store/'c/0/0';assert obj.stat().st_size==source['content_length']==receipt['bytes'];assert sha(obj)==receipt['sha256'];assert receipt['etag']==source['etag']
  meta=R/'provenance'/Path(source['metadata_file']).name;assert sha(meta)==source['metadata_sha256'];assert (store/'zarr.json').read_bytes()==meta.read_bytes()
  a=np.asarray(zarr.open_array(str(store),mode='r')[:]);assert a.dtype==np.uint8
  arrays[role]=a;decodes[role]={'decoded_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'source_sha256':sha(obj),'shape':list(a.shape),'values':{str(int(v)):int(n) for v,n in zip(*np.unique(a,return_counts=True))}}
  kept=load('MASK_DECODE_CHECKS.json' if role=='supervision' else 'LABEL_DECODE_CHECKS.json');k=next(x for x in kept if x['winding']==winding);assert decodes[role]['decoded_sha256']==k['decoded_sha256']==k['official_zarr_decoded_sha256']
 mask=arrays['supervision']!=0; ink=arrays['inklabels']!=0;assert mask.shape==ink.shape;h,w=mask.shape
 rows=[];chunks=set()
 # Independent enumeration starts at the first valid128-aligned core and never enumerates excluded image borders.
 for y in range(128,h-191,128):
  for x in range(128,w-191,128):
   domain=mask[y:y+128,x:x+128];n=int(domain.sum())
   if n<1024:continue
   target=ink[y:y+128,x:x+128];cid=f'{winding}-x{x:05d}-y{y:05d}'
   rows.append({'core_id':cid,'x':x,'y':y,'supervised_pixels':n,'ink_supervised':int(target[domain].sum()),'nonink_supervised':int(domain.sum()-target[domain].sum()),'ink_outside_supervision':int(target[~domain].sum())})
   # Derive touched storage cells from every actual coordinate, independently of intake's upper-bound formula.
   xcells=np.unique(np.arange(x-64,x+192)//128);ycells=np.unique(np.arange(y-64,y+192)//128)
   chunks.update((0,int(cy),int(cx)) for cy in ycells for cx in xcells)
 saved=readcsv(f'{winding}-eligible-cores.csv');savedlabels=readcsv(f'{winding}-eligible-core-class-counts.csv')
 assert [x['core_id'] for x in rows]==[x['core_id'] for x in saved]==[x['core_id'] for x in savedlabels]
 for own,other in zip(rows,savedlabels):
  for key,value in own.items():assert str(value)==other[key],(winding,own['core_id'],key,value,other[key])
 cov=next(s for s in coverage['sites'] if s['winding']==winding);res=next(s for s in result['sites'] if s['winding']==winding)
 assert set(chunks)=={tuple(x) for x in cov['prospective_ct_chunks_zyx']};assert len(chunks)==res['prospective_ct_unique_chunks']==cov['prospective_ct_unique_chunks']
 counts={'total_ink_pixels':int(ink.sum()),'supervised_pixels':int(mask.sum()),'ink_supervised':int(ink[mask].sum()),'nonink_supervised':int(mask.sum()-ink[mask].sum()),'ink_outside_supervision':int(ink[~mask].sum()),'eligible_cores':len(rows),'eligible_supervised_pixels':sum(x['supervised_pixels'] for x in rows),'eligible_ink_supervised':sum(x['ink_supervised'] for x in rows),'eligible_nonink_supervised':sum(x['nonink_supervised'] for x in rows),'eligible_ink_outside_supervision':sum(x['ink_outside_supervision'] for x in rows),'eligible_cores_with_both_classes':sum(x['ink_supervised']>0 and x['nonink_supervised']>0 for x in rows),'eligible_cores_without_ink':sum(x['ink_supervised']==0 for x in rows),'eligible_cores_without_nonink':sum(x['nonink_supervised']==0 for x in rows),'prospective_ct_unique_chunks':len(chunks),'prospective_ct_bytes':len(chunks)*28*128*128}
 for key,value in counts.items():assert value==res[key],(winding,key,value,res[key])
 expected_set.update((winding,*c) for c in chunks)
 sites.append({'winding':winding,'counts':counts,'decoded':decodes,'core_ids':[x['core_id'] for x in rows],'unique_ct_chunk_ids':[list(c) for c in sorted(chunks)]})
 del arrays,mask,ink,a
actual=[(q['winding'],*q['chunk_zyx']) for q in ctp['requests']];assert len(actual)==len(set(actual));assert set(actual)==expected_set
for name,digest in ctp['cohort_pins'].items():assert sha(R/name)==digest
for nm in ctp['native_metadata']:
 assert sha(R/nm['path'])==nm['sha256'];meta=load(nm['path']);assert meta==nm['array'];assert meta['chunks']==[28,128,128] and meta['dtype']=='|u1' and meta['compressor'] is None and meta['order']=='C' and meta['dimension_separator']=='/' and meta['filters'] is None
 for req in (x for x in ctp['requests'] if x['winding']==nm['winding']):
  z,y,x=req['chunk_zyx'];expect=nm['metadata_url'].rsplit('/',1)[0]+f'/{z}/{y}/{x}';assert req['url']==expect;assert req['expected_bytes']==28*128*128;assert req['relative_path']==f'chunks/{nm["winding"]}/{z}/{y}/{x}.bin'
assert sum(q['expected_bytes'] for q in ctp['requests'])==ctp['expected_total_bytes']==len(expected_set)*28*128*128
assert ctp['expected_total_bytes']<=ctp['body_cap_bytes']==256*1024**2
T=lambda s:datetime.datetime.fromisoformat(s)
assert T(freeze['created_utc'])<=T(ledger['started_utc'])<T(ledger['requests'][0]['retrieved_utc'])
assert max(T(q['retrieved_utc']) for q in ledger['requests'] if q['role']=='supervision')<=T(stage['completed_utc'])<min(T(q['retrieved_utc']) for q in ledger['requests'] if q['role']=='inklabels')
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','implementation':'Independent official Zarr decoding and NumPy counting. No import of intake/replay code. No network.','zarr_version':zarr.__version__,'numpy_version':np.__version__,'sites':sites,'aggregate':{'cores':sum(s['counts']['eligible_cores'] for s in sites),'ct_chunks':len(expected_set),'ct_bytes':ctp['expected_total_bytes'],'label_source_bytes':sum(q['bytes'] for q in ledger['requests'])},'chronology':{'freeze_created':freeze['created_utc'],'mask_stage_completed':stage['completed_utc'],'first_label_retrieval_completed':min(q['retrieved_utc'] for q in ledger['requests'] if q['role']=='inklabels'),'freeze_and_stage_hashes_match':True,'chronology_passes':True,'scope':'Receipts timestamp successful completed retrieval, not each HTTP request start. Frozen labels() source checks prior mask-stage receipt and all mask-list hashes before label fetch. Evidence supports recorded guarded execution order; no independent external timestamp attestation.'},'ct_preflight':{'request_plan_sha256':sha(R/'ct/REQUEST_PLAN.json'),'protocol_sha256':sha(R/'ct/PROTOCOL.md'),'request_set_exact':True,'no_additional_chunks':True,'cap_headroom_bytes':ctp['body_cap_bytes']-ctp['expected_total_bytes'],'ct_source_bytes_not_read_by_review':True},'scientific_scope':'Support-count intake only. Every mask-eligible core retained, including single-class. Coverage-selected cohort is not area-unbiased, pristine heldout or evidence of true ink. Whole winding names do not prove disjoint input contexts.'}
(OUT/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'aggregate':report['aggregate'],'sites':[{k:v for k,v in s.items() if k in ['winding','counts']} for s in sites],'chronology':report['chronology']},indent=2))
