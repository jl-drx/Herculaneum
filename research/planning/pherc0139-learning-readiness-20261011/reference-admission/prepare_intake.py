import json,pathlib,urllib.request,hashlib,datetime,concurrent.futures
R=pathlib.Path(__file__).parent;d=json.load(open(R/'CATALOG_EXTRACT.json'));base='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'pixel-free intake plan; HEAD and retained metadata only','maximum_body_bytes':8388608,'maximum_object_body_bytes':1048576,'arrays':[]}
for w in ['w035','w039','w044']:
 s=d['segments'][w];a=next(a for a in s['data'] if a['type']=='ink-labels' and (a.get('parameters') or {}).get('target_volume')=='20250728140407');p=a['origins'][0]['path'].rstrip('/')
 for role in ['supervision','inklabels']:
  m=json.load(open(R/'metadata'/f'{w}-s3-{role}-array.json'));sh=m['chunk_grid']['configuration']['chunk_shape'];inner=m['codecs'][0]['configuration']['chunk_shape'];slots=(sh[0]//inner[0])*(sh[1]//inner[1]);u=base+p+'/'+role+'.zarr/0/c/0/0'
  with urllib.request.urlopen(urllib.request.Request(u,method='HEAD'),timeout=40) as r:
   rec={'winding':w,'role':role,'public_segment':s['long_id'],'url':u,'status':r.status,'content_length':int(r.headers['Content-Length']),'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified'),'shape':m['shape'],'dtype':m['data_type'],'shard_shape':sh,'inner_chunk_shape':inner,'index_entries':slots,'index_bytes_including_crc32c':slots*16+4,'metadata_file':f'metadata/{w}-s3-{role}-array.json','metadata_sha256':hashlib.sha256((R/'metadata'/f'{w}-s3-{role}-array.json').read_bytes()).hexdigest(),'root_metadata_file':f'metadata/{w}-s3-{role}-root.json','native_metadata_file':f'metadata/{w}-native-array.json'}
   assert rec['content_length']<=1048576
   rec['index_byte_range']=[rec['content_length']-rec['index_bytes_including_crc32c'],rec['content_length']-1];out['arrays'].append(rec)
out['compressed_total_bytes']=sum(a['content_length'] for a in out['arrays']);out['supervision_compressed_total_bytes']=sum(a['content_length'] for a in out['arrays'] if a['role']=='supervision')
json.dump(out,open(R/'NATIVE_REFERENCE_INTAKE_PLAN.json','w'),indent=2)
print(json.dumps({'all_six_bytes':out['compressed_total_bytes'],'all_three_supervision_bytes':out['supervision_compressed_total_bytes'],'objects':[{k:a[k] for k in ['winding','role','content_length','index_bytes_including_crc32c']} for a in out['arrays']]},indent=2))
