"""Requires a root GO binding frozen code and exact proposal before any request."""
import argparse,datetime,json,os,pathlib,resource,signal,sys,time
import numpy as np
import geometry_kernels as K
from bounded_transport import Client,record,write_json
ROOT=pathlib.Path(__file__).resolve().parent;P=ROOT.parent;PLAN_SHA='4f180f4507e77c8460c5b3a57c0683e818c8b29957c1a6d75c70dc9d53f6623d';LIMIT_RSS=536870912;LIMIT_OUTPUT=33554432
EXPECTED_RUNTIME={'python':'3.12.14','numpy':'2.5.3','platform':'linux'}
def runtime():return {'python':sys.version.split()[0],'numpy':np.__version__,'platform':sys.platform}
def budget(out,start):
 K.require(time.monotonic()-start<230,'internal wall limit230s');rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;K.require(rss<=LIMIT_RSS,'512MiB RSS limit');size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file());K.require(size<=LIMIT_OUTPUT,'32MiB output limit');return {'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':rss,'output_bytes':size}
def terminate(*a):raise K.ContractError('external termination')
def verify_inputs(plan,freeze):
 for item in freeze['sources']+freeze['inputs']+plan['input_pins']:
  K.require(record(pathlib.Path(item['path']))==item,'pinned input changed')
 # The embedded source is gzip JSON metadata; verify its bytes without accessing arrays.
 native=plan['native_frame'];K.require(record(pathlib.Path(native['source']['path']))==native['source'],'native frame catalog changed')
 K.require(native['volume_id']=='20260102150214' and native['shape_xyz']==[26511,26511,76953],'native original frame')
def verify_plan(plan):
 K.require(plan['maximum_attempts']==6 and plan['maximum_total_reserved_response_body_bytes']==4194304 and len(plan['requests'])==6 and len(plan['contexts'])==2,'literal plan scope');K.require(plan['derived_total_reserved_body_bytes']==2739750 and plan['derived_total_expected_body_bytes']==2739744,'literal planned bytes');K.require(plan['decode_budget']['maximum_peak_rss_bytes']==LIMIT_RSS and plan['decode_budget']['maximum_output_bytes']==LIMIT_OUTPUT,'literal runtime caps')
 specs=[('pherc0139-w016-rank01',[1792,4864,384,384],[357,972,80,80],[5615708,6072331]),('pherc0139-w016-rank02',[3584,5120,384,384],[716,1023,80,80],[5911720,6368343])]
 for i,(name,box,window,span) in enumerate(specs):
  c=plan['contexts'][i];K.require(c['id']==name and c['level2_bbox_xywh']==box and c['mesh_window_xywh']==window and c['fine_raster_bbox_xywh']==[4*v for v in box],'fixed context identity')
  for j,channel in enumerate('xyz'):
   q=plan['requests'][3*i+j];K.require(q['id']==name+'-'+channel+'.tif' and q['context']==name and q['range_inclusive']==span and q['window_xywh']==window and q['expected_body_bytes']==456624 and q['reserved_body_bytes']==456625,'fixed exact request')
   contract=json.loads((P/'structural/run'/f'{channel}.tif.contract.json').read_text());K.require(q['source']==contract['source'],'request source differs from observed contract')
 K.require(sum(q['reserved_body_bytes'] for q in plan['requests'])==2739750,'request reservation sum')
def run(go_path,out):
 start=time.monotonic();signal.signal(signal.SIGTERM,terminate);go_path=pathlib.Path(go_path).resolve();out=pathlib.Path(out).resolve();K.require(out.parent==ROOT and not out.exists(),'output must be new immediate child of implementation directory')
 planpath=P/'public-geometry-plan/COORDINATE_QUERY_PLAN.json';freezepath=ROOT/'CODE_FREEZE.json';K.require(record(planpath)['sha256']==PLAN_SHA,'exact proposal');plan=json.loads(planpath.read_text());freeze=json.loads(freezepath.read_text());go=json.loads(go_path.read_text());K.require(go.get('status')=='GO' and go.get('plan')==record(planpath) and go.get('code_freeze')==record(freezepath),'root GO must bind exact plan and code_freeze');K.require(go.get('authorized_action')=='acquire_and_reconstruct_public_geometry','explicit acquisition/reconstruction GO');K.require(runtime()==EXPECTED_RUNTIME and freeze['runtime']==EXPECTED_RUNTIME,'pinned runtime');verify_inputs(plan,freeze);verify_plan(plan)
 out.mkdir();guard=lambda:budget(out,start);client=Client(plan,out,guard);result={'status':'RUNNING','plan':record(planpath),'root_go':record(go_path),'code_freeze':record(freezepath),'runtime':runtime(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'contexts':[],'decoded_window_channel_count':0,'numerically_decoded_coordinate_bytes':0,'native_CT_requests':0,'mask_label_requests':0,'model_calls':0,'historical_private_validity_authenticated':False,'output_claim':plan['output_claim']};write_json(out/'result.running.json',result)
 try:
  # Complete exactly six authenticated downloads before any coordinate interpretation.
  bodies=[client.fetch(q) for q in plan['requests']];client.ledger['status']='PASS';client.save();guard()
  scale=json.loads((P/'text-run/transformed_mesh_meta.body').read_text())['scale']
  for i,c in enumerate(plan['contexts']):
   guard();channels=[]
   for j in range(3):
    index=3*i+j;body=bodies[index].read_bytes();K.require(record(bodies[index])==client.ledger['requests'][index]['body'],'retained range changed before decode');channels.append(K.decode_window(body,plan['requests'][index]));result['decoded_window_channel_count']+=1;result['numerically_decoded_coordinate_bytes']+=plan['requests'][index]['useful_coordinate_bytes']
   xyz=np.stack(channels,axis=2).astype(np.float32,copy=False);del channels,body
   xyz_path=out/(c['id']+'.xyz-window.npy');np.save(xyz_path,xyz,allow_pickle=False);guard();mesh,bounds,counts=K.reconstruct(xyz,c,scale,plan['native_frame']['shape_xyz'],guard);paths={}
   for key,array in [('mesh-stencil-support',mesh),('mesh-native-bounds-support',bounds)]:
    path=out/(c['id']+'.'+key+'.npy');np.save(path,array,allow_pickle=False);paths[key]=record(path)
   count_path=out/(c['id']+'.counts.json');write_json(count_path,counts);paths['counts']=record(count_path);paths['xyz_window']=record(xyz_path);result['contexts'].append({'id':c['id'],'mesh_window_xywh':c['mesh_window_xywh'],'outputs':paths,'counts':counts});del xyz,mesh,bounds;guard()
  K.require(len(result['contexts'])==2 and len(client.ledger['requests'])==6 and all(q['status']=='PASS' for q in client.ledger['requests']),'final cardinality');verify_inputs(plan,freeze)
  for path,row in zip(bodies,client.ledger['requests']):K.require(record(path)==row['body'],'raw range changed')
  result['status']='PASS'
 except BaseException as e:
  result['status']='STOP';result['error_type']=type(e).__name__;result['stop_reason']=str(e) if isinstance(e,K.ContractError) else type(e).__name__
 finally:
  client.ledger['status']='PASS' if len(client.ledger['requests'])==6 and all(q['status']=='PASS' for q in client.ledger['requests']) else 'STOP';client.save();result['transport_ledger']=record(out/'transport-ledger.json');result['request_count']=len(client.ledger['requests']);result['reserved_bytes_all_attempts']=client.ledger['reserved_bytes_all_attempts'];result['charged_bytes']=client.ledger['charged_bytes'];result['retained_range_body_bytes']=sum(q['retained_body_bytes'] for q in client.ledger['requests']);result['coordinate_decode_accounting']='Exact completed80x80channel conversions; any exception inside a conversion is a STOP with no complete channel claim.'
  result['elapsed_seconds']=time.monotonic()-start;result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;result['pre_final_output_bytes']=sum(f.stat().st_size for f in out.rglob('*') if f.is_file());write_json(out/'result.staged.json',result)
  try:result['final_resource_check']=budget(out,start)
  except BaseException as e:result['status']='STOP';result['stop_reason']=str(e) if isinstance(e,K.ContractError) else type(e).__name__
  finalbytes=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file());K.require(size+len(finalbytes)<=LIMIT_OUTPUT,'final result exceeds output cap');(out/'result.json').write_bytes(finalbytes)
 print(json.dumps({'status':result['status'],'requests':result['request_count'],'charged_bytes':result['charged_bytes'],'result':record(out/'result.json'),'stop_reason':result.get('stop_reason')}));return 0 if result['status']=='PASS' else 2
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--go',required=True);parser.add_argument('--output-dir',required=True);args=parser.parse_args();raise SystemExit(run(args.go,args.output_dir))
