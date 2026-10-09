"""Root-authorized one-pass structural metadata ranges; no coordinate decode."""
import datetime,hashlib,importlib.util,json,math,pathlib,signal,ssl,time,urllib.error,urllib.request
import tiff_metadata as T
ROOT=pathlib.Path(__file__).resolve().parent;P=ROOT.parent;PLAN_SHA='0f45ef77043d518a029ccdef3c102827f3264357f3eb2e86189ba662d8bfc49e'
def rec(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def load_helper(expected):
 T.require(rec(P/'fetch_text.py')==expected,'sealed transport helper changed');s=importlib.util.spec_from_file_location('sealed_text_helper',P/'fetch_text.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Acquisition:
 def __init__(self,plan,out,m):
  self.plan=plan;self.out=out;self.m=m;self.start=time.monotonic();self.ledger={'status':'RUNNING','requests':[],'charged_bytes':0,'reserved_bytes_all_attempts':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};self.opener=urllib.request.build_opener(m.NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
 def get(self,obj,label,lo,length,cap):
  T.extent(lo,length,obj['object_size_bytes']);T.require(cap>length and cap<=32769,'request cap');T.require(time.monotonic()-self.start<510,'global deadline');T.require(len(self.ledger['requests'])<16 and self.ledger['reserved_bytes_all_attempts']+cap<=262144,'request/reservation budget')
  hi=lo+length-1;row={'object':obj['name'],'url':obj['url'],'range':[lo,hi],'if_match':obj['strong_etag'],'reserved_body_bytes':cap,'status':'RESERVED','retained_body_bytes':0};self.ledger['requests'].append(row);self.ledger['reserved_bytes_all_attempts']+=cap;self.ledger['charged_bytes']+=cap;self.save()
  path=self.out/(obj['name']+'.'+label+'.body');start=time.monotonic();signal.alarm(30)
  try:
   req=urllib.request.Request(obj['url'],headers={'Range':f'bytes={lo}-{hi}','If-Match':obj['strong_etag'],'Accept-Encoding':'identity','User-Agent':'Herculaneum-TIFF-structure/1'},method='GET')
   try:r=self.opener.open(req,timeout=30)
   except urllib.error.HTTPError as e:r=e
   with r:
    row['http_status']=r.status;row['response_headers']={k:r.headers.get(k) for k in ['Content-Range','Content-Length','ETag','Content-Encoding','Last-Modified'] if r.headers.get(k) is not None}
    T.require(r.geturl()==obj['url'],'URL changed');T.validate_http(r,obj,lo,hi)
    b=self.m.body_read(r,{'maximum_reserved_body_bytes':cap,'expected_status':206},path,row)
   T.require(len(b)==length,'range body length');row['status']='PASS';self.ledger['charged_bytes']-=cap-len(b);return b
  except BaseException as e:
   row['status']='STOP';row['error_type']=type(e).__name__;row['stop_reason']=str(e) if isinstance(e,(T.ContractError,self.m.Stop)) else type(e).__name__
   if path.exists():row['partial_body']=rec(path);row['retained_body_bytes']=path.stat().st_size
   raise
  finally:signal.alarm(0);row['elapsed_seconds']=time.monotonic()-start;self.save()
 def save(self):self.m.write(self.out/'ledger.json',self.ledger)
def run():
 freeze=json.loads((ROOT/'CODE_FREEZE.json').read_text())
 for record in freeze['sources']:
  T.require(rec(pathlib.Path(record['path']))==record,'frozen source differs')
 planpath=P/'TIFF_STRUCTURAL_PLAN.json';T.require(rec(planpath)['sha256']==PLAN_SHA,'plan hash');plan=json.loads(planpath.read_text());go_path=P.parent/'PHERC0139_TIFF_GO.json';go=json.loads(go_path.read_text());T.require(go['status']=='GO' and go['plan']==rec(planpath) and freeze['root_go']==rec(go_path),'root GO')
 
 for key in ('source_inventory','text_result','exact_renderer_source','exact_loader_source','metadata'):
  pin=plan[key];T.require(rec(pathlib.Path(pin['path']))==pin,'plan input pin '+key)
 T.require(plan['maximum_attempts']==16 and plan['maximum_response_body_bytes_all_attempts']==262144,'literal limits');m=load_helper(freeze['transport_helper']);out=ROOT/'run';out.mkdir(exist_ok=False);a=Acquisition(plan,out,m);signal.signal(signal.SIGALRM,m.alarm);result={'status':'RUNNING','plan':rec(planpath),'root_go':rec(go_path),'freeze':rec(ROOT/'CODE_FREEZE.json'),'objects':[],'coordinate_values_decoded':False}
 try:
  for obj in plan['objects']:
   b=a.get(obj,'header',0,8,9);head=T.header(b,obj['object_size_bytes']);b=a.get(obj,'ifd-count',head['first_ifd_offset'],2,3);n=T.count(b,head);b=a.get(obj,'ifd',head['first_ifd_offset']+2,12*n+4,3077);doc=T.parse_ifd(b,head,n);lay=T.layout(doc);external={}
   for label,tag in [('offsets',lay['offset_tag']),('counts',lay['bytecount_tag'])]:
    e=doc['tags'][tag]
    if not e['inline']:external[tag]=a.get(obj,label,e['extent'][0],e['length'],32769)
   contract=T.finish(doc,lay,external);contract['source']=obj;path=out/(obj['name']+'.contract.json');m.write(path,contract);result['objects'].append(rec(path))
  contracts=[json.loads(pathlib.Path(r['path']).read_text()) for r in result['objects']];signature=lambda d:tuple(d[k] for k in ('width','height','bits_per_sample','sample_format','samples_per_pixel','orientation'))
  T.require(len({signature(d) for d in contracts})==1,'XYZ dimensions/sample representation mismatch');meta=json.loads((P/'text-run/transformed_mesh_meta.body').read_text());sx,sy=meta['scale'];w,h=contracts[0]['width'],contracts[0]['height'];canvas=[math.floor(w/sx+.5),math.floor(h/sy+.5)];result['observed_mesh_width_height']=[w,h];result['computed_render_canvas_width_height']=canvas;result['published_canvas_width_height']=[28880,28080];result['raster_reconciliation']=canvas==[28880,28080];T.require(result['raster_reconciliation'],'public raster mismatch');result['status']='PASS'
 except BaseException as e:result['status']='STOP';result['error_type']=type(e).__name__;result['stop_reason']=str(e) if isinstance(e,(T.ContractError,m.Stop)) else type(e).__name__
 finally:
  a.ledger['status']=result['status'];a.save();result['elapsed_seconds']=time.monotonic()-a.start;result['requests']=len(a.ledger['requests']);result['charged_bytes']=a.ledger['charged_bytes'];result['reserved_bytes_all_attempts']=a.ledger['reserved_bytes_all_attempts'];result['actual_retained_body_bytes']=sum(r.get('retained_body_bytes',0) for r in a.ledger['requests']);result['ledger']=rec(out/'ledger.json');result['missing_geometry_inputs']=['XYZ coordinate payloads remain unread; no per-pixel mesh validity or normals reconstructed.','No native CT chunks/masking observations; historical private optional mask and exact renderer output identity remain unproven.'];m.write(out/'result.json',result)
 print(json.dumps({'status':result['status'],'requests':result['requests'],'body_bytes':result['actual_retained_body_bytes'],'result':rec(out/'result.json'),'dimensions':result.get('observed_mesh_width_height'),'raster':result.get('computed_render_canvas_width_height'),'stop_reason':result.get('stop_reason')}));return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(run())
