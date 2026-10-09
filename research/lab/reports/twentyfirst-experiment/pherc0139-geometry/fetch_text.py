"""One root-authorized, bounded public-text pass. No scientific array imports."""
import datetime,hashlib,http.client,json,os,pathlib,signal,ssl,time,urllib.error,urllib.request,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parent
PLAN_SHA='93673160d75d62d26bba18b2571ebd08667973da210c32590cba5b60a8ee5d0a'
class Stop(ValueError): pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): return None
def require(value,message):
 if not value: raise Stop(message)
def record(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def body_read(response,req,path,receipt):
 cap=req['maximum_reserved_body_bytes']; h=response.headers
 for name in ['Content-Length','Content-Encoding','Content-Type','ETag']:
  values=h.get_all(name,[]); require(len(values)<=1,'duplicate '+name)
 receipt['http_status']=response.status
 receipt['response_headers']={k:h.get(k) for k in ['Content-Length','Content-Encoding','Content-Type','ETag','Last-Modified'] if h.get(k) is not None}
 require(h.get('Content-Encoding','identity').lower()=='identity','nonidentity encoding')
 length=h.get('Content-Length'); length=int(length) if length is not None and length.isdecimal() else None
 if h.get('Content-Length') is not None: require(length is not None,'malformed content length')
 if length is not None: require(length<cap,'content length reaches reserved cap')
 total=0
 with path.open('xb') as out:
  while total<cap:
   try: data=response.read(min(65536,cap-total))
   except http.client.IncompleteRead as e:
    require(len(e.partial)<=cap-total,'partial exceeds remaining reservation');out.write(e.partial);total+=len(e.partial);receipt['retained_body_bytes']=total;raise Stop('truncated response') from None
   if not data:break
   require(len(data)<=cap-total,'read exceeds reservation');out.write(data);total+=len(data);receipt['retained_body_bytes']=total
 require(total<cap,'response reached cap without verified EOF')
 require(length is None or total==length,'content length mismatch')
 require(response.status==req['expected_status'],'unexpected HTTP status')
 receipt['body']=record(path)
 return path.read_bytes()
def parse_text(body,req):
 text=body.decode('utf-8');require('\x00' not in text,'NUL in text')
 if req['kind']=='XML object-name inventory':
  require('<!DOCTYPE' not in text.upper() and '<!ENTITY' not in text.upper(),'XML entity declaration')
  tree=ET.fromstring(text);ns='{http://s3.amazonaws.com/doc/2006-03-01/}'
  require(tree.tag==ns+'ListBucketResult','wrong inventory root')
  prefix='PHerc0139/segments/20250108000004-w029_2025010827/mesh/20250108000004-on-20260102150214-2.399um.tifxyz/'
  require(tree.findtext(ns+'Prefix')==prefix,'wrong inventory prefix');require(tree.findtext(ns+'Delimiter')=='/','wrong delimiter');require(tree.findtext(ns+'IsTruncated')=='false','incomplete inventory')
  keys=[n.findtext(ns+'Key') for n in tree.findall(ns+'Contents')]
  require(len(keys)<=100 and len(keys)==len(set(keys)),'invalid inventory count')
  require(all(isinstance(k,str) and k.startswith(prefix) and '/' not in k[len(prefix):] for k in keys),'non-immediate inventory object')
  return {'keys':keys,'meta_present':prefix+'meta.json' in keys}
 if req['kind']=='JSON metadata':
  value=json.loads(text);require(isinstance(value,dict),'metadata not object');return {'metadata':value}
 require(not text.lstrip().lower().startswith(('<!doctype html','<html')),'HTML instead of source')
 require(len(text)>0,'empty source');return {'utf8_source':True,'lines':len(text.splitlines())}
def alarm(signum,frame):raise Stop('request elapsed deadline')
def run():
 plan=json.loads((ROOT/'TEXT_QUERY_PLAN.json').read_text()); require(record(ROOT/'TEXT_QUERY_PLAN.json')['sha256']==PLAN_SHA,'plan hash')
 go=json.loads((ROOT.parent/'PHERC0139_TEXT_GO.json').read_text());require(go['status']=='GO' and go['plan']==record(ROOT/'TEXT_QUERY_PLAN.json'),'root GO binding')
 freeze=json.loads((ROOT/'FETCH_CODE_FREEZE.json').read_text());require(freeze['source']==record(pathlib.Path(__file__)),'source freeze');require(freeze['plan']==go['plan'],'freeze plan')
 require(plan['maximum_attempts']==10 and plan['maximum_response_body_bytes_all_attempts']==2097152 and sum(r['maximum_reserved_body_bytes'] for r in plan['requests'])<=2097152,'budget literal')
 out=ROOT/'text-run';out.mkdir(exist_ok=False);start=time.monotonic();ledger={'status':'RUNNING','plan':go['plan'],'root_go':record(ROOT.parent/'PHERC0139_TEXT_GO.json'),'code_freeze':record(ROOT/'FETCH_CODE_FREEZE.json'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requests':[],'charged_bytes':0}
 opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
 inventory=None;signal.signal(signal.SIGALRM,alarm)
 try:
  for req in plan['requests']:
   if req.get('condition'):
    require(inventory is not None and inventory['meta_present'],'conditional meta.json unavailable')
   require(len(ledger['requests'])<10 and time.monotonic()-start<330,'global attempt/time cap')
   cap=req['maximum_reserved_body_bytes'];require(ledger['charged_bytes']+cap<=2097152,'global bytes cap')
   row={'id':req['id'],'url':req['url'],'reserved_bytes':cap,'status':'RESERVED','retained_body_bytes':0};ledger['requests'].append(row);ledger['charged_bytes']+=cap;write(out/'ledger.json',ledger)
   attempt=time.monotonic();path=out/(req['id']+'.body');signal.alarm(30)
   try:
    request=urllib.request.Request(req['url'],headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-source-contract-audit/1'},method='GET')
    try: response=opener.open(request,timeout=30)
    except urllib.error.HTTPError as e: response=e
    with response:
     if not hasattr(response,'status'):response.status=response.code
     require(response.geturl()==req['url'],'response URL changed')
     body=body_read(response,req,path,row)
    row['parsed']=parse_text(body,req);row['status']='PASS';ledger['charged_bytes']-=cap-len(body)
    if req['id']=='transformed_mesh_inventory':inventory=row['parsed']
   except BaseException as e:
    row['status']='STOP';row['error_type']=type(e).__name__;row['stop_reason']=str(e) if isinstance(e,Stop) else type(e).__name__
    if path.exists():row['partial_body']=record(path);row['retained_body_bytes']=path.stat().st_size
    raise
   finally:
    signal.alarm(0);row['elapsed_seconds']=time.monotonic()-attempt;write(out/'ledger.json',ledger)
  ledger['status']='PASS'
 except BaseException as e:
  ledger['status']='STOP';ledger['stop_reason']=str(e) if isinstance(e,Stop) else type(e).__name__
 finally:
  ledger['elapsed_seconds']=time.monotonic()-start;ledger['actual_retained_body_bytes']=sum(r.get('retained_body_bytes',0) for r in ledger['requests']);write(out/'ledger.json',ledger);write(out/'result.json',ledger)
 print(json.dumps({'status':ledger['status'],'attempts':len(ledger['requests']),'charged_bytes':ledger['charged_bytes'],'retained_bytes':ledger['actual_retained_body_bytes'],'result':record(out/'result.json'),'stop_reason':ledger.get('stop_reason')}))
 return 0 if ledger['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(run())
