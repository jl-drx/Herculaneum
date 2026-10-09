"""Bounded pinned safetensors acquisition and independent offline value qualification.
No torch, pickle, model factory or implicit loading. Execute only under a root GO.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import ssl
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

class Stop(Exception):
    def __init__(self, code): self.code = code

def sha(data): return hashlib.sha256(data).hexdigest()
def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(65536): h.update(block)
    return h.hexdigest()
def pin(path):
    p = Path(path)
    return {'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': file_hash(p)}
def check_pin(record):
    if pin(record['path']) != record: raise Stop('INPUT_PIN_MISMATCH')
def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def durable_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'x', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path); sync_dir(path.parent)
def no_duplicates(pairs):
    d = {}
    for key, value in pairs:
        if key in d: raise Stop('DUPLICATE_JSON_KEY')
        d[key] = value
    return d
def safe_json(raw):
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(Stop('NONFINITE_JSON')))
    except (UnicodeError, ValueError, TypeError): raise Stop('MALFORMED_JSON') from None

def validate_header(raw, extent, reference):
    header = safe_json(raw)
    if not isinstance(header, dict): raise Stop('HEADER_ROOT_NOT_OBJECT')
    if '__metadata__' in header: raise Stop('UNEXPECTED_METADATA')
    metadata = header.get('__metadata__', {})
    if not isinstance(metadata, dict) or any(type(k) is not str or type(v) is not str for k,v in metadata.items()):
        raise Stop('INVALID_METADATA')
    named = {k:v for k,v in header.items() if k != '__metadata__'}
    expected = {x['name']:x for x in reference['tensor_schema']}
    if len(expected) != reference['expected_named_tensor_count'] or set(named) != set(expected):
        raise Stop('NAME_CARDINALITY_MISMATCH')
    intervals = []; numel_total = 0
    for name, tensor in named.items():
        e = expected[name]
        if not isinstance(tensor, dict) or set(tensor) != {'dtype','shape','data_offsets'}: raise Stop('DESCRIPTOR_FIELDS')
        shape = tensor['shape']; offsets = tensor['data_offsets']
        if tensor['dtype'] != 'F32' or not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape) or shape != e['shape']:
            raise Stop('DTYPE_SHAPE_MISMATCH')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets):
            raise Stop('INVALID_OFFSETS')
        start, end = offsets; n = math.prod(shape)
        if end < start or end-start != n*4 or n != e['numel'] or end-start != e['data_bytes']:
            raise Stop('OFFSET_LENGTH_MISMATCH')
        intervals.append((start, end, name)); numel_total += n
    cursor = 0
    for start,end,_ in sorted(intervals):
        if start != cursor: raise Stop('INTERVAL_OVERLAP_OR_GAP')
        cursor = end
    if cursor != extent-8-len(raw) or cursor != reference['expected_named_data_bytes'] or numel_total != reference['expected_named_numel']:
        raise Stop('DATA_EXTENT_TOTAL_MISMATCH')
    used=set()
    for group in reference['expected_factory_alias_groups']:
        names=group['names']
        if len(names)!=2 or len(set(names))!=2 or any(n not in named or n in used for n in names): raise Stop('REFERENCE_ALIAS_GROUP')
        used.update(names)
        if any(named[n]['shape'] != group['shape'] for n in names): raise Stop('ALIAS_SHAPE_MISMATCH')
        if named[names[0]]['data_offsets'] == named[names[1]]['data_offsets']: raise Stop('ALIASES_NOT_SEPARATELY_SERIALIZED')
    return named

def inspect_file(path, source, reference, retained_prefix, retained_header):
    """Hash and retained-header identity precede every value interpretation."""
    import numpy as np
    path = Path(path)
    if path.stat().st_size != source['expected_file_extent_bytes']: raise Stop('FULL_FILE_EXTENT_MISMATCH')
    digest = file_hash(path)
    if digest != source['expected_full_file_sha256']: raise Stop('FULL_FILE_SHA256_MISMATCH')
    with path.open('rb') as f:
        prefix=f.read(8)
        if prefix != retained_prefix: raise Stop('RETAINED_PREFIX_MISMATCH')
        hlen=struct.unpack('<Q', prefix)[0]
        if hlen != len(retained_header) or hlen%8: raise Stop('PREFIX_HEADER_LENGTH_MISMATCH')
        raw=f.read(hlen)
        if raw != retained_header: raise Stop('RETAINED_HEADER_MISMATCH')
    named=validate_header(raw, path.stat().st_size, reference)
    data_start=8+len(raw); tensor_records=[]; finite_count=0
    with path.open('rb') as f:
        for name in sorted(named):
            start,end=named[name]['data_offsets']; f.seek(data_start+start)
            remaining=end-start; h=hashlib.sha256(); count=0
            while remaining:
                block=f.read(min(65536,remaining))
                if not block or len(block)%4: raise Stop('TENSOR_READ_TRUNCATED')
                h.update(block); values=np.frombuffer(block,dtype='<f4')
                if not bool(np.isfinite(values).all()): raise Stop('NONFINITE_F32_VALUE')
                count += values.size; remaining -= len(block)
            finite_count += count
            tensor_records.append({'name':name,'elements':count,'all_finite':True,'raw_sha256':h.hexdigest()})
    aliases=[]
    with path.open('rb') as a, path.open('rb') as b:
        for group in reference['expected_factory_alias_groups']:
            an,bn=group['names']; ar=named[an]['data_offsets']; br=named[bn]['data_offsets']
            a.seek(data_start+ar[0]); b.seek(data_start+br[0]); remaining=ar[1]-ar[0]
            while remaining:
                n=min(65536,remaining); av=a.read(n); bv=b.read(n)
                if len(av)!=n or len(bv)!=n: raise Stop('ALIAS_READ_TRUNCATED')
                if av!=bv: raise Stop('FACTORY_ALIAS_VALUES_NOT_BYTE_EQUAL')
                remaining-=n
            aliases.append({'names':[an,bn], 'equal_bytes':ar[1]-ar[0], 'byte_equal':True})
    unique_numel=sum(x['numel'] for x in reference['tensor_schema'])-sum(math.prod(g['shape']) for g in reference['expected_factory_alias_groups'])
    unique_groups=len(named)-len(aliases)
    if unique_numel!=reference['retained_factory_unique_parameters'] or unique_groups!=reference['retained_factory_unique_storage_groups']:
        raise Stop('CONDITIONAL_CANONICAL_TOTAL_MISMATCH')
    return {'status':'FULL_FILE_INTEGRITY_AND_VALUES_QUALIFIED_LOAD_UNVERIFIED',
        'checkpoint':pin(path),'retained_prefix_and_header_exact':True,'full_file_sha256_verified':True,
        'tensor_name_count':len(named),'finite_named_elements':finite_count,
        'tensor_records':tensor_records,'alias_pair_count':len(aliases),'alias_pairs':aliases,
        'conditional_factory_unique_groups':unique_groups,'conditional_factory_unique_parameters':unique_numel,
        'shared_serialized_storage_claimed':False,'strict_model_load_verified':False,'model_calls':0,
        'accuracy_or_transfer_measured':False,'new_checkpoint_conversion_or_save':False}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None

@contextlib.contextmanager
def absolute_deadline(seconds):
    def expire(_signal,_frame): raise Stop('ABSOLUTE_REQUEST_DEADLINE')
    previous=signal.signal(signal.SIGALRM,expire); signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)

class Client:
    def __init__(self, plan, output, opener=None, deadline=absolute_deadline):
        self.plan=plan; self.output=Path(output); self.deadline=deadline
        self.opener=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.ledger={'status':'PREPARED','attempts':[],'observed_body_bytes':0,'redirect_count':0,
            'signed_URLs_retained':False,'raw_redirect_error_bodies_retained':False,'automatic_retries':0}
        self.save()
    def save(self): durable_json(self.output/'ledger.json',self.ledger)
    def host(self,url):
        try: p=urllib.parse.urlsplit(url); port=p.port
        except (ValueError,TypeError): raise Stop('MALFORMED_DESTINATION') from None
        if p.scheme!='https' or p.hostname not in self.plan['transport']['allowed_exact_hosts'] or port not in (None,443) or p.username is not None or p.password is not None or p.fragment:
            raise Stop('DESTINATION_OUTSIDE_ALLOWLIST')
        return p.hostname
    @staticmethod
    def single(response,key):
        values=response.headers.get_all(key,[])
        if len(values)!=1: raise Stop('MISSING_OR_DUPLICATE_'+key.upper().replace('-','_'))
        return values[0]
    def length(self,response):
        value=self.single(response,'Content-Length')
        if not re.fullmatch('[0-9]+',value): raise Stop('INVALID_CONTENT_LENGTH')
        return int(value)
    def encoding(self,response):
        if response.headers.get_all('Content-Encoding',[]) not in ([],['identity']): raise Stop('NON_IDENTITY_ENCODING')
        if response.headers.get_all('Transfer-Encoding',[]): raise Stop('TRANSFER_ENCODING_DISALLOWED')
    def consume(self,response,length,event,stream=None):
        if length+self.ledger['observed_body_bytes']>self.plan['limits']['body_bytes']: raise Stop('BODY_BUDGET')
        event['reserved_body_bytes']=length; self.save()
        h=hashlib.sha256(); remaining=length
        try:
            while remaining:
                n=min(65536,remaining)
                block=response.read(n)
                if not block: raise Stop('TRUNCATED_RESPONSE')
                # Real HTTPResponse.read(n) obeys n; reject a broken transport before writing it.
                self.ledger['observed_body_bytes']+=len(block); event['observed_body_bytes']+=len(block); h.update(block)
                if len(block)>n: raise Stop('TRANSPORT_OVERREAD')
                if stream is not None: stream.write(block)
                remaining-=len(block)
            event['body_complete']=True
        finally:
            event['observed_body_sha256']=h.hexdigest()
            if stream is not None: stream.flush(); os.fsync(stream.fileno())
            self.save()
    def download(self):
        source=self.plan['source']; url=source['canonical_url']; extent=source['expected_file_extent_bytes']
        expected_range=f'bytes 0-{extent-1}/{extent}'
        request_range=f'bytes=0-{extent-1}'
        while True:
            host=self.host(url)
            if len(self.ledger['attempts'])>=self.plan['limits']['attempts']: raise Stop('ATTEMPT_BUDGET')
            event={'attempt':len(self.ledger['attempts'])+1,'hostname':host,'method':'GET','range':request_range,
                'canonical_source_url':source['canonical_url'],'state':'DURABLY_RECORDED_BEFORE_OPEN','observed_body_bytes':0}
            remaining_budget=self.plan['limits']['body_bytes']-self.ledger['observed_body_bytes']
            if remaining_budget<extent: raise Stop('INSUFFICIENT_DURABLE_BODY_RESERVATION')
            event['pre_open_reserved_body_bytes']=extent
            self.ledger['attempts'].append(event); self.ledger['status']='ACTIVE'; self.save()
            response=None; started=time.monotonic()
            try:
                with self.deadline(self.plan['limits']['absolute_request_seconds']):
                    request=urllib.request.Request(url,method='GET',headers={'Range':request_range,'Accept-Encoding':'identity','If-Match':source['strong_etag']})
                    try: response=self.opener.open(request,timeout=min(30,self.plan['limits']['absolute_request_seconds']))
                    except urllib.error.HTTPError as e: response=e
                    status=response.status; event['status']=status
                    if status==200: raise Stop('HTTP200_RANGE_IGNORED')
                    self.encoding(response); length=self.length(response); event['declared_body_bytes']=length
                    if status in (301,302,303,307,308):
                        if self.ledger['redirect_count']>=self.plan['limits']['redirects']: raise Stop('REDIRECT_BUDGET')
                        if length>4096: raise Stop('REDIRECT_BODY_LIMIT')
                        target=urllib.parse.urljoin(url,self.single(response,'Location'))
                        target_host=self.host(target)
                        event['redirect_target_hostname']=target_host
                        self.consume(response,length,event)
                        self.ledger['redirect_count']+=1; event['state']='REDIRECT_COMPLETE'; url=target
                    elif status!=206:
                        if length>4096: raise Stop('ERROR_BODY_LIMIT')
                        self.consume(response,length,event); raise Stop('HTTP_STATUS_REJECTED')
                    else:
                        if length!=extent: raise Stop('CONTENT_LENGTH_MISMATCH')
                        if self.single(response,'Content-Range')!=expected_range: raise Stop('CONTENT_RANGE_MISMATCH')
                        if self.single(response,'ETag')!=source['strong_etag']: raise Stop('STRONG_ETAG_MISMATCH')
                        event.update({'Content_Range':expected_range,'ETag':source['strong_etag'],'Content_Encoding':'identity'})
                        partial=self.output/'checkpoint.partial'
                        with partial.open('xb') as f:
                            sync_dir(self.output); self.consume(response,length,event,f)
                        if file_hash(partial)!=source['expected_full_file_sha256']: raise Stop('FULL_FILE_SHA256_MISMATCH')
                        final=self.output/'dense_native_016000.safetensors'
                        if final.exists(): raise Stop('FINAL_ALREADY_EXISTS')
                        os.rename(partial,final); sync_dir(self.output)
                        event['state']='DOWNLOAD_HASH_VERIFIED'; self.ledger['status']='DOWNLOAD_HASH_VERIFIED'; self.save()
                        return final
            except Stop as e:
                event['state']='STOPPED'; event['stop_code']=e.code; raise
            except Exception as e:
                # Do not retain exception text, which may contain a signed URL.
                event['state']='STOPPED'; event['exception_type']=type(e).__name__; raise Stop('TRANSPORT_EXCEPTION') from None
            finally:
                if response is not None: response.close()
                event['elapsed_seconds']=time.monotonic()-started; self.save()

def preflight(plan_path,freeze_path,go_path):
    plan=safe_json(Path(plan_path).read_bytes()); freeze=safe_json(Path(freeze_path).read_bytes()); go=safe_json(Path(go_path).read_bytes())
    if go.get('status')!='GO' or go.get('plan_sha256')!=file_hash(plan_path) or go.get('code_freeze_sha256')!=file_hash(freeze_path): raise Stop('ROOT_GO_REQUIRED')
    for item in plan['inputs']+freeze['files']: check_pin(item)
    return plan

def worker(plan_path,freeze_path,go_path,output):
    plan=preflight(plan_path,freeze_path,go_path)
    resource.setrlimit(resource.RLIMIT_AS,(plan['limits']['address_space_bytes'],)*2)
    output=Path(output); output.mkdir(exist_ok=False); sync_dir(output.parent)
    durable_json(output/'inputs.json',{'plan':pin(plan_path),'code_freeze':pin(freeze_path),'root_GO':pin(go_path)})
    client=Client(plan,output)
    try:
        checkpoint=client.download()
        inputs={x['role']:x for x in plan['input_roles']}
        reference=safe_json(Path(inputs['reference']['path']).read_bytes())
        offline_started=time.monotonic()
        cpu_started=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime
        cpu_limit=math.ceil(cpu_started+plan['limits']['offline_cpu_seconds'])
        resource.setrlimit(resource.RLIMIT_CPU,(cpu_limit,cpu_limit))
        with absolute_deadline(plan['limits']['offline_wall_seconds']):
            result=inspect_file(checkpoint,plan['source'],reference,Path(inputs['prefix']['path']).read_bytes(),Path(inputs['header']['path']).read_bytes())
        result['offline_wall_seconds']=time.monotonic()-offline_started
        result['offline_cpu_seconds']=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime-cpu_started
        if result['offline_cpu_seconds']>plan['limits']['offline_cpu_seconds']: raise Stop('OFFLINE_CPU_LIMIT')
        result['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if result['max_rss_bytes']>plan['limits']['rss_bytes']: raise Stop('RSS_LIMIT')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>plan['limits']['output_bytes']-1048576: raise Stop('OUTPUT_BYTE_LIMIT')
        durable_json(output/'qualification.json',result)
        client.ledger['status']='FULL_FILE_INTEGRITY_AND_VALUES_QUALIFIED_LOAD_UNVERIFIED'; client.save()
        return 0
    except Stop as e:
        client.ledger['status']='STOPPED'; client.ledger['stop_code']=e.code; client.save()
        durable_json(output/'STOPPED.json',{'status':'STOPPED','stop_code':e.code,'no_retry':True})
        return 2

def controller(args):
    plan=preflight(args.plan,args.freeze,args.go); started=time.monotonic()
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--plan',args.plan,'--freeze',args.freeze,'--go',args.go,'--output',args.output]
    child=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    reason=None; observed_max_rss=0
    try:
        while child.poll() is None:
            if time.monotonic()-started>plan['limits']['external_wall_seconds']: reason='EXTERNAL_WALL_LIMIT'; break
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                match=re.search(r'^VmRSS:\s+(\d+) kB$',status,re.M)
                if match: observed_max_rss=max(observed_max_rss,int(match.group(1))*1024)
            except FileNotFoundError: pass
            if observed_max_rss>plan['limits']['rss_bytes']: reason='RSS_LIMIT'; break
            time.sleep(.05)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=2)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
    result={'status':'STOPPED' if reason or child.returncode else 'PASS','stop_code':reason,'returncode':child.returncode,
        'wall_seconds':time.monotonic()-started,'observed_peak_rss_bytes':observed_max_rss,
        'rss_monitor_interval_seconds':.05,'address_space_kernel_limit_bytes':plan['limits']['address_space_bytes'],
        'output':str(Path(args.output).resolve()),'partial_outputs_retained':True,'automatic_retries':0}
    out=Path(args.output)
    if not out.exists(): out.mkdir(exist_ok=False)
    durable_json(out/'controller.json',result)
    print(json.dumps(result,sort_keys=True)); return 0 if result['status']=='PASS' else 2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',required=True); p.add_argument('--freeze',required=True); p.add_argument('--go',required=True); p.add_argument('--output',required=True); p.add_argument('--worker',action='store_true'); args=p.parse_args()
    try: return worker(args.plan,args.freeze,args.go,args.output) if args.worker else controller(args)
    except Stop as e: print(json.dumps({'status':'STOPPED','code':e.code})); return 2
    except Exception as e: print(json.dumps({'status':'STOPPED','exception_type':type(e).__name__})); return 2
if __name__=='__main__': raise SystemExit(main())
