"""Bounded metadata-only discovery; no scientific array reader."""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import ssl
import time
import urllib.error
import urllib.request

BASE='https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/53keV_3.24um_.zarr/'


def require(ok,why):
    if not ok:raise ValueError(why)


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.writing')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def unique_object(pairs):
    value={}
    for key,item in pairs:
        require(key not in value,'Duplicate JSON key');value[key]=item
    return value


def decode_json(body):
    result=json.loads(body.decode('utf-8'),object_pairs_hook=unique_object,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON token')))
    require(isinstance(result,dict),'Metadata root must be an object')
    return result


def safe_dataset(path):
    require(isinstance(path,str) and len(path)<=160 and
            re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*',path) is not None,'Unsafe dataset path')
    return path


def metadata_name(path):
    parts=path.rsplit('/',1)
    require(parts[-1] in ('.zattrs','.zarray','.zgroup','zarr.json'),'Only metadata names allowed')
    if len(parts)==2:safe_dataset(parts[0])
    return path


@contextlib.contextmanager
def deadline():
    def expired(*_):raise TimeoutError('Absolute metadata request deadline')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,30)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kw):return None


class Fetcher:
    def __init__(self,output,opener=None):
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=False)
        self.opener=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.log=dict(base_url=BASE,requests=[],reserved_body_bytes=0,returned_body_bytes=0,retries=0,redirects_followed=0,verified_tls=True)
        self.flush()

    def flush(self):save(self.root/'request-ledger.json',self.log)

    def get(self,name,cap):
        metadata_name(name);require(0<cap<=65536,'Response cap')
        require(len(self.log['requests'])<8 and self.log['reserved_body_bytes']+cap+1<=262144,'Acquisition budget')
        require(name not in [r['relative_path'] for r in self.log['requests']],'Duplicate request/no retries')
        entry=dict(relative_path=name,url=BASE+name,reserved_bytes=cap+1,success=False,returned_bytes=0)
        self.log['requests'].append(entry);self.log['reserved_body_bytes']+=cap+1;self.flush()
        response=None;body=bytearray();tick=time.monotonic()
        try:
            with deadline():
                try:response=self.opener.open(urllib.request.Request(BASE+name,headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-Storage-Metadata-Audit'}),timeout=15)
                except urllib.error.HTTPError as error:response=error
                h=response.headers;entry['status']=response.status
                keys=['Content-Length','Content-Type','Content-Encoding','Transfer-Encoding','ETag','Last-Modified']
                require(sum(len(k)+len(v)+4 for k,v in h.raw_items())<=16384,'Header cap')
                entry['headers']={k:h.get_all(k,[]) for k in keys}
                require(all(len(h.get_all(k,[]))<=1 for k in keys),'Duplicate HTTP framing/identity header')
                require(response.status in (200,404),'Unexpected status/no redirects')
                require(h.get('Content-Encoding','identity').lower() in ('','identity') and h.get('Transfer-Encoding') is None,'Encoded/chunked body refused')
                require(h.get('Content-Length','').isdigit(),'Exact content length required')
                length=int(h['Content-Length']);require(length<=cap,'Response body cap')
                mime=h.get('Content-Type','').split(';')[0].strip().lower()
                if response.status==200:require(mime in ('application/json','text/plain','application/octet-stream'),'Unexpected metadata MIME')
                while len(body)<length:
                    try:block=response.read(length-len(body))
                    except http.client.IncompleteRead as error:
                        body.extend(error.partial);raise ValueError('Incomplete response retained') from None
                    require(block and len(block)<=length-len(body),'Truncated/oversized response');body.extend(block)
                extra=response.read(1);body.extend(extra);require(not extra,'Excess response retained')
                parsed=decode_json(body) if response.status==200 else None
                entry['success']=True
                return parsed
        except Exception as error:
            entry['error']=type(error).__name__+': '+str(error);raise
        finally:
            if response is not None:response.close()
            if body:
                path=self.root/f'{len(self.log["requests"]):02d}-metadata.body';path.write_bytes(body);entry['body']=pin(path)
            entry['returned_bytes']=len(body);entry['elapsed_seconds']=time.monotonic()-tick
            self.log['returned_body_bytes']+=len(body);self.flush()


def first_dataset(attributes):
    if 'multiscales' not in attributes:return ''
    multi=attributes['multiscales'];require(isinstance(multi,list) and len(multi)==1,'One declared multiscale image required')
    datasets=multi[0].get('datasets');require(isinstance(datasets,list) and datasets,'Declared datasets required')
    return safe_dataset(datasets[0]['path'])


def acquire(fetcher):
    attrs=fetcher.get('.zattrs',65536)
    if attrs is not None:
        group=fetcher.get('.zgroup',4096)
        dataset=first_dataset(attrs)
        prefix=dataset+'/' if dataset else ''
        array=fetcher.get(prefix+'.zarray',16384);require(array is not None,'Selected array metadata absent')
        node_attrs=fetcher.get(prefix+'.zattrs',16384) if dataset else attrs
        return dict(storage_format='zarr_v2',root_attributes=attrs,root_group=group,
                    array_path=dataset,array_metadata=array,array_attributes=node_attrs)
    root=fetcher.get('zarr.json',65536);require(root is not None,'No supported root metadata')
    require(root.get('zarr_format')==3,'Unsupported root format')
    dataset=first_dataset(root.get('attributes',{})) if root.get('node_type')=='group' else ''
    require(root.get('node_type')=='array' or dataset,'Group requires a declared first dataset')
    array=fetcher.get(dataset+'/zarr.json',65536) if dataset else root
    require(array is not None and array.get('node_type')=='array','Selected v3 array missing')
    return dict(storage_format='zarr_v3',root_metadata=root,array_path=dataset,array_metadata=array)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--go',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    go=json.loads(Path(args.go).read_text());require(go['status']=='GO_METADATA_ONLY' and go['base_url']==BASE,'Exact metadata GO required')
    for item in go['files']:require(pin(item['path'])==item,'Frozen source changed')
    f=Fetcher(args.output)
    try:
        result=acquire(f)
        for item in go['files']:require(pin(item['path'])==item,'Source changed during acquisition')
        result.update(status='PASS_METADATA_ACQUISITION',base_url=BASE,go=pin(args.go),requests=len(f.log['requests']),
                      returned_body_bytes=f.log['returned_body_bytes'],reserved_body_bytes=f.log['reserved_body_bytes'],candidate_chunks_read=0)
        save(f.root/'METADATA_INVENTORY.json',result)
        print(json.dumps(result,indent=2))
    except Exception as error:
        save(f.root/'STOP.json',dict(status='STOP',reason=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
