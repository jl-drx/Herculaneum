"""Strict fixed-inventory IR and sparse PPM transport. No numerical decoder."""
import datetime
import hashlib
import http.client
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from acquisition_common import ContractError, require, record, save, content_range, parse_multipart


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AcquisitionClient:
    def __init__(self, root, requests, limits, plan_record, opener=None):
        self.root=Path(root)
        require(not self.root.exists(),'Fresh acquisition directory required')
        self.root.mkdir(parents=True); (self.root/'raw').mkdir()
        self.requests=requests; self.limits=limits
        self.opener=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.ledger=dict(schema_version=1,plan=plan_record,limits=limits,requests=[],charged_response_bytes=0,
            retained_body_bytes=0,ir_retained_body_bytes=0,redirects_followed=0,automatic_retries=0,verified_tls=True,
            accounting='Reserve each bounded response before its attempt; successful complete bodies charge actual bytes; failed attempts retain full reservation. Retained body bytes are separate from reservations and are not total wire traffic.')
        save(self.root/'request-ledger.json',self.ledger)

    def get(self, request):
        require(len(self.ledger['requests'])<len(self.requests) and request==self.requests[len(self.ledger['requests'])],
                'Only next frozen request may execute; no retry')
        url=request['url']; mode=request['mode']; ranges=request.get('ranges',[]); pin=request.get('pin')
        parsed=urllib.parse.urlsplit(url)
        require(parsed.scheme=='https' and parsed.netloc=='dl.ash2txt.org' and not parsed.query and not parsed.fragment
                and re.fullmatch(r'/fragments/Frag[123]/PHercParis[12]Fr(?:47|143|34)\.volpkg/working/54keV_exposed_surface/(?:ir\.png|result\.ppm|surface_volume\.ppm)',parsed.path),
                'Exact frozen official IR/PPM source required')
        require(mode in ('full','single','multipart'),'Unsupported request mode')
        if mode=='full':
            require(url.endswith('/ir.png') and not ranges and pin is None,'Only full IR requests supported')
            useful=None; cap=48*1024**2
        else:
            require(url.endswith('.ppm') and all(type(a)is int and type(b)is int and 0<=a<=b for a,b in ranges),'Invalid PPM spans')
            require((mode=='single' and ranges==[[0,1023]] and pin is None) or
                    (mode=='multipart' and len(ranges)==243 and pin is not None),'Exact PPM prefix/sample mode required')
            useful=sum(b-a+1 for a,b in ranges)
            cap=useful+(1 if mode=='single' else 768*len(ranges)+1024)
            if pin: require(all(b<pin['object_total_bytes'] for a,b in ranges),'Range exceeds pinned object')
        require(request['reserved_cap_bytes']==cap and cap<=self.limits['per_response'] and
                len(self.ledger['requests'])<self.limits['attempts'] and self.ledger['charged_response_bytes']+cap<=self.limits['body_bytes'],
                'Frozen attempt/body budget exhausted')
        name=request['request_id']+'.bin'
        require(Path(name).name==name and not name.startswith('.'),'Unsafe artifact name')
        headers={'Accept-Encoding':'identity'}
        if ranges: headers['Range']='bytes='+','.join(f'{a}-{b}'for a,b in ranges)
        if pin: headers['If-Match']=pin['etag']
        entry=dict(request_id=request['request_id'],url=url,mode=mode,ranges_inclusive=ranges,request_headers=headers,
            success=False,reserved_cap_bytes=cap,charged_response_bytes=cap,retained_body_bytes=0,
            started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.ledger['requests'].append(entry);self.ledger['charged_response_bytes']+=cap
        save(self.root/'request-ledger.json',self.ledger)
        partial=self.root/'raw'/(name+'.partial');path=self.root/'raw'/name
        require(not path.exists()and not partial.exists(),'Artifact already exists')
        digest=hashlib.sha256();retained=0;response=None;started=time.monotonic()
        try:
            try: response=self.opener.open(urllib.request.Request(url,headers=headers),timeout=25)
            except urllib.error.HTTPError as error: response=error
            entry['http_status']=response.status
            entry['http_identity']={k.lower():v for k,v in response.headers.items() if k.lower() in
                ('content-length','content-range','content-type','content-encoding','etag','last-modified')}
            for key in ('Content-Length','Content-Range','Content-Type','Content-Encoding','ETag','Transfer-Encoding'):
                values=response.headers.get_all(key) if hasattr(response.headers,'get_all') else ([response.headers[key]]if key in response.headers else [])
                require(len(values or [])<=1,'Duplicate singleton HTTP identity/framing header')
            require(response.status==(200 if mode=='full'else 206),'Unexpected HTTP status rejected before body; no fallback')
            require(response.headers.get('Content-Encoding','identity').lower()in ('','identity'),'Encoded HTTP body rejected')
            require(response.headers.get('Transfer-Encoding') is None,'Transfer encoding unsupported; exact Content-Length required')
            etag=response.headers.get('ETag','')
            require(re.fullmatch(r'"[\x21\x23-\x7e]+"',etag)is not None,'Quoted strong ETag required')
            if pin:require(etag==pin['etag'],'ETag changed')
            declared=response.headers.get('Content-Length','')
            require(declared.isdigit()and 0<int(declared)<=cap,'Exact bounded Content-Length required')
            length=int(declared)
            if mode=='full':
                require(response.headers.get('Content-Range')is None,'Unexpected range on full IR response')
                require(self.ledger['ir_retained_body_bytes']+length<=96*1024**2,'Total IR cap exceeded')
                total=length
            elif mode=='single':
                start,end,total=content_range(response.headers.get('Content-Range'))
                require([start,end]==ranges[0] and length==useful,'Prefix extent/length mismatch')
            else:
                require(response.headers.get('Content-Range')is None,'Top-level multipart range rejected')
                total=pin['object_total_bytes']
            with partial.open('xb')as stream:
                while retained<length:
                    require(time.monotonic()-started<60,'Request body wall limit exceeded')
                    try:block=response.read(min(65536,length-retained))
                    except http.client.IncompleteRead as error:
                        block=error.partial
                        require(len(block)<=length-retained,'Truncated partial exceeds cap')
                        stream.write(block);stream.flush();os.fsync(stream.fileno())
                        digest.update(block);retained+=len(block)
                        raise ContractError('Truncated transfer; partial preserved')from None
                    require(block and len(block)<=length-retained,'Truncated or overflowing bounded body')
                    stream.write(block);digest.update(block);retained+=len(block)
                stream.flush();os.fsync(stream.fileno())
            require(retained==length,'Actual body length mismatch')
            if mode=='multipart':entry['parts']=parse_multipart(partial.read_bytes(),response.headers.get('Content-Type'),ranges,total)
            partial.replace(path)
            entry.update(success=True,etag=etag,object_total_bytes=total,file=record(path),charged_response_bytes=retained)
            self.ledger['charged_response_bytes']-=cap-retained
            if mode=='full':self.ledger['ir_retained_body_bytes']+=retained
            return path,entry
        except Exception as error:
            entry['failure_type']=type(error).__name__
            if isinstance(error,ContractError):entry['stop_reason']=str(error)
            raise
        finally:
            if response is not None:response.close()
            entry.update(retained_body_bytes=retained,body_sha256=digest.hexdigest(),elapsed_seconds=time.monotonic()-started,
                         completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if partial.exists():entry['partial_file']=record(partial)
            self.ledger['retained_body_bytes']+=retained
            save(self.root/'request-ledger.json',self.ledger)
