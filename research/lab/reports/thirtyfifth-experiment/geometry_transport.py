"""Exact-source HTTP ranges with permanent reservation accounting and no retry."""
import http.client, re, resource, signal, ssl, time, urllib.error, urllib.request
from geometry_common import *

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class Client:
    def __init__(self,root,plan,ledger=None,opener=None):
        self.root=Path(root);self.root.mkdir(exist_ok=True)
        self.plan=plan;self.urls=set(plan['urls']);self.start=time.monotonic()
        self.ledger=ledger or dict(requests=[],reserved_body_bytes=0,returned_body_bytes=0,
                                  active_acquisition_seconds=0,retries=0,redirects=0,verified_TLS=True)
        self.previous_seconds=self.ledger['active_acquisition_seconds']
        self.opener=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def limits(self):
        seconds=self.previous_seconds+time.monotonic()-self.start
        require(seconds<self.plan['limits']['acquisition_seconds'],'Acquisition deadline exceeded')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=512*1024**2,'Acquisition RSS exceeded')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=64*1024**2,'Phase disk exceeded')
        return self.plan['limits']['acquisition_seconds']-seconds

    def get(self,url,ranges,name,identity=None):
        require(url in self.urls,'URL outside frozen inventory')
        require(len(ranges)==1 or len(ranges)==108,'Exactly one or108 ranges required')
        require(len(set(map(tuple,ranges)))==len(ranges),'Duplicate request range')
        ordered=sorted(ranges)
        require(all(type(a)is int and type(b)is int and 0<=a<=b for a,b in ordered),'Invalid range')
        require(all(ordered[i-1][1]<ordered[i][0] for i in range(1,len(ordered))),'Overlapping ranges')
        require(Path(name).name==name and not name.startswith('.'),'Invalid file name')
        multipart=len(ranges)>1
        require(not multipart or identity is not None,'Multipart requires prior object identity')
        if identity:require(all(b<identity['total_bytes'] for _,b in ranges),'Range exceeds pinned object')
        useful=sum(b-a+1 for a,b in ranges)
        cap=useful+(768*len(ranges)+1024 if multipart else 0)
        require(cap<=512*1024,'Per-response cap')
        require(len(self.ledger['requests'])<self.plan['limits']['requests'],'Request budget')
        require(self.ledger['reserved_body_bytes']+cap+1<=self.plan['limits']['reserved_body_bytes'],'Reservation budget')
        remaining=self.limits();dest=self.root/name;partial=dest.with_suffix(dest.suffix+'.partial')
        require(not dest.exists() and not partial.exists(),'Fresh destination required')
        headers={'Range':'bytes='+','.join(f'{a}-{b}' for a,b in ranges),'Accept-Encoding':'identity','User-Agent':'Plinius-Regional-PPM-Geometry'}
        if identity:headers['If-Match']=identity['etag']
        entry=dict(url=url,ranges=ranges,request_headers=headers,reserved_body_bytes=cap+1,returned_body_bytes=0,success=False)
        self.ledger['requests'].append(entry);self.ledger['reserved_body_bytes']+=cap+1
        save(self.root/'LEDGER.json',self.ledger)
        started=time.monotonic();response=None
        old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('Absolute HTTP deadline')))
        signal.setitimer(signal.ITIMER_REAL,min(30,remaining))
        try:
            try:response=self.opener.open(urllib.request.Request(url,headers=headers),timeout=25)
            except urllib.error.HTTPError as error:response=error
            entry['status']=response.status
            keys=['Content-Length','Content-Range','Content-Type','Content-Encoding','Transfer-Encoding','ETag','Last-Modified']
            require(sum(len(k)+len(v)+4 for k,v in response.headers.raw_items())<=16384,'Header bound')
            require(all(len(response.headers.get_all(k,[]))<=1 for k in keys),'Duplicate response headers')
            entry['response_headers']={k:response.headers.get(k) for k in keys}
            require(response.status==206,'HTTP200/error/redirect rejected before body read')
            require(response.headers.get('Content-Encoding','identity') in ('','identity'),'Encoded body unsupported')
            require(response.headers.get('Transfer-Encoding') is None,'Chunked body unsupported')
            etag=response.headers.get('ETag','')
            require(re.fullmatch(r'"[\x21\x23-\x7e]+"',etag) is not None,'Strong ETag required')
            if identity:require(etag==identity['etag'],'Object ETag changed')
            if multipart:
                require(response.headers.get('Content-Range') is None,'Unexpected outer range')
                total=identity['total_bytes']
            else:
                a,b,total=content_range(response.headers.get('Content-Range'))
                require([a,b]==list(ranges[0]),'Wrong returned range')
                if identity:require(total==identity['total_bytes'],'Object length changed')
            declared=response.headers.get('Content-Length','')
            require(declared.isdigit() and 0<int(declared)<=cap,'Exact bounded Content-Length required')
            length=int(declared)
            if not multipart:require(length==useful,'Single-range size changed')
            with partial.open('xb') as f:
                while entry['returned_body_bytes']<length:
                    n=min(65536,length-entry['returned_body_bytes'])
                    try:block=response.read(n)
                    except http.client.IncompleteRead as e:
                        f.write(e.partial);entry['returned_body_bytes']+=len(e.partial)
                        raise ValueError('Interrupted body retained') from None
                    require(bool(block) and len(block)<=n,'Truncated/oversized body')
                    f.write(block);entry['returned_body_bytes']+=len(block)
                extra=response.read(1)
                if extra:
                    f.write(extra);entry['returned_body_bytes']+=len(extra)
                    raise ValueError('Excess body byte')
            if multipart:entry['parts']=parse_multipart(partial.read_bytes(),response.headers.get('Content-Type'),ranges,total)
            partial.replace(dest);entry.update(success=True,file=pin(dest),identity=dict(etag=etag,total_bytes=total))
        except Exception as e:
            entry['error']=type(e).__name__+': '+str(e)
            if partial.exists():entry['partial_file']=pin(partial)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
            if response is not None:response.close()
            entry['elapsed_seconds']=time.monotonic()-started
            self.ledger['returned_body_bytes']+=entry['returned_body_bytes']
            self.ledger['active_acquisition_seconds']=self.previous_seconds+time.monotonic()-self.start
            save(self.root/'LEDGER.json',self.ledger)
        self.limits()
        return dest,entry
