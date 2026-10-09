"""Exactly one frozen, bounded PPM multipart request. No retries or fallback."""
import hashlib
import http.client
import os
from pathlib import Path
import re
import ssl
import time
import urllib.error
import urllib.request
from common import LIMITS, ContractError, deadline, now, pin, require, resources, save
from multipart import parse_multipart

URL = "https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/result.ppm"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MultipartClient:
    def __init__(self,root,object,ranges,plan_record,opener=None,limits=None):
        self.root=Path(root);require(not self.root.exists(),"Fresh acquisition directory required")
        self.root.mkdir(parents=True);(self.root/"raw").mkdir()
        self.object,self.ranges=object,ranges
        self.limits=dict(LIMITS if limits is None else limits)
        self.opener=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.ledger=dict(schema="herculaneum.frag6-stage-b-ledger.v1",created_utc=now(),plan=plan_record,
                         requests=[],reserved_response_body_bytes=0,retained_response_body_bytes=0,
                         automatic_retries=0,redirects_followed=0,verified_tls=True,limits=self.limits,
                         accounting="Reserve the complete 222640-byte response-body ceiling before the sole attempt. Content-Length plus one excess-byte probe must fit it; failed reservations are never refunded. Returned partial bytes are retained; not total wire traffic.")
        self.write_ledger()

    def write_ledger(self):
        save(self.root/"request-ledger.json",self.ledger)

    def get(self):
        obj,ranges=self.object,self.ranges
        require(obj["url"]==URL,"Only exact frozen Frag6 PPM URL is supported")
        require(len(ranges)==243 and len(set(map(tuple,ranges)))==243,"Exactly 243 unique ranges required")
        require(all(type(a) is int and type(b) is int and 0<=a<=b<obj["total_bytes"] and b-a+1==144 for a,b in ranges),"Invalid PPM record row span")
        sorted_ranges=sorted(ranges)
        require(all(a[1]<b[0] for a,b in zip(sorted_ranges,sorted_ranges[1:])),"Overlapping scheduled PPM spans")
        require(not self.ledger["requests"] and self.limits["maximum_attempts"]>=1,"Only one attempt; no retry")
        cap=222640
        require(cap<=self.limits["maximum_reserved_response_body_bytes"],"Body reservation budget exhausted")
        resources(self.root,self.limits)
        headers={"Range":"bytes="+",".join(f"{a}-{b}" for a,b in ranges),"If-Match":obj["etag"],
                 "Accept-Encoding":"identity","User-Agent":"Herculaneum-Frag6-Sparse-PPM-StageB"}
        entry=dict(url=obj["url"],method="GET",ranges_inclusive=ranges,request_headers=headers,
                   reserved_response_body_bytes=cap,retained_body_bytes=0,success=False,started_utc=now())
        self.ledger["requests"].append(entry);self.ledger["reserved_response_body_bytes"]=cap
        self.write_ledger()
        target,partial=self.root/"raw/multipart.bin",self.root/"raw/multipart.bin.partial"
        response,retained=None,0;digest=hashlib.sha256();started=time.monotonic()
        try:
            with deadline(self.limits["per_request_wall_seconds"]):
                try:
                    response=self.opener.open(urllib.request.Request(obj["url"],headers=headers,method="GET"),timeout=self.limits["connect_timeout_seconds"])
                except urllib.error.HTTPError as error:
                    response=error
                entry["http_status"]=response.status
                observed=list(response.headers.raw_items())
                require(sum(len(str(k).encode())+len(str(v).encode())+4 for k,v in observed)<=16384,"HTTP metadata bound exceeded")
                entry["response_headers"]=observed
                require(response.status==206,"Expected 206 multipart; no error/redirect/full-object body read")
                for key in ("Content-Length","Content-Range","Content-Type","Content-Encoding","Transfer-Encoding","ETag","Last-Modified"):
                    require(len(response.headers.get_all(key,[]))<=1,"Duplicate singleton response header: "+key)
                require(response.headers.get("Content-Encoding","identity").lower() in ("","identity"),"Encoded body unsupported")
                require(response.headers.get("Transfer-Encoding") is None,"Transfer encoding unsupported")
                require(response.headers.get("Content-Range") is None,"Top-level Content-Range forbidden on multipart response")
                etag=response.headers.get("ETag","")
                require(re.fullmatch(r'"[\x21\x23-\x7e]+"',etag) is not None and etag==obj["etag"],"Strong pinned PPM ETag mismatch")
                text=response.headers.get("Content-Length","")
                require(text.isdigit() and 34992<int(text)<cap,"Bounded exact multipart Content-Length plus excess-byte allowance required")
                length=int(text);entry["declared_body_bytes"]=length
                content_type=response.headers.get("Content-Type","")
                require(re.fullmatch(r'multipart/byteranges\s*;\s*boundary=(?:"([A-Za-z0-9\x27()+_,./:=?-]{1,70})"|([A-Za-z0-9\x27+_.-]{1,70}))\s*',content_type,re.I) is not None,"One bounded multipart boundary required")
                with partial.open("xb") as stream:
                    while retained<length:
                        try:
                            block=response.read(min(self.limits["stream_chunk_bytes"],length-retained))
                        except http.client.IncompleteRead as error:
                            block=error.partial
                            require(len(block)<=cap-retained,"Partial response exceeded full reservation")
                            stream.write(block);stream.flush();os.fsync(stream.fileno());digest.update(block);retained+=len(block)
                            raise ContractError("Incomplete multipart response; partial retained") from None
                        require(block and len(block)<=cap-retained,"Truncated or oversized multipart response")
                        stream.write(block);stream.flush();os.fsync(stream.fileno());digest.update(block);retained+=len(block)
                        require(retained<=length,"Response exceeded declared useful length")
                        resources(self.root,self.limits)
                    extra=response.read(1)
                    require(len(extra)<=1,"Excess-byte probe violated bound")
                    if extra:
                        stream.write(extra);stream.flush();os.fsync(stream.fileno());digest.update(extra);retained+=len(extra)
                        raise ContractError("Unexpected excess multipart byte")
                    stream.flush();os.fsync(stream.fileno())
                body=partial.read_bytes()
                parts=parse_multipart(body,content_type,ranges,obj["total_bytes"])
                partial.replace(target)
                descriptor=os.open(target.parent,os.O_DIRECTORY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
                entry.update(success=True,file=pin(target),parts=parts)
        except Exception as error:
            entry.update(failure_type=type(error).__name__,stop_reason=str(error))
            raise
        finally:
            if response is not None:response.close()
            entry.update(retained_body_bytes=retained,body_sha256=digest.hexdigest(),
                         elapsed_seconds=time.monotonic()-started,completed_utc=now())
            if partial.exists():entry["partial_file"]=pin(partial)
            self.ledger["retained_response_body_bytes"]=retained
            self.write_ledger()
        resources(self.root,self.limits)
        return body,parts,entry
