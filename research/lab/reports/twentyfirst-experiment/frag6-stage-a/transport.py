"""Strict single-range transport; no multipart, retries, redirects or fallback."""
import hashlib
import http.client
import os
from pathlib import Path
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from common import LIMITS, ContractError, deadline, now, pin, require, resources, save


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RangeClient:
    def __init__(self, root, objects, plan_record, opener=None, limits=None):
        self.root = Path(root)
        require(not self.root.exists(), "Fresh acquisition directory required")
        self.root.mkdir(parents=True)
        (self.root/"raw").mkdir()
        self.limits = dict(LIMITS if limits is None else limits)
        self.objects = {o["url"]:o for o in objects}
        self.opener = opener or urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.ledger = dict(schema="herculaneum.frag6-stage-a-ledger.v1", plan=plan_record,
                           created_utc=now(), requests=[], reserved_response_body_bytes=0,
                           retained_response_body_bytes=0, automatic_retries=0,
                           redirects_followed=0, verified_tls=True, limits=self.limits,
                           accounting="Each attempt permanently reserves useful length+1 before connection. Partial and retained bytes are separate; this is not total wire traffic.")
        self.write_ledger()

    def write_ledger(self):
        save(self.root/"request-ledger.json",self.ledger)

    def get(self, url, start, end, name, kind):
        require(url in self.objects, "URL outside frozen object list")
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == "https" and parsed.netloc == "dl.ash2txt.org" and not parsed.query and not parsed.fragment, "Unexpected public source URL")
        obj = self.objects[url]
        require(type(start) is int and type(end) is int and 0 <= start <= end < obj["total_bytes"], "Invalid object byte range")
        length = end-start+1
        require(kind in ("ppm_prefix","tiff_header","ifd_count","ifd_table","description"), "Unsupported request category")
        if kind == "ppm_prefix":
            require(url.endswith("/result.ppm") and (start,end)==(0,1023), "Only structural PPM prefix allowed")
        else:
            require(url.endswith(("/00.tif","/32.tif","/64.tif")), "Only three frozen TIFFs allowed")
            require((kind != "tiff_header" or (start,end)==(0,7)) and
                    (kind != "ifd_count" or length==2) and
                    (kind != "ifd_table" or (6 < length <= 1542 and (length-6)%12 == 0)) and
                    (kind != "description" or 5 <= length <= 65536), "Invalid structural range category")
        cap = length+1
        require(length <= self.limits["maximum_per_range_bytes"], "Per-range cap exceeded")
        require(len(self.ledger["requests"]) < self.limits["maximum_attempts"], "Attempt cap exhausted")
        require(self.ledger["reserved_response_body_bytes"]+cap <= self.limits["maximum_reserved_response_body_bytes"], "Response reservation cap exhausted")
        require(Path(name).name == name and re.fullmatch(r"[a-z0-9-]+\.bin",name) is not None, "Unsafe range filename")
        target, partial = self.root/"raw"/name, self.root/"raw"/(name+".partial")
        require(not target.exists() and not partial.exists(), "Range output already exists; no retry")
        require(not any(e["url"]==url and e["kind"]==kind for e in self.ledger["requests"]), "Duplicate request category; no retry")
        resources(self.root,self.limits)
        headers = {"Range":f"bytes={start}-{end}","If-Match":obj["etag"],"Accept-Encoding":"identity","User-Agent":"Herculaneum-Frag6-Structural-StageA"}
        entry = dict(url=url, method="GET", kind=kind, range_inclusive=[start,end],
                     request_headers=headers, reserved_response_body_bytes=cap,
                     retained_body_bytes=0, success=False, started_utc=now())
        self.ledger["requests"].append(entry)
        self.ledger["reserved_response_body_bytes"] += cap
        self.write_ledger()  # Durable reservation precedes even constructing/opening a connection.
        response, retained = None, 0
        digest = hashlib.sha256()
        started = time.monotonic()
        try:
            with deadline(self.limits["per_request_wall_seconds"]):
                try:
                    response = self.opener.open(urllib.request.Request(url,headers=headers,method="GET"), timeout=self.limits["connect_timeout_seconds"])
                except urllib.error.HTTPError as error:
                    response = error
                entry["http_status"] = response.status
                raw_headers = list(response.headers.raw_items())
                header_bytes = sum(len(str(k).encode())+len(str(v).encode())+4 for k,v in raw_headers)
                entry["response_header_bytes"] = header_bytes
                require(header_bytes <= 16384, "Response headers exceed retained metadata bound")
                entry["response_headers"] = raw_headers
                require(response.status == 206, "Expected 206; no redirect/error/full-object body read")
                for key in ("Content-Length","Content-Range","Content-Type","Content-Encoding","ETag","Transfer-Encoding","Last-Modified"):
                    require(len(response.headers.get_all(key,[])) <= 1, "Duplicate singleton HTTP header: "+key)
                require(response.headers.get("Content-Encoding","identity").lower() in ("","identity"), "Encoded body unsupported")
                require(response.headers.get("Transfer-Encoding") is None, "Transfer encoding unsupported")
                require(response.headers.get("Content-Length","").isdigit() and int(response.headers["Content-Length"]) == length, "Exact Content-Length mismatch")
                match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",response.headers.get("Content-Range",""))
                require(match is not None and tuple(map(int,match.groups())) == (start,end,obj["total_bytes"]), "Exact Content-Range/total mismatch")
                etag = response.headers.get("ETag","")
                require(re.fullmatch(r'"[\x21\x23-\x7e]+"',etag) is not None and etag == obj["etag"], "Strong pinned ETag mismatch")
                mime = response.headers.get("Content-Type","").split(";",1)[0].strip().lower()
                require(mime in (("application/octet-stream",) if kind=="ppm_prefix" else ("image/tiff","application/octet-stream")), "Unsupported response MIME")
                with partial.open("xb") as stream:
                    while retained < length:
                        try:
                            block = response.read(min(self.limits["stream_chunk_bytes"],length-retained))
                        except http.client.IncompleteRead as error:
                            block = error.partial
                            require(len(block) <= cap-retained, "Oversized partial body")
                            stream.write(block); stream.flush(); os.fsync(stream.fileno())
                            digest.update(block); retained += len(block)
                            raise ContractError("Incomplete response body; partial retained") from None
                        require(block and len(block) <= cap-retained, "Truncated or oversized response body")
                        stream.write(block); stream.flush(); os.fsync(stream.fileno())
                        digest.update(block); retained += len(block)
                        require(retained <= length, "Response exceeded declared useful length")
                        resources(self.root,self.limits)
                    extra = response.read(1)
                    require(len(extra) <= 1, "Excess-byte read violated bound")
                    if extra:
                        stream.write(extra); stream.flush(); os.fsync(stream.fileno())
                        digest.update(extra); retained += len(extra)
                        raise ContractError("Unexpected excess response byte")
                    stream.flush(); os.fsync(stream.fileno())
                require(retained == length, "Response length mismatch")
                partial.replace(target)
                d = os.open(target.parent,os.O_DIRECTORY)
                try:
                    os.fsync(d)
                finally:
                    os.close(d)
                entry.update(success=True,file=pin(target))
        except Exception as error:
            entry.update(failure_type=type(error).__name__,stop_reason=str(error))
            raise
        finally:
            if response is not None:
                response.close()
            entry.update(retained_body_bytes=retained,body_sha256=digest.hexdigest(),
                         elapsed_seconds=time.monotonic()-started,completed_utc=now())
            if partial.exists():
                entry["partial_file"] = pin(partial)
            self.ledger["retained_response_body_bytes"] += retained
            self.write_ledger()
        resources(self.root,self.limits)
        return target.read_bytes(),entry
