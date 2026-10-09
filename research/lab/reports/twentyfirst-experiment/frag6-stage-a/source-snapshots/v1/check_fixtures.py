"""Offline synthetic parser, transport, deadline and reservation checks."""
import argparse
import copy
from email.message import Message
import http.client
import json
from pathlib import Path
import socket
import struct
import time
from acquire import run
from common import LIMITS, ROOT, ContractError, now, pin, require, save
from parsers import ascii_description, ppm_prefix, tiff_count, tiff_header, tiff_ifd
from prepare_plan import load_go, proposal, source_records
from transport import RangeClient


def ppm(version="1.0"):
    header = f"# synthetic provenance only\nwidth: 17\nheight: 19\ndim: 6\nordered: true\ntype: double\nversion: {version}\n<>\n".encode()
    return header+b"\xff"*(1024-len(header)), len(header)+17*19*48


def tiff(order="<",description=b"synthetic renderer --unknown\0",omit=(),overrides=None,extra_entries=0):
    tags = {256:(4,1,17),257:(4,1,19),258:(3,1,16),259:(3,1,1),262:(3,1,1),
            273:(4,1,0),277:(3,1,1),278:(4,1,19),279:(4,1,17*19*2)}
    if description is not None:
        tags[270] = (2,len(description),description if len(description)<=4 else 4096)
    for key in omit:
        tags.pop(key)
    for i in range(extra_entries):
        tags[500+i] = (3,1,i)
    tags.update(overrides or {})
    strip = max(8192,4096+(0 if description is None else len(description))+256)
    if overrides is None or 273 not in overrides:
        tags[273] = (4,1,strip)
    total = strip+17*19*2+64
    raw = bytearray(total)
    raw[:8] = (b"II" if order=="<" else b"MM")+struct.pack(order+"HI",42,8)
    table = bytearray(struct.pack(order+"H",len(tags)))
    for tag,(kind,count,value) in sorted(tags.items()):
        if isinstance(value,bytes):
            field = value.ljust(4,b"\0")
        elif kind==3 and count==1:
            field = struct.pack(order+"H",value)+b"\0\0"
        else:
            field = struct.pack(order+"I",value)
        table.extend(struct.pack(order+"HHI",tag,kind,count)+field)
    table.extend(struct.pack(order+"I",0))
    raw[8:8+len(table)] = table
    if description is not None and len(description)>4:
        raw[4096:4096+len(description)] = description
    return bytes(raw)


def parsed_tiff(raw):
    h = tiff_header(raw[:8],len(raw));o=h["ifd_offset"]
    c = tiff_count(raw[o:o+2],h,len(raw))
    return tiff_ifd(raw[o:o+c["table_bytes"]],h,c,len(raw),(17,19))


class FakeResponse:
    def __init__(self,body,headers,status=206,mode=None):
        self.body,self.headers,self.status,self.mode = body,headers,status,mode
        self.position,self.read_calls,self.closed = 0,0,False
    def read(self,number):
        self.read_calls += 1
        if self.mode=="incomplete":
            raise http.client.IncompleteRead(self.body[:3],8)
        if self.mode=="slow":
            time.sleep(0.015);number=1
        if self.mode=="io-error" and self.position:
            raise OSError("synthetic connection break")
        if self.mode=="io-error":
            number=3
        data = self.body[self.position:self.position+number]
        self.position += len(data)
        return data
    def close(self):
        self.closed=True


class FakeOpener:
    def __init__(self,root,objects,bodies,mutation=None,mode=None):
        self.root=Path(root);self.objects={o["url"]:o for o in objects};self.bodies=bodies
        self.mutation=mutation;self.mode=mode;self.responses=[];self.calls=0
    def open(self,request,timeout):
        self.calls += 1
        ledger = json.loads((self.root/"request-ledger.json").read_text())
        entry = ledger["requests"][-1]
        obj = self.objects[request.full_url]
        assert request.method=="GET" and entry["url"]==request.full_url and entry["success"] is False
        begin,end=map(int,request.get_header("Range").removeprefix("bytes=").split("-"))
        assert entry["range_inclusive"]==[begin,end]
        assert entry["reserved_response_body_bytes"]==end-begin+2
        assert ledger["reserved_response_body_bytes"]==sum(e["reserved_response_body_bytes"] for e in ledger["requests"])
        assert request.get_header("If-match")==obj["etag"]
        assert request.get_header("Accept-encoding")=="identity"
        if self.mode=="slow-open":
            time.sleep(0.1)
        headers=Message()
        for key,value in (("Content-Length",str(end-begin+1)),("Content-Range",f"bytes {begin}-{end}/{obj['total_bytes']}"),
                          ("ETag",obj["etag"]),("Content-Type","application/octet-stream" if request.full_url.endswith(".ppm") else "image/tiff")):
            headers[key]=value
        body=self.bodies[request.full_url][begin:end+1]
        state=dict(status=206,body=body,headers=headers)
        if self.mutation:
            self.mutation(state)
        response=FakeResponse(state["body"],state["headers"],state["status"],self.mode)
        self.responses.append(response)
        return response


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",required=True);args=p.parse_args()
    output=Path(args.output_dir).resolve();require(output.is_relative_to(ROOT) and not output.exists(),"Fresh owned fixture directory required")
    output.mkdir(parents=True)
    receipt=dict(schema="herculaneum.frag6-stage-a-fixtures.v1",status="RUNNING",started_utc=now(),
                 code=source_records(),network_calls=0,tests=[])
    save(output/"fixtures.json",receipt)
    old_connect,old_connect_ex,old_create=socket.socket.connect,socket.socket.connect_ex,socket.create_connection
    def deny(*args,**kwargs):
        receipt["network_calls"]+=1
        raise AssertionError("Network forbidden in synthetic fixtures")
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    def check(name,function):
        function();receipt["tests"].append(dict(name=name,status="PASS"))
        save(output/"fixtures.json",receipt)
    def fails(function):
        try:function()
        except ContractError:return
        raise AssertionError("Expected closed failure")
    def eq(a,b):
        assert a==b,(a,b)
    try:
        check("ppm-version-1.0-and-poison-tail-uninterpreted",lambda:eq(ppm_prefix(*ppm())["numerical_payload_decoded"],False))
        check("ppm-version-1-supported",lambda:eq(ppm_prefix(*ppm("1"))["fields"]["version"],"1"))
        raw,total=ppm()
        for name,body,extent in [
            ("ppm-object-size",raw,total+1),
            ("ppm-terminator",raw.replace(b"<>\n",b"xx\n"),total),
            ("ppm-duplicate-field",raw.replace(b"# synthetic provenance only",b"width: 17"),total),
            ("ppm-unknown-field",raw.replace(b"width:",b"other:"),total),
            ("ppm-small-canvas",raw.replace(b"width: 17",b"width: 01"),total),
            ("ppm-invalid-version",raw.replace(b"version: 1.0",b"version: 2.0"),total),
            ("ppm-cr-framing",raw.replace(b"# synthetic",b"\r synthetic"),total),
            ("ppm-utf8",raw.replace(b"# synthetic",b"#\xffsynthetic"),total),
        ]:
            check(name,lambda body=body,extent=extent:fails(lambda:ppm_prefix(body,extent)))
        check("tiff-little-endian",lambda:eq(parsed_tiff(tiff())["byte_order"],"little"))
        check("tiff-big-endian",lambda:eq(parsed_tiff(tiff(">"))["byte_order"],"big"))
        check("tiff-description-absent",lambda:eq(parsed_tiff(tiff(description=None))["description"],None))
        check("tiff-description-inline",lambda:eq(parsed_tiff(tiff(description=b"ok\0"))["description"]["literal_ascii"],"ok\0"))
        check("tiff-specified-defaults",lambda:eq({k:parsed_tiff(tiff(omit=(259,277,278)))["baseline_defaults_applied"][k] for k in ("259","277","278")},{"259":1,"277":1,"278":2**32-1}))
        for tag in (258,259,262,266,274,277,284,317,339):
            val=16 if tag==258 else 1
            check("short-required-"+str(tag),lambda tag=tag,val=val:fails(lambda:parsed_tiff(tiff(overrides={tag:(4,1,val)}))))
        for name,overrides in [
            ("compressed",{259:(3,1,5)}),("orientation",{274:(3,1,2)}),
            ("signed-sample",{339:(3,1,2)}),("predictor",{317:(3,1,2)}),
            ("wrong-bits",{258:(3,1,8)}),("multiple-samples",{277:(3,1,2)}),
            ("wrong-width",{256:(4,1,18)}),("strip-over-header",{273:(4,1,4)}),
            ("strip-over-ifd",{273:(4,1,12)}),("wrong-strip-size",{279:(4,1,100)}),
            ("wrong-description-type",{270:(7,5,4096)}),
            ("description-over-header",{270:(2,8,0)}),
            ("description-over-ifd",{270:(2,8,8)}),
            ("description-over-strip",{270:(2,8,8192)}),
            ("description-outside-object",{270:(2,32,900000)}),
            ("description-over-cap",{270:(2,65537,4096)}),
            ("tiled",{322:(3,1,16)}),("sub-ifd",{330:(4,1,1000)}),
        ]:
            check("tiff-"+name,lambda overrides=overrides:fails(lambda:parsed_tiff(tiff(overrides=overrides))))
        for name,mutator in [
            ("bigtiff",lambda b: b.__setitem__(slice(2,4),struct.pack("<H",43))),
            ("ifd-outside",lambda b:b.__setitem__(slice(4,8),struct.pack("<I",999999))),
            ("ifd-cycle",lambda b:b.__setitem__(slice(8+2+12*struct.unpack_from("<H",b,8)[0],8+6+12*struct.unpack_from("<H",b,8)[0]),struct.pack("<I",8))),
            ("duplicate-tag",lambda b:b.__setitem__(slice(8+2+12,8+2+14),b[10:12])),
        ]:
            b=bytearray(tiff());mutator(b)
            check("tiff-"+name,lambda b=bytes(b):fails(lambda:parsed_tiff(b)))
        check("ascii-description-literal",lambda:eq(ascii_description(b"source=unknown\0")["literal_ascii"],"source=unknown\0"))
        check("ascii-description-non-ascii",lambda:fails(lambda:ascii_description(b"\xff\0")))
        check("ascii-description-no-nul",lambda:fails(lambda:ascii_description(b"abc")))

        objects=copy.deepcopy(proposal()["present_evidence"]["objects"])
        bodies={objects[0]["url"]:ppm()[0]};objects[0]["total_bytes"]=ppm()[1]
        for i,o in enumerate(objects[1:]):
            bodies[o["url"]]=tiff("<" if i!=1 else ">")
            o["total_bytes"]=len(bodies[o["url"]])
        full=output/"full-success";fake=FakeOpener(full,objects,bodies)
        result=run(full,objects,dict(synthetic=True),opener=fake)
        check("full-pipeline-13-requests-and-no-numeric-decode",lambda:(eq(result["status"],"PASS"),eq(fake.calls,13),eq(result["numerical_ppm_records_decoded"],0),eq(result["image_pixels_decoded"],0)))
        check("initial-four-request-order",lambda:eq([e["url"] for e in json.loads((full/"request-ledger.json").read_text())["requests"][:4]],[o["url"] for o in objects]))
        maximum=copy.deepcopy(objects);maxbody={maximum[0]["url"]:ppm()[0]}
        for o in maximum[1:]:
            maxbody[o["url"]]=tiff(description=b"x"*65535+b"\0",extra_entries=118)
            o["total_bytes"]=len(maxbody[o["url"]])
        mroot=output/"maximum-success";mfake=FakeOpener(mroot,maximum,maxbody)
        mresult=run(mroot,maximum,dict(synthetic=True),opener=mfake)
        check("maximum-13-request-202301-reservation",lambda:(eq(mresult["http_attempts"],13),eq(mresult["reserved_response_body_bytes"],202301)))

        def replace(state,key,value):
            if key in state["headers"]:del state["headers"][key]
            state["headers"][key]=value
        mutations=[]
        for status in (200,301,404,412):
            mutations.append(("http-"+str(status),lambda s,status=status:s.update(status=status),0))
        for key,value in [("Content-Length","9"),("Content-Range",f"bytes 1-8/{objects[1]['total_bytes']}"),
                          ("Content-Range","bytes 0-7/99"),("ETag",'W/"weak"'),("ETag",'"changed"'),
                          ("Content-Encoding","gzip"),("Transfer-Encoding","chunked"),("Content-Type","text/html")]:
            mutations.append(("header-"+key+"-"+str(len(mutations)),lambda s,key=key,value=value:replace(s,key,value),0))
        for key in ("content-length","CONTENT-RANGE","etag","Content-Type","Last-Modified"):
            def duplicate(s,key=key):
                canonical={"content-length":"Content-Length","content-range":"Content-Range","etag":"ETag","content-type":"Content-Type","last-modified":"Last-Modified"}[key.lower()]
                if canonical not in s["headers"]:s["headers"][canonical]="synthetic"
                s["headers"][key]=s["headers"][canonical]
            mutations.append(("duplicate-"+key,duplicate,0))
        mutations.extend([("truncated",lambda s:s.update(body=s["body"][:3]),3),
                          ("excess",lambda s:s.update(body=s["body"]+b"x"),9)])
        for index,(name,mutation,retained) in enumerate(mutations):
            root=output/("failure-"+str(index));opener=FakeOpener(root,objects,bodies,mutation)
            client=RangeClient(root,objects,dict(synthetic=True),opener=opener)
            fails(lambda:client.get(objects[1]["url"],0,7,"header.bin","tiff_header"))
            receipt0=client.ledger["requests"][0]
            check(name,lambda receipt0=receipt0,retained=retained,client=client,opener=opener:(eq(receipt0["success"],False),eq(receipt0["retained_body_bytes"],retained),eq(client.ledger["reserved_response_body_bytes"],9),eq(opener.calls,1),eq(opener.responses[0].closed,True)))
            if retained==0:
                check(name+"-no-body-read",lambda opener=opener:eq(opener.responses[0].read_calls,0))
        for mode in ("incomplete","io-error","slow","slow-open"):
            root=output/("failure-"+mode);opener=FakeOpener(root,objects,bodies,mode=mode)
            limits=dict(LIMITS,per_request_wall_seconds=0.035) if mode.startswith("slow") else LIMITS
            client=RangeClient(root,objects,dict(synthetic=True),opener=opener,limits=limits)
            begun=time.monotonic()
            try:client.get(objects[1]["url"],0,7,"header.bin","tiff_header")
            except (ContractError,OSError):pass
            else:raise AssertionError("Expected injected failure")
            elapsed=time.monotonic()-begun;entry=client.ledger["requests"][0]
            check("failure-"+mode,lambda client=client,entry=entry,elapsed=elapsed:(eq(entry["success"],False),eq(client.ledger["reserved_response_body_bytes"],9),require(elapsed<0.5,"Absolute deadline fixture took too long")))
            if mode in ("incomplete","io-error"):
                check("partial-retention-"+mode,lambda entry=entry:eq(Path(entry["partial_file"]["path"]).read_bytes(),bodies[objects[1]["url"]][:3]))
        for name,change in [("attempt-budget",dict(maximum_attempts=0)),("byte-budget",dict(maximum_reserved_response_body_bytes=8)),
                            ("range-budget",dict(maximum_per_range_bytes=7)),("rss-budget",dict(maximum_rss_bytes=1)),("disk-budget",dict(maximum_artifact_bytes=1))]:
            root=output/name;opener=FakeOpener(root,objects,bodies);client=RangeClient(root,objects,dict(synthetic=True),opener=opener,limits=dict(LIMITS,**change))
            fails(lambda:client.get(objects[1]["url"],0,7,"header.bin","tiff_header"))
            check(name+"-before-connect",lambda opener=opener,client=client:(eq(opener.calls,0),eq(len(client.ledger["requests"]),0)))
        go=output/"invalid-go.json";save(go,dict(status="GO",stage="A_headers",plan_sha256="wrong",allow_live_requests=True))
        check("mismatched-root-go-rejected",lambda:fails(lambda:load_go(go,"expected")))
        check("all-fixtures-network-free",lambda:eq(receipt["network_calls"],0))
        require(receipt["code"]==source_records(),"Source changed during fixtures")
        receipt["status"]="PASS"
    except Exception as error:
        receipt.update(status="FAIL",failure_type=type(error).__name__,failure_message=str(error))
        raise
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection=old_connect,old_connect_ex,old_create
        receipt.update(completed_utc=now(),test_count=len(receipt["tests"]))
        save(output/"fixtures.json",receipt)
    print(json.dumps(dict(status=receipt["status"],tests=receipt["test_count"],network_calls=receipt["network_calls"],receipt=pin(output/"fixtures.json"))))


if __name__ == "__main__":
    main()
