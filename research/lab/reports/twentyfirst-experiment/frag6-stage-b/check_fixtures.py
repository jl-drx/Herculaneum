"""Offline Stage B multipart, schedule, conditional geometry and failure checks."""
import argparse
from email.message import Message
import http.client
import json
import math
from pathlib import Path
import socket
import struct
import time
import acquire
from common import LIMITS,ROOT,ContractError,now,pin,require,save
from geometry import assemble,candidates,derive_points,describe
from multipart import parse_multipart
from prepare_plan import fixed_schedule,load_go,source_records
from transport import MultipartClient

BOUNDARY="FRAG6-SYNTHETIC"
CONTENT_TYPE="multipart/byteranges; boundary="+BOUNDARY


def synthetic_records(points,order="<"):
    n=(-.001,-.004,.02);norm=math.hypot(*n);n=tuple(v/norm for v in n)
    values=[]
    for point in points:
        y,x=point["center_yx"]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                xx,yy=x+dx,y+dy
                values.extend((.2*xx,.1*yy,500+.01*xx+.02*yy,*n))
    return struct.pack(order+str(len(values))+"d",*values)


def mime_body(ranges,total,raw,indices=None,part_type="application/octet-stream"):
    indices=list(range(len(ranges))) if indices is None else list(indices)
    chunks=[]
    for index in indices:
        a,b=ranges[index]
        chunks.append((f"--{BOUNDARY}\r\nContent-Type: {part_type}\r\nContent-Range: bytes {a}-{b}/{total}\r\nContent-Length: {b-a+1}\r\n\r\n").encode()+raw[index*144:(index+1)*144]+b"\r\n")
    return b"".join(chunks)+(f"--{BOUNDARY}--\r\n").encode()


class FakeResponse:
    def __init__(self,body,headers,status=206,mode=None):
        self.body,self.headers,self.status,self.mode=body,headers,status,mode
        self.position=self.read_calls=0;self.closed=False
    def read(self,n):
        self.read_calls+=1
        if self.mode=="incomplete":raise http.client.IncompleteRead(self.body[:7],len(self.body))
        if self.mode=="slow":time.sleep(.015);n=1
        if self.mode=="io-error" and self.position:raise OSError("synthetic transport break")
        if self.mode=="io-error":n=7
        b=self.body[self.position:self.position+n];self.position+=len(b);return b
    def close(self):self.closed=True


class FakeOpener:
    def __init__(self,root,schedule,body,mutation=None,mode=None):
        self.root=Path(root);self.schedule=schedule;self.body=body;self.mutation=mutation;self.mode=mode
        self.calls=0;self.responses=[]
    def open(self,request,timeout):
        self.calls+=1
        ledger=json.loads((self.root/"request-ledger.json").read_text())
        assert len(ledger["requests"])==1 and ledger["reserved_response_body_bytes"]==222640
        assert request.method=="GET" and request.full_url==self.schedule["object"]["url"]
        assert request.get_header("If-match")==self.schedule["object"]["etag"]
        assert request.get_header("Range")=="bytes="+",".join(f"{a}-{b}" for a,b in self.schedule["ranges"])
        if self.mode=="slow-open":time.sleep(.1)
        h=Message()
        for k,v in (("Content-Length",str(len(self.body))),("ETag",self.schedule["object"]["etag"]),("Content-Type",CONTENT_TYPE)):
            h[k]=v
        state=dict(body=self.body,status=206,headers=h)
        if self.mutation:self.mutation(state)
        r=FakeResponse(state["body"],state["headers"],state["status"],self.mode);self.responses.append(r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",required=True);args=p.parse_args()
    root=Path(args.output_dir).resolve();require(root.is_relative_to(ROOT) and not root.exists(),"Fresh owned fixture output required")
    root.mkdir(parents=True)
    receipt=dict(schema="herculaneum.frag6-stage-b-fixtures.v1",status="RUNNING",started_utc=now(),code=source_records(),network_calls=0,tests=[])
    save(root/"fixtures.json",receipt)
    old=(socket.socket.connect,socket.socket.connect_ex,socket.create_connection)
    def deny(*args,**kwargs):receipt["network_calls"]+=1;raise AssertionError("Synthetic fixtures forbid network")
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
    def eq(a,b):assert a==b,(a,b)
    def fails(f):
        try:f()
        except ContractError:return
        raise AssertionError("Expected closed failure")
    def check(name,f):
        f();receipt["tests"].append(dict(name=name,status="PASS"));save(root/"fixtures.json",receipt)
    try:
        s=fixed_schedule();points,ranges=s["points"],s["ranges"];total=s["object"]["total_bytes"]
        raw=synthetic_records(points);body=mime_body(ranges,total,raw)
        check("fixed-counts",lambda:(eq(len(points),81),eq(len(ranges),243),eq(len(raw),34992)))
        check("fixed-extreme-centers",lambda:(eq(points[0]["center_yx"],[885,620]),eq(points[-1]["center_yx"],[7966,5583])))
        check("fixed-native-sites",lambda:eq([p["lattice_ij"] for p in points if p["potential_native_site"]],[[i,j] for i in (1,4,7) for j in (1,4,7)]))
        check("reject-overlapping-small-canvas",lambda:fails(lambda:derive_points(11,11,73,73+11*11*48)))
        for label,raw0,expected in (("little",raw,"little"),("big",synthetic_records(points,">"),"big")):
            value=candidates(raw0);g=describe(raw0,value,points)
            check("unique-"+label,lambda value=value,expected=expected:eq(value["selected_byte_order"],expected))
            check("flat-geometry-"+label,lambda g=g:(eq(g["center_mapped_count"],81),eq(g["complete_mapped_neighborhood_count"],81),eq(g["fixed_native_complete_count"],9),eq(g["records_inside_declared_primary_bounds"],729),eq(g["stage_c_authorized"],False)))
            for index in (0,40,80):
                row=g["samples"][index]
                check(f"plane-jacobian-{label}-{index}",lambda row=row:require(all(abs(row["jacobian_xyz_by_xy"][k][a]-[[.2,0],[0,.1],[.01,.02]][k][a])<1e-10 for k in range(3) for a in range(2)),"Synthetic affine Jacobian mismatch"))
                check(f"plane-normal-{label}-{index}",lambda row=row:require(abs(row["normal_dot_unit_raster_cross"]-1)<1e-10 and abs(row["normal_dot_unit_tangent_x"])<1e-10 and abs(row["normal_dot_unit_tangent_y"])<1e-10,"Synthetic normal mismatch"))
        zero=bytes(len(raw));both=bytes.fromhex("3ff000000000f03f")
        ambiguous=(bytes(24)+both+bytes(16))*729
        for name,raw0 in [("all-zero",zero),("ambiguous-endian",ambiguous),
                          ("nan",struct.pack("<d",float("nan"))+raw[8:]),
                          ("infinity",struct.pack("<d",float("inf"))+raw[8:]),
                          ("nonunit",raw[:24]+struct.pack("<3d",2,0,0)+raw[48:])]:
            val=candidates(raw0)
            check(name+"-stop",lambda val=val:eq(val["status"],"STOP"))
            check(name+"-no-geometry",lambda val=val,raw0=raw0:fails(lambda:describe(raw0,val,points)))
        sparse=bytearray(raw);sparse[10*9*48:11*9*48]=bytes(9*48)
        g=describe(bytes(sparse),candidates(bytes(sparse)),points)
        check("sentinel-neighborhood-retained-no-replacement",lambda:(eq(g["mapped_records"],720),eq(g["center_mapped_count"],80),eq(g["fixed_native_complete_count"],8),eq(len(g["samples"]),81)))
        center=bytearray(raw);center[4*48:5*48]=bytes(48);g0=describe(bytes(center),candidates(bytes(center)),points)
        check("zero-center-normal-diagnostics-null",lambda:eq(g0["samples"][0]["normal_dot_unit_raster_cross"],None))
        g1=bytearray(raw)
        for i,point in enumerate(points):
            if point["potential_native_site"]:g1[i*9*48:(i+1)*9*48]=bytes(9*48)
        gg=describe(bytes(g1),candidates(bytes(g1)),points)
        check("native-site-shortfall-does-not-replace-sites",lambda:(eq(gg["fixed_native_complete_count"],0),eq(gg["minimum_six_fixed_native_sites_condition_met"],False),eq(gg["stage_c_authorized"],False)))
        parts=parse_multipart(body,CONTENT_TYPE,ranges,total)
        check("ordered-mime-exact-assembly",lambda:eq(assemble(body,parts,ranges),raw))
        reverse=mime_body(ranges,total,raw,reversed(range(243)));reverse_parts=parse_multipart(reverse,CONTENT_TYPE,ranges,total)
        check("out-of-order-mime-canonical-assembly",lambda:eq(assemble(reverse,reverse_parts,ranges),raw))
        check("optional-leading-crlf",lambda:eq(len(parse_multipart(b"\r\n"+body,CONTENT_TYPE,ranges,total)),243))
        delimiter=bytearray(raw);text=("\r\n--"+BOUNDARY+"--\r\n").encode();delimiter[10:10+len(text)]=text
        dbody=mime_body(ranges,total,bytes(delimiter))
        check("binary-delimiter-like-content-safe",lambda:eq(assemble(dbody,parse_multipart(dbody,CONTENT_TYPE,ranges,total),ranges),bytes(delimiter)))
        a,b=ranges[0]
        bad=[("missing-part",mime_body(ranges,total,raw,range(242)),CONTENT_TYPE),
             ("duplicate-part",mime_body(ranges,total,raw,[0]*243),CONTENT_TYPE),
             ("extra-part",mime_body(ranges,total,raw,list(range(243))+[0]),CONTENT_TYPE),
             ("trailing-data",body+b"x",CONTENT_TYPE),
             ("preamble",b"x"+body,CONTENT_TYPE),
             ("truncated",body[:-4],CONTENT_TYPE),
             ("wrong-total",body.replace(str(total).encode(),str(total+1).encode(),1),CONTENT_TYPE),
             ("coalesced-span",body.replace(f"bytes {a}-{b}/".encode(),f"bytes {a}-{b+144}/".encode(),1),CONTENT_TYPE),
             ("unknown-part-header",body.replace(b"Content-Length:",b"Other-Encoding:",1),CONTENT_TYPE),
             ("duplicate-part-header",body.replace(b"Content-Type: application/octet-stream",b"Content-Type: application/octet-stream\r\ncontent-type: application/octet-stream",1),CONTENT_TYPE),
             ("wrong-part-mime",mime_body(ranges,total,raw,part_type="text/html"),CONTENT_TYPE),
             ("nonascii-header",body.replace(b"Content-Type:",b"Content-Typ\xff:",1),CONTENT_TYPE),
             ("oversized-part-header",body.replace(b"Content-Type: application/octet-stream",b"Content-Type: "+b"x"*600,1),CONTENT_TYPE),
             ("wrong-boundary",body,"multipart/byteranges; boundary=OTHER"),
             ("extra-boundary-param",body,CONTENT_TYPE+"; x=y"),
             ("boundary-too-long",body,"multipart/byteranges; boundary="+"x"*71),
             ("non-multipart",body,"application/octet-stream")]
        for name,badbody,ct in bad:
            check("mime-"+name,lambda badbody=badbody,ct=ct:fails(lambda:parse_multipart(badbody,ct,ranges,total)))
        live=root/"full-synthetic";fake=FakeOpener(live,s,body)
        result=acquire.run(live,s["object"],ranges,points,dict(synthetic=True),opener=fake)
        check("complete-synthetic-one-request",lambda:(eq(result["status"],"PASS"),eq(fake.calls,1),eq(result["reserved_response_body_bytes"],222640),eq(result["image_pixels_decoded"],0)))
        stopped=root/"all-zero-stop";zf=FakeOpener(stopped,s,mime_body(ranges,total,zero))
        fails(lambda:acquire.run(stopped,s["object"],ranges,points,dict(synthetic=True),opener=zf))
        check("ambiguity-retains-result-and-raw",lambda:(eq(json.loads((stopped/"SPARSE_GEOMETRY.json").read_text())["status"],"STOP"),eq((stopped/"raw/records.bin").read_bytes(),zero)))
        def replace(st,key,value):
            if key in st["headers"]:del st["headers"][key]
            st["headers"][key]=value
        mutations=[]
        for status in (200,301,412,416):mutations.append(("status-"+str(status),lambda st,status=status:st.update(status=status),0))
        for key,value in [("Content-Length","222640"),("Content-Length","34992"),("Content-Length","-1"),("ETag",'W/"weak"'),("ETag",'"changed"'),("Transfer-Encoding","chunked"),("Content-Encoding","gzip"),("Content-Range",f"bytes {a}-{b}/{total}"),("Content-Type","application/octet-stream")]:
            mutations.append(("header-"+key+"-"+str(len(mutations)),lambda st,key=key,value=value:replace(st,key,value),0))
        for key in ("Content-Length","Content-Type","ETag","Content-Encoding","Last-Modified"):
            def duplicate(st,key=key):
                if key not in st["headers"]:st["headers"][key]="x"
                st["headers"][key.lower()]=st["headers"][key]
            mutations.append(("duplicate-"+key,duplicate,0))
        mutations.extend([("truncated",lambda st:st.update(body=st["body"][:7]),7),
                          ("excess-byte",lambda st:st.update(body=st["body"]+b"x"),len(body)+1),
                          ("bad-mime-body",lambda st:st.update(body=st["body"][:-2]+b"xx"),len(body))])
        for index,(name,mutation,count) in enumerate(mutations):
            output=root/("http-"+str(index));opener=FakeOpener(output,s,body,mutation);client=MultipartClient(output,s["object"],ranges,dict(synthetic=True),opener=opener)
            fails(client.get);entry=client.ledger["requests"][0]
            check(name,lambda entry=entry,count=count,opener=opener:(eq(entry["success"],False),eq(entry["retained_body_bytes"],count),eq(entry["reserved_response_body_bytes"],222640),eq(opener.calls,1),eq(opener.responses[0].closed,True)))
            if count==0:check(name+"-no-body-read",lambda opener=opener:eq(opener.responses[0].read_calls,0))
            fails(client.get);check(name+"-no-retry",lambda opener=opener:eq(opener.calls,1))
        for mode in ("incomplete","io-error","slow","slow-open"):
            output=root/("mode-"+mode);opener=FakeOpener(output,s,body,mode=mode)
            limits=dict(LIMITS,per_request_wall_seconds=.035) if mode.startswith("slow") else LIMITS
            client=MultipartClient(output,s["object"],ranges,dict(synthetic=True),opener=opener,limits=limits);begun=time.monotonic()
            try:client.get()
            except (ContractError,OSError):pass
            else:raise AssertionError("Expected injected failure")
            elapsed=time.monotonic()-begun;entry=client.ledger["requests"][0]
            check(mode,lambda elapsed=elapsed,entry=entry:(require(elapsed<.5,"Absolute timer did not stop"),eq(entry["success"],False),eq(entry["reserved_response_body_bytes"],222640)))
            if mode in ("incomplete","io-error"):check(mode+"-partial-retained",lambda entry=entry:eq(Path(entry["partial_file"]["path"]).read_bytes(),body[:7]))
        for name,limits in [("attempt",dict(LIMITS,maximum_attempts=0)),("reservation",dict(LIMITS,maximum_reserved_response_body_bytes=222639)),("rss",dict(LIMITS,maximum_rss_bytes=1)),("disk",dict(LIMITS,maximum_artifact_bytes=1))]:
            output=root/("budget-"+name);opener=FakeOpener(output,s,body);client=MultipartClient(output,s["object"],ranges,dict(synthetic=True),opener=opener,limits=limits)
            fails(client.get);check(name+"-before-connect",lambda opener=opener:eq(opener.calls,0))
        go=root/"invalid-go.json";save(go,dict(status="GO",stage="A_headers",plan_sha256="expected",allow_live_requests=True))
        check("wrong-stage-go-rejected",lambda:fails(lambda:load_go(go,"expected")))
        check("zero-network",lambda:eq(receipt["network_calls"],0))
        require(receipt["code"]==source_records(),"Sources changed during fixtures");receipt["status"]="PASS"
    except Exception as error:
        receipt.update(status="FAIL",failure_type=type(error).__name__,failure_message=str(error));raise
    finally:
        socket.socket.connect,socket.socket.connect_ex,socket.create_connection=old
        receipt.update(completed_utc=now(),test_count=len(receipt["tests"]));save(root/"fixtures.json",receipt)
    print(json.dumps(dict(status=receipt["status"],tests=receipt["test_count"],network_calls=receipt["network_calls"],receipt=pin(root/"fixtures.json"))))


if __name__=="__main__":main()
