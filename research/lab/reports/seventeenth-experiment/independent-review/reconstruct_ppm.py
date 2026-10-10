"""Independent sparse scheduling, retained HTTP accounting and conditional PPM math."""
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time
import numpy as np
from independent_ppm_v2 import candidate,decode_records,select
from reconstruct_ct import read_pin,digest,json_dump,PINS,read_array

P=Path('/workspace/herculaneum-lab/reports/seventeenth-experiment')
LATTICE=[32,72,112,152,192,232,272,312,351]
NEWPINS={
 P/'PPM_PREDECODE_AMENDMENT.md':'ed3096856274505990fb9b4069ce6fdbb657eaeb93ae52b6bfae2144de6d9f9e',
 P/'PPM_ZERO_MAPPING_CLARIFICATION.md':'114e3791f2a7549b2e1d82192ef454865224f3ea9506550a0513f4eaec591d1a',
 P/'acquisition/header-run/result.json':'c2dd0d11e7321900426e77c6b43eab83ef1fb69a4c82b9dbaffa2e16f54b43c4',
 P/'acquisition/sample-run/result.json':'d9b0d166a9b42154895d3ded1520d54e858bd678a1e7333727769353437b3526',
}


def multipart(body,content_type):
    boundary=content_type.split('boundary=',1)[1].strip().strip('"').encode('ascii')
    token=b'--'+boundary
    cursor=2 if body.startswith(b'\r\n') else 0
    parts=[]
    while True:
        assert body[cursor:cursor+len(token)]==token
        cursor+=len(token)
        if body[cursor:cursor+2]==b'--':
            cursor+=2
            assert body[cursor:] in [b'',b'\r\n']
            break
        assert body[cursor:cursor+2]==b'\r\n';cursor+=2
        end=body.index(b'\r\n\r\n',cursor)
        headers={}
        for line in body[cursor:end].decode('ascii').split('\r\n'):
            key,value=line.split(':',1);key=key.lower();assert key not in headers;headers[key]=value.strip()
        m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',headers['content-range']);assert m
        start,stop,total=map(int,m.groups());assert 0<=start<=stop<total
        cursor=end+4;length=stop-start+1;data=body[cursor:cursor+length];assert len(data)==length
        parts.append(dict(range_inclusive=[start,stop],payload_offset=cursor,payload_bytes=length,payload_sha256=digest(data),headers=headers,data=data))
        cursor+=length;assert body[cursor:cursor+2]==b'\r\n';cursor+=2
    return parts


def point_geometry(a):
    zeros=np.all(a==0,axis=2);finite=np.isfinite(a).all(axis=2)
    valid=bool(not zeros[1,1] and finite[1,1])
    p=a[1,1,:3];n=a[1,1,3:];nn=math.hypot(*n)
    du=(a[1,2,:3]-a[1,0,:3])/2 if valid and not zeros[1,0]and not zeros[1,2]and finite[1,0]and finite[1,2]else None
    dv=(a[2,1,:3]-a[0,1,:3])/2 if valid and not zeros[0,1]and not zeros[2,1]and finite[0,1]and finite[2,1]else None
    lu=math.hypot(*du)if du is not None else None;lv=math.hypot(*dv)if dv is not None else None
    cross=None;area=None;sn=None;td=None
    if du is not None and dv is not None:
        cross=np.array([du[1]*dv[2]-du[2]*dv[1],du[2]*dv[0]-du[0]*dv[2],du[0]*dv[1]-du[1]*dv[0]])
        area=math.hypot(*cross)
        if area and nn:sn=math.fsum(float(cross[i])*float(n[i])for i in range(3))/(area*nn)
        if lu and lv:td=math.fsum(float(du[i])*float(dv[i])for i in range(3))/(lu*lv)
    return dict(valid_center=valid,center_xyz=p.tolist(),center_normal=n.tolist(),normal_norm=nn,
                zero_sentinel=bool(zeros[1,1]),neighborhood_zero_count=int(zeros.sum()),
                sampled_normal_norms=[[math.hypot(*a[y,x,3:])for x in range(3)]for y in range(3)],
                du=du.tolist()if du is not None else None,dv=dv.tolist()if dv is not None else None,
                du_norm=lu,dv_norm=lv,cross_product=cross.tolist()if cross is not None else None,
                cross_product_area=area,signed_normal_dot=sn,tangent_dot=td)


def main():
    start=time.monotonic();resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    # Header/range grammar synthetic checks precede any scientific PPM interpretation.
    payload=b'\x00'*144;synthetic=b'--abc\r\nContent-Range: bytes 100-243/1000\r\n\r\n'+payload+b'\r\n--abc--\r\n'
    fixture=multipart(synthetic,'multipart/byteranges; boundary=abc');assert fixture[0]['data']==payload and fixture[0]['range_inclusive']==[100,243]
    for bad in [synthetic[:-3],synthetic.replace(b'100-243',b'100-244'),synthetic+b'junk']:
        try:multipart(bad,'multipart/byteranges; boundary=abc')
        except (AssertionError,ValueError):pass
        else:raise AssertionError('bad multipart accepted')
    for path,sha in {**PINS,**NEWPINS}.items():assert digest(path.read_bytes())==sha,str(path)
    header=json.loads((P/'acquisition/header-run/result.json').read_text());sample=json.loads((P/'acquisition/sample-run/result.json').read_text())
    desc=json.loads(next(p for p in PINS if p.name=='pooled-descriptor.json').read_text());cases={c['id']:c for c in desc['cases']}
    output=P/'independent-review/ppm-reconstruction';output.mkdir(exist_ok=False)
    json_dump(output/'start-receipt.json',dict(input_pins={str(k):v for k,v in {**PINS,**NEWPINS}.items()},sources={q.name:digest(q.read_bytes())for q in[Path(__file__),Path(__file__).with_name('independent_ppm_v2.py')]},synthetic_multipart_cases=4,network_calls=0))
    ledger_reports=[];ledger_entries={}
    for result in [header,sample]:
        assert result['status']=='PASS'
        plan=json.loads(read_pin(result['plan']));ledger=json.loads(read_pin(result['ledger']))
        assert ledger['plan']==result['plan'] and len(ledger['requests'])==result['http_attempts']==len(plan['requests'])==len(result['objects'])
        assert ledger['automatic_retries']==ledger['redirects_followed']==0 and ledger['verified_tls']
        assert ledger['retained_body_bytes']==sum(e['retained_body_bytes']for e in ledger['requests'])==result['retained_body_bytes']
        assert ledger['charged_response_bytes']==sum(e['charged_response_bytes']for e in ledger['requests'])==result['charged_response_bytes']
        assert ledger['retained_body_bytes']<=plan['limits']['body_bytes']
        for req,e,obj in zip(plan['requests'],ledger['requests'],result['objects'],strict=True):
            assert req['request_id']==e['request_id']==obj['request_id']
            assert e['success']and e['url']==req['url']==obj['url']and e['reserved_cap_bytes']==req['reserved_cap_bytes']
            body=read_pin(obj['file']);assert len(body)==e['retained_body_bytes']==e['charged_response_bytes']<=e['reserved_cap_bytes']
            assert e['body_sha256']==digest(body)and e['etag']==obj['etag']and e['http_identity']['etag']==e['etag']
            assert not e['etag'].startswith('W/')and e['http_status']==(200 if req['mode']=='full'else 206)
            assert e['request_headers']['Accept-Encoding']=='identity'and int(e['http_identity']['content-length'])==len(body)
            assert e['ranges_inclusive']==req['ranges']
            if req['mode']!='full':assert e['request_headers']['Range']=='bytes='+','.join(f'{a}-{b}'for a,b in req['ranges'])
            if req['mode']=='multipart':assert e['request_headers']['If-Match']==req['pin']['etag']==e['etag']
            ledger_entries[e['request_id']]=e
        ledger_reports.append(dict(stage=result['stage'],requests=len(ledger['requests']),body_bytes=ledger['retained_body_bytes'],charged_bytes=ledger['charged_response_bytes']))
    contracts={}
    for obj in header['objects']:
        if obj['object_id'].endswith('-ir'):continue
        data=read_pin(obj['file']);assert len(data)==1024
        offset=data.index(b'<>\n')+3;fields={}
        for line in data[:offset-3].decode('ascii').splitlines():
            if line.startswith('#'):continue
            k,v=line.split(': ',1);assert k not in fields;fields[k]=v
        assert fields['dim']=='6'and fields['ordered']=='true'and fields['type']=='double'
        w,h=int(fields['width']),int(fields['height']);assert offset+48*w*h==obj['object_total_bytes']
        contracts[obj['object_id']]=dict(width=w,height=h,offset=offset,etag=obj['etag'],total=obj['object_total_bytes'])
    plan=json.loads(read_pin(sample['plan']));requests={r['request_id']:r for r in plan['requests']}
    chunks={};expected_xy={};points={};parts_count=0
    for obj in sample['objects']:
        caseid=obj['case_id'];variant=obj['variant'];oid=obj['object_id'];case=cases[caseid];c=contracts[oid]
        assert obj['etag']==c['etag']and obj['object_total_bytes']==c['total']
        req=requests[obj['request_id']];e=ledger_entries[obj['request_id']];body=read_pin(obj['file'])
        parsed=multipart(body,e['http_identity']['content-type']);assert len(parsed)==len(obj['parts'])==243
        data_by_range={}
        for physical,record in zip(parsed,obj['parts'],strict=True):
            assert {k:v for k,v in physical.items()if k!='data'}==record
            span=tuple(record['range_inclusive']);assert span not in data_by_range;data_by_range[span]=physical['data'];parts_count+=1
        x0,y0,w,h=case['bbox_source_xywh'];schedule=[];request_points=[]
        for iy in LATTICE:
            for ix in LATTICE:
                cy,cx=y0+1+3*iy,x0+1+3*ix
                raw_rows=[];xy=[];ranges=[]
                for dy in [-1,0,1]:
                    begin=c['offset']+48*((cy+dy)*c['width']+cx-1);span=[begin,begin+143]
                    assert span[1]<c['total'];schedule.append(span);ranges.append(span)
                    raw_rows.append(data_by_range[tuple(span)])
                    xy.extend([[cx+dx,cy+dy]for dx in[-1,0,1]])
                point=dict(coarse_yx=[iy,ix],native_center_yx=[cy,cx],row_ranges=ranges)
                request_points.append(point)
                chunk=b''.join(raw_rows);chunks.setdefault(oid,[]).append(chunk);expected_xy.setdefault(oid,[]).extend(xy)
                points.setdefault(oid,[]).append(dict(case_id=caseid,variant=variant,coarse_yx=[iy,ix],native_center_yx=[cy,cx],chunk=chunk))
        assert schedule==req['ranges']and request_points==req['points']and len(data_by_range)==len(schedule)
    assert parts_count==2916
    objects=[]
    for oid in sorted(chunks):
        raw=b''.join(chunks[oid]);xy=np.asarray(expected_xy[oid],np.float64);variant=oid.split('-',1)[1]
        checks={e:candidate(raw,e,variant,xy)for e in ['<','>']};chosen=select(checks)
        out=dict(object_id=oid,variant=variant,records_sha256=digest(raw),candidate_checks=checks,selected_endian=chosen,points=[])
        if chosen:
            for point in points[oid]:
                chunk=point['chunk'];a=decode_records(chunk,chosen).reshape(3,3,6)
                out['points'].append({k:v for k,v in point.items()if k!='chunk'}|dict(geometry=point_geometry(a)))
        objects.append(out)
        print(oid,chosen,[x['zero_sentinels']for x in checks.values()],flush=True)
    r=dict(status='PASS',objects=objects,ledger_checks=ledger_reports,independent_ranges=parts_count,useful_ppm_bytes=parts_count*144,
           fixed_centers=972,ppm_records=8748,conditional_objects=sum(o['selected_endian']is not None for o in objects),
           wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,network_calls=0)
    json_dump(output/'independent-ppm-diagnostics.json',r)
    print(json.dumps({k:v for k,v in r.items()if k!='objects'},indent=2))

if __name__=='__main__':main()
