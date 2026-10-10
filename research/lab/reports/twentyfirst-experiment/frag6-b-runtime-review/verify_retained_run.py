"""Independent retained-byte verification; stdlib only, no production imports."""
from collections import defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import statistics
import struct
import time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
BASE=PHASE/'frag6-stage-b'
RUN=BASE/'run-v1'
ACQ=RUN/'acquisition'
PLAN_SHA='d56b1e1a9b3ecffcc54de01cf574ff00b926ba2b83a9c857cb566d1211d563f8'
WORKER_SHA='586de6fe885c608fba92eda106b737ca2bfc71a9c5762793ce341e8541be25a1'
RESULT_SHA='584ad931220ebd6394937c40833af9e337bc727b266985322d9b4e49bc59db50'
ATOL=RTOL=5e-12


def pin(path):
    data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def norm(vector):
    if any(math.isnan(v) for v in vector):return float('nan')
    if any(math.isinf(v) for v in vector):return float('inf')
    scale=max(abs(v) for v in vector)
    return 0.0 if scale==0 else scale*math.sqrt(math.fsum((v/scale)**2 for v in vector))


def cosine(a,b):
    na,nb=norm(a),norm(b)
    if not (math.isfinite(na) and math.isfinite(nb) and na>0 and nb>0):return None
    result=math.fsum((x/na)*(y/nb) for x,y in zip(a,b))
    return result if math.isfinite(result) else None


def main():
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(dict(name=name,passed=True))
    def verify(record):
        observed=pin(Path(record['path']))
        check('pin:'+record['path'],observed['sha256']==record['sha256']
              and observed['bytes']==record.get('bytes',record.get('size_bytes')))
    def close(name,observed,expected):
        if isinstance(expected,(tuple,list)):
            check(name+':shape',isinstance(observed,list) and len(observed)==len(expected))
            for i,(a,b) in enumerate(zip(observed,expected)):close(name+f'[{i}]',a,b)
        elif expected is None:check(name,observed is None)
        elif isinstance(expected,bool):check(name,observed is expected)
        else:check(name,isinstance(observed,(int,float)) and math.isfinite(observed)
                   and abs(observed-expected)<=ATOL+RTOL*abs(expected))
    def load(path,anchor=None):
        if anchor:check('external_anchor:'+path.name,pin(path)['sha256']==anchor)
        return json.loads(path.read_text())
    plan=load(BASE/'execution-plan.json',PLAN_SHA)
    worker=load(ACQ/'worker-result.json',WORKER_SHA)
    result=load(ACQ/'SPARSE_GEOMETRY.json',RESULT_SHA)
    controller=load(RUN/'controller-result.json')
    start=load(RUN/'controller-start.json')
    ledger=load(ACQ/'request-ledger.json')
    go=load(PHASE/'FRAG6_STAGE_B_GO.json')
    for item in plan['code']+[plan[k] for k in ('proposal','stage_a','stage_a_review','adaptation','synthetic_fixtures')]:verify(item)
    for item in [controller['worker_result'],worker['root_go'],worker['result'],result['plan'],result['records'],
                 result['request_ledger'],start['root_go'],start['plan'],go['plan'],go['technical_review']]:verify(item)
    check('root_go_exact_scope',go['status']=='GO' and go['stage']=='B_sparse_ppm' and go['plan_sha256']==PLAN_SHA
          and go['allow_conditional_ppm_geometry'] and go['maximum_attempts']==1
          and go['maximum_reserved_response_body_bytes']==222640)
    check('all_run_receipts_pass',worker['status']==result['status']==controller['status']=='PASS'
          and worker['source_pins_reverified'] and controller['worker_returncode']==0)
    check('runtime_limits_and_no_extra_scope',controller['elapsed_seconds']<1800
          and controller['maximum_observed_worker_rss_bytes']<=268435456
          and result['resources']['peak_rss_bytes']<=268435456
          and sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())<=33554432
          and all(result[k]==0 for k in ('image_pixels_decoded','labels_loaded','model_calls','gpu_spend_gbp'))
          and result['stage_c_authorized'] is False)
    check('single_request_no_retry',len(ledger['requests'])==result['http_attempts']==1
          and ledger['automatic_retries']==ledger['redirects_followed']==0)
    request=ledger['requests'][0]
    points=[];ranges=[]
    for i in range(9):
        y=(i+1)*8852//10
        for j in range(9):
            x=(j+1)*6204//10
            rows=[]
            for dy in (-1,0,1):
                begin=73+48*((y+dy)*6205+x-1);rows.append([begin,begin+143])
            ranges.extend(rows)
            points.append(dict(lattice_ij=[i,j],center_yx=[y,x],row_ranges=rows,potential_native_site=i in (1,4,7) and j in (1,4,7)))
    check('exact_independent_ranges_and_centers',plan['schedule']['ranges']==ranges and plan['schedule']['points']==points
          and request['ranges_inclusive']==ranges and len(ranges)==243 and len(set(map(tuple,ranges)))==243
          and all(a[1]<b[0] for a,b in zip(sorted(ranges),sorted(ranges)[1:])))
    check('request_identity',request['url']==plan['schedule']['object']['url'] and request['method']=='GET'
          and request['request_headers']['If-Match']=='"656fa2a5-9d2a0879"'
          and request['request_headers']['Accept-Encoding']=='identity'
          and request['request_headers']['Range']=='bytes='+','.join(f'{a}-{b}' for a,b in ranges)
          and request['http_status']==206 and request['success'])
    headers=defaultdict(list)
    for name,value in request['response_headers']:headers[name.lower()].append(value)
    for key in ('content-length','content-range','content-type','content-encoding','transfer-encoding','etag','last-modified'):
        check('singleton_header:'+key,len(headers[key])<=1)
    check('response_identity',headers['etag']==['"656fa2a5-9d2a0879"'] and not headers['content-range']
          and not headers['transfer-encoding'] and headers['content-encoding'] in ([],['identity']))
    verify(request['file'])
    body=(ACQ/'raw/multipart.bin').read_bytes()
    check('exact_body_length_and_reservation',headers['content-length']==[str(len(body))]
          and len(body)==request['declared_body_bytes']==request['retained_body_bytes']==ledger['retained_response_body_bytes']==result['retained_response_body_bytes']
          and len(body)<222640 and request['reserved_response_body_bytes']==ledger['reserved_response_body_bytes']==result['reserved_response_body_bytes']==222640
          and request['elapsed_seconds']<60 and hashlib.sha256(body).hexdigest()==request['body_sha256'])
    match=re.fullmatch(r'multipart/byteranges; boundary=([A-Za-z0-9_.-]{1,70})',headers['content-type'][0])
    check('observed_outer_mime',match is not None)
    boundary=match[1].encode();marker=b'--'+boundary
    cursor=2 if body.startswith(b'\r\n') else 0
    parts={};observed_parts=[];framing_sum=0
    # Independent cursor parser consumes exactly 144 bytes per requested row.
    while True:
        check('boundary_at:'+str(cursor),body[cursor:cursor+len(marker)]==marker)
        cursor+=len(marker)
        if body[cursor:cursor+4]==b'--\r\n':
            cursor+=4;check('exact_final_termination',cursor==len(body));break
        check('part_boundary_crlf:'+str(cursor),body[cursor:cursor+2]==b'\r\n');cursor+=2
        header_start=cursor;end=body.find(b'\r\n\r\n',cursor)
        check('bounded_mime_headers:'+str(cursor),end>=cursor and end-cursor<=512)
        fields={}
        for line in body[cursor:end].decode('ascii').split('\r\n'):
            name,value=line.split(':',1);name=name.lower();value=value.strip()
            check('part_header_unique:'+str(cursor)+name,name not in fields and name in ('content-range','content-length','content-type'))
            fields[name]=value
        check('part_mime:'+str(cursor),'content-type' not in fields or fields['content-type']=='application/octet-stream')
        m=re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)',fields['content-range'])
        check('part_exact_range:'+str(cursor),m is not None)
        a,b,total=map(int,m.groups());span=(a,b)
        check('part_membership:'+str(cursor),span in set(map(tuple,ranges)) and span not in parts and total==2636777593 and b-a+1==144)
        check('part_optional_length:'+str(cursor),'content-length' not in fields or fields['content-length']=='144')
        cursor=end+4;payload_start=cursor;payload=body[cursor:cursor+144]
        check('part_payload_extent:'+str(cursor),len(payload)==144)
        parts[span]=payload;cursor+=144
        framing=end+4-header_start+2+len(marker)+2;framing_sum+=framing
        check('part_framing_cap:'+str(cursor),framing<=768)
        observed_parts.append(dict(range_inclusive=[a,b],payload_offset=payload_start,payload_bytes=144,
            payload_sha256=hashlib.sha256(payload).hexdigest(),headers=fields,framing_bytes=framing))
        check('payload_followed_by_crlf:'+str(cursor),body[cursor:cursor+2]==b'\r\n');cursor+=2
    check('exact_243_part_records',len(parts)==243 and observed_parts==request['parts']
          and 0<=len(body)-34992-framing_sum<=1024)
    raw=b''.join(parts[tuple(span)] for span in ranges)
    check('reassembled_payload_exact',raw==(ACQ/'raw/records.bin').read_bytes() and len(raw)==34992
          and hashlib.sha256(raw).hexdigest()==result['validation']['records_sha256'])
    candidates={};decoded={}
    for name,prefix in [('little','<'),('big','>')]:
        rows=[struct.unpack(prefix+'6d',raw[offset:offset+48]) for offset in range(0,len(raw),48)]
        decoded[name]=rows;zero=[];finite_bad=unit_bad=0;finite_errors=[]
        for row in rows:
            empty=all(v==0.0 for v in row);zero.append(empty)
            if empty:continue
            finite_bad+=not all(math.isfinite(v) for v in row)
            error=abs(norm(row[3:])-1.0)
            unit_bad+=not (math.isfinite(error) and error<=1e-5)
            if math.isfinite(error):finite_errors.append(error)
        report=dict(record_count=729,zero_sentinel_count=sum(zero),mapped_count=729-sum(zero),
            nonfinite_mapped_count=finite_bad,nonunit_mapped_count=unit_bad,
            maximum_finite_normal_norm_error=max(finite_errors) if finite_errors else None,
            normal_absolute_tolerance=1e-5,passed=sum(zero)<729 and finite_bad==0 and unit_bad==0)
        candidates[name]=report
        observed=result['validation']['candidates'][name]
        for key,value in report.items():
            if key=='maximum_finite_normal_norm_error':close('candidate:'+name+':'+key,observed[key],value)
            else:check('candidate:'+name+':'+key,observed[key]==value)
    passing=[name for name,value in candidates.items() if value['passed']]
    check('unique_invariant_interpretation',len(passing)==1 and passing==result['validation']['passing_candidates']
          and passing[0]==result['validation']['selected_byte_order'] and result['validation']['status']=='CONDITIONAL')
    rows=decoded[passing[0]];summaries=[];mapped_total=inside_total=0
    geometry=result['geometry']
    check('fixed_geometry_shape',len(geometry['samples'])==81)
    for idx,(point,observed) in enumerate(zip(points,geometry['samples'])):
        for key,value in point.items():check('point:'+str(idx)+':'+key,observed[key]==value)
        neighborhood=rows[idx*9:idx*9+9];mapped=[not all(v==0.0 for v in row) for row in neighborhood]
        for k,(row,valid) in enumerate(zip(neighborhood,mapped)):
            x,y,z=row[:3];inside=valid and 0<=x<6300 and 0<=y<2260 and 0<=z<8855
            record=observed['records_row_major'][k];mapped_total+=valid;inside_total+=inside
            check(f'record:{idx}:{k}',record['mapped']==valid and record['xyz']==(list(row[:3]) if valid else None)
                  and record['normal']==list(row[3:]) and record['inside_declared_primary_volume_bounds']==(inside if valid else None))
            close(f'record_norm:{idx}:{k}',record['normal_norm'],norm(row[3:]))
        dx=[(neighborhood[5][d]-neighborhood[3][d])*.5 for d in range(3)] if mapped[3] and mapped[5] else None
        dy=[(neighborhood[7][d]-neighborhood[1][d])*.5 for d in range(3)] if mapped[1] and mapped[7] else None
        area=None if dx is None or dy is None else [dx[1]*dy[2]-dx[2]*dy[1],dx[2]*dy[0]-dx[0]*dy[2],dx[0]*dy[1]-dx[1]*dy[0]]
        n=neighborhood[4][3:] if mapped[4] else None
        computed=dict(center_mapped=mapped[4],complete_mapped_neighborhood=all(mapped),
            jacobian_xyz_by_xy=None if dx is None or dy is None else [[dx[k],dy[k]] for k in range(3)],
            tangent_x=dx,tangent_y=dy,tangent_x_norm=norm(dx) if dx is not None else None,
            tangent_y_norm=norm(dy) if dy is not None else None,local_area_scale=norm(area) if area is not None else None,
            tangent_cosine=cosine(dx,dy) if dx is not None and dy is not None else None,
            normal_dot_unit_tangent_x=cosine(n,dx) if n is not None and dx is not None else None,
            normal_dot_unit_tangent_y=cosine(n,dy) if n is not None and dy is not None else None,
            normal_dot_unit_raster_cross=cosine(n,area) if n is not None and area is not None else None)
        for key,value in computed.items():close(f'geometry:{idx}:{key}',observed[key],value)
        summaries.append(dict(**point,**computed))
    sites=[{key:s[key] for key in ('lattice_ij','center_yx','center_mapped','complete_mapped_neighborhood')}
           for s in summaries if s['potential_native_site']]
    complete=sum(s['complete_mapped_neighborhood'] for s in summaries)
    site_complete=sum(s['complete_mapped_neighborhood'] for s in sites)
    check('global_geometry_counts',mapped_total==geometry['mapped_records'] and inside_total==geometry['records_inside_declared_primary_bounds']
          and sum(s['center_mapped'] for s in summaries)==geometry['center_mapped_count']
          and complete==geometry['complete_mapped_neighborhood_count'])
    check('nine_fixed_sites_unreplaced',sites==geometry['fixed_potential_native_sites'] and len(sites)==9
          and site_complete==geometry['fixed_native_complete_count']
          and (site_complete>=6)==geometry['minimum_six_fixed_native_sites_condition_met'])
    check('unresolved_provenance_preserved',geometry['stage_c_authorized'] is False
          and all(geometry[key] is None and result[key] is None for key in
                  ('source_volume_binding','plane_offset_rule','anatomical_normal','physical_ink_depth')))
    distributions={}
    for key in ('tangent_x_norm','tangent_y_norm','local_area_scale','tangent_cosine',
                'normal_dot_unit_tangent_x','normal_dot_unit_tangent_y','normal_dot_unit_raster_cross'):
        values=[s[key] for s in summaries if s[key] is not None]
        distributions[key]=dict(finite_count=len(values),minimum=min(values),median=statistics.median(values),
                                 maximum=max(values),maximum_absolute=max(map(abs,values)),zero_count=sum(v==0 for v in values)) if values else dict(finite_count=0)
    summary=dict(endian_candidates=candidates,selected_conditional_byte_order=passing[0],mapped_records=mapped_total,
        zero_sentinel_records=729-mapped_total,records_inside_declared_primary_bounds=inside_total,
        complete_neighborhoods=complete,unmapped_neighborhoods=81-complete,fixed_sites=sites,
        fixed_complete_sites=site_complete,six_site_condition_met=site_complete>=6,diagnostics=distributions)
    report=dict(kind='phase21-independent-frag6-b-retained-runtime-v1',status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        check_count=len(checks),checks=checks,summary=summary,
        anchors=dict(plan=pin(BASE/'execution-plan.json'),worker=pin(ACQ/'worker-result.json'),result=pin(ACQ/'SPARSE_GEOMETRY.json')),
        retained_inputs=[pin(p) for p in [RUN/'controller-start.json',RUN/'controller-result.json',ACQ/'request-ledger.json',
            ACQ/'raw/multipart.bin',ACQ/'raw/records.bin',PHASE/'FRAG6_STAGE_B_GO.json',HERE/'NUMERICAL_CONTRACT.md',HERE/'verify_retained_run.py']],
        execution=dict(new_network_attempts=0,retained_ppm_records_independently_decoded=729,endian_interpretations=2,
                       native_ct_pixels=0,labels=0,model_calls=0,gpu_spend_gbp=0),
        resources=dict(wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        numerical_comparison=dict(absolute_tolerance=ATOL,relative_tolerance=RTOL,admission_unit_normal_tolerance=1e-5),
        limitations=['Sparse serialization/geometry compatibility only; no source UUID or anatomical depth is authenticated.',
                     'Seven complete fixed sites can satisfy only the six-site coverage prerequisite; missing render/source provenance still closes Stage C.',
                     'No native comparison, CT input contract, label check, scoring or tuning was performed.'])
    with (HERE/'verification.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    with (HERE/'independent-geometry.json').open('x') as f:json.dump(dict(summary=summary,neighborhoods=summaries),f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks),summary=summary,resources=report['resources'])))


if __name__=='__main__':main()
