"""Independent synthetic arithmetic/MIME controls; no real payload or network."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
AUTHOR = HERE.parent / 'frag6-stage-b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-pins', type=Path, required=True)
    args = parser.parse_args()
    start_time = time.monotonic()
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    pins = json.loads(args.source_pins.read_text())
    for p in pins['author_sources']:
        data=Path(p['path']).read_bytes()
        assert len(data)==p['bytes'] and hashlib.sha256(data).hexdigest()==p['sha256']
    sys.path.insert(0, str(AUTHOR))
    import geometry
    import multipart
    from common import ContractError
    checks=[]
    def check(name, condition, **details):
        assert condition, name
        checks.append(dict(name=name,passed=True,**details))
    def reject(name, action):
        try: action()
        except ContractError as error: check(name,True,reason=str(error))
        else: raise AssertionError(name)

    width,height,offset,total=6205,8853,73,2636777593
    oracle_points=[];oracle_ranges=[];coordinates=[]
    for i in range(9):
        cy=((i+1)*8852)//10
        for j in range(9):
            cx=((j+1)*6204)//10
            rows=[]
            for y in range(cy-1,cy+2):
                a=73+(y*6205+cx-1)*6*8
                rows.append([a,a+3*6*8-1])
                for x in range(cx-1,cx+2): coordinates.append((x,y))
            oracle_points.append(dict(lattice_ij=[i,j],center_yx=[cy,cx],row_ranges=rows,
                                      potential_native_site=i in (1,4,7) and j in (1,4,7)))
            oracle_ranges.extend(rows)
    points,ranges=geometry.derive_points(width,height,offset,total)
    check('independent_81_center_and_243_span_identity',points==oracle_points and ranges==oracle_ranges)
    check('independent_extent_and_nonoverlap',len(ranges)==243 and len(coordinates)==729
          and sum(b-a+1 for a,b in ranges)==34992
          and all(a[1]<b[0] for a,b in zip(sorted(ranges),sorted(ranges)[1:])))
    for label,args_bad in [('small_overlapping_canvas',(11,11,73,73+11*11*48)),
                           ('wrong_extent',(width,height,offset,total+1)),
                           ('bool_dimension',(True,height,offset,total)),
                           ('zero_offset',(width,height,0,width*height*48))]:
        reject('derive_reject_'+label,lambda a=args_bad:geometry.derive_points(*a))

    cross=(-.001,-.004,.02)
    area=math.sqrt(sum(v*v for v in cross))
    normal=tuple(v/area for v in cross)
    values=[(.2*x,.1*y,500+.01*x+.02*y,*normal) for x,y in coordinates]
    def encode(records,order='<'):return b''.join(struct.pack(order+'6d',*r) for r in records)
    raw=encode(values)
    for name,order in [('little','<'),('big','>')]:
        trial=encode(values,order)
        choice=geometry.candidates(trial)
        check(name+'_selected_unique_invariant',choice['status']=='CONDITIONAL' and choice['passing_candidates']==[name]
              and choice['candidates'][name]['mapped_count']==729)
        result=geometry.describe(trial,choice,points)
        check(name+'_all_fixed_neighborhoods_preserved',len(result['samples'])==81 and result['mapped_records']==729
              and result['complete_mapped_neighborhood_count']==81 and result['fixed_native_complete_count']==9)
        for k,item in enumerate(result['samples']):
            check(name+'_analytic_geometry_'+str(k),
                  max(abs(a-b) for a,b in zip(item['tangent_x'],(.2,0,.01)))<1e-11
                  and max(abs(a-b) for a,b in zip(item['tangent_y'],(0,.1,.02)))<1e-11
                  and abs(item['local_area_scale']-area)<1e-11
                  and abs(item['normal_dot_unit_tangent_x'])<1e-10
                  and abs(item['normal_dot_unit_tangent_y'])<1e-10
                  and abs(item['normal_dot_unit_raster_cross']-1)<1e-10)
        check(name+'_claims_remain_qualified',result['records_inside_declared_primary_bounds']==729
              and not result['stage_c_authorized'] and result['source_volume_binding'] is None
              and result['plane_offset_rule'] is None and result['anatomical_normal'] is None
              and result['physical_ink_depth'] is None)
    for label,replacement in [('nan_coordinate',(float('nan'),1,2,*normal)),
                              ('infinite_coordinate',(float('inf'),1,2,*normal)),
                              ('nonunit_normal',(1,2,3,0,0,1.00002)),
                              ('zero_normal_mapped',(1,2,3,0,0,0))]:
        altered=values.copy();altered[728]=replacement
        candidate=geometry.candidates(encode(altered))
        check('reject_whole_interpretation_'+label,candidate['status']=='STOP' and not candidate['passing_candidates'])
    candidate=geometry.candidates(bytes(34992))
    check('all_zero_unmapped_stops',candidate['status']=='STOP'
          and all(v['zero_sentinel_count']==729 for v in candidate['candidates'].values()))
    palindrome=bytes.fromhex('3ff000000000f03f')
    ambiguous_record=bytes(24)+palindrome+bytes(16)
    candidate=geometry.candidates(ambiguous_record*729)
    check('both_endians_valid_stops',candidate['status']=='STOP' and candidate['passing_candidates']==['little','big'])
    reject('ambiguous_geometry_forbidden',lambda:geometry.describe(ambiguous_record*729,candidate,points))
    origin=values.copy();origin[0]=(0,0,0,0,0,1)
    candidate=geometry.candidates(encode(origin))
    check('mapped_origin_not_zero_sentinel',candidate['candidates']['little']['mapped_count']==729)
    near=values.copy();near[0]=(1,2,3,0,0,1.000005)
    check('within_frozen_normal_tolerance_allowed',geometry.candidates(encode(near))['selected_byte_order']=='little')
    fixed_indices=[i*9+j for i in (1,4,7) for j in (1,4,7)]
    for missing in (1,3,4):
        altered=values.copy()
        for site in fixed_indices[:missing]: altered[site*9]=(0,0,0,0,0,0)
        trial=encode(altered);choice=geometry.candidates(trial);result=geometry.describe(trial,choice,points)
        check('sentinels_retained_without_site_replacement_'+str(missing),result['mapped_records']==729-missing
              and result['complete_mapped_neighborhood_count']==81-missing and result['fixed_native_complete_count']==9-missing
              and result['minimum_six_fixed_native_sites_condition_met']==(missing<=3)
              and [s['lattice_ij'] for s in result['fixed_potential_native_sites']]==[[i,j] for i in (1,4,7) for j in (1,4,7)])
    outside=values.copy();outside[0]=(6300,0,0,*normal)
    trial=encode(outside);choice=geometry.candidates(trial);result=geometry.describe(trial,choice,points)
    check('candidate_bounds_are_diagnostic_not_endian_selection',choice['selected_byte_order']=='little'
          and result['records_inside_declared_primary_bounds']==728)
    reject('wrong_raw_extent_rejected',lambda:geometry.candidates(raw[:-1]))
    valid=geometry.candidates(raw)
    reject('changed_raw_validation_digest_rejected',lambda:geometry.describe(bytes(34992),valid,points))

    boundary=b'IndependentStage21'
    mime='multipart/byteranges; boundary='+boundary.decode()
    def message(order=None, source=raw, override=None):
        order=list(range(243)) if order is None else order
        chunks=[]
        for k in order:
            a,b=ranges[k]
            headers=(f'Content-Type: application/octet-stream\r\nContent-Range: bytes {a}-{b}/{total}\r\nContent-Length: 144').encode()
            if override is not None: headers=override(k,headers)
            chunks.append(b'--'+boundary+b'\r\n'+headers+b'\r\n\r\n'+source[k*144:(k+1)*144]+b'\r\n')
        return b''.join(chunks)+b'--'+boundary+b'--\r\n'
    body=message()
    parts=multipart.parse_multipart(body,mime,ranges,total)
    check('exact_243_parts_reassemble_original_bytes',len(parts)==243 and geometry.assemble(body,parts,ranges)==raw)
    check('explicit_framing_allowances',all(p['framing_bytes']<=768 for p in parts)
          and len(body)-sum(p['payload_bytes']+p['framing_bytes'] for p in parts)<=1024 and len(body)<222640)
    reversed_body=message(order=list(reversed(range(243))))
    parsed=multipart.parse_multipart(reversed_body,mime,ranges,total)
    check('multipart_reordering_preserves_frozen_schedule',geometry.assemble(reversed_body,parsed,ranges)==raw)
    check('quoted_boundary_and_initial_crlf_allowed',len(multipart.parse_multipart(b'\r\n'+body,
          'Multipart/ByteRanges; boundary="'+boundary.decode()+'"',ranges,total))==243)
    no_optional=message(override=lambda k,h:h.split(b'\r\n')[1])
    check('only_required_content_range_part_header_supported',len(multipart.parse_multipart(no_optional,mime,ranges,total))==243)
    needle=b'\r\n--'+boundary+b'\r\n'
    embedded=needle+raw[len(needle):]
    embedded_body=message(source=embedded)
    parsed=multipart.parse_multipart(embedded_body,mime,ranges,total)
    check('delimiter_like_binary_payload_is_length_delimited',geometry.assemble(embedded_body,parsed,ranges)==embedded)
    malformed={
      'missing_part':message(order=list(range(242))),
      'duplicate_part':message(order=[0,0,*range(2,243)]),
      'extra_part':message(order=[*range(243),0]),
      'trailing_epilogue':body+b'X', 'arbitrary_preamble':b'junk'+body,
      'missing_final_crlf':body[:-2], 'truncated_part':body[:500],
      'lf_only_headers':body.replace(b'\r\n',b'\n'),
      'wrong_part_mime':message(override=lambda k,h:h.replace(b'application/octet-stream',b'text/html') if k==0 else h),
      'duplicate_header':message(override=lambda k,h:h+b'\r\nContent-Length: 144' if k==0 else h),
      'unknown_header':message(override=lambda k,h:h+b'\r\nX-Foo: bar' if k==0 else h),
      'transfer_encoding':message(override=lambda k,h:h+b'\r\nTransfer-Encoding: chunked' if k==0 else h),
      'non_ascii_header':message(override=lambda k,h:h+b'\r\nX-Invalid: \xff' if k==0 else h),
      'wrong_part_length':message(override=lambda k,h:h.replace(b'Content-Length: 144',b'Content-Length: 143') if k==0 else h),
      'changed_total':message(override=lambda k,h:h.replace(str(total).encode(),str(total+1).encode()) if k==0 else h),
      'unrequested_range':message(override=lambda k,h:h.replace(f'{ranges[0][0]}-{ranges[0][1]}'.encode(),f'{ranges[0][0]+1}-{ranges[0][1]+1}'.encode()) if k==0 else h),
      'coalesced_range':message(override=lambda k,h:h.replace(str(ranges[0][1]).encode(),str(ranges[1][1]).encode()) if k==0 else h),
      'oversized_header':message(override=lambda k,h:h+b'\r\nX-Long: '+b'a'*513 if k==0 else h),
      'body_cap':body+b'X'*222641,
    }
    for name,bad in malformed.items(): reject('MIME_reject_'+name,lambda b=bad:multipart.parse_multipart(b,mime,ranges,total))
    for name,bad_mime in [('wrong_type','text/html'),('missing_boundary','multipart/byteranges'),
       ('duplicate_boundary',mime+'; boundary=second'),('empty_boundary','multipart/byteranges; boundary=""'),
       ('oversized_boundary','multipart/byteranges; boundary='+('x'*71)),
       ('unclosed_quote','multipart/byteranges; boundary="bad')]:
        reject('outer_MIME_reject_'+name,lambda t=bad_mime:multipart.parse_multipart(body,t,ranges,total))
    overlap=[s.copy() for s in ranges];overlap[1]=[overlap[0][1],overlap[0][1]+143]
    reject('overlapping_schedule_rejected',lambda:multipart.parse_multipart(body,mime,overlap,total))
    duplicate=[s.copy() for s in ranges];duplicate[1]=duplicate[0]
    reject('duplicate_schedule_rejected',lambda:multipart.parse_multipart(body,mime,duplicate,total))
    check('resource_bounds',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<256*1024**2
          and time.monotonic()-start_time<120)
    result=dict(status='PASS',kind='phase21-independent-frag6-b-pure-fixtures-v1',check_count=len(checks),checks=checks,
        source_pins_sha256=hashlib.sha256(args.source_pins.read_bytes()).hexdigest(),
        resources=dict(wall_seconds=time.monotonic()-start_time,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        execution=dict(network_calls=0,real_ppm_records_decoded=0,model_calls=0,annotation_reads=0),
        synthetic_planar_data_sha256=hashlib.sha256(raw).hexdigest())
    with (HERE/'pure-fixtures-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','check_count','resources','execution')}))


if __name__=='__main__': main()
