"""Independent production range/row arithmetic; synthetic inputs until root authorizes."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent


def require(value,message):
    if not value:raise ValueError(message)


def load(path,sha,name):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==sha,'Sealed independent helper differs')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


MIME=load(ROOT/'production-preparation/multipart_256.py','3a901cb45dab6e680ac8d25c4892fce5f277b25889e9eb1bb50ef9dbcadceccc','production_independent_MIME')
TIFF=load(ROOT/'endian-v2/tiff_math_v2.py','0157c1b5bde34395e0af958d72d0892ac97823b87ab476bc6841c2da6f2a613a','production_independent_TIFF')
POOL=load(ROOT/'endian-v2/pooling_math_v2.py','fa2d34fed845f75dfd3aa744987cf5427e08339da600b3526f33fb0f27a6b794','production_independent_POOL')


def expected_spans(layout,x,y,row_count,width=1152):
    require(layout['orientation']==1 and layout['photometric']=='min-is-black'and width==1152 and
            row_count in (128,256)and 0<=x and 0<=y and x+width<=layout['width']and y+row_count<=layout['height'],
            'Declared source crop/layout outside bounds')
    start=layout['strip_offset'];stop=start+layout['strip_bytes'];canvas=layout['width']
    spans=[[start+2*((y+i)*canvas+x),start+2*((y+i)*canvas+x+width)-1]for i in range(row_count)]
    require(all(start<=a<=b<stop<=layout['object_total_bytes']for a,b in spans),'Raw row spans outside strip/object')
    return spans


def validate_batch(body,case,layout,batch):
    x,y,w,h=case['bbox_source_xywh'];origin=batch['native_row_start'];n=batch['row_count']
    allowed={(y+s,n)for s,n in [(0,256),(256,256),(512,256),(768,256),(1024,128)]}
    require((origin,n)in allowed and w==h==1152 and batch['native_x_start']==x and batch['native_crop_width']==w,
            'Fixed batch routing/bbox differs')
    require(batch['case_id']==case['case_id']and batch['fragment']==case['fragment']and
            batch['source_url']==layout['url']and batch['etag']==layout['etag']and
            batch['object_total_bytes']==layout['object_total_bytes']and batch['status_code']==206,
            'Batch source/routing/HTTP identity differs')
    spans=expected_spans(layout,x,origin,n)
    result=MIME.parse(body,batch['content_type'],spans,layout['object_total_bytes'],maximum_body=1048576)
    parts=[dict(**m,payload_sha256=hashlib.sha256(result['parts'][tuple(m['range_inclusive'])]).hexdigest())
           for m in result['part_metadata']]
    require(batch['parts']==parts,'Physical part order/headers/offsets differ')
    by_span={tuple(m['range_inclusive']):m for m in parts}
    require(len(batch['rows'])==n and len({r['native_y']for r in batch['rows']})==n,'Duplicate/missing routed native rows')
    wanted={origin+i:dict(native_y=origin+i,range_inclusive=span,payload_offset=by_span[tuple(span)]['payload_offset'],
            payload_bytes=2304,payload_sha256=by_span[tuple(span)]['payload_sha256'])for i,span in enumerate(spans)}
    require({r['native_y']:r for r in batch['rows']}==wanted,'Native row mapping/offset/hash differs')
    return result


def decode_batch(result,layout,x,y,row_count):
    require((layout['dtype'],layout['byte_order'],layout['struct_endian'])in
            [('<u2','little','<'),('>u2','big','>')],'Validated per-file byte order/dtype required')
    spans=expected_spans(layout,x,y,row_count)
    values=np.empty((row_count,1152),np.uint32)
    for i,span in enumerate(spans):
        payload=result['parts'][tuple(span)]
        require(len(payload)==2304,'Exact uint16 source row required')
        values[i]=np.frombuffer(payload,dtype=layout['dtype']).astype(np.uint32)
    require(np.all(values<=65535),'Decoded uint16 value range differs')
    return values


def validate_inventory(index,cases):
    require(len(index)==1530 and len(cases)==6,'Exact six-context1530batch inventory required')
    expected={(case['case_id'],case['fragment'],plane,y+offset,n)
        for case in cases for plane in range(7,58)
        for y in [case['bbox_source_xywh'][1]]
        for offset,n in [(0,256),(256,256),(512,256),(768,256),(1024,128)]}
    actual={(b['case_id'],b['fragment'],b['plane_index'],b['native_row_start'],b['row_count'])for b in index}
    ids=[b['request_id']for b in index]
    require(actual==expected and len(actual)==len(ids)==len(set(ids))==1530 and
            all(isinstance(i,str)and i for i in ids),'Logical row coverage/identity duplicated or missing')
    useful=sum(b['row_count']*2304 for b in index)
    reserved=sum(b['row_count']*(2304+768)+1024 for b in index)
    require(useful==812187648 and reserved==1084483584,'Exact useful/reservation inventory totals differ')
    return dict(logical_batches=1530,source_context_layers=306,useful_bytes=useful,reserved_bytes=reserved,
                every_context_layer_row_covered_once=True)


def fixtures():
    checked=rejected=0
    for order,dtype,prefix in [('little','<u2','<'),('big','>u2','>')]:
        layout=dict(dtype=dtype,byte_order=order,struct_endian=prefix,width=4000,height=4000,
                    orientation=1,photometric='min-is-black',strip_offset=260,strip_bytes=32000000,
                    object_total_bytes=32000260,url='https://example.invalid/synthetic.tif',etag='"fixture"')
        case=dict(case_id='synthetic',fragment='frag1',bbox_source_xywh=[2304,1152,1152,1152])
        for origin,n in [(1152,256),(2176,128)]:
            spans=expected_spans(layout,2304,origin,n)
            logical=((np.arange(n)[:,None]*1000+np.arange(1152)[None,:]*3)%65536).astype(np.uint16)
            payloads=[logical[i].astype(dtype).tobytes()for i in range(n)]
            encoded=MIME.fixture_body([(tuple(span),payload,layout['object_total_bytes'])for span,payload in zip(spans,payloads)][::-1])
            parsed=MIME.parse(encoded,'multipart/byteranges; boundary=fixed',spans,layout['object_total_bytes'])
            parts=[dict(**m,payload_sha256=hashlib.sha256(parsed['parts'][tuple(m['range_inclusive'])]).hexdigest())for m in parsed['part_metadata']]
            by_span={tuple(p['range_inclusive']):p for p in parts}
            rows=[dict(native_y=origin+i,range_inclusive=span,payload_offset=by_span[tuple(span)]['payload_offset'],payload_bytes=2304,
                       payload_sha256=by_span[tuple(span)]['payload_sha256'])for i,span in enumerate(spans)]
            batch=dict(case_id='synthetic',fragment='frag1',native_row_start=origin,row_count=n,native_x_start=2304,
                native_crop_width=1152,source_url=layout['url'],etag=layout['etag'],object_total_bytes=layout['object_total_bytes'],
                status_code=206,content_type='multipart/byteranges; boundary=fixed',parts=parts,rows=rows[::-1])
            result=validate_batch(encoded,case,layout,batch)
            values=decode_batch(result,layout,2304,origin,n)
            require(np.array_equal(values,logical.astype(np.uint32)),'Endian/row-order decoded values differ');checked+=1
            for field,value in [('native_row_start',origin+1),('native_x_start',2305),('etag','"wrong"')]:
                bad=dict(batch);bad[field]=value
                try:validate_batch(encoded,case,layout,bad)
                except ValueError:rejected+=1
                else:raise ValueError('Incorrect synthetic source routing accepted')
    cases=[dict(case_id=f'frag{f}-context{c}',fragment=f'frag{f}',bbox_source_xywh=[0,1152*c,1152,1152])
           for f in range(1,4)for c in range(1,3)]
    index=[]
    for case in cases:
        for plane in range(7,58):
            for ordinal,(offset,n)in enumerate([(0,256),(256,256),(512,256),(768,256),(1024,128)]):
                index.append(dict(request_id=f"{case['case_id']}-p{plane}-b{ordinal}",case_id=case['case_id'],
                    fragment=case['fragment'],plane_index=plane,native_row_start=case['bbox_source_xywh'][1]+offset,row_count=n))
    validate_inventory(index,cases)
    invalid=[index[:-1],index[:-1]+[index[0]],index[:-1]+[dict(index[-1],native_row_start=index[-1]['native_row_start']+1)],
             index[:-1]+[dict(index[-1],request_id=index[0]['request_id'])]]
    for wrong in invalid:
        try:validate_inventory(wrong,cases)
        except ValueError:pass
        else:raise ValueError('Incomplete/duplicate/gapped synthetic coverage accepted')
    return dict(status='PASS',accepted_endian_row_routing_cases=checked,rejected_routing_cases=rejected,
        accepted_complete_inventory_cases=1,rejected_inventory_cases=4,
        frozen_pooling_fixtures=POOL.fixtures(),actual_production_body_reads=0,actual_CT_arrays_decoded=0,
        network_calls=0,model_calls=0)
