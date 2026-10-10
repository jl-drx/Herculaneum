"""Independent offline production-byte, TIFF, coverage and exact FP32 pooling check.

Source/synthetic preparation does not inspect future production files. Root supplies
three explicit entry pins and authorizes one saved-output pass after acquisition.
No producer module is imported, and checkpoints are neither opened nor decoded.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import sys
import time

sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='4'
def offline(*args,**kwargs):raise RuntimeError('Independent production verifier has no network')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
import numpy as np
import reconstruction_math as M

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
PINS={}
BLOCKED={'.pt','.pth','.ckpt','.safetensors','.npz'}
REFS={'design':'5c942839223b69b8667da841db82babc15dbf62eaed0bf37005d76130e5315d8',
      'schema':'70c810cbfc6bc1ed174f9c3d7c5654c6e3e8f4223ec6ea101aea7dd373be2f5c',
      'benchmark':'a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735',
      'source':'53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890',
      'header_result':'dde73506ee5de6b40979125303c5956a52e42052b1b340a74201b3314487ed1b',
      'serving_result':'82fe4e347c3a41ed934e9b14571675a8dad9fff245e453c448ede226d16519da',
      'large_result':'de9d67d9c3b091e73246178de5621b2db4d4a27ef7d1e295d211fef4b33439c6'}
SHAPES={'frag1':[8181,6330],'frag2':[14830,9506],'frag3':[7606,5249]}
IDENTITIES={'frag1':('PHercParis2Fr47','evaluation','Paris2'),
            'frag2':('PHercParis2Fr143','evaluation','Paris2'),
            'frag3':('PHercParis1Fr34','calibration','Paris1')}
BOXES={'frag1-context1':[2304,1152,1152,1152],'frag1-context2':[0,6912,1152,1152],
       'frag2-context1':[1152,1152,1152,1152],'frag2-context2':[8064,12672,1152,1152],
       'frag3-context1':[2304,1152,1152,1152],'frag3-context2':[3456,5760,1152,1152]}
GROUPS=[(0,256),(256,256),(512,256),(768,256),(1024,128)]
require=M.require


def identity(item):
    sizes=[item[k]for k in ('bytes','size_bytes','byte_length')if k in item]
    require(sizes and all(type(v)is int and v>=0 and v==sizes[0]for v in sizes),'Invalid file-size identity')
    require(isinstance(item['sha256'],str)and re.fullmatch('[0-9a-f]{64}',item['sha256']),'Invalid SHA256')
    path=Path(item['path']).resolve(strict=True)
    require(path.suffix not in BLOCKED,'Checkpoint/archive reads forbidden')
    return dict(path=str(path),size_bytes=sizes[0],sha256=item['sha256'])


def record(path):
    path=Path(path).resolve(strict=True);require(path.suffix not in BLOCKED,'Checkpoint reads forbidden')
    with path.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)


def pin(item,cap=None):
    expected=identity(item)
    require(cap is None or expected['size_bytes']<=cap,'Input exceeds its byte cap')
    require(expected['path']not in PINS or PINS[expected['path']]==expected,'Conflicting input pin')
    if expected['path']not in PINS:require(record(expected['path'])==expected,'Input pin changed: '+Path(expected['path']).name)
    PINS[expected['path']]=expected
    return Path(expected['path'])


def data(item,cap):
    expected=identity(item);require(expected['size_bytes']<=cap,'Body byte cap exceeded')
    path=Path(expected['path']);require(path.stat().st_size==expected['size_bytes'],'Body length changed')
    with path.open('rb')as stream:body=stream.read(cap+1)
    require(len(body)==expected['size_bytes']and hashlib.sha256(body).hexdigest()==expected['sha256'],'Body pin changed')
    require(expected['path']not in PINS or PINS[expected['path']]==expected,'Conflicting body pin')
    PINS[expected['path']]=expected;return body


def read(item,cap=64*1024**2):return json.loads(data(item,cap))
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def transport(http,request,terminal,body):
    spans=request['ranges'];n=len(spans);total=terminal['object_total_bytes'];etag=terminal['etag'];headers=http['http_identity']
    cap=sum(b-a+1 for a,b in spans)+(1 if request['mode']=='single'else 768*n+1024)
    require(http['success']is True and http['http_status']==terminal['status_code']==206 and
        http['url']==request['url']and http['mode']==request['mode']and http['ranges_inclusive']==spans,
        'Transport206/request identity differs')
    require(request['reserved_cap_bytes']==http['reserved_cap_bytes']==cap and 0<len(body)<=cap and
        len(body)==http['retained_body_bytes']==http['charged_response_bytes']==terminal['retained_body_bytes']==terminal['charged_response_bytes'],
        'Transport reservation/body accounting differs')
    require(http['body_sha256']==hashlib.sha256(body).hexdigest()and identity(http['file'])==identity(terminal['body']),
        'Transport body hash/file differs')
    require(re.fullmatch(r'"[\x21\x23-\x7e]+"',etag)is not None and http['etag']==headers['etag']==etag and
        http['object_total_bytes']==total,'Strong HTTP object identity differs')
    req=http['request_headers'];expected_pin=request.get('pin')
    require(req['Range']=='bytes='+','.join(f'{a}-{b}'for a,b in spans)and req['Accept-Encoding']=='identity'and
        req.get('If-Match')==(expected_pin['etag']if expected_pin else None),'Request headers/If-Match differ')
    if expected_pin:require(expected_pin=={'etag':etag,'object_total_bytes':total},'Pinned source identity differs')
    require(headers.get('content-encoding','identity').lower()in ('','identity'),'Encoded transport unsupported')
    if 'content-length'in headers:require(headers['content-length'].isdigit()and int(headers['content-length'])==len(body),'Content-Length differs')
    require(terminal['content_type']==headers.get('content-type'),'Content-Type receipt differs')
    if request['mode']=='single':
        require(n==1 and headers['content-range']==f'bytes {spans[0][0]}-{spans[0][1]}/{total}'and
            len(body)==spans[0][1]-spans[0][0]+1,'Exact single range differs')
    else:require(n in (128,256)and 'content-range'not in headers,'Multipart transport layout differs')


def ledger(result,plan,count,stage):
    value=read(result['request_ledger']);entries=value['requests'];limits={'metadata':(450,524288,16384,300),'pixels':(1530,1342177280,1048576,900)}[stage]
    require(result['status']=='PASS'and result['stage']==stage and result['http_attempts']==len(entries)==count and
        value['stage']==stage and identity(value['plan'])==identity(result['plan']),'Completed ledger stage/plan differs')
    require(value['verified_tls']is True and value['automatic_retries']==value['redirects_followed']==0 and
        value['maximum_concurrent_requests']==result['maximum_concurrent_requests']==4,'Recorded transport scope differs')
    require(value['limits']==dict(attempts=limits[0],body_bytes=limits[1],per_response=limits[2],wall_seconds=limits[3]),'Stage limits differ')
    require(all(e['status']=='PASS'and e['returncode']==0 and e['reserved_cap_bytes']<=limits[2]for e in entries),
        'Unconfirmed/failed request or response cap exceeded')
    require(len({e['request_id']for e in entries})==count and sum(e['charged_response_bytes']for e in entries)==
        value['charged_response_bytes']==result['charged_response_bytes']<=limits[1]and
        sum(e['retained_body_bytes']for e in entries)==value['retained_body_bytes']==result['retained_body_bytes'],
        'Ledger byte totals/uniqueness differ')
    require(result['wall_seconds']<=limits[3]and result['numeric_pixel_decodes']==result['model_calls']==0 and
        result['production_elapsed_wall_seconds']<=1500,'Recorded production scope/resource acceptance differs')
    terminals=read(result['terminal_receipts'])['terminals']
    require({e['request_id']:identity(e['terminal'])for e in entries}==
        {e['request_id']:identity(e['terminal'])for e in terminals}and len(terminals)==count,'Terminal inventory differs')
    return {e['request_id']:e for e in entries}


def request_receipt(entry,plan,expected,body):
    config=read(entry['request'],1024**2);request=config['request'];terminal=read(entry['terminal'],1024**2)
    require(config['schema_version']==1 and config['python']=='3.12.14'and config['stage']==plan['stage']and
        config['code']==plan['code']and config['limits']==plan['limits']and
        identity(config['plan'])==identity(entry['_plan'])and request==expected,'Scheduled config/plan/request differs')
    require(terminal['status']=='PASS'and terminal['request_id']==request['request_id']==entry['request_id']and
        identity(terminal['request'])==identity(entry['request'])and terminal['numeric_pixel_decodes']==terminal['model_calls']==0,
        'Terminal request/scope differs')
    tl=read(entry['transport_ledger'],2*1024**2)
    require(identity(entry['transport_ledger'])==identity(terminal['transport_ledger'])and
        identity(tl['plan'])==identity(entry['request'])and len(tl['requests'])==1 and
        tl['verified_tls']is True and tl['redirects_followed']==tl['automatic_retries']==0,'Single isolated transport differs')
    require(tl['limits']==dict(attempts=1,body_bytes=request['reserved_cap_bytes'],
        per_response=16384 if plan['stage']=='metadata'else 1048576),'Child transport byte limits differ')
    http=tl['requests'][0];transport(http,request,terminal,body)
    require(entry['url']==request['url']and entry['mode']==request['mode']and
        entry['reserved_cap_bytes']==request['reserved_cap_bytes']and
        entry['charged_response_bytes']==terminal['charged_response_bytes']==tl['charged_response_bytes']and
        entry['retained_body_bytes']==terminal['retained_body_bytes']==tl['retained_body_bytes'],'Parent/child reservation accounting differs')
    for key in ('stdout','stderr'):pin(entry[key],1024**2)
    return terminal,http


def metadata_layout(contract,shape):
    raw=contract['raw_metadata'];header=data(raw['header'],8);count=data(raw['ifd_count'],2);ifd=data(raw['ifd'],16384)
    require(len(header)==8 and len(count)==2 and count==ifd[:2],'Header/count/IFD byte identity differs')
    rebuilt=M.TIFF.inspect(header,ifd,shape,contract['object_total_bytes'])
    for key in ('dtype','byte_order','strip_offset','strip_bytes','ifd_offset','orientation','photometric'):
        require(contract[key]==rebuilt[key],'TIFF reconstructed scalar differs: '+key)
    require(contract['width']==shape[1]and contract['height']==shape[0]and contract['ifd_bytes']==len(ifd)and
        contract['struct_endian']==('<'if rebuilt['byte_order']=='little'else '>'),'TIFF canvas/order/table differs')
    require(contract['specification_defaults_used']==rebuilt['specification_defaults_applied']and
        contract['auxiliary_metadata_extents']==[dict(tag=e['tag'],extent_half_open=[e['start'],e['stop_exclusive']],dereferenced=False)
            for e in rebuilt['external_metadata_extents']],'TIFF defaults/metadata extents differ')
    require(contract['tags']=={key:dict(**value,storage_bytes=M.TIFF.TYPE_BYTES[value['type']]*value['count'])
        for key,value in rebuilt['tags'].items()},'TIFF raw tag bytes/values differ')
    return header,count,ifd,rebuilt


def execute(args):
    started=time.perf_counter();signal.alarm(420)
    resource.setrlimit(resource.RLIMIT_CPU,(600,)*2);resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
    require(list(sys.version_info[:3])==[3,12,14]and np.__version__=='2.5.3','Frozen NumPy/Python runtime differs')
    freeze=read(record(HERE/'code-freeze.json'))
    require(freeze['no_further_edits']is True and freeze['actual_production_inputs_read']==0,'Independent source seal differs')
    for item in freeze['files']:pin(item)
    ep={name:record(getattr(args,name))for name in ('metadata','raw','descriptor')}
    require(all(ep[name]['sha256']==getattr(args,name+'_sha256')for name in ep),'Root entry pins differ')
    meta,raw,descriptor=[read(ep[name])for name in ('metadata','raw','descriptor')]
    mp,pp=read(meta['plan']),read(raw['plan'])
    require(mp['ready']is True and pp['ready']is True and mp['stage']=='metadata'and pp['stage']=='pixels'and
        identity(pp['metadata_result'])==ep['metadata']and identity(pp['metadata_plan'])==identity(meta['plan']),
        'Metadata-before-pixel ready barriers differ')
    require(meta['fixed_inputs']==raw['fixed_inputs']==mp['fixed_inputs']==pp['fixed_inputs']and
        meta['tiff_contracts']==raw['tiff_contracts']==pp['tiff_contracts'],'Pinned fixed inventory/layout differs')
    fixed=raw['fixed_inputs'];refs=fixed['references']
    require(all(refs[k]['sha256']==v for k,v in REFS.items()),'Frozen design/schema/source/probe pins differ')
    for item in refs.values():pin(item)
    benchmark=read(refs['benchmark']);catalog=read(refs['source']);cases=benchmark['selected_contexts']
    require(cases==fixed['cases']and [c['case_id']for c in cases]==list(BOXES),'Literal benchmark case inventory/order differs')
    fragments={v['legacy_alias'].lower():v for v in catalog['eligible_candidates']}
    normalized=[]
    for c in cases:
        alias=c['case_id'].split('-')[0];physical,role,parent=IDENTITIES[alias];x,y,w,h=BOXES[c['case_id']]
        require(c['fragment']==physical and c['role']==role and c['parent_papyrus']==parent and
            c['bbox_source_xywh']==[x,y,w,h]and c['native_bbox_yxhw']==[y,x,h,w],'Physical role/bbox differs')
        normalized.append(dict(c,fragment=alias))
    inventory=M.validate_inventory(raw['batches'],normalized)
    expected_objects=[dict(object_id=f'{alias}-z{z:02d}',fragment=alias,plane_index=z,
        url=fragments[alias]['surface_volume_url']+f'{z:02d}.tif',shape_yx=SHAPES[alias])
        for alias in SHAPES for z in range(7,58)]
    require(fixed['objects']==expected_objects and len(raw['tiff_contracts'])==153,'Literal source153-object inventory differs')
    for plan in (mp,pp):
        for item in plan['code']:pin(item)
        for item in plan['reviews']:pin(item)
    me=ledger(meta,mp,450,'metadata');pe=ledger(raw,pp,1530,'pixels')
    for e in me.values():e['_plan']=meta['plan']
    for e in pe.values():e['_plan']=raw['plan']
    require(set(me)=={f"m-{o['object_id']}-{phase}"for o in expected_objects if o['plane_index']!=7 for phase in ('header','count','ifd')},
        'Metadata450-ID inventory differs')
    require(meta['production_started_epoch_seconds']==raw['production_started_epoch_seconds']==pp['production_started_epoch_seconds'],
        'Shared production timing origin differs')
    prior=read(refs['header_result']);carried={v['fragment']:v for v in prior['tiff_contracts']};layouts={};orders={alias:{}for alias in SHAPES}
    require(fixed['carried_contracts']==[dict(c,plane_index=7,object_id=f"{c['fragment']}-z07")for c in prior['tiff_contracts']],
        'Plane07 carried contract provenance differs')
    for obj,c in zip(expected_objects,raw['tiff_contracts'],strict=True):
        require(all(c[k]==obj[k]for k in ('object_id','fragment','plane_index','url')),'TIFF153-object order/identity differs')
        header,count,ifd,rebuilt=metadata_layout(c,obj['shape_yx']);alias=obj['fragment'];z=obj['plane_index']
        require(re.fullmatch(r'"[\x21\x23-\x7e]+"',c['etag'])is not None,'Per-file strong validator required')
        if z==7:require(c==dict(carried[alias],plane_index=7,object_id=obj['object_id']),'Carried plane07 metadata changed')
        else:
            require(c['prefix_reused']is False,'New metadata must not silently reuse plane07')
            offset=rebuilt['ifd_offset'];strong={'etag':c['etag'],'object_total_bytes':c['object_total_bytes']}
            for phase,span,body in [('header',[0,7],header),('count',[offset,offset+1],count),('ifd',[offset,offset+len(ifd)-1],ifd)]:
                rid=f"m-{obj['object_id']}-{phase}";want=dict(request_id=rid,object_id=obj['object_id'],phase=phase,
                    url=obj['url'],mode='single',ranges=[span],pin=None if phase=='header'else strong,reserved_cap_bytes=span[1]-span[0]+2)
                terminal,_=request_receipt(me[rid],mp,want,body)
                require(identity(terminal['body'])==identity(c['raw_metadata'][{'header':'header','count':'ifd_count','ifd':'ifd'}[phase]])and
                    terminal['etag']==c['etag']and terminal['object_total_bytes']==c['object_total_bytes'],'Metadata body/object binding differs')
        layouts[(alias,z)]=c;orders[alias][c['dtype']]=orders[alias].get(c['dtype'],0)+1
    require(descriptor['status']=='pooled_reference_complete_no_inference'and descriptor['metadata_object_count']==153 and
        descriptor['pixel_batch_count']==1530 and descriptor['independent_CT_verification_pending']is True and descriptor['model_calls']==0 and
        identity(descriptor['raw_provenance'])==ep['raw']and identity(descriptor['pixel_plan'])==identity(raw['plan']),
        'Canonical pooled descriptor/raw binding differs')
    pooling_plan=read(descriptor['pooling_plan']);pooling_go=read(descriptor['pooling_source_go']);pooling_freeze=read(descriptor['pooling_code_freeze'])
    require(pooling_plan['ready_for_execution']is True and identity(pooling_plan['raw_provenance'])==ep['raw']and
        pooling_go['status']=='GO'and identity(pooling_go['code_freeze'])==identity(descriptor['pooling_code_freeze'])and
        pooling_freeze['no_further_edits']is True,'Pooling ready/source gates differ')
    for item in pooling_freeze['files']:pin(item)
    require(pooling_plan['sources']['inference_contract']['sha256']=='6985845aa7d1d43ed633058c6fabd6c6e8b8df0f1b3eae124cd4a86986afe89f'and
        pooling_plan['sources']['inference_code_freeze']['sha256']=='06e138276d5bc9fbb9db1fa657a62b4b6814f4db2cb3023fb8e65baaac344dbb',
        'Frozen inference preprocessing source linkage differs')
    inference_contract=read(pooling_plan['sources']['inference_contract'])
    require(descriptor['preprocessing']==inference_contract['preprocessing']and descriptor['production_schema']['sha256']==REFS['schema']and
        descriptor['production_design']['sha256']==REFS['design'],'Descriptor preprocessing/design differs')
    require([v['id']for v in descriptor['cases']]==list(BOXES),'Descriptor case order differs')
    pixel_by_id={r['request_id']:r for r in pp['requests']};index_by_id={b['request_id']:b for b in raw['batches']}
    require(set(pixel_by_id)==set(index_by_id)==set(pe)and len(pixel_by_id)==1530,'Plan/ledger/batch identity sets differ')
    schedule={(b['case_id'],b['plane_index'],b['native_row_start']):b for b in raw['batches']}
    raw_verification=read(descriptor['raw_verification'])
    require(raw_verification['status']=='PASS'and raw_verification['object_count']==153 and raw_verification['batch_count']==1530 and
        raw_verification['useful_sample_bytes']==812187648 and raw_verification['all_raw_verified_before_numeric_decode']is True and
        identity(raw_verification['plan'])==identity(descriptor['pooling_plan'])and identity(raw_verification['raw_provenance'])==ep['raw'],
        'Published raw-before-decode receipt binding differs')
    author_expected=[ep['raw'],identity(raw['plan']),identity(raw['request_ledger']),identity(raw['terminal_receipts'])]
    author_expected.extend(identity(v)for v in pp['input_pins']+mp['input_pins']+pp['reviews'])
    author_expected.extend([ep['metadata'],identity(meta['plan']),identity(meta['request_ledger']),identity(meta['terminal_receipts'])])
    for c in raw['tiff_contracts']:author_expected.extend(identity(v)for v in c['raw_metadata'].values())
    encoded_bytes=0;row_count=0;exact_results=[]
    for original,case,output in zip(cases,normalized,descriptor['cases'],strict=True):
        require(all(output[k]==original[k]for k in ('native_bbox_yxhw','bbox_source_xywh','fragment','role','parent_papyrus'))and
            output['id']==original['case_id']and set(output['arrays'])=={'surface','labels','primary_mask','known_mask','surface_mask'},
            'Descriptor physical role/array inventory differs')
        names={'labels':'labels.npy','primary_mask':'primary-mask.npy','known_mask':'known-mask.npy','surface_mask':'surface-mask.npy'}
        for role,name in names.items():
            require(identity(output['arrays'][role])==identity(original['files'][name]),'Literal fifteenth array alias changed')
            a=np.load(pin(output['arrays'][role]),allow_pickle=False,mmap_mode='r')
            require(a.shape==(384,384)and a.dtype==np.dtype('uint8'if role=='labels'else 'bool')and a.flags.c_contiguous,
                'Literal label/mask encoding differs')
            if role=='labels':require(set(np.unique(a))<={0,1},'Noncanonical supplied labels')
            author_expected.append(identity(original['files'][name]))
            del a
        native=np.empty((51,1152,1152),np.uint16);alias=case['fragment'];x,y,w,h=case['bbox_source_xywh']
        for depth,z in enumerate(range(7,58)):
            c=layouts[(alias,z)]
            for ordinal,(offset,n)in enumerate(GROUPS):
                index=schedule[(case['case_id'],z,y+offset)];rid=f"p-{case['case_id']}-z{z:02d}-b{ordinal}"
                require(index['request_id']==rid,'Deterministic pixel ID differs')
                batch=read(index['batch'],1024**2);body=data(batch['multipart'],1024**2)
                require(all(index[k]==batch[k]for k in ('request_id','case_id','fragment','plane_index','native_row_start','row_count')),
                    'Raw index/batch route differs')
                spans=M.expected_spans(c,x,y+offset,n)
                want=dict(request_id=rid,object_id=c['object_id'],case_id=case['case_id'],fragment=alias,plane_index=z,url=c['url'],
                    mode='multipart',ranges=spans,native_row_start=y+offset,row_count=n,native_x_start=x,native_crop_width=w,
                    pin={'etag':c['etag'],'object_total_bytes':c['object_total_bytes']},reserved_cap_bytes=2304*n+768*n+1024)
                require(pixel_by_id[rid]==want,'Independent pixel scheduling/ranges differ')
                terminal,http=request_receipt(pe[rid],pp,want,body)
                require(identity(terminal['body'])==identity(batch['multipart'])and identity(pe[rid]['terminal'])==identity(batch['terminal_receipt'])and
                    terminal['content_type']==batch['content_type']and http['parts']==batch['parts'],'Batch/terminal transport part binding differs')
                parsed=M.validate_batch(body,case,c,batch);values=M.decode_batch(parsed,c,x,y+offset,n)
                author_expected.extend([identity(index['batch']),identity(batch['multipart']),identity(batch['terminal_receipt']),
                    identity(terminal['request']),identity(terminal['transport_ledger'])])
                native[depth,offset:offset+n]=values;encoded_bytes+=len(body);row_count+=n
                del values,parsed,body,batch,http,terminal
        pooled=M.POOL.pool51(native);surface_item=output['arrays']['surface'];surface=np.load(pin(surface_item),allow_pickle=False,mmap_mode='r')
        require(surface.shape==pooled.shape==(17,384,384)and surface.dtype==np.dtype('float32')and surface.dtype.isnative and
            surface.flags.c_contiguous and surface_item['shape']==[17,384,384]and surface_item['dtype']=='float32'and
            surface_item['byte_order']==sys.byteorder,'Canonical CT surface encoding differs')
        require(np.array_equal(surface.view(np.uint32),pooled.view(np.uint32)),'CT pooling FP32 bits differ: '+case['case_id'])
        exact_results.append(dict(case_id=case['case_id'],shape=[17,384,384],surface=identity(surface_item),
            all_FP32_bits_exact=True,minimum=float(pooled.min()),maximum=float(pooled.max())))
        del native,pooled,surface;gc.collect()
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=3*1024**3,'Independent CT verification RSS exceeds3GiB')
    require(row_count==6*51*1152 and encoded_bytes==raw['retained_body_bytes'],'Decoded row/encoded byte total differs')
    expected_by_path={v['path']:v for v in author_expected}
    require(all(expected_by_path[v['path']]==v for v in author_expected),'Conflicting published raw-input inventory pin')
    require({identity(v)['path']:identity(v)for v in raw_verification['inputs']}==expected_by_path and
        len(raw_verification['inputs'])==len(expected_by_path)and raw_verification['encoded_multipart_bytes']==encoded_bytes,
        'Published complete raw-verification input inventory differs')
    for item in expected_by_path.values():pin(item)
    for item in PINS.values():require(record(item['path'])==item,'Input changed during independent pass')
    require('torch'not in sys.modules,'Torch imported by independent checker')
    signal.alarm(0)
    return dict(status='PASS',descriptor=ep['descriptor'],all_CT_arrays_exact=True,metadata=ep['metadata'],raw_provenance=ep['raw'],
        TIFF_objects=153,metadata_requests_verified=450,pixel_requests_verified=1530,routed_rows=row_count,
        byte_orders_by_fragment=orders,coverage=inventory,encoded_multipart_bytes=encoded_bytes,exact_arrays=exact_results,
        input_pins=list(PINS.values()),all_verified_inputs_unchanged=True,wall_seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,network_calls=0,model_calls=0,
        checkpoint_reads=0,Torch_imports=0,new_scientific_predictions=0,
        limitations=['HTTP/TLS, scheduling concurrency and producer execution claims are reviewed from pinned source and saved receipts; no network replay.',
            'Exact CT pooling does not validate optical annotations or demonstrate model performance; inference is a separate saved-output review.'])


def self_test(output):
    body=b'abcd';path=output/'synthetic-body.bin';path.write_bytes(body);r=record(path)
    req=dict(url='https://example.invalid/fixture',ranges=[[10,13]],mode='single',reserved_cap_bytes=5,
        pin={'etag':'"fixture"','object_total_bytes':30})
    term=dict(body=r,status_code=206,etag='"fixture"',object_total_bytes=30,content_type='image/tiff',retained_body_bytes=4,charged_response_bytes=4)
    http=dict(file=r,success=True,http_status=206,url=req['url'],ranges_inclusive=[[10,13]],mode='single',reserved_cap_bytes=5,
        body_sha256=r['sha256'],retained_body_bytes=4,charged_response_bytes=4,etag='"fixture"',object_total_bytes=30,
        http_identity={'etag':'"fixture"','content-type':'image/tiff','content-range':'bytes 10-13/30','content-length':'4'},
        request_headers={'Range':'bytes=10-13','Accept-Encoding':'identity','If-Match':'"fixture"'})
    transport(http,req,term,body);rejected=0
    for key,value in [('http_status',200),('body_sha256','0'*64),('reserved_cap_bytes',6),('object_total_bytes',31),('charged_response_bytes',3)]:
        bad=dict(http);bad[key]=value
        try:transport(bad,req,term,body)
        except ValueError:rejected+=1
        else:raise ValueError('Incorrect synthetic transport accepted')
    for bad in [dict(req,pin=None),dict(req,pin={'etag':'"wrong"','object_total_bytes':30})]:
        try:transport(http,bad,term,body)
        except ValueError:rejected+=1
        else:raise ValueError('Incorrect synthetic If-Match accepted')
    accepted_metadata=rejected_metadata=0
    for order in ('little','big'):
        for short_tags in ((),(258,259,262,274)):
            header,ifd,shape,total=M.TIFF.fixture_bytes(order,short_tags=short_tags)
            values={}
            for name,encoded in [('header',header),('ifd_count',ifd[:2]),('ifd',ifd)]:
                path=output/f'synthetic-{order}-{len(short_tags)}-{name}.bin';path.write_bytes(encoded);values[name]=record(path)
            rebuilt=M.TIFF.inspect(header,ifd,shape,total)
            contract=dict(object_total_bytes=total,raw_metadata=values,width=shape[1],height=shape[0],
                dtype=rebuilt['dtype'],byte_order=order,struct_endian='<'if order=='little'else '>',
                strip_offset=rebuilt['strip_offset'],strip_bytes=rebuilt['strip_bytes'],ifd_offset=rebuilt['ifd_offset'],
                ifd_bytes=len(ifd),orientation=rebuilt['orientation'],photometric=rebuilt['photometric'],
                specification_defaults_used=rebuilt['specification_defaults_applied'],
                auxiliary_metadata_extents=[dict(tag=e['tag'],extent_half_open=[e['start'],e['stop_exclusive']],dereferenced=False)
                    for e in rebuilt['external_metadata_extents']],
                tags={key:dict(**value,storage_bytes=M.TIFF.TYPE_BYTES[value['type']]*value['count'])
                    for key,value in rebuilt['tags'].items()})
            metadata_layout(contract,list(shape));accepted_metadata+=1
            bad=dict(contract,dtype='<u2'if order=='big'else '>u2')
            try:metadata_layout(bad,list(shape))
            except ValueError:rejected_metadata+=1
            else:raise ValueError('Wrong metadata byte-order declaration accepted')
    return dict(status='PASS',transport_fixture_cases=8,rejected_transport_cases=rejected,
        metadata_schema_fixture_cases=accepted_metadata+rejected_metadata,accepted_metadata_schema_cases=accepted_metadata,
        rejected_metadata_schema_cases=rejected_metadata,
        reconstruction=M.fixtures(),actual_production_inputs_read=0,network_calls=0,model_calls=0,checkpoint_reads=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--output-dir',type=Path,required=True)
    for name in ('metadata','raw','descriptor'):p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-sha256')
    args=p.parse_args();require(HERE in args.output_dir.resolve().parents and not args.output_dir.exists(),'Fresh owned output directory required')
    args.output_dir.mkdir()
    try:result=self_test(args.output_dir)if args.self_test else execute(args)
    except BaseException as error:
        write(args.output_dir/'failure.json',dict(status='STOP',failure_type=type(error).__name__,reason=str(error),
            partial_verified_input_pins=list(PINS.values()),network_calls=0,model_calls=0));raise
    write(args.output_dir/'verification.json',result)
    print(json.dumps(dict(status=result['status'],verification=record(args.output_dir/'verification.json'))))


if __name__=='__main__':main()
