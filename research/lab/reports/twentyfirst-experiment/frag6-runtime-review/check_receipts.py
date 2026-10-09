"""Read-only verification of saved receipts/hashes; no request or binary decode."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SOURCE=PHASE/'frag6-stage-a'
RUN=SOURCE/'run-v1'
seen={}


def pin(path):
    path=Path(path).resolve()
    value=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    seen[str(path)]=value
    return value


def verify(record):
    actual=pin(record['path'])
    assert actual['sha256']==record['sha256']
    assert actual['bytes']==record.get('bytes',record.get('size_bytes'))
    return Path(record['path'])


def read(path):
    pin(path);return json.loads(Path(path).read_text())


def main():
    plan_path=SOURCE/'execution-plan.json'
    plan=read(plan_path)
    assert pin(plan_path)['sha256']=='ae44baaed05f7faf85ee7fa592b88cc9d2931cf3a76667a914c2fa4b738bedeb'
    for record in plan['code']:verify(record)
    verify(plan['proposal']);verify(plan['synthetic_fixtures'])
    go=read(PHASE/'FRAG6_STAGE_A_GO.json')
    assert go['status']=='GO' and go['stage']=='A_headers' and go['allow_live_requests'] is True
    assert go['plan_sha256']==pin(plan_path)['sha256']
    verify(go['plan']);verify(go['technical_review'])
    start=read(RUN/'controller-start.json');controller=read(RUN/'controller-result.json')
    worker=read(verify(controller['worker_result']));result=read(verify(worker['result']))
    ledger=read(verify(result['request_ledger']))
    assert controller['status']==worker['status']==result['status']=='PASS'
    assert controller['worker_returncode']==0 and worker['source_pins_reverified'] is True
    for x in (start,worker):assert verify(x['root_go'])==PHASE/'FRAG6_STAGE_A_GO.json'
    for x in (start,result,ledger):assert x['plan']==pin(plan_path)
    for x in (start,worker,ledger):assert x['limits']==plan['limits']
    limits=plan['limits'];objects={o['url']:o for o in plan['objects']}
    assert len(ledger['requests'])==result['http_attempts']==10<=limits['maximum_attempts']
    assert ledger['reserved_response_body_bytes']==result['reserved_response_body_bytes']==1514
    assert ledger['retained_response_body_bytes']==result['retained_response_body_bytes']==1504
    assert ledger['automatic_retries']==ledger['redirects_followed']==0
    assert ledger['verified_tls'] is True
    expected=[(plan['objects'][0]['url'],'ppm_prefix',[0,1023])]
    expected.extend((o['url'],'tiff_header',[0,7]) for o in plan['objects'][1:])
    for tiff in result['tiffs']:
        offset=tiff['header']['ifd_offset'];length=tiff['count']['table_bytes']
        expected.extend([(tiff['object']['url'],'ifd_count',[offset,offset+1]),
                         (tiff['object']['url'],'ifd_table',[offset,offset+length-1])])
    assert [(r['url'],r['kind'],r['range_inclusive']) for r in ledger['requests']]==expected
    for r in ledger['requests']:
        obj=objects[r['url']];a,b=r['range_inclusive'];n=b-a+1
        assert r['success'] is True and r['method']=='GET' and r['http_status']==206
        assert r['request_headers']['Range']==f'bytes={a}-{b}'
        assert r['request_headers']['If-Match']==obj['etag']
        assert r['request_headers']['Accept-Encoding']=='identity'
        assert r['reserved_response_body_bytes']==n+1 and r['retained_body_bytes']==n
        verify(r['file']);assert r['file']['sha256']==r['body_sha256'] and r['file']['bytes']==n
        headers={}
        for key,value in r['response_headers']:headers.setdefault(key.lower(),[]).append(value)
        for key in ('content-length','content-range','content-type','content-encoding','etag','transfer-encoding','last-modified'):
            assert len(headers.get(key,[]))<=1
        assert headers['content-length']==[str(n)]
        assert headers['content-range']==[f'bytes {a}-{b}/{obj["total_bytes"]}']
        assert headers['etag']==[obj['etag']] and 'transfer-encoding' not in headers
        assert headers.get('content-encoding',['identity'])==['identity']
        assert r['elapsed_seconds']<=limits['per_request_wall_seconds'] and r['response_header_bytes']<=16384
    assert sum(r['reserved_response_body_bytes'] for r in ledger['requests'])==1514<=202301
    assert sum(r['retained_body_bytes'] for r in ledger['requests'])==1504
    ppm=result['ppm'];p=ppm['parsed'];verify(ppm['prefix'])
    assert ppm['object']==plan['objects'][0]
    assert (p['width'],p['height'],p['data_offset'],p['record_bytes'])==(6205,8853,73,48)
    assert p['data_offset']+p['width']*p['height']*48==ppm['object']['total_bytes']
    assert hashlib.sha256(p['header_text'].encode()).hexdigest()==p['header_sha256']
    assert p['numerical_payload_decoded'] is False and p['numerical_byte_order'] is None
    assert p['source_volume_binding'] is None and p['plane_offset_rule'] is None
    assert result['canvas_shape_yx']==[8853,6205]
    assert [t['layer_index'] for t in result['tiffs']]==[0,32,64]
    for t in result['tiffs']:
        for key in ('header_file','count_file','ifd_file'):verify(t[key])
        q=t['parsed'];assert t['object']==objects[t['object']['url']]
        assert q['width']==p['width'] and q['height']==p['height'] and q['dtype']=='uint16'
        assert q['byte_order']==t['header']['byte_order']=='little'
        assert q['strip_offset']==8 and q['strip_bytes']==p['width']*p['height']*2==109865730
        assert q['strip_offset']+q['strip_bytes']==q['ifd_offset']==109865738
        assert q['ifd_entries']==t['count']['entries']==12 and q['ifd_bytes']==t['count']['table_bytes']==150
        assert q['ifd_offset']+q['ifd_bytes']<=t['object']['total_bytes']
        assert q['rows_per_strip']==8853 and q['orientation']==1
        assert q['description'] is None and q['description_range'] is None and q['image_pixels_decoded'] is False
        assert q['baseline_defaults_applied']=={'266':1,'274':1,'317':1}
        tags={x['tag']:x for x in q['tags']};assert len(tags)==12 and 270 not in tags
        software=tags[305]
        assert software['payload_offset']==109865888 and software['payload_bytes']==27 and software['payload_acquired'] is False
        assert software['payload_offset']+software['payload_bytes']==t['object']['total_bytes']
        assert all(v is None for v in q['unknown_render_fields'].values())
    assert all(v is None for v in result['unknown_render_fields'].values())
    for key in ('numerical_ppm_records_decoded','image_pixels_decoded','model_calls','gpu_spend_gbp'):assert result[key]==0
    assert result['automatic_stage_advance'] is False
    assert controller['elapsed_seconds']<=1800
    assert max(controller['maximum_observed_worker_rss_bytes'],result['resources']['peak_rss_bytes'])<=268435456
    disk=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    assert disk<=33554432 and controller['artifact_bytes_before_final_receipt']<=disk
    receipt=dict(status='PASS',scope='Saved receipt, source pin, body hash, range and storage-field consistency only; no binary format replay or numerical payload decoding.',
        plan=pin(plan_path),root_go=pin(PHASE/'FRAG6_STAGE_A_GO.json'),
        structural_contract=pin(RUN/'acquisition/STRUCTURAL_CONTRACT.json'),ledger=pin(RUN/'acquisition/request-ledger.json'),
        controller=pin(RUN/'controller-result.json'),worker=pin(RUN/'acquisition/worker-result.json'),
        input_pins=list(seen.values()),checker=pin(Path(__file__).resolve()),
        observed_http_attempts=10,reserved_response_body_bytes=1514,retained_response_body_bytes=1504,
        canvas_shape_yx=[8853,6205],description_absent_all_three=True,
        software_tag_payload_unacquired=True,render_geometry_unresolved=True,
        controller_wall_seconds=controller['elapsed_seconds'],maximum_reported_rss_bytes=max(controller['maximum_observed_worker_rss_bytes'],result['resources']['peak_rss_bytes']),
        run_artifact_bytes=disk,new_requests=0,new_binary_decodes=0,scientific_arrays_read=0,model_calls=0,
        limitations=['Consistency check does not independently replay network execution or process limits.',
                     'No PPM numeric byte order, native source UUID, offsets, signed normals, geometry, labels or image accuracy is established.'])
    with (HERE/'verification.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='PASS',verification=pin(HERE/'verification.json'),requests=10,reserved_bytes=1514,body_bytes=1504)))


if __name__=='__main__':main()
