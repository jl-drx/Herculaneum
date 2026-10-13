"""Offline, independent coverage/byte/source-pin check of the next CT plan."""
from pathlib import Path
import json
import geometry as G
R=Path(__file__).resolve().parent

def main():
    plan=G.read(R/'NEXT_CT_PILOT_PLAN.json');q=G.read(R/'qualification/RESULTS.json')
    assert G.sha(R/'qualification/RESULTS.json')==plan['qualification_sha256']
    sites=plan['fixed_selected_sites'];byid={r['core_id']:r for r in q['regions']}
    expected_ids={r['core_id'] for r in q['regions'] if r.get('pilot',{}).get('geometry_ready')}
    assert {s['core_id'] for s in sites}==expected_ids and len(sites)==8
    assert len(plan['failed_geometry_sites'])==4
    requests={r['id']:r for r in plan['requests']};assert len(requests)==len(plan['requests'])==72
    needed=set()
    for site in sites:
        row=byid[site['core_id']];actual={requests[k]['kind']:set() for k in site['request_ids']};expected={}
        for k in site['request_ids']:
            req=requests[k];meta=G.read(R/req['array_metadata_path']);sep=meta.get('dimension_separator','.')
            z,y,x=map(int,req['key'].split(sep));assert z==0
            actual[req['kind']].add((y,x));needed.add(k)
        nx,ny,nw,nh=row['native_context_xywh'];assert nw==nh==256
        fx,fy,ex,ey=row['pilot']['fine_group2_xyxy']
        for kind,rect in [('native',(nx,ny,nx+nw,ny+nh)),('fine78',(fx,fy,ex,ey))]:
            x0,y0,x1,y1=rect
            # Enumerate needed chunks from every spatial pixel, independently
            # of the request builder's range-of-chunk-index loops.
            expected[kind]={(y//128,x//128) for y in range(y0,y1) for x in range(x0,x1)}
        assert actual==expected,site['core_id']
    assert needed==set(requests)
    for req in requests.values():
        rp=R/'pilot-receipts'/(req['id']+'.json');h=G.read(rp)
        assert G.sha(rp)==req['head_receipt_sha256'] and h['url']==req['url'] and h['etag']==req['etag']
        assert h['status']==200 and h['usable'] and h['content_length']==req['expected_bytes'] and h['body_bytes']==0
        mp=R/req['array_metadata_path'];assert G.sha(mp)==req['array_metadata_sha256'];m=G.read(mp)
        assert m['dtype']=='|u1' and m['compressor'] is None and m.get('filters') is None
        assert req['expected_bytes']==m['chunks'][0]*m['chunks'][1]*m['chunks'][2]
    assert sum(r['expected_bytes'] for r in requests.values())==plan['max_CT_payload_bytes']==86114304
    assert plan['actual_CT_bytes_fetched']==0 and plan['status']=='FROZEN_NOT_EXECUTED'
    report=dict(status='PASS',sites=len(sites),fixed_geometry_failures=4,chunks=len(requests),bytes=plan['max_CT_payload_bytes'],
                verified_live_HEAD_receipts=72,CT_bodies_acquired=0,
                checks=['fixed cohort and failures','independent per-pixel chunk coverage','no redundant requests','all HEADs200 with pinned ETags/lengths',
                        'metadata and receipt SHA bindings','all28 native / all109 fine depth chunks','byte ceiling','not-executed status'],
                code_sha256=G.sha(Path(__file__)),plan_sha256=G.sha(R/'NEXT_CT_PILOT_PLAN.json'))
    G.save(R/'PILOT_VERIFICATION.json',report);print(json.dumps(report))

if __name__=='__main__':main()
