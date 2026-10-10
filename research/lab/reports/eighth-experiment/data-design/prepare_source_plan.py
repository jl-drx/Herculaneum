"""Prepare exact source-read plan only after annotation selection; no network."""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'byte_length':p.stat().st_size}
def guard(records):
    for r in records:
        p=Path(r['path'])
        if p.stat().st_size!=r['byte_length'] or sha(p)!=r['sha256']:raise ValueError('Pinned input changed')

def main(selection_path,label_plan_path,target):
    selection_path=Path(selection_path).resolve();label_plan_path=Path(label_plan_path).resolve();target=Path(target).resolve()
    if target.exists():raise ValueError('Refuse overwrite of source proposal')
    s=json.loads(selection_path.read_text());labels=json.loads(label_plan_path.read_text())
    assert s['status']=='selection_complete_no_scans_or_inference'
    assert s['plan_record']==rec(label_plan_path)
    guard(labels['input_records']);guard([s['selection_code_record'],s['http_module_record'],s['label_request_ledger_record']])
    inputs=labels['input_records']+[
        rec(selection_path),rec(label_plan_path),s['selection_code_record'],s['http_module_record'],
        s['label_request_ledger_record'],rec(ROOT/'assemble_reserve_sources.py'),rec(Path(__file__))]
    crops=[];objects=[];metadata=[];seen_raw=set()
    for case in labels['case_order']:
        spec=labels['cases'][case];selected=s['cases'][case]['selected']
        assert len(selected)==2
        allowed={tuple(c['bbox_source_xywh']):c['rank'] for c in spec['candidates']}
        for a in selected:
            bbox=a['bbox_source_xywh'];x,y,w,h=bbox
            assert tuple(bbox) in allowed and a['rank']==allowed[tuple(bbox)] and w==h==384
            assert a['counts']['primary_positive']>=256 and a['counts']['primary_negative']>=256
            inputs+=list(a['arrays'].values())
            for source in a['source_objects']:
                p=selection_path.parent/'raw-labels'/source['local_file']
                if p not in seen_raw:
                    inputs.append({'path':str(p),'sha256':source['sha256'],'byte_length':source['byte_length']});seen_raw.add(p)
            chunk_keys=[]
            for yy in range(y//128,y//128+3):
                for xx in range(x//128,x//128+3):
                    key=case+f'/0/{yy}/{xx}';chunk_keys.append(key)
                    objects.append({'key':key,'case':case,'chunk_zyx':[0,yy,xx],
                        'url':spec['source_surface_url']+f'/2/0/{yy}/{xx}',
                        'expected_byte_length':109*128*128})
            crops.append({'crop_id':a['crop_id'],'case':case,'specimen':spec['specimen'],
                'rank':a['rank'],'evaluation_participation':'supervision_mask',
                'evaluation_participation_role':'supplied_supervision','bbox_source_xywh':bbox,'label_arrays':a['arrays'],
                'label_counts':a['counts'],'binary_foreground':a['binary_foreground'],
                'observed_annotation_values':a['observed_raw_values'],'source_chunk_keys':chunk_keys,
                'source_surface_url':spec['source_surface_url'],'public_segment':spec['public_segment'],
                'source_shape_zyx':spec['source_shape_zyx'],'shape_zyx':[21,384,384],
                'pixel_spacing_um':9.596,'source_depth_slice':[13,97],'model_layer_slice':[2,19]})
        if len(selected)==2:
            from select_reserve_labels import gap
            assert gap(selected[0]['bbox_source_xywh'],selected[1]['bbox_source_xywh'])>=128
            for a in selected:
                assert all(gap(a['bbox_source_xywh'],old['bbox_source_xywh'])>=128 for old in labels['old_used_contexts'])
        for record,suffix in zip(spec['source_metadata_records'],['/.zattrs','/2/.zarray']):
            metadata.append({'case':case,'url':spec['source_surface_url']+suffix,
                'expected_byte_length':record['byte_length'],'expected_sha256':record['sha256'],
                'role':'zattrs' if suffix=='/.zattrs' else 'level2-zarray'})
    unique={obj['key']:obj for obj in objects}
    if len(unique)!=len(objects):raise ValueError('Unexpected source-chunk reuse between nonoverlap contexts')
    guard(inputs)
    expected=sum(o['expected_byte_length'] for o in objects)+2*sum(o['expected_byte_length'] for o in metadata)
    max_requests=6*(len(objects)+2*len(metadata))
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'source_proposal_no_source_reads',
        'selection_record':rec(selection_path),'label_plan_record':rec(label_plan_path),
        'source_response_body_cap_bytes':40*1024*1024,'maximum_requests_per_object':6,
        'maximum_http_request_attempts':max_requests,'expected_final_response_body_bytes':expected,
        'maximum_redirect_body_bytes':(len(objects)+2*len(metadata))*5*4096,
        'source_chunk_count':len(objects),'scan_payload_bytes':sum(o['expected_byte_length'] for o in objects),
        'crops':crops,'objects':objects,'metadata_objects':metadata,'input_records':inputs,
        'http_module_record':rec(ROOT/'bounded_http.py'),
        'source_code_record':rec(ROOT/'assemble_reserve_sources.py'),
        'missing_source_policy':'Stop on any non-200 or missing object; no fill or automatic retry.',
        'depth_orientation':'Increasing published source Z; no fitted reversal, offsets or CT normal assertion.',
        'old_used_contexts':labels['old_used_contexts'],'minimum_context_gap_px':128,
        'evaluation_participation':'supervision_mask','geometry_validity':{'status':'not_acquired','mask_file':None},
        'claim_scope':labels['claim_scope']}
    assert expected+plan['maximum_redirect_body_bytes']<plan['source_response_body_cap_bytes']
    target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'path':str(target),'sha256':sha(target),'scan_chunks':len(objects),
        'scan_bytes':plan['scan_payload_bytes'],'expected_final_bodies':expected,'maximum_requests':max_requests}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--label-plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.selection,a.label_plan,a.output)
