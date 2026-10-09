"""Prepare metadata-first production plans; root creates ready stages after source GO."""
import argparse
import json
from pathlib import Path
from production_common import *


def fixed_inputs():
    base=ROOT.parent
    paths={'design':base/'PRODUCTION_INPUT_DESIGN.md','schema':base/'production-pooling/RAW_PROVENANCE_SCHEMA.json',
           'large_result':base/'retrieval-large/run/result.json',
           'header_result':base/'retrieval-v2/header-run/result.json',
           'serving_result':base/'retrieval-v2/serving-run/result.json',
           'benchmark':base.parent/'fifteenth-experiment/BENCHMARK_CONTRACT.json',
           'source':base.parent/'fifteenth-experiment/SOURCE_CONTRACT.json'}
    hashes={'design':DESIGN_SHA,'schema':SCHEMA_SHA,'large_result':LARGE_RESULT_SHA,
            'header_result':HEADER_RESULT_SHA,'serving_result':SERVING_RESULT_SHA,
            'benchmark':BENCHMARK_SHA,'source':SOURCE_SHA}
    refs={k:record(p)for k,p in paths.items()}
    require(all(refs[k]['sha256']==hashes[k]for k in refs),'Frozen prerequisite/design/schema changed')
    values={k:json.loads(verify(v).read_text())for k,v in refs.items()if k!='design'}
    h,s,l=values['header_result'],values['serving_result'],values['large_result']
    require(h['status']==s['status']==l['status']=='PASS' and h['stage']=='headers'
            and s['stage']=='serving' and l['stage']=='large-serving','All three completed probe stages required')
    require(h['sources']==s['sources']==l['sources'] and h['tiff_contracts']==s['tiff_contracts']==l['tiff_contracts']
            and len(h['sources'])==3,'Probe source/layout identity changed')
    for i,(small,large)in enumerate(zip(s['serving_comparisons'],l['serving_comparisons'],strict=True)):
        require(len(small['rows'])==8 and len(large['rows'])==256
                and all(r['bitwise_equal']is True for r in small['rows']+large['rows'])
                and small['fragment']==large['fragment']==h['sources'][i]['fragment'],'Every observed comparison must pass')
    require(len(s['serving_comparisons'])==len(l['serving_comparisons'])==3,'All three source comparisons required')
    sources=h['sources']
    cases=values['benchmark']['selected_contexts']
    expected=[('frag1-context1',[2304,1152,1152,1152]),('frag1-context2',[0,6912,1152,1152]),
              ('frag2-context1',[1152,1152,1152,1152]),('frag2-context2',[8064,12672,1152,1152]),
              ('frag3-context1',[2304,1152,1152,1152]),('frag3-context2',[3456,5760,1152,1152])]
    require([(c['case_id'],c['bbox_source_xywh'])for c in cases]==expected,'Six frozen context boxes/order changed')
    objects=[]
    for source in sources:
        index=verify(source['source_index']).read_text()
        for z in range(7,58):
            require(f'href="{z:02d}.tif"'in index,'Required plane not in frozen source inventory')
            objects.append(dict(object_id=f"{source['fragment']}-z{z:02d}",fragment=source['fragment'],
                                plane_index=z,url=source['url'][:-6]+f'{z:02d}.tif',shape_yx=source['shape_yx']))
    carried=[dict(c,plane_index=7,object_id=f"{c['fragment']}-z07")for c in h['tiff_contracts']]
    records=list(refs.values())
    # Preserve prior successful/failure traffic separately; no decoding of raw bodies here.
    for result in (h,s,l):
        records += [record(verify(r))for r in nested_records(result)]
        prior=json.loads(verify(result['plan']).read_text())
        records += [record(verify(r))for r in prior['input_pins']]
    records += [record(verify(r))for r in nested_records(values['source'])]
    return dict(references=refs,sources=sources,cases=cases,objects=objects,carried_contracts=carried,
                prior_traffic={'original_prefix':{'attempts':1,'retained_bytes':8},
                'v2_metadata':{'attempts':h['http_attempts'],'retained_bytes':h['retained_body_bytes']},
                'small_serving':{'attempts':s['http_attempts'],'retained_bytes':s['retained_body_bytes']},
                'large_serving':{'attempts':l['http_attempts'],'retained_bytes':l['retained_body_bytes']},
                'scope':'Probe traffic retained as prerequisites; production fetches complete selected inputs separately.'}),records


def pixel_requests(cases,objects,contracts):
    by_object={c['object_id']:c for c in contracts}
    require(len(by_object)==len(contracts)==153,'All153 unique TIFF contracts required')
    output=[]
    for obj in objects:
        c=by_object[obj['object_id']]
        require(c['url']==obj['url'] and [c['height'],c['width']]==obj['shape_yx'],'Metadata object identity/shape changed')
        for case in cases:
            if case['case_id'].split('-')[0]!=obj['fragment']:
                continue
            x,y,w,h=case['bbox_source_xywh']
            require(w==h==1152 and 0<=x and x+w<=c['width'] and 0<=y and y+h<=c['height'],'Fixed crop exceeds TIFF')
            offset=0
            for b,n in enumerate([256,256,256,256,128]):
                spans=[]
                for row in range(y+offset,y+offset+n):
                    start=c['strip_offset']+2*(row*c['width']+x)
                    require(c['strip_offset']<=start and start+2304<=c['strip_offset']+c['strip_bytes']
                            and start+2304<=c['object_total_bytes'],'Pixel span exceeds declared strip/object')
                    spans.append([start,start+2303])
                output.append(dict(request_id=f"p-{case['case_id']}-z{obj['plane_index']:02d}-b{b}",
                                   object_id=obj['object_id'],case_id=case['case_id'],fragment=obj['fragment'],
                                   plane_index=obj['plane_index'],url=obj['url'],mode='multipart',ranges=spans,
                                   native_row_start=y+offset,row_count=n,native_x_start=x,native_crop_width=w,
                                   pin={'etag':c['etag'],'object_total_bytes':c['object_total_bytes']},
                                   reserved_cap_bytes=2304*n+768*n+1024))
                offset+=n
            require(offset==1152,'Fixed row coverage changed')
    require(len(output)==1530 and len({r['request_id']for r in output})==1530,'Exact1530 request IDs required')
    require(sum(2304*r['row_count']for r in output)==EXPECTED_USEFUL_BYTES
            and sum(r['reserved_cap_bytes']for r in output)==EXPECTED_RESERVED_BYTES,'Pixel byte accounting changed')
    return output


def metadata_inventory(objects):
    return [dict(request_id=f"m-{o['object_id']}-{phase}",object_id=o['object_id'],phase=phase)
            for o in objects if o['plane_index']!=7 for phase in ('header','count','ifd')]


def source_go(reviews,code):
    bound=False
    for r in reviews:
        data=json.loads(verify(r).read_text())
        if data.get('status')=='GO':
            pins=list(nested_records(data))
            require(all(any(Path(p['path']).resolve()==Path(c['path']) and p['sha256']==c['sha256']for p in pins)
                        for c in code),'Source GO must bind all six exact files')
            bound=True
    require(bound,'Source-bound technical GO required')


def load_plan(path,ready=True):
    plan=json.loads(Path(path).read_text())
    require(plan['schema_version']==1 and plan['stage']in ('metadata','pixels')and(not ready or plan['ready']is True),
            'Production plan not ready')
    require(plan['limits']==LIMITS and plan['python']==sys.version.split()[0]=='3.12.14','Pinned runtime/limits changed')
    for r in plan['input_pins']:
        verify(r)
    code=[record(ROOT/n)for n in CODE_NAMES]
    require(plan['code']==code,'Executed source changed')
    fixed,_=fixed_inputs()
    require(plan['fixed_inputs']==fixed,'Frozen scientific inventory/prerequisites changed')
    if plan['stage']=='metadata':
        require(plan['requests']==metadata_inventory(fixed['objects']),'Metadata logical request inventory changed')
    else:
        metadata=json.loads(verify(plan['metadata_result']).read_text())
        require(metadata['status']=='PASS'and metadata['stage']=='metadata'and len(metadata['tiff_contracts'])==153,
                'All metadata must pass before pixels')
        require(metadata['fixed_inputs']==fixed and metadata['plan']['sha256']==plan['metadata_plan']['sha256'],
                'Metadata contract/ready plan linkage changed')
        require(plan['production_started_epoch_seconds']==metadata['production_started_epoch_seconds'],
                'Shared production start timestamp changed')
        require(plan['tiff_contracts']==metadata['tiff_contracts']and plan['requests']==pixel_requests(
                fixed['cases'],fixed['objects'],metadata['tiff_contracts']),'Pixel schedule/layout changed')
    if ready:
        source_go(plan['reviews'],code)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['metadata','pixels'])
    parser.add_argument('--metadata-result')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--review',action='append',default=[])
    parser.add_argument('--ready',action='store_true')
    args=parser.parse_args()
    root=Path(args.output_dir).resolve()
    require(not root.exists(),'Fresh plan directory required')
    fixed,records=fixed_inputs()
    code,reviews=[record(ROOT/n)for n in CODE_NAMES],[record(p)for p in args.review]
    if args.ready:
        source_go(reviews,code)
    records += code+reviews
    for review in reviews:
        records += [record(verify(r))for r in nested_records(json.loads(verify(review).read_text()))]
    plan=dict(schema_version=1,stage=args.stage,ready=args.ready,python='3.12.14',limits=LIMITS,
              fixed_inputs=fixed,code=code,reviews=reviews,numeric_pixel_decodes=0,model_calls=0,
              stage_barrier='All153 layouts before separately frozen ready pixel plan; no automatic stage continuation.')
    if args.stage=='metadata':
        plan['requests']=metadata_inventory(fixed['objects'])
    else:
        require(args.metadata_result,'Completed metadata result required')
        mr=record(args.metadata_result)
        m=json.loads(verify(mr).read_text())
        require(m['status']=='PASS'and m['stage']=='metadata'and m['fixed_inputs']==fixed,'Metadata stage incomplete/changed')
        plan.update(metadata_result=mr,metadata_plan=m['plan'],tiff_contracts=m['tiff_contracts'],
                    requests=pixel_requests(fixed['cases'],fixed['objects'],m['tiff_contracts']),
                    useful_sample_bytes=EXPECTED_USEFUL_BYTES,maximum_reserved_body_bytes=EXPECTED_RESERVED_BYTES,
                    production_started_epoch_seconds=m['production_started_epoch_seconds'])
        records += [mr]+[record(verify(r))for r in nested_records(m)]
        metadata_plan=json.loads(verify(m['plan']).read_text())
        records += [record(verify(r))for r in metadata_plan['input_pins']]
    unique={}
    for r in records:
        require(r['path']not in unique or unique[r['path']]==r,'Conflicting input pins')
        unique[r['path']]=r
    plan['input_pins']=list(unique.values())
    root.mkdir(parents=True)
    path=root/'acquisition-plan.json';save(path,plan)
    load_plan(path,ready=args.ready)
    print(json.dumps({'plan':record(path),'stage':args.stage,'ready':args.ready}))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__,
                          **({'stop_reason':str(error)}if isinstance(error,ContractError)else {})}))
        raise SystemExit(1)
