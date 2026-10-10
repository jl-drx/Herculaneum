"""Freeze bounded regional IR/prefix stage, then separately derive fixed sparse samples."""
import argparse
import json
import sys
from pathlib import Path
from acquisition_common import *

BENCHMARK_SHA='a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735'
SOURCE_SHA='53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890'
DESIGN_SHA='08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1'
FORMAT_SOURCE_SHA='679866b52e98a64947f17970b08c7fcc8b0eda4eebc71593f076c536edbc55b9'
AMENDMENT_SHA='ed3096856274505990fb9b4069ce6fdbb657eaeb93ae52b6bfae2144de6d9f9e'
FORMAT_EVIDENCE_SHA='be6284825f10a6b4bf807f419e4e0a383dbcea9ad7346b4b4042e684d2d3b8eb'
CLARIFICATIONS_SHA='22ed3e0749d56bd61113b6a2cd74f6ae7373082a34a1c090352ccba99eb2c643'
ZERO_MAPPING_SHA='114e3791f2a7549b2e1d82192ef454865224f3ea9506550a0513f4eaec591d1a'


def format_source():
    ref=record(ROOT.parent/'source-review/historical-inkid-ppm.py')
    require(ref['sha256']==FORMAT_SOURCE_SHA and ref['bytes']==7494,'Verified historical PPM format source required')
    raw=verify(ref).read_bytes()
    require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='05ef3b591fc593df6fbc5531a429c9ebd870d62c',
            'Historical PPM source Git blob changed')
    return ref


def inventory():
    reports=ROOT.parent.parent
    source=record(reports/'fifteenth-experiment/SOURCE_CONTRACT.json')
    benchmark=record(reports/'fifteenth-experiment/BENCHMARK_CONTRACT.json')
    require(source['sha256']==SOURCE_SHA and benchmark['sha256']==BENCHMARK_SHA,'Immutable source/benchmark changed')
    s=json.loads(verify(source).read_text());b=json.loads(verify(benchmark).read_text())
    require([(c['case_id'],c['bbox_source_xywh'])for c in b['selected_contexts']]==EXPECTED_CASES,'Fixed six cases changed')
    amendment=record(ROOT.parent/'PPM_PREDECODE_AMENDMENT.md')
    evidence=record(ROOT.parent/'source-review/PPM_SERIALIZATION_EVIDENCE.json')
    clarifications=record(ROOT.parent/'PREEXECUTION_CLARIFICATIONS.md')
    zero_mapping=record(ROOT.parent/'PPM_ZERO_MAPPING_CLARIFICATION.md')
    require(amendment['sha256']==AMENDMENT_SHA and evidence['sha256']==FORMAT_EVIDENCE_SHA,'Frozen PPM amendment/evidence changed')
    require(clarifications['sha256']==CLARIFICATIONS_SHA,'Frozen pre-execution clarifications changed')
    require(zero_mapping['sha256']==ZERO_MAPPING_SHA,'Frozen unmapped-point clarification changed')
    objects=[];inputs=[source,benchmark,format_source(),amendment,evidence,clarifications,zero_mapping]
    for candidate in s['eligible_candidates']:
        frag=candidate['legacy_alias'].lower()
        require(frag in ('frag1','frag2','frag3'),'Unexpected eligible fragment')
        listing=record(reports/f'fifteenth-experiment/source-audit/{frag}-exposed-index.html')
        original=next(r for r in s['network_source_files']if Path(r['path']).resolve()==Path(listing['path']))
        verify(original)
        require(original['sha256']==listing['sha256'],'Source listing changed')
        text=verify(listing).read_text();inputs.append(listing)
        for filename in ('ir.png','result.ppm','surface_volume.ppm'):
            require(f'href="{filename}"'in text,'Object absent from retained official listing')
            objects.append(dict(object_id=f'{frag}-{filename[:-4]}',fragment=frag,filename=filename,
                url=candidate['exposed_surface_url']+filename,shape_yx=candidate['shape_yx'],listing=listing))
    require(len(objects)==9,'Exact9 object inventory required')
    return dict(objects=objects,cases=[dict(case_id=c['case_id'],bbox_source_xywh=c['bbox_source_xywh'])for c in b['selected_contexts']]),inputs


def header_requests(fixed):
    requests=[]
    for obj in fixed['objects']:
        full=obj['filename']=='ir.png'
        requests.append(dict(request_id=('ir-'if full else'prefix-')+obj['object_id'],object_id=obj['object_id'],
            url=obj['url'],mode='full'if full else'single',ranges=[]if full else[[0,1023]],reserved_cap_bytes=48*1024**2 if full else 1025))
    return requests


def sample_requests(fixed,contracts):
    by_object={c['object_id']:c for c in contracts};output=[]
    require(len(by_object)==len(contracts)==6,'Six distinct validated PPM contracts required')
    for case in fixed['cases']:
        frag=case['case_id'].split('-')[0];x,y,w,h=case['bbox_source_xywh']
        for variant in ('result','surface_volume'):
            c=by_object[f'{frag}-{variant}']
            points=[];spans=[]
            for iy in LATTICE:
                for ix in LATTICE:
                    cy=y+1+3*iy;cx=x+1+3*ix
                    require(0<=cy-1<cy+1<c['height']and 0<=cx-1<cx+1<c['width'],'Fixed lattice exceeds PPM canvas')
                    point=dict(coarse_yx=[iy,ix],native_center_yx=[cy,cx],row_ranges=[])
                    for dy in (-1,0,1):
                        offset=c['data_offset']+48*((cy+dy)*c['width']+cx-1)
                        span=[offset,offset+3*48-1]
                        require(span[1]<c['object_total_bytes'],'Fixed sample exceeds PPM object')
                        spans.append(span);point['row_ranges'].append(span)
                    points.append(point)
            cap=sum(b-a+1 for a,b in spans)+768*len(spans)+1024
            require(len(points)==81 and len(spans)==243 and cap==222640,'Fixed lattice request accounting changed')
            output.append(dict(request_id=f"sample-{case['case_id']}-{variant}",case_id=case['case_id'],
                object_id=c['object_id'],variant=variant,url=c['url'],mode='multipart',ranges=spans,
                pin=dict(etag=c['etag'],object_total_bytes=c['object_total_bytes']),
                points=points,reserved_cap_bytes=cap))
    require(len(output)==12 and sum(r['reserved_cap_bytes']for r in output)==2671680,'Exact12 sample inventory required')
    return output


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    require(plan['schema_version']==1 and plan['stage']in STAGE_LIMITS,'Unsupported acquisition plan')
    require(plan['limits']==STAGE_LIMITS[plan['stage']],'Frozen HTTP caps changed')
    require(plan['python']==sys.version.split()[0] and plan['design']['sha256']==DESIGN_SHA,'Frozen runtime/design changed')
    for ref in plan['input_pins']:verify(ref)
    require(plan['code']==[record(ROOT/n)for n in CODE_NAMES],'Frozen acquisition source changed')
    fixed,_=inventory();require(fixed==plan['fixed_inventory'],'Fixed source/case inventory changed')
    if plan['stage']=='headers-ir':expected=header_requests(fixed)
    else:
        h=json.loads(verify(plan['headers_result']).read_text())
        require(h['status']=='PASS'and h['stage']=='headers-ir','Successful prefix/IR stage required')
        contracts=validate_headers(fixed,h,plan['format_evidence'])
        require(contracts==plan['ppm_contracts'],'Derived PPM contracts changed')
        expected=sample_requests(fixed,contracts)
    require(plan['requests']==expected,'Frozen request inventory changed')
    return plan


def validate_headers(fixed, result, evidence):
    source=format_source();audit=json.loads(verify(evidence).read_text())
    require(evidence['sha256']==FORMAT_EVIDENCE_SHA,'Frozen authoritative serialization evidence required')
    require(any(r['sha256']==source['sha256'] and verify(r)==Path(source['path']) for r in nested_records(audit)),
            'Serialization evidence must bind verified historical reader')
    require(result['fixed_inventory']==fixed,'Prefix stage source inventory changed')
    require(len(result['objects'])==9,'All9 prior objects required')
    by_object={o['object_id']:o for o in result['objects']}
    require(len(by_object)==9,'Distinct prior objects required')
    contracts=[]
    for obj in fixed['objects']:
        if obj['filename']=='ir.png':continue
        observed=by_object[obj['object_id']]
        require(observed['url']==obj['url'],'Prefix URL identity changed')
        contract=parse_ppm_prefix(verify(observed['file']).read_bytes(),observed['object_total_bytes'],obj['shape_yx'])
        contracts.append(dict(contract,object_id=obj['object_id'],url=obj['url'],etag=observed['etag'],
            object_total_bytes=observed['object_total_bytes'],prefix=observed['file'],format_source=source))
    return contracts


def parse_ppm_prefix(prefix,total,shape_yx):
    """Interpret bounded ASCII schema only; never decode one binary double."""
    require(len(prefix)==1024,'Exact1024-byte PPM prefix required')
    end=prefix.find(b'<>\n')
    require(0<=end<1022 and (end==0 or prefix[end-1:end]==b'\n'),'PPM LF terminator absent from fixed prefix')
    raw=prefix[:end]
    require(b'\r'not in raw and b'\x00'not in raw,'Unexpected PPM header framing')
    try:text=raw.decode('utf-8')
    except UnicodeDecodeError:raise ContractError('Invalid UTF-8 PPM header')from None
    fields={}
    for line in text.splitlines():
        if line.startswith('#'):continue
        match=re.fullmatch(r'(width|height|dim|ordered|type|version): ([^\r\n]+)',line)
        require(match is not None,'Unknown/malformed PPM header field')
        key,value=match.groups();require(key not in fields,'Duplicate PPM header field');fields[key]=value
    require(set(fields)=={'width','height','dim','ordered','type','version'},'Incomplete PPM schema')
    require(all(re.fullmatch(r'[1-9][0-9]*',fields[k])for k in ('width','height','dim')),'Invalid PPM dimensions')
    w,h,dim=(int(fields[k])for k in ('width','height','dim'))
    require([h,w]==shape_yx and dim==6 and fields['ordered']=='true'and fields['type']=='double'
            and fields['version']in ('1','1.0'),'Unsupported PPM canvas/layout/version')
    offset=end+3;require(total==offset+h*w*48,'PPM total extent differs from48-byte ordered records')
    return dict(width=w,height=h,dim=dim,type='double',ordered=True,version=fields['version'],
        data_offset=offset,record_bytes=48,header_text=text+'<>\n',header_sha256=hashlib.sha256(prefix[:offset]).hexdigest(),
        record_order='row-major y,x,component',components=['x','y','z','nx','ny','nz'],
        byte_order='native-unspecified',dtype=None,numerical_interpretation_authorized=False,
        byte_order_limitation='Verified historical source uses struct.pack/unpack("d") with native byte order; producer platform is not authenticated.',
        binary_numeric_decodes=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['headers-ir','samples']);p.add_argument('--design',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--headers-result');p.add_argument('--format-evidence')
    args=p.parse_args();out=Path(args.output_dir).resolve()
    require(not out.exists(),'Fresh plan directory required')
    fixed,inputs=inventory();code=[record(ROOT/n)for n in CODE_NAMES];design=record(args.design)
    require(design['sha256']==DESIGN_SHA,'Root frozen research design required')
    plan=dict(schema_version=1,stage=args.stage,design=design,code=code,fixed_inventory=fixed,
        ppm_predecode_amendment=record(ROOT.parent/'PPM_PREDECODE_AMENDMENT.md'),
        preexecution_clarifications=record(ROOT.parent/'PREEXECUTION_CLARIFICATIONS.md'),
        ppm_zero_mapping_clarification=record(ROOT.parent/'PPM_ZERO_MAPPING_CLARIFICATION.md'),
        format_evidence=record(ROOT.parent/'source-review/PPM_SERIALIZATION_EVIDENCE.json'),
        limits=STAGE_LIMITS[args.stage],python=sys.version.split()[0],numeric_image_decodes=0,numeric_ppm_decodes=0,
        scientific_analysis_authorized=False,model_calls=0,raw_CT_requests=0,
        stage_barrier='Root executes frozen plan; samples require completed prefix stage and separate frozen sample plan.')
    inputs+=code+[design]
    if args.stage=='headers-ir':plan['requests']=header_requests(fixed)
    else:
        require(args.headers_result and args.format_evidence,'Headers result and authoritative format evidence required')
        hr=record(args.headers_result);fe=record(args.format_evidence);h=json.loads(verify(hr).read_text())
        require(h['status']=='PASS'and h['stage']=='headers-ir','Completed headers stage required')
        contracts=validate_headers(fixed,h,fe)
        plan.update(headers_result=hr,format_evidence=fe,ppm_contracts=contracts,requests=sample_requests(fixed,contracts))
        inputs += [hr,fe]+list(nested_records(h))+list(nested_records(json.loads(verify(fe).read_text())))
    unique={}
    for ref in inputs:
        normalized=record(verify(ref))
        require(normalized['path']not in unique or unique[normalized['path']]==normalized,'Conflicting frozen input pins')
        unique[normalized['path']]=normalized
    plan['input_pins']=list(unique.values())
    out.mkdir(parents=True);path=out/'acquisition-plan.json';save(path,plan);load_plan(path)
    print(json.dumps(dict(plan=record(path),stage=args.stage,requests=len(plan['requests']))))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='STOP',failure_type=type(error).__name__,**({'reason':str(error)}if isinstance(error,ContractError)else{}))))
        raise SystemExit(1)
