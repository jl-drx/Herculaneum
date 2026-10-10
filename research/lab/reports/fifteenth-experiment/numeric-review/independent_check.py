"""Single label-only verification using independent banded integer support."""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[option]='1'

import numpy as np
from integral_support import (require,native_known,candidates,choose,context_arrays,fixtures as support_fixtures)
from png_decode import decode,structure,fixtures as png_fixtures
from review_io import Review,record,identity

HERE=Path(__file__).resolve().parent
DESIGN_SHA='9bf490e33e039ccacfdfbf99ab89b207d373ad1a08f0975afc9c0893f87d42e4'
SOURCE_SHA='53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890'
EXPECTED=[('Frag1','PHercParis2Fr47','Paris2','evaluation',[8181,6330]),
          ('Frag2','PHercParis2Fr143','Paris2','evaluation',[14830,9506]),
          ('Frag3','PHercParis1Fr34','Paris1','calibration',[7606,5249])]
RSS_CAP=3*1024**3
AS_CAP=12*1024**3
LIMITS=dict(http_attempts=6,maximum_response_bytes=524288,total_response_bytes=2097152,
    maximum_pixels=150000000,worker_wall_seconds=120,worker_cpu_seconds=360,threads=4,
    worker_peak_rss_bytes=3221225472,worker_address_space_bytes=12884901888,
    selection_wall_seconds=420,acquisition_wall_seconds=180,
    total_output_bytes=8388608,selection_output_bytes=6291456)


def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(value<=RSS_CAP,'Independent review RSS cap exceeded')
    return value


def published_candidate(row):
    return dict(y=row['y'],x=row['x'],native_inside_count=row['inside_count'],native_area=1152*1152,
        surface_fraction=row['inside_count']/(1152*1152),primary_positive=row['positive'],
        primary_negative=row['negative'],primary_count=row['primary_count'],
        coarse_surface_count=row['coarse_surface_count'],coarse_known_count=row['coarse_known_count'],eligible=row['eligible'])


def review_outputs(args,sources):
    started=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(AS_CAP,AS_CAP))
    review=Review()
    pins={name:record(getattr(args,name)) for name in ('plan','acquisition','panel')}
    for name,pin in pins.items():require(pin['sha256']==getattr(args,name+'_sha256'),'Root entry hash differs: '+name)
    plan=review.json(pins['plan']);source=review.json(plan['source_contract'])
    require(plan['schema_version']==1 and plan['ready'] is True and plan['design']['sha256']==DESIGN_SHA and
            plan['source_contract']['sha256']==SOURCE_SHA,'Frozen ready source/design required')
    require(plan['limits']==LIMITS,'Frozen acquisition/selection/resource limits changed')
    require(plan['fragments']==source['eligible_candidates'] and
            [(f['legacy_alias'],f['id'],f['parent_papyrus'],f['role'],f['shape_yx']) for f in plan['fragments']]==EXPECTED,
            'Declared fragment/alias/parent/role/canvas changed')
    expected_objects=[dict(fragment=f['legacy_alias'].lower(),kind=kind,url=f[key],shape_yx=f['shape_yx'])
        for f in plan['fragments'] for kind,key in [('labels','labels_url'),('mask','mask_url')]]
    require(plan['objects']==expected_objects and len(plan['objects'])==6,'Exact six-object identity/order changed')
    require(plan['selection']==dict(native_size=1152,stride=1152,square_size=33,xy_factor=3,
        center_offset=1,coarse_size=384,border=32,positive_min=512,negative_min=8192,separated_y_min=2304,
        surface_gate='10*inside_count >= 9*1152*1152'),'Selection recipe changed')
    require(plan['scientific_calls']==0 and plan['limitations']['CT_ready'] is False and
            plan['deferred']==source['deferred_candidates'],'Excluded/deferred source roles changed')
    seen=source['existing_seen_diagnostic']
    require(seen['alias']=='Frag4' and seen['id']=='PHercParis1Fr39' and seen['parent_papyrus']=='Paris1' and
            {f['id'] for f in plan['fragments']}.isdisjoint({seen['id'],*[d['id'] for d in plan['deferred']]}),
            'Seen/deferred aliases entered benchmark')
    acquisition=review.json(pins['acquisition']);panel=review.json(pins['panel'])
    require(acquisition['status']==panel['status']=='PASS','Completed acquisition and panel required')
    review.same(acquisition['plan'],pins['plan'],'Acquisition plan')
    review.same(panel['plan'],pins['plan'],'Panel plan')
    review.same(panel['acquisition'],pins['acquisition'],'Panel acquisition')
    require(acquisition['image_pixel_decodes']==acquisition['model_calls']==0 and acquisition['http_attempts']==6,
            'Acquisition accounting differs')
    require([{k:o[k] for k in ('fragment','kind','url','shape_yx')} for o in acquisition['objects']]==expected_objects,
            'Acquired identities differ')
    ledger=review.json(acquisition['request_ledger'])
    review.same(ledger['plan'],pins['plan'],'Request ledger plan')
    require(ledger['verified_tls'] is True and ledger['redirects_followed']==ledger['automatic_retries']==0 and
            len(ledger['requests'])==6 and ledger['maximums']==LIMITS,'Transport receipt scope differs')
    total=0;encoded_facts=[]
    # Validate all six immutable complete PNG byte streams before pixel decoding.
    for i,(obj,request) in enumerate(zip(acquisition['objects'],ledger['requests']),1):
        require(request['object']==expected_objects[i-1] and request['attempt']==i and
                request['success'] is True and request['http_status']==200,'HTTP identity/success/order differs')
        review.same(request['file'],obj['file'],'Transport file identity')
        filepin=identity(obj['file']);body=review.pin(obj['file']).read_bytes()
        require(0<len(body)<=524288 and request['body_bytes']==len(body)==filepin['size_bytes'] and
                request['body_sha256']==filepin['sha256'],'Transport body/hash/count differs')
        headers=request['http_identity']
        require(headers.get('content-encoding','identity').lower() in ('','identity'),'HTTP encoding differs')
        if 'content-length' in headers:require(headers['content-length'].isdigit() and int(headers['content-length'])==len(body),'Declared HTTP length differs')
        facts=structure(body,obj['shape_yx'])
        require(request['png_structure']==facts,'Transport PNG structure facts differ')
        encoded_facts.append(dict(fragment=obj['fragment'],kind=obj['kind'],file=filepin,structure=facts))
        total+=len(body)
    require(total==acquisition['response_body_bytes']==ledger['response_body_bytes']<=2097152,'Combined body count differs')
    if 'charged_response_bytes' in acquisition:
        require(acquisition['charged_response_bytes']==ledger['charged_response_bytes']==total,'Successful charged-body accounting differs')
    require(panel['automatic_retries']==panel['network_requests']==panel['model_calls']==panel['CT_pixel_decodes']==0 and
            panel['selected_context_count']==6 and panel['selected_array_count']==24 and panel['CT_ready'] is False and
            panel['benchmark_predictions']==0,'Panel method/accounting differs')
    require([w['fragment'] for w in panel['workers']]==['frag1','frag2','frag3'] and
            all(w['returncode']==0 for w in panel['workers']),'Once-only worker ordering/status differs')
    selected_all=[];fragment_receipts={};array_count=0
    filenames=dict(labels='labels.npy',surface_mask='surface-mask.npy',known_mask='known-mask.npy',primary_mask='primary-mask.npy')
    for fragment,worker in zip(plan['fragments'],panel['workers']):
        alias=fragment['legacy_alias'].lower();fragment_started=time.perf_counter()
        reported=review.json(worker['result'])
        require(reported['status']=='PASS' and reported['fragment']==alias and reported['identity']==fragment and
                reported['CT_pixel_decodes']==reported['model_calls']==reported['network_requests']==0,
                'Worker fragment/role/method receipt differs')
        review.same(reported['plan'],pins['plan'],'Worker plan');review.same(reported['acquisition'],pins['acquisition'],'Worker acquisition')
        objects=[o for o in acquisition['objects'] if o['fragment']==alias]
        require(reported['source_objects']==objects,'Worker raw source identities differ')
        by_kind={o['kind']:o for o in objects}
        labels,label_facts=decode(review.pin(by_kind['labels']['file']),fragment['shape_yx']);rss()
        surface,mask_facts=decode(review.pin(by_kind['mask']['file']),fragment['shape_yx']);rss()
        # Bands avoid a full-native temporary for inside/outside reference counts.
        inside_positive=outside_positive=0
        for start in range(0,labels.shape[0],128):
            l,m=labels[start:start+128],surface[start:start+128]
            inside_positive+=int(np.count_nonzero(l&m));outside_positive+=int(np.count_nonzero(l&~m))
        expected_quality=dict(labels=label_facts,mask=mask_facts,label_positive_inside_surface=inside_positive,
            label_positive_outside_surface=outside_positive,
            outside_labels_policy='record and exclude; outside-positive count is not an eligibility gate')
        require(reported['quality']==expected_quality,'PNG encoding/palette/quality/outside-label facts differ')
        coarse_known,native_known_count=native_known(labels,surface);rss()
        reconstructed=candidates(labels,surface,coarse_known)
        published=[published_candidate(row) for row in reconstructed]
        require(reported['candidates']==published and reported['candidate_count']==len(published) and
                reported['eligible_count']==sum(r['eligible'] for r in published),'All grid candidate counts/gates differ')
        first,last=choose(reconstructed)
        require(len(reported['selected'])==2,'Exactly two selected contexts required')
        selected_receipts=[]
        for index,(selected,row) in enumerate(zip(reported['selected'],[first,last]),1):
            xywh=[row['x'],row['y'],1152,1152];yxhw=[row['y'],row['x'],1152,1152]
            require({k:selected[k] for k in published_candidate(row)}==published_candidate(row),'Chosen candidate counts differ')
            require(selected['case_id']==alias+'-context'+str(index) and selected['fragment']==fragment['id'] and
                    selected['role']==fragment['role'] and selected['parent_papyrus']==fragment['parent_papyrus'] and
                    selected['bbox_source_xywh']==xywh and selected['native_bbox_yxhw']==yxhw and
                    selected['source_pixel_center_mapping']==dict(x=f"{row['x']}+1+3*x_coarse",y=f"{row['y']}+1+3*y_coarse"),
                    'Selected identity/bbox/center mapping/role differs')
            expected=context_arrays(labels,surface,coarse_known,row)
            require(set(selected['files'])==set(filenames.values()),'Four exact selected arrays required')
            array_pins={}
            for name,field in expected.items():
                item=selected['files'][filenames[name]]
                require(item['shape_yx']==[384,384] and item['dtype']==str(field.dtype),'Selected array metadata differs')
                actual=review.array(item,shape=(384,384),dtype=field.dtype)
                require(np.array_equal(actual,field),'Selected canonical array differs: '+name)
                array_pins[name]=identity(item);array_count+=1
            selected_receipts.append(dict(case_id=selected['case_id'],role=selected['role'],parent_papyrus=selected['parent_papyrus'],
                bbox_source_xywh=xywh,candidate=published_candidate(row),arrays=array_pins,all_arrays_exact=True))
            selected_all.append(selected)
        require(last['y']-first['y']>=2304,'Selected native y separation differs')
        require(reported['peak_rss_bytes']<=RSS_CAP and reported['wall_seconds']<=120,'Worker resource receipt exceeds frozen cap')
        fragment_receipts[alias]=dict(status='PASS',identity=fragment,encoding_quality=expected_quality,
            all_candidates=published,native_known_count=native_known_count,selected=selected_receipts,
            reconstruction_wall_seconds=time.perf_counter()-fragment_started,review_peak_rss_bytes=rss())
        del labels,surface,coarse_known,expected,actual,l,m;gc.collect()
    require(panel['selected']==selected_all and array_count==24,'Aggregated selected arrays/order differ')
    require([s['role'] for s in selected_all]==['evaluation']*4+['calibration']*2 and
            [s['parent_papyrus'] for s in selected_all]==['Paris2']*4+['Paris1']*2,'Parent-level role exclusion differs')
    require(panel['wall_seconds']<=LIMITS['selection_wall_seconds'],'Selection wall receipt exceeds frozen cap')
    selection_bytes=sum(p.stat().st_size for p in Path(pins['panel']['path']).parent.rglob('*') if p.is_file())
    acquisition_bytes=sum(p.stat().st_size for p in Path(pins['acquisition']['path']).parent.rglob('*') if p.is_file())
    require(selection_bytes<=LIMITS['selection_output_bytes'] and
            selection_bytes+acquisition_bytes<=LIMITS['total_output_bytes'],'Completed output bytes exceed frozen cap')
    for item in sources:review.pin(item)
    review.unchanged()
    require('torch' not in sys.modules and 'scipy.ndimage' not in sys.modules,'Forbidden model/author filter dependency imported')
    return dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),inputs=pins,
        verifier_sources=sources,transport_receipts=encoded_facts,response_body_bytes=total,
        selection_output_bytes=selection_bytes,acquisition_output_bytes=acquisition_bytes,
        fragments=fragment_receipts,selected_array_count=24,all_selected_arrays_exact=True,
        all_candidates_counts_and_integer_gates_exact=True,first_last_spacing_exact=True,
        roles_parents_alias_exclusions_verified=True,PNG_binary_encoding_palette_and_CRC_verified=True,
        integral_method='Signed int32 prefix sums in256 output-row bands with16-pixel halos; full-canvas33-square known reconstructed without SciPy min/max or full-native integral allocation.',
        review_peak_rss_bytes=rss(),review_address_space_cap_bytes=AS_CAP,wall_seconds=time.perf_counter()-started,
        source_pins_unchanged=True,verified_input_pins=list(review.files.values()),
        actual_operations=dict(raw_label_mask_PNG_decodes=6,CT_image_decodes=0,Torch_imports=0,model_calls=0,
                              checkpoint_reads=0,network_calls=0,new_predictions_calibration_or_training=0),
        limitations=['This reconstructs label-supported deterministic selection, not model accuracy or CT readiness.',
            'HTTP/TLS/resource/source execution claims remain pinned source and recorded worker/transport evidence; this reviewer performs no network.',
            'Three physical fragments span two parents; Paris2 has two evaluation fragments, while calibration Paris1 already contains seen Frag4.',
            'Optical registration residual and complete upstream teacher/pretraining exposure remain unmeasured/unknown.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--authorized-retained-read',action='store_true')
    for name in ('plan','acquisition','panel'):
        parser.add_argument('--'+name,type=Path);parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    require(HERE in args.output_dir.resolve().parents and not args.output_dir.exists(),'Fresh owned output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources=[record(p) for p in sorted(HERE.glob('*.py'))]
    try:
        if args.self_test:
            result=dict(status='PASS',support_fixtures=support_fixtures(),PNG_fixtures=png_fixtures(),sources=sources,
                actual_real_image_decodes=0,network_calls=0,Torch_imports=0,model_calls=0)
        else:
            require(args.authorized_retained_read,'Root authorization after scientific completion required')
            result=review_outputs(args,sources)
    except Exception as error:
        result=dict(status='STOPPED',reason=str(error),error_type=type(error).__name__,verifier_sources=sources,
            no_partial_PASS_claim=True,network_calls=0,model_calls=0,Torch_imports=0,checkpoint_reads=0)
        (args.output_dir/'independent-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        raise
    target=args.output_dir/('synthetic-fixtures.json' if args.self_test else 'independent-verification.json')
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',receipt=record(target))))


if __name__=='__main__':main()
