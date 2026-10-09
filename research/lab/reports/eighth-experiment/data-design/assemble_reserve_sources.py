"""Execute a separately reviewed source plan, preserving observed zero values."""
from pathlib import Path
import argparse, hashlib, json, re
from datetime import datetime, timezone
import numpy as np
import bounded_http

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'byte_length':p.stat().st_size}
def guard(records):
    for r in records:
        p=Path(r['path'])
        if p.stat().st_size!=r['byte_length'] or sha(p)!=r['sha256']:raise ValueError('Frozen source input changed')
def array(folder,name,values):
    p=folder/(name+'.npy')
    if p.exists():raise ValueError('Refuse overwrite of source array')
    np.save(p,values,allow_pickle=False)
    return {**rec(p),'shape':list(values.shape),'dtype':str(values.dtype)}
def load(r):
    a=np.load(r['path'],allow_pickle=False)
    if list(a.shape)!=r['shape'] or str(a.dtype)!=r['dtype']:raise ValueError('Input array contract mismatch')
    return a

def run(plan_path,output):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve();plan_sha=sha(plan_path)
    plan=json.loads(plan_path.read_text());guard(plan['input_records'])
    guard([plan['http_module_record'],plan['source_code_record'],plan['selection_record'],plan['label_plan_record']])
    if plan['source_code_record']!=rec(Path(__file__)) or plan['http_module_record']!=rec(ROOT/'bounded_http.py'):
        raise ValueError('Executable does not match frozen proposal')
    if plan['source_response_body_cap_bytes']!=40*1024*1024 or len(plan['crops'])!=2 or plan['evaluation_participation']!='supervision_mask':
        raise ValueError('Unsupported panel contract')
    selection=json.loads(Path(plan['selection_record']['path']).read_text())
    selected={a['crop_id']:a for c in selection['cases'].values() for a in c['selected']}
    if set(selected)!={c['crop_id'] for c in plan['crops']}:raise ValueError('Source plan differs from selected crop set')
    for crop in plan['crops']:
        original=selected[crop['crop_id']]
        if crop['evaluation_participation']!='supervision_mask' or original['evaluation_participation']!='supervision_mask':
            raise ValueError('Participation role differs from selected support')
        if crop['bbox_source_xywh']!=original['bbox_source_xywh'] or crop['label_arrays']!=original['arrays']:
            raise ValueError('Source crop/labels differ from annotation selection')
        if crop['shape_zyx']!=[21,384,384] or crop['source_depth_slice']!=[13,97] or crop['model_layer_slice']!=[2,19]:
            raise ValueError('Source pooling contract changed')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'panel-provenance.json').exists():raise ValueError('Panel already published')
    raw=output/'raw-source';raw.mkdir(exist_ok=True)
    def fetch(obj,namespace):
        return bounded_http.fetch_object(obj,raw,plan_sha,cap=plan['source_response_body_cap_bytes'],
            ledger_name='source-request-ledger.json',maximum_requests=plan['maximum_http_request_attempts'],
            maximum_requests_per_object=plan['maximum_requests_per_object'],namespace=namespace)
    checked_metadata=[]
    def metadata(stage):
        for obj in plan['metadata_objects']:
            body,retained=fetch(obj,stage+':'+obj['case'])
            if hashlib.sha256(body).hexdigest()!=obj['expected_sha256']:
                raise ValueError('Live source metadata differs from pinned evidence')
            checked_metadata.append({'stage':stage,'case':obj['case'],'role':obj['role'],**retained})
    metadata('before')
    cubes={};retained_objects={}
    for obj in plan['objects']:
        body,retained=fetch(obj,'scan:'+obj['case'])
        values=np.frombuffer(body,dtype=np.uint8)
        if values.size!=109*128*128:raise ValueError('Wrong source chunk length')
        cubes[obj['key']]=values.reshape(109,128,128)
        retained_objects[obj['key']]=retained
    metadata('after');guard(plan['input_records'])
    if sha(plan_path)!=plan_sha:raise ValueError('Source plan changed during acquisition')
    panel={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'incomplete',
        'source_plan_record':rec(plan_path),'selection_record':plan['selection_record'],
        'claim_scope':plan['claim_scope'],'crops':[],'checked_metadata':checked_metadata}
    object_by_key={o['key']:o for o in plan['objects']}
    fresh_cases=[]
    for crop in plan['crops']:
        x,y,w,h=crop['bbox_source_xywh'];source=np.empty((109,384,384),np.uint8)
        observed=np.zeros((384,384),bool);crop_sources=[]
        for key in crop['source_chunk_keys']:
            obj=object_by_key[key];zz,yy,xx=obj['chunk_zyx'];ly=yy*128-y;lx=xx*128-x
            if zz!=0 or not(0<=ly<=256 and 0<=lx<=256):raise ValueError('Source chunk outside selected context')
            source[:,ly:ly+128,lx:lx+128]=cubes[key]
            observed[ly:ly+128,lx:lx+128]=True
            crop_sources.append({'key':key,'chunk_zyx':obj['chunk_zyx'],**retained_objects[key]})
        if not observed.all():raise ValueError('Incomplete observed source footprint')
        surface=np.rint(source[13:97].reshape(21,4,384,384).mean(axis=1,dtype=np.float32)).astype(np.uint8)
        occupancy=np.any(surface[2:19]!=0,axis=0)
        labels=load(crop['label_arrays']['inklabels']);sup=load(crop['label_arrays']['supervision_mask']);val=load(crop['label_arrays']['validation_mask'])
        if any(a.shape!=(384,384) or a.dtype!=np.uint8 or not set(np.unique(a).tolist()).issubset({0,1}) for a in [labels,sup,val]):
            raise ValueError('Unexpected source annotation encoding/shape')
        interior=np.zeros((384,384),bool);interior[64:320,64:320]=True
        primary=(sup>0)&observed&interior
        primary_positive=int((primary&(labels>0)).sum());primary_negative=int((primary&(labels==0)).sum())
        if primary_positive!=crop['label_counts']['primary_positive'] or primary_negative!=crop['label_counts']['primary_negative']:
            raise ValueError('Post-acquisition annotation class counts changed')
        if min(primary_positive,primary_negative)<256:raise ValueError('Observed primary class gate failed')
        folder=output/crop['crop_id']
        if folder.exists():raise ValueError('Refuse overwrite of source crop directory')
        folder.mkdir()
        arrays=dict(crop['label_arrays'])
        arrays.update({'surface':array(folder,'surface',surface),
            'acquisition_observed_mask':array(folder,'acquisition-observed-mask',observed),
            'source_observed_nonzero_mask':array(folder,'source-observed-nonzero-mask',occupancy),
            'primary_annotation_mask':array(folder,'primary-annotation-mask',primary),
            'evaluation_participation_mask':array(folder,'evaluation-participation-mask',(sup>0)&observed)})
        provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'source_complete_no_inference',
            'crop_id':crop['crop_id'],'case':crop['case'],'segment':crop['case'],'sample_id':crop['specimen'],
            'public_segment':crop['public_segment'],'segment_long_id':crop['public_segment'],
            'target_volume_id':re.search(r'volume-(\d+)',crop['source_surface_url']).group(1),
            'source_surface_url':crop['source_surface_url'],'source_shape_zyx':crop['source_shape_zyx'],
            'source_level':2,'bbox_source_xywh':crop['bbox_source_xywh'],'shape_zyx':[21,384,384],
            'pixel_spacing_um':9.596,'spacing_zyx_um':[9.596]*3,'dtype':'uint8',
            'source_depth_slice':[13,97],'depth_aggregation':'float32 mean of consecutive groups of 4; np.rint nearest-even; uint8',
            'pooled_plane_source_ranges':[[13+4*z,17+4*z] for z in range(21)],
            'model_layer_slice':[2,19],'model_source_depth_slice':[21,89],'annotation_center_layer':10,
            'depth_orientation':plan['depth_orientation'],'arrays':arrays,
            'binary_foreground':crop['binary_foreground'],'observed_annotation_values':crop['observed_annotation_values'],
            'evaluation_participation':'supervision_mask','evaluation_participation_role':'supplied_supervision',
            'counts':{**crop['label_counts'],'acquisition_observed_pixels':int(observed.sum()),
                'source_observed_nonzero_pixels_selected17':int(occupancy.sum()),
                'primary_observed_zero_profile_pixels':int((primary&~occupancy).sum())},
            'geometry_validity':{'status':'not_acquired','mask_file':None},
            'actual_historical_training_overlap':{'status':'unknown','mask_file':None},
            'mask_semantics':{'primary':'supervision>0 & HTTP200-acquired source & fixed 64-pixel interior; later common finite/evaluated support required',
                'nonzero_occupancy':'np.any(surface[2:19]!=0,axis=0); diagnostic stratum only, zeros retained in primary',
                'current_loader_label_mask':'public supervision>0 & ~(validation>0), conditional current-loader semantics only'},
            'physical_independence':'Entire PHerc0814 excluded from our new PHerc0139 adaptation. Fresh full contexts have at least 128-pixel source XY gap from old used contexts and each other; native 3D coincidence and historical teacher exposure unverified.',
            'old_used_contexts':plan['old_used_contexts'],'minimum_context_gap_px':128,
            'annotation_lineage':'results-fixed, transferred manual-plus-iterative-pseudo public reference annotation; complete human/teacher provenance unverified',
            'claim_scope':plan['claim_scope'],'source_objects':crop_sources,
            'selection_record':plan['selection_record'],'source_plan_record':rec(plan_path)}
        target=folder/'provenance.json';target.write_text(json.dumps(provenance,indent=2)+'\n')
        panel['crops'].append({'crop_id':crop['crop_id'],'case':crop['case'],'specimen':crop['specimen'],
            'bbox_source_xywh':crop['bbox_source_xywh'],'provenance_record':rec(target),
            'counts':provenance['counts']})
        fresh_cases.append({'id':crop['crop_id'],'physical_group':'PHerc0814','role':'reserved_from_new_adaptation',
            'evaluation_participation_role':'supplied_supervision','evaluation_participation':'supervision_mask',
            'bbox_source_xywh':crop['bbox_source_xywh'],'shape_zyx':[21,384,384],
            'source_depth_slice':[13,97],'model_layer_slice':[2,19],'annotation_center_layer':10,
            'pixel_spacing_um':9.596,'source_surface_url':crop['source_surface_url'],'public_segment':crop['public_segment'],
            'provenance_record':rec(target),'arrays':arrays,'counts':provenance['counts']})
    guard(plan['input_records']);guard([plan['http_module_record'],plan['source_code_record']])
    if sha(plan_path)!=plan_sha:raise ValueError('Frozen plan changed before publication')
    ledger_path=raw/'source-request-ledger.json';ledger=json.loads(ledger_path.read_text())
    panel.update({'status':'panel_source_complete_no_inference','source_request_ledger_record':rec(ledger_path),
        'response_body_bytes':ledger['body_bytes'],'http_request_attempts':len(ledger['requests']),
        'finished_at_utc':datetime.now(timezone.utc).isoformat()})
    descriptor=output/'fresh-descriptor.json'
    descriptor.write_text(json.dumps({'status':'fresh_reference_complete_no_inference','cases':fresh_cases,
        'source_plan_record':rec(plan_path),'claim_scope':plan['claim_scope'],
        'evaluation_participation_role':'supplied_supervision'},indent=2)+'\n')
    panel['fresh_descriptor_record']=rec(descriptor)
    guard(plan['input_records']);guard([plan['http_module_record'],plan['source_code_record']])
    if sha(plan_path)!=plan_sha:raise ValueError('Frozen plan changed before final publication')
    target=output/'panel-provenance.json';target.write_text(json.dumps(panel,indent=2)+'\n')
    print(json.dumps({'status':panel['status'],'sha256':sha(target),'crop_count':len(panel['crops']),
                      'body_bytes':panel['response_body_bytes'],'requests':panel['http_request_attempts']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.plan,a.output)
