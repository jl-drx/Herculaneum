"""Execute the reviewed fixed annotation-only candidate plan; never read scans."""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone
import numpy as np
import numcodecs
import bounded_http

ROOT=Path(__file__).resolve().parent
ROLES=('inklabels','supervision_mask','validation_mask')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard(records):
    for rec in records:
        p=Path(rec['path'])
        if p.stat().st_size!=rec['byte_length'] or sha(p)!=rec['sha256']:
            raise ValueError('Pinned source input changed')
def record(p):return {'path':str(p.resolve()),'sha256':sha(p),'byte_length':p.stat().st_size}
def save_array(folder,name,array):
    p=folder/(name+'.npy')
    if p.exists():raise ValueError('Refuse overwrite of selected annotation')
    np.save(p,array,allow_pickle=False)
    return {**record(p),'shape':list(array.shape),'dtype':str(array.dtype)}
def overlap(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    return max(x,xx)<min(x+w,xx+ww) and max(y,yy)<min(y+h,yy+hh)

def run(plan_path,output):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve()
    plan_sha=sha(plan_path);code_sha=sha(Path(__file__))
    plan=json.loads(plan_path.read_text());guard(plan['input_records'])
    if sha(ROOT/'bounded_http.py')!=plan['http_module_sha256']:
        raise ValueError('HTTP helper differs from pinned implementation')
    if (plan['crop_size_xy']!=[384,384] or plan['primary_margin_px']!=64 or
        plan['minimum_per_class']!=256 or plan['maximum_selected_per_case']!=2 or
        plan['minimum_selected_per_case']!=1 or plan['annotation_z']!=10 or
        plan['raw_annotation_allowed_values']!=[0,1]):
        raise ValueError('Unsupported selection contract')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():raise ValueError('Selection already published')
    raw=output/'raw-labels';raw.mkdir(exist_ok=True)
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'incomplete','plan_record':record(plan_path),'selection_code_record':record(Path(__file__)),
        'http_module_record':record(ROOT/'bounded_http.py'),'case_order':plan['case_order'],
        'selection_before_scans_and_predictions':True,'cases':{},
        'claim_scope':plan['claim_scope'],'actual_historical_training_overlap':{'status':'unknown','mask_file':None},
        'geometry_validity':{'status':'not_acquired','mask_file':None}}
    for case in plan['case_order']:
        spec=plan['cases'][case];objects={o['key']:o for o in spec['objects']}
        codec=numcodecs.get_codec(json.loads(Path(spec['label_zarray_record']['path']).read_text())['compressor'])
        accepted=[];evaluations=[];cache={}
        for candidate in spec['candidates']:
            if len(accepted)==2:break
            bbox=candidate['bbox_source_xywh'];x,y,w,h=bbox
            if any(overlap(bbox,a['bbox_source_xywh']) for a in accepted):
                evaluations.append({'rank':candidate['rank'],'bbox_source_xywh':bbox,
                    'status':'ineligible_full_context_overlaps_accepted','annotation_bytes_read':False})
                continue
            arrays={role:np.zeros((384,384),np.uint8) for role in ROLES}
            used=[]
            for key in candidate['label_object_keys']:
                obj=objects[key]
                if key not in cache:
                    body,retained=bounded_http.fetch_object(obj,raw,plan_sha,
                        cap=plan['label_response_body_cap_bytes'],ledger_name='label-request-ledger.json',
                        maximum_requests=plan['maximum_http_request_attempts'],
                        maximum_requests_per_object=plan['maximum_requests_per_object'],namespace=case)
                    decoded=np.frombuffer(codec.decode(body),dtype=np.uint8)
                    if decoded.size!=21*128*128:raise ValueError('Wrong decoded label chunk length')
                    cube=decoded.reshape(21,128,128)
                    unique=set(np.unique(cube).tolist())
                    if not unique.issubset({0,1}):raise ValueError('Unexpected annotation encoding')
                    if np.any(cube[:10]) or np.any(cube[11:]):
                        raise ValueError('Annotation outside declared center plane')
                    cache[key]=(cube[10].copy(),retained)
                plane,retained=cache[key];yy,xx=obj['chunk_yx'];local_y=yy*128-y;local_x=xx*128-x
                if not (0<=local_y<=256 and 0<=local_x<=256):raise ValueError('Chunk outside fixed candidate')
                arrays[obj['role']][local_y:local_y+128,local_x:local_x+128]=plane
                used.append({'key':key,**retained})
            valid=arrays['validation_mask']>0;labels=arrays['inklabels']>0;sup=arrays['supervision_mask']>0
            interior=np.zeros((384,384),bool);interior[64:320,64:320]=True
            primary=valid&interior
            counts={'whole_validation':int(valid.sum()),
                'whole_validation_positive':int((valid&labels).sum()),
                'whole_validation_negative':int((valid&~labels).sum()),
                'primary_annotation_pixels':int(primary.sum()),
                'primary_positive':int((primary&labels).sum()),'primary_negative':int((primary&~labels).sum()),
                'supervision':int(sup.sum()),'raw_supervision_validation_overlap':int((sup&valid).sum()),
                'conditional_current_loader_training_pixels':int((sup&~valid).sum())}
            adequate=counts['primary_positive']>=256 and counts['primary_negative']>=256
            evaluation={'rank':candidate['rank'],'bbox_source_xywh':bbox,
                'status':'accepted' if adequate else 'rejected_class_count','counts':counts,
                'observed_raw_values':{role:np.unique(a).tolist() for role,a in arrays.items()}}
            evaluations.append(evaluation)
            if not adequate:continue
            ident=case+'-rank'+str(candidate['rank']).zfill(2)
            folder=output/ident
            if folder.exists():raise ValueError('Selected crop folder already exists')
            folder.mkdir()
            records={role:save_array(folder,role,arrays[role]) for role in ROLES}
            records.update({'validation_membership':save_array(folder,'validation-membership',valid),
                'supervision_membership':save_array(folder,'supervision-membership',sup),
                'current_loader_label_mask':save_array(folder,'current-loader-label-mask',sup&~valid),
                'primary_annotation_mask':save_array(folder,'primary-annotation-mask',primary)})
            accepted.append({'crop_id':ident,'case':case,'specimen':spec['specimen'],
                'rank':candidate['rank'],'bbox_source_xywh':bbox,'arrays':records,'counts':counts,
                'observed_raw_values':evaluation['observed_raw_values'],'binary_foreground':{role:1 for role in ROLES},
                'annotation_z':10,'acquisition_observed_mask':'pending source acquisition',
                'source_objects':used})
        result['cases'][case]={'selected':accepted,'evaluations':evaluations,
            'complete':len(accepted)>=1,'unexamined_fixed_candidates':[
                c['rank'] for c in spec['candidates'] if c['rank'] not in {e['rank'] for e in evaluations}]}
    guard(plan['input_records'])
    if sha(plan_path)!=plan_sha or sha(Path(__file__))!=code_sha or sha(ROOT/'bounded_http.py')!=plan['http_module_sha256']:
        raise ValueError('Frozen executable or plan changed')
    ledger=json.loads((raw/'label-request-ledger.json').read_text())
    result['label_request_ledger_record']=record(raw/'label-request-ledger.json')
    result['response_body_bytes']=ledger['body_bytes'];result['http_request_attempts']=len(ledger['requests'])
    result['status']='selection_complete_no_scans_or_inference' if all(c['complete'] for c in result['cases'].values()) else 'selection_inadequate_no_scan_authorization'
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    target=output/'selection.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'selection_sha256':sha(target),
        'selected':{c:[a['bbox_source_xywh'] for a in d['selected']] for c,d in result['cases'].items()},
        'body_bytes':result['response_body_bytes'],'requests':result['http_request_attempts']}))
    return 0 if result['status']=='selection_complete_no_scans_or_inference' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raise SystemExit(run(a.plan,a.output))
