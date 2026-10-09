"""Offline annotation/exclusion design. No models, network or new data reads."""
from pathlib import Path
import hashlib,json,re
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[2]
PREVIOUS=LAB/'reports/seventh-experiment/panel-sources'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'byte_length':p.stat().st_size}
def guard(records):
    for r in records:
        p=Path(r['path']);assert p.stat().st_size==r['byte_length'] and sha(p)==r['sha256']
def gap(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    return max(max(xx-x-w,x-xx-ww,0),max(yy-y-h,y-yy-hh,0))

def main():
    inputs=[];training=[]
    for name in ['pherc0139-w016-rank01','pherc0139-w016-rank02']:
        p=PREVIOUS/'acquisition'/name/'provenance.json';inputs.append(rec(p));d=json.loads(p.read_text())
        roles={k:d['arrays'][k] for k in ['inklabels','supervision_mask','validation_mask','acquisition_observed_mask']}
        guard(list(roles.values()));inputs+=list(roles.values())
        a={k:np.load(r['path'],allow_pickle=False) for k,r in roles.items()}
        assert all(x.shape==(384,384) for x in a.values())
        assert all(set(np.unique(a[k]).tolist()).issubset({0,1}) for k in ['inklabels','supervision_mask','validation_mask'])
        manual=(a['supervision_mask']>0)&(a['validation_mask']==0)&a['acquisition_observed_mask']
        ink=a['inklabels']>0
        tiles=[]
        for y in range(0,257,64):
            for x in range(0,257,64):
                m=manual[y:y+128,x:x+128];i=ink[y:y+128,x:x+128]
                pos=int((m&i).sum());neg=int((m&~i).sum())
                tiles.append({'bbox_local_xywh':[x,y,128,128],'bbox_source_xywh':[d['bbox_source_xywh'][0]+x,d['bbox_source_xywh'][1]+y,128,128],
                              'manual_positive':pos,'manual_negative':neg,'eligible_minimum256_each':min(pos,neg)>=256})
        training.append({'crop_id':name,'specimen':'PHerc0139','provenance_record':rec(p),'bbox_source_xywh':d['bbox_source_xywh'],
            'manual_mask_rule':'public supervision>0 & validation==0 & acquired source',
            'unique_manual_pixels':int(manual.sum()),'unique_manual_positive':int((manual&ink).sum()),
            'unique_manual_negative':int((manual&~ink).sum()),'eligible128_stride64_tiles':sum(t['eligible_minimum256_each'] for t in tiles),
            'all25_tile_counts':tiles,'source_array_record':d['arrays']['surface'],
            'label_lineage':'Public manual-reference root with transferred/iterative pseudo history; not proven human-only truth'})
    excluded=[]
    for name in ['pherc0814-46527-rank01','pherc0814-46527-rank07']:
        p=PREVIOUS/'acquisition'/name/'provenance.json';inputs.append(rec(p));d=json.loads(p.read_text())
        excluded.append({'crop_id':name,'bbox_source_xywh':d['bbox_source_xywh'],'provenance_record':rec(p),
            'reason':'Already used for seventh-study predictions and new hypothesis development'})
    inventory={}
    for role in ['inklabels','supervision_mask','validation_mask']:
        inv={}
        for p in PREVIOUS.glob('source-*-pherc0814-46527-'+role+'-inventory-*.json'):
            inputs.append(rec(p))
            for obj in json.loads(p.read_text()):
                n=obj['path'].split('/')[-1]
                if re.fullmatch(r'0\.\d+\.\d+',n):
                    _,y,x=map(int,n.split('.'));assert(y,x)not in inv;inv[y,x]=obj
        assert len(inv)==459;inventory[role]=inv
    previous_plan=PREVIOUS/'label-selection-plan.proposal.json';inputs.append(rec(previous_plan))
    spec=json.loads(previous_plan.read_text())['cases']['pherc0814-46527']
    inputs+=spec['source_metadata_records']+[spec['label_zarray_record']]
    for role in ['inklabels','supervision_mask','validation_mask']:
        inputs.append(rec(next(PREVIOUS.glob('source-*-pherc0814-46527-'+role+'-zattrs.json'))))
    assert spec['source_shape_zyx']==[109,2130,3455]
    ranked=[]
    for y in range(0,2130-383,128):
        for x in range(0,3455-383,128):
            bbox=[x,y,384,384];gaps=[gap(bbox,p['bbox_source_xywh']) for p in excluded]
            if min(gaps)<128:continue
            keys={(yy,xx)for yy in range(y//128,y//128+3)for xx in range(x//128,x//128+3)}
            proxy={role:sum(inventory[role][k]['size']!=78 for k in keys)for role in inventory}
            if proxy['validation_mask'] or proxy['supervision_mask']:
                ranked.append({'bbox_source_xywh':bbox,'old_context_gap_pixels':gaps,
                    'potential_validation_chunks':proxy['validation_mask'],'potential_supervision_chunks':proxy['supervision_mask'],
                    'potential_ink_chunks':proxy['inklabels']})
    ranked.sort(key=lambda x:(-x['potential_validation_chunks'],-x['potential_supervision_chunks'],-x['potential_ink_chunks'],
                              x['bbox_source_xywh'][1],x['bbox_source_xywh'][0]))
    assert len(ranked)==35
    candidates=ranked[:16];keys=set()
    for rank,c in enumerate(candidates,1):
        c['rank']=rank;x,y,w,h=c['bbox_source_xywh']
        c['label_object_keys']=[f'{role}/0.{yy}.{xx}'for role in inventory for yy in range(y//128,y//128+3)for xx in range(x//128,x//128+3)]
        keys.update((role,yy,xx)for role in inventory for yy in range(y//128,y//128+3)for xx in range(x//128,x//128+3))
    objects=[]
    for role,y,x in sorted(keys):
        o=inventory[role][y,x]
        objects.append({'key':f'{role}/0.{y}.{x}','role':role,'chunk_yx':[y,x],
            'url':'https://huggingface.co/buckets/scrollprize/datasets/resolve/'+o['path'],
            'expected_byte_length':o['size'],'expected_etag':o['xetHash']})
    guard(inputs)
    proposal={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'metadata_and_retained_annotations_only_proposal',
        'adaptation_specimen':'PHerc0139','adaptation_support':training,
        'excluded_from_all_new_adaptation':'PHerc0814, including all representations, teacher targets and prior contexts',
        'reserve_case':'pherc0814-46527','reserve_specimen':'PHerc0814','source_specification':spec,
        'old_used_contexts':excluded,'source_xy_gap_rule':'For closed rectangle extents, max(horizontal gap,vertical gap)>=128. A touching boundary has gap0. This conservative L-infinity rule is applied to every oldcontext and between accepted freshcontexts.',
        'minimum_context_gap_px':128,'crop_size_xy':[384,384],'primary_margin_px':64,
        'minimum_class_count_each':256,'required_new_reserve_contexts':2,
        'candidate_ranking':'Descending non78 validation-chunk count, then non78 supervision count, then non78 ink count, then ascendingY/X. Non78 is only a metadata proxy.',
        'candidates':candidates,'all_gap_supported_origins_count':35,'fresh_validation_proxy_supported_origins':0,
        'evaluation_participation_priority':['validation_mask','supervision_mask'],
        'selection_policy':'Global first pass in fixed16 order using suppliedvalidation. If fewer than2adequate gap-separated contexts, second pass over identical16using suppliedsupervision. Accept only>=256positive and>=256negative within fixed[64:320]^2; do notrelaxgeometryorclasses. Retain masks/role explicitly; exacttwo required beforetraining.',
        'objects':objects,'maximum_unique_label_objects':len(objects),
        'expected_candidate_label_payload_bytes':sum(o['expected_byte_length']for o in objects),
        'label_response_body_cap_bytes':2*1024*1024,'maximum_label_http_attempts':6*len(objects),
        'source_response_body_cap_bytes':40*1024*1024,'source_scan_payload_maximum_bytes':18*109*128*128,
        'expected_source_metadata_bytes':2*sum(m['byte_length']for m in spec['source_metadata_records']),
        'maximum_source_http_attempts':6*(18+4),'maximum_source_redirect_bytes':5*4096*(18+4),
        'input_records':inputs,
        'physical_split_limits':['Different specimens separate newadaptation fromreserve physically','SourceXY gap does notruleout within0814 native3D coincidence ofrenderedfaces/depthsupport','Seed42student andseed43teacher historicallyinclude0814; no globalmodel/teacherheldoutclaim','Reference labels maycontain model-assisted annotation; no opticalgroundtruth'],
        'terms':'Private noncommercial research; publicCTCC-BY-NC4.0; legacyannotation termsreach unresolved before anyrelease',
        'teacher_contract':'Root proposes retainedseed43 maps on PHerc0139 adaptationcrops only; no teacher inference/targetselectionperformedhere'}
    target=ROOT/'data-proposal.json';assert not target.exists();target.write_text(json.dumps(proposal,indent=2)+'\n')
    print(json.dumps({'candidate_first':candidates[0]['bbox_source_xywh'],'max_labels':len(objects),
        'expected_labels':proposal['expected_candidate_label_payload_bytes'],'max_label_requests':proposal['maximum_label_http_attempts'],
        'source_scan_bytes':proposal['source_scan_payload_maximum_bytes'],'source_metadata':proposal['expected_source_metadata_bytes'],
        'input_records_verified':len(inputs),'sha256':sha(target)}))

if __name__=='__main__':main()
