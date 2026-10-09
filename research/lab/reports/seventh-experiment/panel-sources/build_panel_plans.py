"""Create fixed metadata-ranked panel proposals without reading labels or scans."""
from pathlib import Path
import hashlib, json, re, shutil
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[2]
DOC=LAB/'data/experiments/known-ink-control/upstream-dataset-README.md'
CONFIG=LAB/'reports/frontier-methods-audit-20261008/source-evidence/official-contributions/russo-configs-train_baseline_config.json'
EXCONFIG=CONFIG.with_name('russo-configs-train_pseudo_ex016_config.json')
DIFF=CONFIG.with_name('russo-results-label_version_diff.json')
OLD_HTTP=LAB/'reports/sixth-experiment/data-contract/bounded_http_v2.py'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return {'path':str(p),'sha256':sha(p),'byte_length':p.stat().st_size}
def metadata(name):return next(ROOT.glob('source-*-'+name))
def read(name):return json.loads(metadata(name).read_text())
def dump(name,data):
    p=ROOT/name
    if p.exists():raise ValueError('Refuse overwrite: '+name)
    p.write_text(json.dumps(data,indent=2)+'\n')

def main():
    rows=[]
    for line in DOC.read_text().splitlines():
        m=re.match(r'\| (pherc[^ |]+|w\d+) \| ([^|]+) \| .*\((https://[^)]+)\)',line)
        if not m:continue
        alias,scroll,url=m.groups();source=url.split('index.html#',1)[1].rstrip('/')
        native=not alias.startswith('pherc')
        canonical='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+source
        public_segment=source.split('/')[2]
        rows.append({'annotation_alias':alias,'specimen':source.split('/')[0],
            'representation_family':'native_9p362_level0' if native else 'public_2p4_level2_zmean4',
            'public_segment':public_segment,'physical_group_by_public_segment':source.split('/')[0]+'/'+public_segment,
            'source_surface_url':canonical,'source_volume_id':re.search(r'volume-(\d+)',source).group(1),
            'supplied_validation_mask':alias in ('pherc0139-w016','pherc1667-w029','pherc0814-46527'),
            'all_four_model_configs_include_representation':True,
            'complete_historical_training_or_teacher_exposure':'unverified'})
    assert len(rows)==29
    refs=[record(p) for p in [DOC,CONFIG,EXCONFIG,DIFF,OLD_HTTP,
        LAB/'reports/seventh-experiment/model-exposure-ledger.json',
        LAB/'reports/sixth-experiment/data-contract/SOURCE_AUDIT.md',
        LAB/'reports/sixth-experiment/data-contract/selection-v2/provenance.json']]
    created=datetime.now(timezone.utc).isoformat()
    plan={'created_at_utc':created,'status':'proposal_no_labels_scans_or_predictions_read',
        'case_order':['pherc0139-w016','pherc0814-46527'],
        'selection_rule':'For each case in fixed order, inspect only the sixteen fixed metadata-ranked candidates. Accept first with >=256 positive and >=256 negative validation pixels in local[64:320,64:320]; accept next adequate whose entire384x384 context has no 2D intersection with the first. Stop aftertwo accepted per case. Overlappinglatercandidates are recorded as ineligible without reading labels. Require at least one per case, otherwise no panel-complete claim.',
        'candidate_ranking':'Descending number of primary-interior-intersecting validation chunks whose inventory size !=78; then analogous ink-label chunk count; then ascending sourceY/X. Size !=78 is only a metadata proxy, never decoded annotation truth.',
        'crop_size_xy':[384,384],'primary_margin_px':64,'minimum_per_class':256,
        'maximum_selected_per_case':2,'minimum_selected_per_case':1,
        'label_response_body_cap_bytes':2*1024*1024,'maximum_requests_per_object':6,
        'source_scan_cap_bytes':64*1024*1024,'source_chunk_bytes':109*128*128,
        'source_maximum_scan_payload_bytes':4*9*109*128*128,
        'cases':{},'input_records':refs+[record(ROOT/'metadata-ledger.json')],
        'raw_annotation_allowed_values':[0,1],'binary_foreground':1,'annotation_z':10,
        'claim_scope':'Reserved from our future tuning only. Existing checkpoints already use these scrolls/onlinevalidation. Public reference annotations are not independently authenticated ink truth.',
        'http_module_sha256':sha(OLD_HTTP)}
    for case in plan['case_order']:
        attrs=read(case+'-validation_mask-zattrs.json')
        surface=read(case+'-surface-level2-zarray.json')
        label_arrays=[read(case+'-'+role+'-zarray.json') for role in ['inklabels','supervision_mask']]+[read(case+'-validation-zarray.json')]
        assert surface['shape']==attrs['source_shape_zyx'] and surface['chunks']==[109,128,128]
        assert surface['dtype']=='|u1' and surface['compressor'] is None and surface['dimension_separator']=='/'
        assert attrs['source_z_slice']==[13,97] and attrs['annotation_center_channel']==10
        assert all(a==label_arrays[0] for a in label_arrays)
        assert label_arrays[0]['shape']==[21,*surface['shape'][1:]] and label_arrays[0]['chunks']==[21,128,128]
        assert all(read(case+'-'+role+'-zattrs.json')==attrs for role in ['inklabels','supervision_mask'])
        inventory={};local=[]
        for role in ['inklabels','supervision_mask','validation_mask']:
            inv={}
            for p in sorted(ROOT.glob('source-*-'+case+'-'+role+'-inventory-*.json')):
                local.append(record(p))
                for obj in json.loads(p.read_text()):
                    n=obj['path'].split('/')[-1]
                    if not re.fullmatch(r'0\.\d+\.\d+',n):continue
                    _,y,x=map(int,n.split('.'));key=(y,x)
                    if key in inv:raise ValueError('Duplicate inventory coordinate')
                    inv[key]=obj
            h,w=surface['shape'][1:]
            assert set(inv)=={(y,x) for y in range((h+127)//128) for x in range((w+127)//128)}
            inventory[role]=inv
        proxies={role:{key for key,obj in inventory[role].items() if obj['size']!=78}
                 for role in ['validation_mask','inklabels']}
        ranked=[]
        for y in range(0,h-383,128):
            for x in range(0,w-383,128):
                keys={(yy,xx) for yy in range(y//128,(y+383)//128+1) for xx in range(x//128,(x+383)//128+1)}
                v=len(keys&proxies['validation_mask']);ink=len(keys&proxies['inklabels'])
                if v:ranked.append({'bbox_source_xywh':[x,y,384,384],
                    'potential_validation_chunk_count':v,'potential_ink_chunk_count':ink})
        ranked.sort(key=lambda c:(-c['potential_validation_chunk_count'],-c['potential_ink_chunk_count'],
                                  c['bbox_source_xywh'][1],c['bbox_source_xywh'][0]))
        ranked=ranked[:16]
        assert len(ranked)==16
        union=set();objects=[]
        for n,c in enumerate(ranked,1):
            c['rank']=n;x,y,wc,hc=c['bbox_source_xywh']
            keys=[(yy,xx) for yy in range(y//128,y//128+3) for xx in range(x//128,x//128+3)]
            c['label_object_keys']=[f'{role}/0.{yy}.{xx}' for role in ['inklabels','supervision_mask','validation_mask'] for yy,xx in keys]
            union.update((role,yy,xx) for role in ['inklabels','supervision_mask','validation_mask'] for yy,xx in keys)
        for role,y,x in sorted(union):
            obj=inventory[role][(y,x)]
            objects.append({'key':f'{role}/0.{y}.{x}',
                'url':'https://huggingface.co/buckets/scrollprize/datasets/resolve/'+obj['path'],
                'expected_byte_length':obj['size'],'expected_etag':obj['xetHash'],'chunk_yx':[y,x],'role':role})
        local += [record(metadata(case+'-'+name)) for name in [
            'inklabels-zattrs.json','supervision_mask-zattrs.json','validation_mask-zattrs.json',
            'inklabels-zarray.json','supervision_mask-zarray.json','validation-zarray.json',
            'surface-zattrs.json','surface-level2-zarray.json']]
        plan['input_records']+=local
        source=attrs['source_surface_volume'].replace('s3.amazonaws.com','s3.us-east-1.amazonaws.com')
        plan['cases'][case]={'specimen':source.split('/')[3],
            'public_segment':source.split('/')[5],'source_surface_url':source,
            'source_shape_zyx':surface['shape'],'source_level':2,'label_shape_zyx':label_arrays[0]['shape'],
            'source_depth_slice':[13,97],'pooled_shape_depth':21,'model_layer_slice':[2,19],
            'spacing_zyx_um':[9.596]*3,'label_zarray_record':record(metadata(case+'-validation-zarray.json')),
            'source_metadata_records':[record(metadata(case+'-surface-zattrs.json')),record(metadata(case+'-surface-level2-zarray.json'))],
            'candidates':ranked,'objects':objects,'label_inventory_chunk_counts':{role:len(inv) for role,inv in inventory.items()},
            'historical_overlap':'unknown; online validation/training universe inclusion known',
            'author_all_seven_pseudolabel_model_overlap_px':2924 if case=='pherc0139-w016' else 0,
            'author_overlap_scope':'Author-reported all-seven pseudo-label model, not one of this panel\'s four arms. Actual historical exposure for each selected model remains unknown.'}
    plan['maximum_http_request_attempts']=6*sum(len(c['objects']) for c in plan['cases'].values())
    plan['maximum_unique_label_objects']=sum(len(c['objects']) for c in plan['cases'].values())
    plan['expected_complete_candidate_label_payload_bytes']=sum(o['expected_byte_length'] for c in plan['cases'].values() for o in c['objects'])
    target=ROOT/'bounded_http.py'
    if target.exists():raise ValueError('Refuse overwrite of helper')
    shutil.copyfile(OLD_HTTP,target);assert sha(target)==sha(OLD_HTTP)
    dump('label-selection-plan.proposal.json',plan)
    panel={'created_at_utc':created,'status':'metadata_only_proposed_panel_not_executed',
        'development':{'case':'pherc1667-w029','specimen':'PHerc1667','reuse_existing_results':True,
                       'bbox_source_xywh':[6912,3200,256,256],'provenance_record':record(LAB/'reports/sixth-experiment/data-contract/selection-v2/provenance.json')},
        'reserved_from_our_tuning':{case:{k:v for k,v in value.items() if k!='objects'} for case,value in plan['cases'].items()},
        'available_curated_representations':rows,'retained_evidence_records':refs,
        'source_identity_pin':'metadata/plaintextSHAandpublic object ETag; mutablebucketnotGitversion',
        'official_inference_source_commit':'e0bbb8b40a2db58b1d71864f286eb85717e59e64',
        'russo_author_source_commit':'dd5bc8fb300c810c6ec042cb7e068be78829d086',
        'label_lineage':'Public transferred manual-plus-iterative-pseudo reference annotations; per-stroke/complete teacher lineage unverified.',
        'physical_separation':'PHerc1667,PHerc0139,PHerc0814 are different physical specimens. Within-case entire 2D contexts will not intersect; native CT/mesh coincidence has not been checked.',
        'native_secondary':'Five native9.362um PHerc0139 references are actual coarse acquisition, not pooled fine. Samepublicsegment aliases ofalignedfine maps are same physical groups; nativew044==alignedpherc0139-w028. No supplied native validation masks; all models include them.',
        'metadata_budget':{'body_bytes':json.loads((ROOT/'metadata-ledger.json').read_text())['body_bytes'],
                          'http_requests':len(json.loads((ROOT/'metadata-ledger.json').read_text())['requests']),
                          'cap_bytes':8*1024*1024,'cap_requests':80}}
    dump('candidate-panel.json',panel)
    print(json.dumps({'candidate_firsts':{k:v['candidates'][0]['bbox_source_xywh'] for k,v in plan['cases'].items()},
                      'max_unique_labels':plan['maximum_unique_label_objects'],'expected_labels':plan['expected_complete_candidate_label_payload_bytes'],
                      'requests_cap':plan['maximum_http_request_attempts']}))

if __name__=='__main__':main()
