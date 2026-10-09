"""Read only eight PHerc0139 Boolean/participation arrays; never infer CT geometry validity."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import socket
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT.parents[1]
TRAINING=REPORTS/'thirteenth-experiment/implementation/final-plan/training-plan.json'
PLAN_SHA='22d083c2359c1d9e55e6f5c806e087340ec550fbb6bc11de73b093d124f88766'
PIN_NAMES=['acquisition_observed_mask','supervision_mask','validation_mask','source_observed_nonzero_mask']

def require(ok,message):
    if not ok:raise ValueError(message)

def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),sha256=digest,size_bytes=path.stat().st_size)

def verify(pin):
    observed=record(pin['path'])
    require(observed['sha256']==pin['sha256'] and observed['size_bytes']==pin.get('size_bytes',pin.get('byte_length')),'Source identity differs')
    return Path(pin['path'])

def read_json(pin):return json.loads(verify(pin).read_text())

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def augment(a,code):
    require(type(code)is int and 0<=code<8,'D4 code invalid')
    b=np.rot90(a,code%4,axes=(-2,-1))
    if code>=4:b=np.flip(b,axis=-1)
    return np.ascontiguousarray(b)

def array_record(path,a):
    require(not path.exists(),'Refuse overwrite')
    np.save(path,a,allow_pickle=False)
    return dict(**record(path),shape=list(a.shape),dtype=str(a.dtype),raw_sha256=hashlib.sha256(a.tobytes()).hexdigest())

def hist(values):
    keys,counts=np.unique(values,return_counts=True)
    return {str(int(k)):int(v) for k,v in zip(keys,counts)}

def summaries(masks,counts):
    union=counts>0
    return {name:dict(whole_context_pixels=int(mask.sum()),unique_exposed_pixels=int((mask&union).sum()),
        repeated_output_exposures=int(counts[mask].sum()),excess_repeats_beyond_first=int(counts[mask].sum())-int((mask&union).sum()),
        exposure_multiplicity_histogram=hist(counts[mask]),
        exposure_multiplicity_histogram_exposed_only=hist(counts[mask&union])) for name,mask in masks.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(args.output.resolve().parent==ROOT and not args.output.exists(),'Only fresh owned output permitted')
    started=time.monotonic();signal.alarm(60);resource.setrlimit(resource.RLIMIT_CPU,(180,180))
    def offline(*a,**k):raise RuntimeError('No network permitted')
    socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
    planpin=record(TRAINING)
    plan=json.loads(TRAINING.read_text());cases=plan['adaptation']
    require([c['id'] for c in cases]==['pherc0139-w016-rank01','pherc0139-w016-rank02'],'Only two exact training contexts')
    schedule=read_json(plan['schedule']);require([x['step'] for x in schedule]==list(range(1,33)),'Exactly32 batches')
    histories={k:[json.loads(x) for x in verify(plan['historical'][k]['steps']).read_text().splitlines()] for k in ['A_preserve','B_preserve']}
    regions=[];sources=[planpin,plan['schedule']]+[plan['historical'][k]['steps'] for k in histories]
    for c in cases:
        require(c['physical_group']=='PHerc0139','Forbidden specimen')
        provenance=read_json(c['provenance']);sources.append(c['provenance'])
        require(provenance['geometry_validity']==dict(status='not_acquired',mask_file=None),'Unexpected geometry evidence; stop to review')
        a={}
        for key in PIN_NAMES:
            pin=c['inputs'][key];require('pherc0139-w016-rank' in pin['path'],'Forbidden scientific array path')
            value=np.load(verify(pin),allow_pickle=False)
            require(value.shape==(384,384) and value.dtype in [np.dtype('bool'),np.dtype('uint8')],'Unexpected mask shape/dtype')
            require(set(np.unique(value).tolist())<={0,1},'Unexpected mask encoding')
            a[key]=value>0;sources.append(pin)
        O=a['acquisition_observed_mask'];S=a['supervision_mask'];V=a['validation_mask']
        require(O.all(),'Missing local acquired source support')
        masks=dict(acquisition_observed=O,manual=O&S&~V,retired_validation=O&V,
                   annotation_unknown=O&~S&~V,outside_current_manual=O&~(S&~V),
                   raw_supervision_validation_overlap=S&V,nonzero_occupancy_diagnostic=a['source_observed_nonzero_mask'])
        require(not bool((masks['manual']&masks['retired_validation']).any()) and
                np.array_equal(masks['manual']|masks['retired_validation']|masks['annotation_unknown'],O), 'Participation partition differs')
        x,y,w,h=provenance['bbox_source_xywh'];shape=provenance['source_shape_zyx']
        require(w==h==384 and 0<=x and x+w<=shape[2] and 0<=y and y+h<=shape[1] and shape[0]==109,'Source bounds differ')
        expected={(0,yy,xx) for yy in range(y//128,y//128+3) for xx in range(x//128,x//128+3)}
        require({tuple(o['chunk_zyx']) for o in provenance['source_objects']}==expected and
                all(o['verified'] and o['byte_length']==109*128*128 for o in provenance['source_objects']), 'Original source chunk coverage receipt differs')
        regions.append(dict(case=c,provenance=provenance,masks=masks,counts=np.zeros((384,384),np.uint16),tile_frequency={},augmentation_frequency={}))
    batch={name:np.empty((32,2,1,128,128),np.bool_) for name in ['manual','acquisition_observed','retired_validation','annotation_unknown']}
    batch['source_index']=np.empty((32,2,1,128,128),np.int32)
    local_index=np.arange(384*384,dtype=np.int32).reshape(384,384);per_batch=[]
    for step in schedule:
        require([p['region'] for p in step['patches']]==[0,1],'Region order differs')
        rows=[]
        for patch in step['patches']:
            r=patch['region'];region=regions[r];x,y,w,h=patch['bbox'];code=patch['augmentation']
            require(w==h==128 and patch['bbox'] in plan['exposure'][r]['eligible_tiles'],'Scheduled tile differs')
            region['counts'][y:y+h,x:x+w]+=1
            tile=','.join(map(str,patch['bbox']));region['tile_frequency'][tile]=region['tile_frequency'].get(tile,0)+1
            region['augmentation_frequency'][str(code)]=region['augmentation_frequency'].get(str(code),0)+1
            mask_counts={}
            for name in batch:
                values=local_index if name=='source_index' else region['masks'][name]
                transformed=augment(values[y:y+h,x:x+w],code)
                batch[name][step['step']-1,r,0]=transformed
                if name!='source_index':mask_counts[name]=int(transformed.sum())
                require(np.array_equal(np.sort(transformed.ravel()),np.sort(values[y:y+h,x:x+w].ravel())),'D4 changed membership values')
            for history in histories.values():
                prior=history[step['step']-1]
                require(prior['step']==step['step'] and prior['patches']==step['patches'],'Historical schedule mismatch')
                manual_exposures=prior['exposures'][r]['manual']
                require(sum(manual_exposures.values())==mask_counts['manual'],'Historical manual participation changed')
            require(sum(mask_counts[k] for k in ['manual','retired_validation','annotation_unknown'])==128*128,'Batch participation partition differs')
            rows.append(dict(region=r,patch=patch,counts=mask_counts,manual_fraction=mask_counts['manual']/(128*128),
                             geometry_valid_pixels=None,geometry_invalid_pixels=None))
        per_batch.append(dict(step=step['step'],regions=rows,
                              counts={name:sum(row['counts'][name] for row in rows) for name in batch if name!='source_index'}))
    args.output.mkdir()
    outputs={name:array_record(args.output/('batch-'+name.replace('_','-')+'.npy'),array) for name,array in batch.items()}
    region_reports=[]
    for r,region in enumerate(regions):
        c=region['case'];stats=summaries(region['masks'],region['counts'])
        require(stats['manual']['unique_exposed_pixels']==plan['exposure'][r]['unique_manual_pixels'],'Prior unique manual count differs')
        countpin=array_record(args.output/(c['id']+'-exposure-counts.npy'),region['counts'])
        region_reports.append(dict(id=c['id'],bbox_source_xywh=region['provenance']['bbox_source_xywh'],statistics=stats,
            source_coverage=dict(original_chunks=9,all_selected_chunks_verified_in_retained_provenance=True,
                                 retained_chunk_payload_bytes=9*109*128*128,crop_entirely_within_source_canvas=True,
                                 local_missing_chunk_fill_pixels=0,local_crop_padding_pixels=0,
                                 source_renderer_unmapped_or_masked_pixels=None,authenticated_geometry_valid_pixels=None,
                                 authenticated_geometry_invalid_pixels=None),
            exposure_counts=countpin,tile_frequency=region['tile_frequency'],augmentation_frequency=region['augmentation_frequency']))
    categories=list(regions[0]['masks'])
    totals={name:{key:sum(rr['statistics'][name][key] for rr in region_reports) for key in
                  ['whole_context_pixels','unique_exposed_pixels','repeated_output_exposures','excess_repeats_beyond_first']} for name in categories}
    require(totals['acquisition_observed']['repeated_output_exposures']==32*2*128*128,'Exposure total differs')
    sources.extend(record(REPORTS/relative) for relative in [
        'seventh-experiment/panel-sources/assemble_panel_sources.py',
        'seventh-experiment/panel-sources/prepare_source_plan.py',
        'seventh-experiment/panel-sources/bounded_http.py',
        'seventh-experiment/panel-sources/source-acquisition-plan.json',
        'seventh-experiment/panel-sources/label-selection-plan.proposal.json',
        'eighth-experiment/implementation/experiment_common.py',
        'eighth-experiment/implementation/training_mechanics.py',
        'twentieth-experiment/implementation/depth_data.py',
        'twentieth-experiment/implementation/depth_adapter.py',
        'twentieth-experiment/NEXT_STUDY_DRAFT.md'])
    panel=REPORTS/'seventh-experiment/panel-sources'
    for suffix in ['surface-zattrs.json','surface-level2-zarray.json','supervision_mask-zattrs.json','validation_mask-zattrs.json']:
        paths=list(panel.glob('source-*-pherc0139-w016-'+suffix));require(len(paths)==1,'Metadata pin ambiguous');sources.append(record(paths[0]))
    for source in sources:verify(source)
    result=dict(status='PASS',kind='phase21-training-participation-audit-v1',
        broader_valid_support_ready=False,geometry_validity_status='not_acquired',
        authenticated_geometry_valid_pixels=None,authenticated_geometry_invalid_pixels=None,
        machine_readable_valid_support_mask=None,existing_manual_support_reproduced=True,
        batch_arrays=outputs,regions=region_reports,totals=totals,per_batch=per_batch,
        source_index_encoding='Local y*384+x within each region; second axis selects rank01/rank02; exact crop then D4 transform',
        array_semantics=dict(manual='O & S & ~V; existing supplied supervision participation only, not geometry authentication',
            acquisition_observed='O: bytes acquired from source chunks; not geometry validity',
            retired_validation='O & V: retired annotation membership, no retired ink target read',
            annotation_unknown='O & ~S & ~V: no participation established; zeros are not background'),
        masks_read=8,inklabel_arrays_read=0,ct_arrays_decoded=0,teacher_or_prediction_arrays_read=0,
        checkpoint_loads=0,model_calls=0,optimization_steps=0,network_calls=0,paris_scientific_array_reads=0,
        current_source_file=record(Path(__file__)),source_pins=sources,
        wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        limitations=['Transport completeness excludes local missing-chunk fill, not upstream masked/unmapped voxels.',
                     'All geometry-valid and geometry-invalid counts remain unknown, never zero by assumption.',
                     'Supplied supervision/validation masks and mapping-policy prose are not per-pixel mapping validity.',
                     'Retired-validation and annotation-unknown outputs remain separate from manual support.',
                     'No semantic ink target, score, teacher or CT intensity selected any support.'])
    write(args.output/'support-contract.json',result)
    print(json.dumps(dict(status='PASS',contract=record(args.output/'support-contract.json'),totals=totals,
                          broader_valid_support_ready=False)))

if __name__=='__main__':main()
