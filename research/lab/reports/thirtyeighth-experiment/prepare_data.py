"""Model-free audit and fixed class-exposure-matched coverage schedule."""
import copy,platform
import numpy as np
import numcodecs
from common import *

def main():
    out=HERE/'data';out.mkdir(exist_ok=False)
    prior=REPORTS/'thirtyseventh-experiment';p=json.loads((prior/'PLAN.json').read_text())
    for r in json.loads((prior/'GO.json').read_text())['files']:verify(r)
    base=json.loads(verify(p['training_plan']).read_text());old=legacy()
    import torch
    require(platform.python_version()=='3.12.14' and np.__version__=='2.5.3' and torch.__version__=='2.12.0+cpu','Runtime matches historical control')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    helper=module(verify(base['helper']),'phase38_prepare_helper');runtime=helper.load_runtime()
    src=REPORTS.parent/'data/experiments/known-ink-control'
    provenance=json.loads((src/'provenance.frozen.json').read_text())
    require(provenance['bbox_source_xywh']==[3968,1408,256,256] and provenance['source_depth_slice']==[13,97],'Retained PHerc1667 identity')
    require(provenance['sample_id']=='PHerc1667','Second physical training parent')
    attrs=json.loads((src/'supervision.zattrs.json').read_text());la=json.loads((src/'inklabels.level0.zarray.json').read_text());sa=json.loads((src/'surface.level2.zarray.json').read_text())
    require(attrs==json.loads((src/'inklabels.zattrs.json').read_text()),'Same label/support contract')
    require(attrs['online_validation'] is False and attrs['source_z_slice']==[13,97] and attrs['annotation_center_channel']==10 and attrs['source_surface_volume_level']==2,'Declared w013 training-only supplied masks')
    require(attrs['case']=='pherc1667-w013' and attrs['output_resolution_um_zyx']==[9.596]*3,'Matched physical scale')
    require(la['chunks']==[21,128,128] and sa['chunks']==[109,128,128] and sa['dtype']=='|u1' and sa['dimension_separator']=='.','Raw decoding contract')
    surface=np.load(src/'surface.npy',allow_pickle=False);labels=np.load(src/'inklabels.npy',allow_pickle=False);support=np.load(src/'supervision-mask.npy',allow_pickle=False)>0
    for name in ['surface.npy','inklabels.npy','supervision-mask.npy']:
        require(pin(src/name)['sha256']==provenance['outputs'][name]['sha256'],'Prior assembled array identity')
    decoded_surface=np.empty((21,256,256),np.uint8);decoded_labels=np.empty((256,256),np.uint8);decoded_support=np.empty((256,256),np.uint8);rawpins=[]
    records={r['local_file']:r for r in provenance['source_records']}
    for yy in [11,12]:
        for xx in [31,32]:
            sl=(slice((yy-11)*128,(yy-10)*128),slice((xx-31)*128,(xx-30)*128))
            for role in ['surface','inklabels','supervision']:
                path=src/f'{role}-chunk-0-{yy}-{xx}.bin';rec=records[path.name];actual=pin(path)
                require(rec['status']==200 and actual['sha256']==rec['sha256'] and actual['size_bytes']==rec['size_bytes'],'Actually delivered raw chunk identity')
                rawpins.append(actual)
                if role=='surface':
                    body=path.read_bytes();decoded=numcodecs.get_codec(sa['compressor']).decode(body) if sa['compressor'] else body
                    a=np.frombuffer(decoded,dtype=np.uint8).reshape(109,128,128)
                    pooled=np.stack([np.rint(a[z:z+4].astype(np.float32).mean(0)).astype(np.uint8) for z in range(13,97,4)])
                    decoded_surface[(slice(None),*sl)]=pooled
                else:
                    a=np.frombuffer(numcodecs.get_codec(la['compressor']).decode(path.read_bytes()),dtype=np.uint8).reshape(21,128,128)
                    require(not np.any(np.delete(a,10,axis=0)) and set(np.unique(a))<={0,1},'Only central binary labels')
                    (decoded_labels if role=='inklabels' else decoded_support)[sl]=a[10]
    require(np.array_equal(decoded_surface,surface) and np.array_equal(decoded_labels,labels) and np.array_equal(decoded_support>0,support) and support.all(),'Independent source pooling and masks exact')
    old_w029=REPORTS/'sixth-experiment/data-contract/selection-v2/supervision-mask.npy'
    require(not np.load(old_w029,allow_pickle=False).any(),'Cached w029 rejected: no supplied training supervision')
    regions=[old.training_region(c) for c in base['adaptation']];schedule=json.loads(verify(base['schedule']).read_text());hist=[json.loads(l) for l in verify(base['historical']['A_preserve']['steps']).read_text().splitlines()]
    original=[];mixed=[];specs=[];use={};unique=[np.zeros((384,384),bool),np.zeros((384,384),bool),np.zeros((256,256),bool)]
    # Same global coordinate ranking across all patches: no model-dependent or per-cycle resampling.
    ranks=np.array([int.from_bytes(hashlib.sha256(f'phase38-support-v1:{y}:{x}'.encode()).digest()[:8],'big') for y in range(256) for x in range(256)],dtype=np.uint64).reshape(256,256)
    tiles=[(x,y) for y in range(0,129,32) for x in range(0,129,32)]
    for i,step in enumerate(schedule):
        b,identity,exposures=old.batch(step,regions,runtime,torch);require(identity==hist[i]['input_target_sha256'],'Original batch identity')
        x,m,t=(v.numpy().copy() for v in b[:3]);original.append((x.copy(),m.copy(),t.copy()))
        slot=i%2;other=1-slot;removed=step['patches'][slot];pos=int(((t[slot]==1)&m[slot]).sum());neg=int(((t[slot]==0)&m[slot]).sum())
        eligible=[]
        for xx,yy in tiles:
            yylabels=labels[yy:yy+128,xx:xx+128];mm=support[yy:yy+128,xx:xx+128]
            if int((mm&(yylabels==1)).sum())>=pos and int((mm&(yylabels==0)).sum())>=neg:eligible.append((xx,yy))
        require(eligible,'Every original class-count requirement has an eligible replacement')
        xx,yy=min(eligible,key=lambda c:(use.get(c,0),c[1],c[0]));use[(xx,yy)]=use.get((xx,yy),0)+1
        yylabels=labels[yy:yy+128,xx:xx+128];mm=np.zeros((128,128),bool);rank=ranks[yy:yy+128,xx:xx+128]
        for value,count in [(1,pos),(0,neg)]:
            indices=np.flatnonzero((yylabels==value)&support[yy:yy+128,xx:xx+128]);chosen=indices[np.argsort(rank.ravel()[indices],kind='stable')[:count]];mm.ravel()[chosen]=True
        target=np.full((128,128),-1.,np.float32);target[mm]=yylabels[mm]
        code=removed['augmentation'];image=np.ascontiguousarray(runtime['normalize_robust'](surface[2:19,yy:yy+128,xx:xx+128]),dtype=np.float32)
        x[slot]=old.augment(image,code)[None];m[slot]=old.augment(mm,code)[None];t[slot]=old.augment(target,code)[None]
        require(all(np.array_equal(v[other],o[other]) for v,o in zip((x,m,t),original[-1])),'Retained PHerc0139 patch exact')
        require(int((m&(t==1)).sum())==sum(e['manual']['positive'] for e in exposures) and int((m&(t==0)).sum())==sum(e['manual']['negative'] for e in exposures),'Per-update positive/negative counts exactly matched')
        require(np.isfinite(x).all() and set(np.unique(t[m]))=={0.,1.},'Finite normalized input and binary selected labels')
        kept=step['patches'][other];kx,ky,kw,kh=kept['bbox'];unique[kept['region']][ky:ky+kh,kx:kx+kw]|=regions[kept['region']][1][ky:ky+kh,kx:kx+kw];unique[2][yy:yy+128,xx:xx+128]|=mm
        mixed.append((x,m,t));specs.append(dict(step=i+1,replaced_slot=slot,original_patch=removed,new_bbox=[xx,yy,128,128],augmentation=code,positive=pos,negative=neg,eligible_tiles=len(eligible),original_batch_fingerprint=identity))
    for kind,rows in [('original',original),('mixed',mixed)]:
        for index,name in enumerate(['inputs','mask','target']):np.save(out/f'{kind}-{name}.npy',np.stack([b[index] for b in rows]),allow_pickle=False)
    schedule_record=dict(rule='Alternate replaced slot; least-used eligible stride32 tile, tie source Y/X; deterministic SHA256 global-coordinate support order within each class',steps=specs,tile_use=[dict(x=x,y=y,count=n) for (x,y),n in sorted(use.items())],unique_manual_pixels_by_region=[int(a.sum()) for a in unique])
    save(out/'schedule.json',schedule_record)
    audit=dict(status='PASS_OPERATIONAL_PUBLIC_REFERENCE_SUPERVISION',created_at_utc=now(),accepted_parent='PHerc1667',accepted_case='pherc1667-w013',accepted_bbox=[3968,1408,256,256],new_parent_positive=int(labels[support].sum()),new_parent_negative=int((labels[support]==0).sum()),source_pooling_exact=True,label_decode_exact=True,source_chunks_delivered=4,raw_source_records=rawpins,validation_policy='w013 online_validation=false; no public validation array by dataset contract; w016 validation remains excluded',cached_w029_rejected_no_supervision=True,supervision_per_step_class_counts_matched=True,unique_manual_pixels_by_region=schedule_record['unique_manual_pixels_by_region'],per_cycle_positive=sum(int((b[1]&(b[2]==1)).sum()) for b in mixed),per_cycle_negative=sum(int((b[1]&(b[2]==0)).sum()) for b in mixed),new_model_forwards=0,network_requests=0,annotation_ancestry='manual-plus-iterative public reference; not authenticated human-only truth',no_new_geometry_or_optical_registration_claim=True,inputs=[pin(src/n) for n in ['provenance.frozen.json','inklabels.zattrs.json','supervision.zattrs.json','inklabels.level0.zarray.json','supervision.level0.zarray.json','surface.level2.zarray.json','surface.zattrs.json','surface.npy','inklabels.npy','supervision-mask.npy','upstream-dataset-README.md']]+[pin(old_w029)])
    save(HERE/'DATA_AUDIT.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['raw_source_records','inputs']}))
if __name__=='__main__':main()
