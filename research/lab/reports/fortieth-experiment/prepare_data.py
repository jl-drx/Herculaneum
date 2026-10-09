"""Reconstruct one admitted crop and repeat the unchanged phase38 mixing rule."""
import numpy as np
import numcodecs
from common import *

SOURCE=HERE/'admission-stage2/acquisition'
def main():
    out=HERE/'data';out.mkdir(exist_ok=False)
    receipt=json.loads((SOURCE/'COMPLETE.json').read_text());require(receipt['status']=='PASS','Acquisition admission')
    for r in receipt['records']:verify({k:r[k] for k in ['path','sha256','size_bytes']})
    surface=np.load(verify(receipt['arrays']['surface']),allow_pickle=False);labels=np.load(verify(receipt['arrays']['inklabels']),allow_pickle=False);support=np.load(verify(receipt['arrays']['supervision-mask']),allow_pickle=False)>0
    bx,by,_,_=receipt['selected']['bbox_source_xywh']
    require(bx+256<=3968-256 or bx>=3968+512 or by+256<=1408-256 or by>=1408+512,'Same-surface coordinate separation')
    sa=json.loads((SOURCE/'surface.level2.zarray.json').read_text());la=json.loads((SOURCE/'inklabels.level0.zarray.json').read_text())
    raw=np.empty((109,256,256),np.uint8);ys=np.empty((256,256),np.uint8);ms=np.empty_like(ys)
    for dy in range(2):
        for dx in range(2):
            target=(slice(128*dy,128*(dy+1)),slice(128*dx,128*(dx+1)))
            for role,a in [('surface',raw),('inklabels',ys),('supervision',ms)]:
                spec=sa if role=='surface' else la;body=(SOURCE/f'{role}-0-{by//128+dy}-{bx//128+dx}.bin').read_bytes()
                decoded=numcodecs.get_codec(spec['compressor']).decode(body) if spec['compressor'] else body
                z=np.frombuffer(decoded,np.uint8).reshape(spec['chunks'])
                if role=='surface':a[(slice(None),*target)]=z
                else:
                    require(not np.delete(z,10,axis=0).any() and np.isin(z,[0,1]).all(),'Central binary labels')
                    a[target]=z[10]
    independent=np.rint(raw[13:97].reshape(21,4,256,256).sum(axis=1,dtype=np.uint16)/4).astype(np.uint8)
    require(np.array_equal(independent,surface) and np.array_equal(ys,labels) and np.array_equal(ms>0,support),'Independent raw pooling/label/support reconstruction')
    require(np.any(surface,axis=0)[support].all(),'Every supplied training pixel has observed nonzero CT; no mask shrink')
    prior=json.loads((REPORTS/'thirtyeighth-experiment/PLAN-V2.json').read_text());base=json.loads(verify(prior['training_plan']).read_text());old=legacy()
    helper=module(verify(base['helper']),'phase40_data_helper');runtime=helper.load_runtime()
    arrays={k:np.load(verify(prior['batch_arrays']['original'][k]),allow_pickle=False) for k in ['inputs','mask','target']}
    original={k:v.copy() for k,v in arrays.items()};regions=[old.training_region(c) for c in base['adaptation']]
    schedule=json.loads(verify(base['schedule']).read_text());prior_specs=json.loads(verify(prior['schedule38']).read_text())['steps']
    ranks=np.array([int.from_bytes(hashlib.sha256(f'phase38-support-v1:{y}:{x}'.encode()).digest()[:8],'big') for y in range(256) for x in range(256)],dtype=np.uint64).reshape(256,256)
    use={};specs=[];unique=[np.zeros((384,384),bool),np.zeros((384,384),bool),np.zeros((256,256),bool)]
    for i,step in enumerate(schedule):
        slot=i%2;other=1-slot;removed=step['patches'][slot];pos=int((original['mask'][i,slot]&(original['target'][i,slot]==1)).sum());neg=int((original['mask'][i,slot]&(original['target'][i,slot]==0)).sum())
        require((pos,neg)==(prior_specs[i]['positive'],prior_specs[i]['negative']),'Unchanged replacement requirements')
        eligible=[]
        for yy in range(0,129,32):
            for xx in range(0,129,32):
                y=labels[yy:yy+128,xx:xx+128];m=support[yy:yy+128,xx:xx+128]
                if int((m&(y==1)).sum())>=pos and int((m&(y==0)).sum())>=neg:eligible.append((xx,yy))
        require(eligible,'All replacements class-feasible');xx,yy=min(eligible,key=lambda v:(use.get(v,0),v[1],v[0]));use[(xx,yy)]=use.get((xx,yy),0)+1
        y=labels[yy:yy+128,xx:xx+128];rank=ranks[yy:yy+128,xx:xx+128];m=np.zeros((128,128),bool)
        for value,count in [(1,pos),(0,neg)]:
            index=np.flatnonzero((y==value)&support[yy:yy+128,xx:xx+128]);selected=index[np.argsort(rank.ravel()[index],kind='stable')[:count]];m.ravel()[selected]=True
        target=np.full((128,128),-1,np.float32);target[m]=y[m];code=removed['augmentation'];image=np.ascontiguousarray(runtime['normalize_robust'](surface[2:19,yy:yy+128,xx:xx+128]),dtype=np.float32)
        for name,a in [('inputs',image),('mask',m),('target',target)]:arrays[name][i,slot]=old.augment(a,code)[None]
        require(all(np.array_equal(arrays[k][i,other],original[k][i,other]) for k in arrays),'Retained original patch bitwise unchanged')
        for value in [0,1]:require(int((arrays['mask'][i]&(arrays['target'][i]==value)).sum())==int((original['mask'][i]&(original['target'][i]==value)).sum()),'Exact per-update class exposure')
        kept=step['patches'][other];kx,ky,kw,kh=kept['bbox'];unique[kept['region']][ky:ky+kh,kx:kx+kw]|=regions[kept['region']][1][ky:ky+kh,kx:kx+kw];unique[2][yy:yy+128,xx:xx+128]|=m
        specs.append(dict(step=i+1,replaced_slot=slot,original_patch=removed,new_bbox=[xx,yy,128,128],augmentation=code,positive=pos,negative=neg,eligible_tiles=len(eligible),original_batch_fingerprint=prior_specs[i]['original_batch_fingerprint']))
    for kind,values in [('original',original),('mixed',arrays)]:
        for name,a in values.items():np.save(out/f'{kind}-{name}.npy',a,allow_pickle=False)
    schedule_record=dict(rule='Identical phase38 alternate-slot/least-use tile and crop-global SHA256 class selection',steps=specs,tile_use=[dict(x=x,y=y,count=n) for (x,y),n in sorted(use.items())],unique_manual_pixels_by_region=[int(v.sum()) for v in unique])
    save(out/'schedule.json',schedule_record)
    audit=dict(status='PASS_OPERATIONAL_PUBLIC_REFERENCE_SUPERVISION',accepted_parent='PHerc1667',accepted_case='pherc1667-w013',accepted_bbox=[bx,by,256,256],spatial_replication_within_same_surface=True,minimum_axis_gap_from_original=256,old_crop_used_in_replication=False,source_pooling_exact=True,label_decode_exact=True,per_update_class_counts_matched=True,unique_manual_pixels_by_region=schedule_record['unique_manual_pixels_by_region'],per_cycle_positive=int((arrays['mask']&(arrays['target']==1)).sum()),per_cycle_negative=int((arrays['mask']&(arrays['target']==0)).sum()),source=pin(SOURCE/'COMPLETE.json'),source_arrays=receipt['arrays'],new_model_forwards=0,annotation_ancestry='supplied manual-plus-iterative; incomplete human/teacher ancestry',source_specific_original_exposure_matched=False,validation_mask_promoted=False,independent_ink_accuracy=False)
    require(audit['per_cycle_positive']==230340 and audit['per_cycle_negative']==387868 and audit['unique_manual_pixels_by_region'][:2]==[25967,28734],'Same cycle exposure and original support')
    save(HERE/'DATA_AUDIT.json',audit);print(json.dumps(audit))
if __name__=='__main__':main()
