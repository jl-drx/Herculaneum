"""Reconstruct selected support and geometry from the frozen raw-derived crop."""
import numpy as np
from common import *
def main():
    src=REPORTS.parent/'data/experiments/known-ink-control';labels=np.load(src/'inklabels.npy');support=np.load(src/'supervision-mask.npy')>0;surface=np.load(src/'surface.npy')
    old=legacy();p=json.loads((REPORTS/'thirtyseventh-experiment/PLAN.json').read_text());base=json.loads(verify(p['training_plan']).read_text());helper=module(verify(base['helper']),'phase38_data_check_helper');runtime=helper.load_runtime()
    a={k:np.load(HERE/f'data/mixed-{k}.npy',mmap_mode='r',allow_pickle=False) for k in ['inputs','mask','target']};spec=json.loads((HERE/'data/schedule.json').read_text());unique=np.zeros((256,256),bool);uses={};spatial=[]
    for i,s in enumerate(spec['steps']):
        slot=s['replaced_slot'];x,y,w,h=s['new_bbox'];m=np.zeros((128,128),bool)
        candidates=[]
        for yy in range(0,129,32):
            for xx in range(0,129,32):
                z=labels[yy:yy+128,xx:xx+128];mm=support[yy:yy+128,xx:xx+128]
                if int((mm&(z==1)).sum())>=s['positive'] and int((mm&(z==0)).sum())>=s['negative']:candidates.append((xx,yy))
        require((x,y)==min(candidates,key=lambda c:(uses.get(c,0),c[1],c[0])),'Independent least-used eligible tile');uses[(x,y)]=uses.get((x,y),0)+1
        for label,count in [(1,s['positive']),(0,s['negative'])]:
            indices=[(yy,xx) for yy in range(128) for xx in range(128) if support[y+yy,x+xx] and labels[y+yy,x+xx]==label]
            indices.sort(key=lambda c:(hashlib.sha256(f'phase38-support-v1:{y+c[0]}:{x+c[1]}'.encode()).digest()[:8],c[0],c[1]))
            for yy,xx in indices[:count]:m[yy,xx]=True
        target=np.full((128,128),-1,np.float32);target[m]=labels[y:y+128,x:x+128][m]
        require(np.array_equal(old.augment(m,s['augmentation']),a['mask'][i,slot,0]) and np.array_equal(old.augment(target,s['augmentation']),a['target'][i,slot,0]),'Independent support/target coordinate reconstruction')
        normalized=np.ascontiguousarray(runtime['normalize_robust'](surface[2:19,y:y+128,x:x+128]),dtype=np.float32)
        require(np.array_equal(old.augment(normalized,s['augmentation']),a['inputs'][i,slot,0]),'Input crop/augmentation/normalizer exact, independent of supervision')
        unique[y:y+128,x:x+128]|=m
    require(int(unique.sum())==55313 and len(spec['steps'])==32,'Expanded unique support')
    save(HERE/'DATA_VERIFICATION.json',dict(status='PASS',replacement_patches=32,exact_class_exposure=True,source_coordinate_reconstruction=True,augmentation_and_normalization_verified=True,unique_added_supervision_pixels=int(unique.sum()),new_model_forwards=0,new_optimizer_updates=0,source=pin(__file__)))
    print(json.dumps(dict(status='PASS',reconstructed_patches=32,unique_added_supervision_pixels=int(unique.sum()))))
if __name__=='__main__':main()
