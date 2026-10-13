"""Read-only verifier for the sealed supervision panel and patch identities."""
import argparse,json
from pathlib import Path
import numpy as np
import intake


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.work;intake.verify_freeze(root)
    selection=intake.load(root/'SELECTION.json');inputs=intake.load(root/'INPUTS_COMPLETE.json');patches=intake.load(root/'PATCHES_COMPLETE.json')
    assert patches['selection_sha256']==inputs['selection_sha256']==intake.sha(root/'SELECTION.json')
    assert patches['input_receipt_sha256']==intake.sha(root/'INPUTS_COMPLETE.json')
    assert patches['model_forwards']==inputs['model_forwards']==0
    rows=[];files=0;hybrid=hecate=0
    for c in selection['cases']:
        native,known=intake.source_canvas(root,c);d,h,w=native.shape
        with np.load(root/'support'/(c['id']+'.npz')) as z:
            s=z['supervision'];assert np.array_equal(s,z['participation'])
        ic=next(x for x in inputs['cases'] if x['id']==c['id']);pc=next(x for x in patches['cases'] if x['id']==c['id'])
        assert len(c['cores'])==len(ic['cores'])==6
        assert pc['hybrid_layers']==list(range(d//2-8,d//2+9))
        total=pos=neg=defined=0
        for core,record in zip(c['cores'],ic['cores']):
            assert core['id']==record['id'];x,y,cw,ch=core['core_xywh']
            assert x%128==y%128==0 and cw==ch==128 and int(s[y:y+128,x:x+128].sum())>=4096
            q=np.load(root/'cores'/(core['id']+'.npz'))
            assert np.array_equal(q['origin_xy'],[x,y])
            assert np.array_equal(q['participation'],s[y:y+128,x:x+128])
            valid=native[d//2,y:y+128,x:x+128]>0
            assert np.array_equal(q['native_centre_nonzero'],valid)
            score=s[y:y+128,x:x+128]&valid
            assert np.array_equal(q['score_mask'],score) and intake.ash(score)==record['score_mask_sha256']
            assert intake.ash(q['ink'])==record['ink_sha256']
            np_,nn_=int((score&q['ink']).sum()),int((score&~q['ink']).sum())
            assert np_==record['positive_pixels'] and nn_==record['negative_pixels'] and np_+nn_==record['score_pixels']
            total+=np_+nn_;pos+=np_;neg+=nn_;defined+=int(np_>0 and nn_>0)
        assert pos>0 and neg>0,'Segment cannot support AP/AUC ranking'
        for item in pc['patches']:
            path=root/item['path'];assert intake.pin(path)=={k:item[k] for k in ['bytes','sha256']};files+=1
            v=np.load(path,allow_pickle=False);assert v.dtype==np.uint8 and intake.ash(v)==item['decoded_sha256']
            assert bool(v.any())==item['raw_nonempty'];y,x=item['origin_yx']
            if item['family']=='hybrid':
                assert v.shape==(17,128,128)
                assert np.array_equal(v,native[d//2-8:d//2+9,y:y+128,x:x+128])
                assert known[y:y+128,x:x+128].all();hybrid+=1
            else:
                assert item['family']=='hecate' and v.shape==(16,64,64);hecate+=1
        rows.append(dict(id=c['id'],score_pixels=total,positive_pixels=pos,negative_pixels=neg,
                         two_class_cores=defined,single_class_cores=6-defined,positive_fraction=pos/total,
                         source_context_verified=True))
        del native,known
    result={'status':'PASS','cases':rows,'core_count':24,'hybrid_patches':hybrid,'hecate_patches':hecate,
            'patch_files_verified':files,'score_pixels':sum(r['score_pixels'] for r in rows),
            'positive_pixels':sum(r['positive_pixels'] for r in rows),'all_segments_two_classes':True,
            'model_forwards':0,'hecate_scope':'Patch hashes + earlier full-canvas footprint and synthetic preparation parity, not an independent real-CT interpolation recomputation',
            'claim':'Read-only source/score/patch verification; not accuracy, registration or neural-output parity'}
    intake.write(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
