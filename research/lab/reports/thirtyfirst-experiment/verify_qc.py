"""Separate root count/support calculation from Pillow crops, no QC-helper import."""
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json,resource,time
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent

def main():
    started=time.monotonic()
    run=HERE/'qc-v1';summary=json.loads((run/'QC_SUMMARY.json').read_text())
    panel=json.loads((run/'PANEL.json').read_text());table=json.loads((run/'TILE_TABLE.json').read_text())
    for obj in panel['source_objects'].values():
        p=Path(obj['path']);assert p.stat().st_size==obj['size_bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==obj['sha256']
    with Image.open(HERE/'acquisition-v1/mask.png') as mask, Image.open(HERE/'acquisition-v1/inklabels.png') as lab, Image.open(HERE/'acquisition-v1/ir.png') as ir:
        assert mask.mode=='L' and lab.mode=='P' and ir.mode=='RGBA'
        assert lab.getpalette()==[0,0,0,255,255,255] and 'transparency' not in lab.info
        mh,lh=mask.histogram(),lab.histogram()
        assert sum(mh[1:255])==0 and sum(lh[2:])==0
        assert [mh[0],mh[255],0]==summary['mask_counts']
        assert [lh[0],lh[1],0]==summary['label_counts']
        joint=np.zeros((2,2),dtype=np.int64);alpha_hist=np.zeros(256,dtype=np.int64)
        nonopaque_supported=chromatic=0
        for y in range(0,8853,97):
            box=(0,y,6205,min(y+97,8853))
            m=np.asarray(mask.crop(box));l=np.asarray(lab.crop(box));o=np.asarray(ir.crop(box))
            for mi,raw in enumerate((0,255)):
                for li in (0,1):joint[mi,li]+=np.count_nonzero((m==raw)&(l==li))
            alpha_hist+=np.bincount(o[...,3].ravel(),minlength=256)
            nonopaque_supported+=int(np.count_nonzero((m==255)&(o[...,3]!=255)))
            chromatic+=int(np.count_nonzero((o[...,0]!=o[...,1])|(o[...,0]!=o[...,2])))
        assert joint.tolist()==[r[:2] for r in summary['joint_counts_mask_row_label_column'][:2]]
        selected=[]
        for band in range(4):
            eligible=[]
            for y in range(32,8853-32-511,512):
                for x in range(32,6205-32-511,512):
                    if 4*(y+256)//8853!=band:continue
                    box=(x,y,x+512,y+512);m=np.asarray(mask.crop(box));l=np.asarray(lab.crop(box))
                    pos=int(np.count_nonzero((m==255)&(l==1)));neg=int(np.count_nonzero((m==255)&(l==0)))
                    t=next(t for t in table if (t['y'],t['x'])==(y,x))
                    assert (pos,neg)==(t['positive'],t['negative'])
                    ok=pos+neg>=int(np.ceil(.95*512**2)) and pos>=256 and neg>=256
                    assert ok==t['eligible']
                    if ok:eligible.append((y,x))
            if eligible:selected.append(min(eligible))
        assert selected==[(t['y'],t['x']) for t in panel['tiles']]
        diagnostics=json.loads((run/'OPTICAL_DIAGNOSTICS.json').read_text())
        for t,d in zip(panel['tiles'],diagnostics,strict=True):
            y,x=t['y'],t['x'];box=(x,y,x+512,y+512)
            m=np.asarray(mask.crop(box));l=np.asarray(lab.crop(box));o=np.asarray(ir.crop(box))
            expected=l.copy();expected[m!=255]=2
            actual=np.load(run/(t['id']+'-labels.npy'),allow_pickle=False)
            assert np.array_equal(actual,expected)
            assert np.all(o[...,3][m==255]==255),'Selected optical support must be opaque'
            # Source RGB is grayscale; no color conversion ambiguity on selected support.
            assert np.array_equal(o[...,0],o[...,1]) and np.array_equal(o[...,1],o[...,2])
            for name,cl in [('supplied_positive',1),('supplied_negative',0)]:
                values=o[...,0][expected==cl];h=np.bincount(values,minlength=256)
                assert h.tolist()==d['stats'][name]['histogram']
                assert float(values.mean())==d['stats'][name]['mean']
        xmp=lab.info['XML:com.adobe.xmp'];root=ET.fromstring(xmp)
        fields={}
        for el in root.iter():
            fields.update({k.rsplit('}',1)[-1]:v for k,v in el.attrib.items() if k.rsplit('}',1)[-1] in ['CreatorTool','CreateDate','ModifyDate','MetadataDate','OriginalDocumentID','DocumentID','InstanceID']})
        # Metadata describes editing software, not an authenticated annotation author.
    result=dict(status='PASS',reviewer='root separate implementation; not independent-agent review',
        source_hashes=3,whole_canvas_histograms_verified=True,joint_counts_verified=True,
        tile_counts_verified=len(table),tile_selection_verified=len(selected),label_arrays_verified=len(selected),
        optical_histograms_verified=8,optical_alpha_histogram=alpha_hist.tolist(),
        nonopaque_optical_pixels_on_white_mask=nonopaque_supported,optical_chromatic_pixels=chromatic,
        png_xmp_selected_fields=fields,elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        completed_utc=datetime.now(timezone.utc).isoformat())
    assert result['peak_rss_bytes']<=1024**3 and result['elapsed_seconds']<=240
    with (HERE/'QC_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='optical_alpha_histogram'},indent=2))

if __name__=='__main__':main()
