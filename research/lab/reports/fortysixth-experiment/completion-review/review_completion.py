"""Independent saved-output composition and region audit; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zarr

ap = argparse.ArgumentParser()
ap.add_argument('--lab', type=Path, required=True)
ap.add_argument('--prior', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
p = a.lab / 'reports/fortysixth-experiment'
old = a.lab / 'reports/fortyfifth-experiment'
work = p / 'work'
stem = 'concat_w047-w048_R5B2_z9500-11000'
area = np.s_[105:2560,380:6021]
valid_full = np.load(work / 'valid.npy', allow_pickle=False)
valid = valid_full[area]
stats = json.loads((p / 'FIXED_DISPLAY_STATS.json').read_text())
side = json.loads((work / 'image' / (stem+'.json')).read_text())
refside = json.loads((old / 'source-evidence/pherc343/outputs' / (stem+'.json')).read_text())
for key in ['orientation','area','overlay']:
    assert side[key] == refside[key], key
assert side['ink']['normalisation'] == refside['ink']['normalisation']
assert [x['model'] for x in side['ink']['maps']] == ['dnative','hecate','ink9-42','ink9-43']
assert all(x['depth']==1 and x['direction']=='reverse' for x in side['ink']['maps'])

def gray(path):
    im = np.array(Image.open(path))
    assert im.ndim==2 and im.dtype==np.uint8
    return im

acc = None
for rec in side['ink']['maps']:
    f = work / 'maps' / rec['file']
    assert hashlib.sha256(f.read_bytes()).hexdigest() == rec['sha256']
    raw = gray(f)
    if rec['model']=='hecate':
        m = cv2.resize(raw.astype(np.float32)/255, (6720,2560), interpolation=cv2.INTER_LINEAR)
        hecate_merged = m
    else:
        m = np.clip((raw.astype(np.float32)-64)/128,0,1)
    st = stats['models'][rec['model']]
    z = (m[area] - st['mean']) / st['sd']
    acc = z if acc is None else acc + z
ensemble = acc / 4
lo, hi = stats['ensemble']['p1'], stats['ensemble']['p99_5']
ink = (np.clip((ensemble-lo)/(hi-lo),0,1)*255).astype(np.uint8)
ink[~valid] = 0
assert np.array_equal(ink[::-1,::-1],gray(work/'image'/(stem+'_ink.png')))
del acc, ensemble, z, m, raw

volume = zarr.open(str(work/(stem+'_66.zarr')), mode='r')
if not hasattr(volume, 'shape'):
    volume = volume['0']
assert volume.shape == (66,2560,6720)
ct = np.asarray(volume[34,105:2560,380:6021])
ctsha = hashlib.sha256(ct.tobytes()).hexdigest()
assert ctsha == 'df1a3e8d4e61de845eca0b94952dc207b73605c1f6093312e5283980b895f8f8'
ct_u8 = (np.clip((ct.astype(np.float32)-26)/138,0,1)*255).astype(np.uint8)
ct_u8[~valid] = 0
assert np.array_equal(ct_u8[::-1,::-1],gray(work/'image'/(stem+'_ct.png')))
del ct, ct_u8

def native_panel(file, metadata):
    im = np.array(Image.open(file).convert('RGB'))
    b = metadata['outputs']['image']['data_panel']
    return im[b['y']:b['y']+b['height'], b['x']:b['x']+b['width']][::-1,::-1].copy()

ref = native_panel(old/'source-evidence'/(stem+'.png'),refside)
actual = native_panel(work/'image'/(stem+'.png'),side)
delta = np.abs(actual.astype(np.int16)-ref.astype(np.int16))

def yellow(im):
    r,g,b = np.moveaxis(im.astype(np.int16),-1,0)
    return (r-b>12)&(g-b>8)&(np.abs(r-g)<50)

yr, ya = yellow(ref), yellow(actual)
hp = {}
for phase in ['plain','padded']:
    im = gray(next((work/('maps_'+phase)).glob('*__hecate__*.png')))
    if phase=='padded':
        im = im[36:,180:]
    assert im.shape == (2304,6048)
    hp[phase] = cv2.resize(im.astype(np.float32)/255,(6720,2560),interpolation=cv2.INTER_LINEAR)

def phase_stats(x,y,v):
    xx,yy = x[v].astype(np.float64),y[v].astype(np.float64)
    k = int(np.ceil(.1*len(xx)))
    tx,ty = np.partition(xx,len(xx)-k)[len(xx)-k], np.partition(yy,len(yy)-k)[len(yy)-k]
    bx,by = xx>=tx,yy>=ty
    return {'pearson':float(np.corrcoef(xx,yy)[0,1]),
            'mae_uint8_equivalent':float(np.abs(xx-yy).mean()*255),
            'top10_tie_inclusive_dice':float(2*(bx&by).sum()/(bx.sum()+by.sum())),
            'selected_fractions':[float(bx.mean()),float(by.mean())]}

regions = [{'id':'published_crop','bounds_xyxy':[380,105,6021,2560]}]
regions += json.loads(a.prior.read_text())['regions']
records = []
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
sheets = [Image.new('RGB',(2100,1470),'white') for _ in range(3)]
for sheet in sheets:
    draw = ImageDraw.Draw(sheet)
    draw.text((10,8),'Fixed published display: reference | reproduced | absolute RGB difference x16 (fixed) | Hecate plain | aligned padded | merged',font=font,fill='black')
    draw.text((10,30),'All 18 pre-existing boxes, reading orientation. Hecate fixed grayscale /255 after author linear resize; no fitted contrast or ink truth.',font=font,fill='black')
for i,reg in enumerate(regions):
    x,y,X,Y = reg['bounds_xyxy']
    sl = np.s_[y-105:Y-105,x-380:X-380]
    v = valid[sl]
    d = delta[sl]
    r,b = yr[sl]&v,ya[sl]&v
    union,intersection = int((r|b).sum()),int((r&b).sum())
    record = {'id':reg['id'],'bounds_xyxy':[x,y,X,Y],'valid_pixels':int(v.sum()),
              'rgb_mae':float(d[v].mean()),'max_rgb_difference':int(d[v].max()),
              'display_mask_intersection':intersection,'display_mask_union':union,
              'display_mask_iou':intersection/union if union else None,
              'hecate_phase':phase_stats(hp['plain'][y:Y,x:X],hp['padded'][y:Y,x:X],v)}
    records.append(record)
    if i:
        ims = [ref[sl],actual[sl],np.minimum(d*16,255).astype(np.uint8)]
        ims += [np.clip(h[y:Y,x:X]*255,0,255).astype(np.uint8) for h in [hp['plain'],hp['padded'],hecate_merged]]
        sheet = sheets[(i-1)//6]
        oy = 65 + ((i-1)%6)*232
        draw = ImageDraw.Draw(sheet)
        for col,im in enumerate(ims):
            pic = Image.fromarray(im[::-1,::-1])
            pic.thumbnail((335,188), Image.Resampling.NEAREST)
            sheet.paste(pic,(col*350+(350-pic.width)//2,oy+36+(188-pic.height)//2))
            label = f'Box {reg["id"]}'
            if col==1: label+=f' RGB MAE {record["rgb_mae"]:.4f}'
            if col==2: label+=f' max {record["max_rgb_difference"]}'
            if col==4: label+=f' phase r={record["hecate_phase"]["pearson"]:.3f}'
            draw.text((col*350+6,oy+8),label,font=font,fill='black')
for i,sheet in enumerate(sheets):
    sheet.save(a.out/f'regions_{i+1}.png')

# Record every maximum-difference pixel and whether it appears in any old box.
maxd = np.where(valid,delta.max(axis=2),-1)
coords = np.argwhere(maxd==maxd.max())
worst = [{'native_yx':[int(y+105),int(x+380)],
          'reference_rgb':ref[y,x].tolist(),'reproduction_rgb':actual[y,x].tolist(),
          'boxes':[r['id'] for r in regions[1:] if r['bounds_xyxy'][0]<=x+380<r['bounds_xyxy'][2]
                   and r['bounds_xyxy'][1]<=y+105<r['bounds_xyxy'][3]]} for y,x in coords]
overview=Image.new('RGB',(1800,1660),'white')
for row,(title,im) in enumerate([('Published reference',ref),('Completed unchanged four-model reproduction',actual)]):
    pic=Image.fromarray(im[::-1,::-1]);pic.thumbnail((1780,780))
    overview.paste(pic,((1800-pic.width)//2,row*830+40))
    ImageDraw.Draw(overview).text((12,row*830+10),title,font=font,fill='black')
overview.save(a.out/'overview.png')
report={'status':'PASS','ct_layer34_sha256':ctsha,'ct_component_decoded_exact':True,
        'ink_component_independent_decoded_exact':True,'reference_recipe_metadata_equal':True,
        'regions':records,'largest_rgb_difference_locations':worst,
        'hecate_metric_scope':'Each phase aligned at 36/180, /255 then author linear resize to native. MAE scaled by255, ranking performed separately within each support.',
        'scope':'Implementation and descriptive fixed-display reproduction. No AP, authenticity, raw-reference equality, new inference or independent test regions.'}
(a.out/'COMPLETION_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','whole_crop':records[0],'worst':worst,'regions':len(records)-1},indent=2))
