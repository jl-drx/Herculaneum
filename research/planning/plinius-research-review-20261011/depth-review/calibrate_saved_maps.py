"""Read-only PHerc343 fixed-display calibration; no neural imports or inference."""
from pathlib import Path
import hashlib
import json
import math
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)
OUT = Path(__file__).resolve().parent / 'calibration'
OUT.mkdir(exist_ok=False)
LAB = Path('/workspace/herculaneum-lab/reports')
P = LAB / 'fortysixth-experiment'
W = P / 'work'
S = LAB / 'fortyfifth-experiment/source-evidence'
STEM = 'concat_w047-w048_R5B2_z9500-11000'
MODELS = ['dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43', 'hecate']
PHASES = ['plain', 'padded']
manifest = []
def sha(b): return hashlib.sha256(b).hexdigest()
def read_bytes(p):
    b = p.read_bytes()
    manifest.append({'path':str(p), 'bytes':len(b), 'sha256':sha(b)})
    return b
def js(p): return json.loads(read_bytes(p))
def pin(p): read_bytes(p)
boxes = js(S/'pherc343/inputs'/f'{STEM}_letter_boxes.json')['letters']
assert len(boxes)==18 and [x['id'] for x in boxes] == list(range(1,19))
integrity = js(P/'completion-review/ACTUAL_FINAL_MAP_CHECK.json')
assert integrity['status'] == 'FINAL_MAP_INTEGRITY_PASS'
expected = {(x['model'], x['phase']):x for x in integrity['maps']}
maps, checks = {}, []
for model in MODELS:
    for phase in PHASES:
        rs = list((W/f'maps_{phase}').glob(f'*__{model}__*.receipt.json'))
        assert len(rs)==1
        r = js(rs[0])
        assert (r['model'],r['phase'],r['status'])==(model,phase,'PASS')
        file = rs[0].with_suffix('').with_suffix('')
        b = read_bytes(file)
        raw = np.asarray(Image.open(file))
        e = expected[model,phase]
        assert sha(b)==r['output']['sha256']==e['encoded_sha256']
        assert len(b)==r['output']['bytes']
        assert sha(raw.tobytes())==r['decoded_sha256']==e['decoded_sha256']
        assert list(raw.shape)==r['shape']==e['shape'] and raw.dtype==np.uint8
        h,w = (2560,6720) if phase=='plain' else (2600,6920)
        if model=='hecate':
            aligned = cv2.resize(raw.astype(np.float32)/255, (w,h), interpolation=cv2.INTER_LINEAR)
        else:
            assert raw.shape==(h,w)
            aligned = raw.astype(np.float32)/255
        if phase=='padded': aligned=aligned[40:2600,200:6920]
        assert aligned.shape==(2560,6720)
        maps[model,phase]=np.rint(aligned*255).astype(np.uint8)
        checks.append({'model':model,'phase':phase,'encoded_sha256':sha(b),'decoded_sha256':sha(raw.tobytes()),'shape':list(raw.shape),'status':'PASS'})
valid_path=W/'valid.npy'
pin(valid_path)
valid=np.load(valid_path,allow_pickle=False)
assert valid.dtype==bool and valid.shape==(2560,6720)
render = js(P/'RENDER_VERIFICATION.json')
assert sha(np.packbits(valid[105:2560,380:6021]).tobytes())=='f60549218edb9871e5acb8930e176a09fe1aaafef28c10a2261014f70a58dcbc'
refmeta=js(S/'pherc343/outputs'/f'{STEM}.json')
refpath=S/f'{STEM}.png'; pin(refpath)
ref=np.asarray(Image.open(refpath).convert('RGB'))
b=refmeta['outputs']['image']['data_panel']
ref=ref[b['y']:b['y']+b['height'],b['x']:b['x']+b['width']][::-1,::-1]
assert ref.shape==(2455,5641,3)
brown=np.array([147,101,90],dtype=np.uint8)
reference=np.broadcast_to(brown,(2560,6720,3)).copy()
reference[105:2560,380:6021]=ref
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
def image_map(m,sl):
    z=np.repeat(m[sl][...,None],3,axis=2)
    z[~valid[sl]]=brown
    return Image.fromarray(z)
def put(pic,im,xywh):
    x,y,w,h=xywh
    im.thumbnail((w,h),Image.Resampling.NEAREST)
    pic.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
regions=[]
for box in boxes:
    ident=box['id'];x,y,X,Y=box['box'];bounds=[math.floor(x),math.floor(y),math.ceil(X),math.ceil(Y)]
    x,y,X,Y=bounds;sl=np.s_[y:Y,x:X]
    pic=Image.new('RGB',(1480,1250),'#f4f4f4');d=ImageDraw.Draw(pic)
    d.text((12,8),f'PHerc343 box {ident:02d} | source x{x}:{X}, y{y}:{Y} | raw unmerged maps | fixed 0..255',font=font,fill='black')
    d.text((12,33),'All four models; same source orientation. Brown = invalid. No fitted contrast. Reported structures, not ink truth.',font=small,fill='black')
    for j,model in enumerate(MODELS):
        for i,phase in enumerate(PHASES):
            xx=j*370;yy=65+i*405
            d.text((xx+8,yy),f'{model} {phase}',font=small,fill='black')
            put(pic,image_map(maps[model,phase],sl),(xx+8,yy+25,354,370))
    d.text((12,885),'Original published composite (different, unchanged display; comparison only)',font=font,fill='black')
    put(pic,Image.fromarray(reference[sl]),(10,915,720,325))
    pic.save(OUT/f'box-{ident:02d}.png')
    cx=(box['box'][0]+box['box'][2])/2;cy=(box['box'][1]+box['box'][3])/2
    sx=max(0,min(6720-1536,math.floor(cx-768)));sy=max(0,min(2560-512,math.floor(cy-256)))
    sx2,sy2=sx+1536,sy+512;cs=np.s_[sy:sy2,sx:sx2]
    context=Image.new('RGB',(1560,1515),'#f4f4f4');dc=ImageDraw.Draw(context)
    dc.text((10,8),f'Box {ident:02d} fixed context x{sx}:{sx2}, y{sy}:{sy2} |1536x512 native pixels; display half scale',font=font,fill='black')
    dc.text((10,34),'Left plain / right padded. Row models fixed. Same source orientation. Brown = invalid. All-map 0..255 display.',font=small,fill='black')
    for i,model in enumerate(MODELS):
        yy=65+i*290
        for j,phase in enumerate(PHASES):
            xx=j*780;dc.text((xx+6,yy),f'{model} {phase}',font=small,fill='black')
            tile=image_map(maps[model,phase],cs).resize((768,256),Image.Resampling.NEAREST)
            context.paste(tile,(xx+6,yy+23))
    dc.text((10,1234),'Published composite: unchanged reference display; comparison only. Grey outside panel is not a negative label.',font=small,fill='black')
    put(context,Image.fromarray(reference[cs]),(6,1260,1536,250))
    context.save(OUT/f'context-{ident:02d}.png')
    regions.append({'id':ident,'full_box_xyxy':bounds,'context_xyxy':[sx,sy,sx2,sy2],'context_encloses_full_box':sx<=x and sy<=y and X<=sx2 and Y<=sy2,'box_valid_pixels':int(valid[sl].sum()),'box_pixels':int(valid[sl].size),'context_valid_pixels':int(valid[cs].sum())})
result={'status':'PASS','maps':checks,'valid_published_crop_packbits_sha256':sha(np.packbits(valid[105:2560,380:6021]).tobytes()),'regions':regions,'source_manifest':manifest,'new_model_forwards':0,'limitations':'Exposed candidate boxes; support is inherited full-depth valid mask, not newly computed all-model context mask; reference is published display not ink truth; only d+1 reading order saved. No study52 maps read. Hecate full-canvas float32/255 linear resize then native padded crop; nearest display downsampling; no fitted contrast.'}
(OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['CALIBRATION_PLAN.md','calibrate_saved_maps.py']:
    p=OUT.parent/name;manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())})
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PASS','maps':len(checks),'boxes':len(boxes),'context_not_enclosing':[r['id'] for r in regions if not r['context_encloses_full_box']],'out':str(OUT)}))
