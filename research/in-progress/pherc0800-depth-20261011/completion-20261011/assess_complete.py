"""Fixed study52 completed-map reporting; no models, input changes or fitted display."""
from pathlib import Path
import argparse, datetime, hashlib, itertools, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
MODELS = ['dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43', 'hecate']
PHASES = ['plain', 'padded']
ORDERS = ['forward', 'reverse']
DEPTHS = [-15, 0, 15]
BROWN = (139, 90, 43)

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def pin(p):
    p = Path(p)
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=sha(p))

def key(m, p, o, d):
    return f'{m}-{p}-{o}-d{d:+03}'

def freeze():
    assert not (HERE / 'ASSESSMENT_PLAN.json').exists()
    inputs = []
    for name in ['PROTOCOL.md', 'PLAN.json', 'INFERENCE_GO.json', 'INFERENCE_RESULTS.json', 'BASELINES.json']:
        inputs.append(pin(STUDY / name))
    inputs.append(pin(STUDY / 'work/output-support.npz'))
    maps = []
    baselines = read(STUDY / 'BASELINES.json')['maps']
    results = read(STUDY / 'INFERENCE_RESULTS.json')['maps']
    for m, p, o, d in itertools.product(MODELS, PHASES, ORDERS, DEPTHS):
        if d == 0:
            rec = next(r for r in baselines if (r['model'], r['phase'], r['direction']) == (m, p, o))['crop']
        else:
            rec = next(r for r in results if (r['model'], r['phase'], r['direction'], r['depth']) == (m, p, o, d))['outputs']['aligned']
        assert pin(rec['path']) == rec
        maps.append(dict(key=key(m,p,o,d), model=m, phase=p, order=o, depth=d, input=rec))
    pairs = []
    for d in DEPTHS:
        for m, o in itertools.product(MODELS, ORDERS):
            pairs.append(dict(kind='padding', left=key(m,'plain',o,d), right=key(m,'padded',o,d)))
        for m, p in itertools.product(MODELS, PHASES):
            pairs.append(dict(kind='order', left=key(m,p,'forward',d), right=key(m,p,'reverse',d)))
        for (a,b), p, o in itertools.product(itertools.combinations(MODELS,2), PHASES, ORDERS):
            pairs.append(dict(kind='cross_model', left=key(a,p,o,d), right=key(b,p,o,d)))
    for m, p, o, (a,b) in itertools.product(MODELS, PHASES, ORDERS, itertools.combinations(DEPTHS,2)):
        pairs.append(dict(kind='depth', left=key(m,p,o,a), right=key(m,p,o,b)))
    assert len(maps) == 48 and len(pairs) == 168
    protocol = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), scope='Completed outputs only. Reporting plan fixed before completion metrics or images; completed-map numeric/hash verification occurred first.', source_inputs=inputs, maps=maps, pairs=pairs, core_xywh=[512,1024,1536,512], cells=[dict(name=f'cell{i+1}', local_xywh=[i*512,0,512,512], source_xywh=[512+i*512,1024,512,512]) for i in range(3)], support='The inherited common mask, identical for all maps and comparisons.', display=dict(mapping='round(255 * aligned_probability); fixed 0 to 255 grayscale', unknown_rgb=list(BROWN), exact_panels='One source pixel per displayed pixel; three depth rows per model/phase/order, both full strip and each fixed cell.', contact_sheets='Overview only: uniform 4x spatial reduction of all maps, no intensity adjustment.'), metrics=['Pearson correlation on original float32 aligned maps converted to float64 over inherited common support', 'Mean absolute difference in uint8-equivalent levels: mean(abs(a-b))*255'], scientific_gate='A coherent nonzero-offset stroke group at the same source location in both padding phases, corroborated by Villa and Hecate, and not resolved at d0; both orders remain visible, reversal invariance is not required. This is only a candidate for independent review, not authenticated text.', negative_action='Close the three-position screen as inconclusive; no intermediate depth or region hunt.', no_new_inference=True, code=pin(Path(__file__)))
    (HERE / 'ASSESSMENT_PLAN.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(status='FROZEN',plan_sha256=sha(HERE/'ASSESSMENT_PLAN.json'),maps=48,pairs=168)))

def metric(a,b,mask):
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    ax, by = x-x.mean(), y-y.mean()
    den = np.sqrt(np.sum(ax*ax)*np.sum(by*by))
    return dict(pixels=len(x), pearson=None if den==0 else float(np.sum(ax*by)/den), mean_absolute_difference_uint8_levels=float(np.mean(np.abs(x-y))*255))

def run():
    plan = read(HERE / 'ASSESSMENT_PLAN.json')
    assert pin(Path(__file__)) == plan['code']
    assert not (HERE / 'METRICS.json').exists()
    for p in plan['source_inputs']:
        assert pin(p['path']) == p
    support = np.load(STUDY/'work/output-support.npz',allow_pickle=False)
    common = support['common']
    assert common.dtype==bool and common.shape==(512,1536)
    assert np.array_equal(common, np.logical_and.reduce([support[k] for k in ['villa_plain','villa_padded','hecate_plain','hecate_padded']]))
    arrays, images = {}, {}
    for row in plan['maps']:
        p = row['input']
        assert pin(p['path']) == p
        a = np.load(p['path'],allow_pickle=False)
        assert a.dtype==np.float32 and a.shape==common.shape and np.isfinite(a).all()
        assert ((a>=0)&(a<=1)).all()
        arrays[row['key']] = a
        gray = np.rint(a*255).astype(np.uint8)
        rgb = np.repeat(gray[:,:,None],3,axis=2)
        rgb[~common] = BROWN
        images[row['key']] = Image.fromarray(rgb)
    rows = []
    for pair in plan['pairs']:
        a,b = arrays[pair['left']], arrays[pair['right']]
        scores = {'full_strip':metric(a,b,common)}
        for cell in plan['cells']:
            x,y,w,h = cell['local_xywh'];s = np.s_[y:y+h,x:x+w]
            scores[cell['name']] = metric(a[s],b[s],common[s])
        rows.append(dict(**pair,scores=scores))
    fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontfile,18) if Path(fontfile).exists() else ImageFont.load_default()
    small=ImageFont.truetype(fontfile,13) if Path(fontfile).exists() else ImageFont.load_default()
    folders={k:HERE/k for k in ['full-strips','cells','contact-sheets']}
    for folder in folders.values():folder.mkdir(exist_ok=True)
    panel_rows=[]
    for m,p,o in itertools.product(MODELS,PHASES,ORDERS):
        for cell in [None,*plan['cells']]:
            x,y,w,h=(0,0,1536,512) if cell is None else cell['local_xywh']
            image=Image.new('RGB',(w+32,3*(h+36)+58),(245,245,245));draw=ImageDraw.Draw(image)
            name='full' if cell is None else cell['name']
            draw.text((16,10),f'{m} | {p} | {o} | {name}',font=font,fill=(0,0,0))
            for j,d in enumerate(DEPTHS):
                k=key(m,p,o,d);top=48+j*(h+36)
                draw.text((16,top),f'd={d:+d} | fixed 0–255; brown = unknown',font=small,fill=(0,0,0))
                crop=images[k].crop((x,y,x+w,y+h));image.paste(crop,(16,top+24))
                assert np.array_equal(np.asarray(image)[top+24:top+24+h,16:16+w],np.asarray(crop))
            out=(folders['full-strips'] if cell is None else folders['cells'])/f'{m}-{p}-{o}-{name}.png'
            assert not out.exists();image.save(out)
            panel_rows.append(dict(path=str(out.relative_to(HERE)),model=m,phase=p,order=o,region=name,width=image.width,height=image.height,source_pixel_scale=1))
    for m in MODELS:
        im=Image.new('RGB',(4*(384+16)+16,3*(128+32)+68),(245,245,245));draw=ImageDraw.Draw(im)
        draw.text((16,8),f'{m} — overview only (uniform 4x spatial reduction)',font=font,fill=(0,0,0))
        for col,(p,o) in enumerate(itertools.product(PHASES,ORDERS)):
            for row,d in enumerate(DEPTHS):
                left=16+col*400;top=44+row*160
                draw.text((left,top),f'{p}, {o}, d={d:+d}',font=small,fill=(0,0,0))
                im.paste(images[key(m,p,o,d)].resize((384,128),Image.Resampling.BOX),(left,top+22))
        out=folders['contact-sheets']/f'{m}.png';assert not out.exists();im.save(out)
    report=dict(status='PASS_COMPLETE_FIXED_REPORT',plan_sha256=sha(HERE/'ASSESSMENT_PLAN.json'),common_support_pixels=int(common.sum()),unknown_pixels=int((~common).sum()),maps=48,pair_comparisons=168,scope_scores_per_pair=4,metrics=rows,panel_layouts=panel_rows,no_new_inference=True,ink_accuracy_claim=False)
    (HERE/'METRICS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    manifest=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='OUTPUT_MANIFEST.json':
            r=pin(p);r['path']=str(p.relative_to(HERE));manifest.append(r)
    (HERE/'OUTPUT_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],maps=48,pairs=168,exact_panels=len(panel_rows),contact_sheets=4,common_support_pixels=int(common.sum()),unknown_pixels=int((~common).sum()))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','run'])
    args=parser.parse_args()
    freeze() if args.mode=='freeze' else run()
