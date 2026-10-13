"""Independent read-only recomputation of frozen study52 completion reporting.
Does not import or execute the reporting generator or model inference code.
"""
from pathlib import Path
import collections, datetime, hashlib, itertools, json, math
import numpy as np
from PIL import Image

SRC = Path('/workspace/herculaneum-lab/reports/fiftysecond-experiment/completion-review')
OUT = Path(__file__).resolve().parent
EXPECTED_PLAN = '0706ea76bdec21fc925f0a5ee6615c8a872050aae593275e06c7566b0982d143'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1048576): h.update(chunk)
    return h.hexdigest()

def verify_pin(rec, root=None):
    p = Path(rec['path']) if root is None else root / rec['path']
    assert p.stat().st_size == rec['size_bytes'], ('size', p)
    assert digest(p) == rec['sha256'], ('hash', p)

assert digest(SRC/'ASSESSMENT_PLAN.json') == EXPECTED_PLAN
plan = json.loads((SRC/'ASSESSMENT_PLAN.json').read_text())
report = json.loads((SRC/'METRICS.json').read_text())
manifest = json.loads((SRC/'OUTPUT_MANIFEST.json').read_text())
assert report['plan_sha256'] == EXPECTED_PLAN
for rec in plan['source_inputs']: verify_pin(rec)
verify_pin(plan['code'])
assert len(manifest['files']) == len({r['path'] for r in manifest['files']})
for rec in manifest['files']:
    p = Path(rec['path'])
    assert not p.is_absolute() and '..' not in p.parts
    verify_pin(rec, SRC)
assert set(r['path'] for r in manifest['files']) == {str(p.relative_to(SRC)) for p in SRC.rglob('*') if p.is_file() and p.name != 'OUTPUT_MANIFEST.json'}

models = ('dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43', 'hecate')
phases = ('plain', 'padded')
orders = ('forward', 'reverse')
depths = (-15, 0, 15)
expected_combinations = set(itertools.product(models, phases, orders, depths))
actual_combinations = [(m['model'],m['phase'],m['order'],m['depth']) for m in plan['maps']]
assert len(actual_combinations) == 48 and set(actual_combinations) == expected_combinations
records = {r['key']: r for r in plan['maps']}
assert len(records) == 48
combination_to_key = {c: r['key'] for c,r in zip(actual_combinations,plan['maps'])}

mask_record = next(r for r in plan['source_inputs'] if r['path'].endswith('output-support.npz'))
with np.load(mask_record['path'],allow_pickle=False) as masks:
    known = masks['common']
    assert known.dtype == np.bool_ and known.shape == (512,1536)
    independently_known = np.ones((512,1536), bool)
    for k in ('villa_plain','villa_padded','hecate_plain','hecate_padded'):
        independently_known &= masks[k]
    assert np.array_equal(known, independently_known)
assert int(known.sum()) == report['common_support_pixels']
assert int(known.size-known.sum()) == report['unknown_pixels']
assert plan['core_xywh'] == [512,1024,1536,512]
assert [(c['name'],c['local_xywh'],c['source_xywh']) for c in plan['cells']] == [(f'cell{i+1}', [i*512,0,512,512], [512+i*512,1024,512,512]) for i in range(3)]
scopes = {'full_strip': known}
for i in range(3):
    m = np.zeros(known.shape, bool)
    m[:,i*512:(i+1)*512] = known[:,i*512:(i+1)*512]
    scopes[f'cell{i+1}'] = m

arrays = {}
rgb = {}
map_checks = []
for key, r in records.items():
    verify_pin(r['input'])
    a = np.load(r['input']['path'],allow_pickle=False)
    assert a.shape == known.shape and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    arrays[key] = a
    # Explicit nearest-even conversion after the declared float32 multiply.
    levels = a * np.float32(255)
    lo = np.floor(levels)
    fraction = levels-lo
    gray = (lo + ((fraction > .5) | ((fraction == .5) & (lo % 2 == 1)))).astype(np.uint8)
    pixels = np.empty((*a.shape,3), np.uint8)
    for channel in range(3): pixels[:,:,channel] = gray
    pixels[~known,:] = [139,90,43]
    rgb[key] = pixels
    map_checks.append({'key':key,'shape':list(a.shape),'dtype':str(a.dtype),'min':float(a.min()),'max':float(a.max()),'input_sha256':r['input']['sha256']})

# Derive required comparisons by differing factor rather than reproduce generator loops.
expected_pairs = {}
for left,right in itertools.combinations(sorted(records),2):
    a,b = records[left],records[right]
    differences = [f for f in ('model','phase','order','depth') if a[f]!=b[f]]
    if len(differences)==1:
        kind = {'model':'cross_model','phase':'padding','order':'order','depth':'depth'}[differences[0]]
        expected_pairs[(left,right)] = kind
assert collections.Counter(expected_pairs.values()) == {'padding':24,'order':24,'cross_model':72,'depth':48}
for collection in (plan['pairs'],report['metrics']):
    observed = {}
    for r in collection:
        pair = tuple(sorted((r['left'],r['right'])))
        assert pair not in observed, ('duplicate pair',pair)
        observed[pair] = r['kind']
    assert observed == expected_pairs

recomputed = []
max_pearson_error = max_mad_error = 0.
for r in report['metrics']:
    assert set(r['scores']) == set(scopes)
    a,b = arrays[r['left']], arrays[r['right']]
    scores = {}
    for name,mask in scopes.items():
        x = a[mask].astype(np.float64)
        y = b[mask].astype(np.float64)
        n = x.size
        # Long-double raw moments use a different reduction path from centred float64 sums.
        xx, yy = x.astype(np.longdouble), y.astype(np.longdouble)
        sx,sy = xx.sum(),yy.sum()
        covariance = (xx*yy).sum() - sx*sy/n
        variance_x = (xx*xx).sum() - sx*sx/n
        variance_y = (yy*yy).sum() - sy*sy/n
        corr = float(covariance/np.sqrt(variance_x*variance_y)) if variance_x>0 and variance_y>0 else None
        delta = xx-yy
        mad = float((delta[delta>=0].sum()-delta[delta<0].sum())*255/n)
        claimed = r['scores'][name]
        assert claimed['pixels'] == n
        if corr is None:
            assert claimed['pearson'] is None
            ce = 0.
        else:
            ce = abs(corr-claimed['pearson'])
            assert ce < 1e-10, ('pearson',r,name,ce)
        me = abs(mad-claimed['mean_absolute_difference_uint8_levels'])
        assert me < 1e-10, ('mad',r,name,me)
        max_pearson_error=max(max_pearson_error,ce)
        max_mad_error=max(max_mad_error,me)
        scores[name]={'pixels':n,'pearson':corr,'mean_absolute_difference_uint8_levels':mad,'pearson_absolute_error':ce,'mad_absolute_error':me}
    recomputed.append({'kind':r['kind'],'left':r['left'],'right':r['right'],'scores':scores})

expected_panels = set(itertools.product(models,phases,orders,('full','cell1','cell2','cell3')))
observed_panels = set()
panel_checks = []
for panel in report['panel_layouts']:
    combo = panel['model'],panel['phase'],panel['order'],panel['region']
    assert combo not in observed_panels
    observed_panels.add(combo)
    m,p,o,region = combo
    width = 1536 if region=='full' else 512
    x = 0 if region=='full' else (int(region[-1])-1)*512
    png = np.asarray(Image.open(SRC/panel['path']).convert('RGB'))
    assert png.shape == (1702,width+32,3)
    assert panel['width']==width+32 and panel['height']==1702 and panel['source_pixel_scale']==1
    for j,depth in enumerate(depths):
        key=combination_to_key[(m,p,o,depth)]
        actual=png[72+548*j:72+548*j+512,16:16+width,:]
        expected=rgb[key][:,x:x+width,:]
        assert np.array_equal(actual,expected), ('panel pixels',panel['path'],depth)
        if np.any(~known[:,x:x+width]):
            assert np.all(actual[~known[:,x:x+width],:]==[139,90,43])
    panel_checks.append({'path':panel['path'],'verified_depth_rows':3,'source_pixel_scale':1,'fixed_uint8_mapping':True,'unknown_brown':True})
assert observed_panels==expected_panels and len(observed_panels)==64
contact_checks=[]
for m in models:
    path=SRC/'contact-sheets'/f'{m}.png'
    png=np.asarray(Image.open(path).convert('RGB'))
    assert png.shape==(548,1616,3)
    for col,(p,o) in enumerate(itertools.product(phases,orders)):
        for row,d in enumerate(depths):
            key=combination_to_key[(m,p,o,d)]
            expected=np.asarray(Image.fromarray(rgb[key]).resize((384,128),resample=Image.Resampling.BOX))
            left=16+400*col; top=66+160*row
            assert np.array_equal(png[top:top+128,left:left+384,:],expected)
    contact_checks.append({'path':str(path.relative_to(SRC)),'verified_tiles':12,'spatial_reduction':4,'no_intensity_adjustment':True})
assert report['maps']==48 and report['pair_comparisons']==168 and report['scope_scores_per_pair']==4
assert report['no_new_inference'] is True and report['ink_accuracy_claim'] is False
result={'status':'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent read-only mathematical and pixel-content verification; no inference or scientific ink claim. Generator code was read for interface and panel locations, never imported or executed.','assessment_plan_sha256':EXPECTED_PLAN,'source_manifest_files_verified':len(manifest['files']),'maps_verified':48,'distinct_pairs_verified':168,'pair_counts':dict(collections.Counter(expected_pairs.values())),'scope_scores_verified':672,'exact_panels_verified':64,'exact_map_regions_verified':192,'contact_sheets_verified':4,'contact_tiles_verified':48,'common_support_pixels':int(known.sum()),'unknown_pixels':int(known.size-known.sum()),'maximum_pearson_absolute_error':max_pearson_error,'maximum_mad_uint8_levels_absolute_error':max_mad_error,'maps':map_checks,'metrics':recomputed,'panels':panel_checks,'contacts':contact_checks,'limitations':['No independent reconstruction of CT volumes, source model predictions, support geometry, or upstream source correctness.','No blinded visual detection, ink accuracy or authenticated letter judgment.','Pixel display correctness and statistical reproducibility do not establish model validity.']}
(OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
summary = {k:v for k,v in result.items() if k not in ('maps','metrics','panels','contacts')}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
