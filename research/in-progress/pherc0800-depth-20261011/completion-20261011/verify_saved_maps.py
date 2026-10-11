#!/usr/bin/env python3
"""Read-only portable verification of all saved study52 completion maps and reports.

Requires only this packet, NumPy and Pillow. Never loads original absolute
paths, model weights or CT volumes, and writes no files. All saved input
bindings must match the frozen assessment plan. V1/v2 historical verifiers and
the reporting generator remain separate, unchanged evidence.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, itertools, json, re, sys
import numpy as np
from PIL import Image

EXPECTED_PLAN = '0706ea76bdec21fc925f0a5ee6615c8a872050aae593275e06c7566b0982d143'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=Path(__file__).resolve().parent,
                        help='Completion-review packet directory; defaults to script directory.')
    args=parser.parse_args()
    SRC=args.packet.resolve(strict=True)
    assert SRC.is_dir(), 'Packet must be a directory'
    bindings={}

    def packet_path(value):
        assert isinstance(value,str) and value, 'Empty or invalid relative path'
        assert not value.startswith('/') and '\\' not in value, ('Nonportable path',value)
        assert all(part not in ('', '.', '..') for part in value.split('/')), ('Unsafe path',value)
        candidate=SRC
        for part in value.split('/'):
            candidate=candidate/part
            assert not candidate.is_symlink(), ('Symlink not allowed',value)
        candidate=candidate.resolve(strict=True)
        assert candidate.is_relative_to(SRC) and candidate.is_file(), ('Path escapes packet or is not file',value)
        return candidate

    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            while chunk:=f.read(1048576): h.update(chunk)
        return h.hexdigest()

    def validate_record(rec):
        assert type(rec['size_bytes']) is int and rec['size_bytes']>=0
        assert isinstance(rec['sha256'],str) and re.fullmatch('[0-9a-f]{64}',rec['sha256'])

    def bound_path(rec):
        assert rec['path'] in bindings, ('Missing frozen input binding',rec['path'])
        binding=bindings[rec['path']]
        assert rec['size_bytes']==binding['size_bytes'] and rec['sha256']==binding['sha256'], ('Input binding differs from frozen pin',rec['path'])
        return packet_path(binding['path'])

    def verify_pin(rec, root=None):
        validate_record(rec)
        assert root is None or root==SRC
        p=bound_path(rec) if root is None else packet_path(rec['path'])
        assert p.stat().st_size==rec['size_bytes'], ('size',str(p))
        assert digest(p)==rec['sha256'], ('hash',str(p))

    def load_saved_inputs(plan):
        saved=json.loads(packet_path('SAVED_INPUTS_MANIFEST.json').read_text())
        assert saved['schema_version']==1
        assert saved['source_assessment_plan_sha256']==EXPECTED_PLAN
        rows=saved['files']
        assert len(rows)==55
        assert len({r['path'] for r in rows})==55, 'Duplicate packet paths'
        assert len({r['original_path'] for r in rows})==55, 'Duplicate original paths'
        pins=plan['source_inputs']+[r['input'] for r in plan['maps']]+[plan['code']]
        assert len(pins)==55 and len({r['path'] for r in pins})==55
        assert {r['original_path'] for r in rows}=={r['path'] for r in pins}, 'Saved manifest differs from required pin set'
        for row in rows:
            assert isinstance(row['original_path'],str)
            validate_record(row)
            path=packet_path(row['path'])
            assert path.stat().st_size==row['size_bytes'] and digest(path)==row['sha256'], ('Saved manifest hash/size',row['path'])
            bindings[row['original_path']]=row
        for pin in pins: verify_pin(pin)
        assert len({bindings[r['input']['path']]['path'] for r in plan['maps']})==48

    assert digest(packet_path('ASSESSMENT_PLAN.json')) == EXPECTED_PLAN
    plan = json.loads(packet_path('ASSESSMENT_PLAN.json').read_text())
    load_saved_inputs(plan)
    report = json.loads(packet_path('METRICS.json').read_text())
    manifest = json.loads(packet_path('OUTPUT_MANIFEST.json').read_text())
    assert report['plan_sha256'] == EXPECTED_PLAN
    for rec in plan['source_inputs']: verify_pin(rec)
    verify_pin(plan['code'])
    assert len(manifest['files']) == len({r['path'] for r in manifest['files']})
    for rec in manifest['files']:
        p = Path(rec['path'])
        assert not p.is_absolute() and '..' not in p.parts
        verify_pin(rec, SRC)
    # The original reporting manifest covers the frozen generated evidence, not later
    # recovery notes or independent verification reports added to this directory.
    expected_generated = {'assess_complete.py', 'ASSESSMENT_PLAN.json', 'METRICS.json'}
    for model, phase, order in itertools.product(
            ('dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43', 'hecate'),
            ('plain', 'padded'), ('forward', 'reverse')):
        expected_generated.add(f'full-strips/{model}-{phase}-{order}-full.png')
        for cell in ('cell1', 'cell2', 'cell3'):
            expected_generated.add(f'cells/{model}-{phase}-{order}-{cell}.png')
    for model in ('dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43', 'hecate'):
        expected_generated.add(f'contact-sheets/{model}.png')
    assert len(expected_generated) == 71
    assert set(r['path'] for r in manifest['files']) == expected_generated
    for subdir in ('full-strips', 'cells', 'contact-sheets'):
        expected_here = {p for p in expected_generated if p.startswith(subdir + '/')}
        assert {str(p.relative_to(SRC)) for p in (SRC/subdir).rglob('*') if p.is_file()} == expected_here


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
    with np.load(bound_path(mask_record),allow_pickle=False) as masks:
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
        a = np.load(bound_path(r['input']),allow_pickle=False)
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
        assert combo in expected_panels, ('Unexpected panel factors',combo)
        assert combo not in observed_panels
        observed_panels.add(combo)
        m,p,o,region = combo
        width = 1536 if region=='full' else 512
        x = 0 if region=='full' else (int(region[-1])-1)*512
        expected_panel_path = f'full-strips/{m}-{p}-{o}-full.png' if region=='full' else f'cells/{m}-{p}-{o}-{region}.png'
        assert panel['path']==expected_panel_path, ('Unexpected panel path',panel['path'])
        png = np.asarray(Image.open(packet_path(panel['path'])).convert('RGB'))
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
        path=packet_path(f'contact-sheets/{m}.png')
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
    result={'status':'PASS','verifier_version':'portable-1','manifest_scope':'Exactly the 71 frozen generator/code/plan/metrics/PNG files; later recovery, visual-review and independent-check documents are outside this original manifest and require the parent portable manifest.','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only portable mathematical and pixel-content replay derived from the independent v2 verifier; all input pins resolve exclusively through the packet saved-inputs manifest. No inference or scientific ink claim.','assessment_plan_sha256':EXPECTED_PLAN,'source_manifest_files_verified':len(manifest['files']),'maps_verified':48,'distinct_pairs_verified':168,'pair_counts':dict(collections.Counter(expected_pairs.values())),'scope_scores_verified':672,'exact_panels_verified':64,'exact_map_regions_verified':192,'contact_sheets_verified':4,'contact_tiles_verified':48,'common_support_pixels':int(known.sum()),'unknown_pixels':int(known.size-known.sum()),'maximum_pearson_absolute_error':max_pearson_error,'maximum_mad_uint8_levels_absolute_error':max_mad_error,'maps':map_checks,'metrics':recomputed,'panels':panel_checks,'contacts':contact_checks,'limitations':['No independent reconstruction of CT volumes, source model predictions, support geometry, or upstream source correctness.','No blinded visual detection, ink accuracy or authenticated letter judgment.','Pixel display correctness and statistical reproducibility do not establish model validity.']}
    result['packet'] = str(SRC)
    result['saved_inputs_manifest_sha256'] = digest(packet_path('SAVED_INPUTS_MANIFEST.json'))
    result['saved_input_bindings_verified'] = len(bindings)
    result['original_path_fallback'] = False
    result['outputs_written'] = False
    summary = {k:v for k,v in result.items() if k not in ('maps','metrics','panels','contacts')}
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc),'outputs_written':False},indent=2))
        sys.exit(1)
