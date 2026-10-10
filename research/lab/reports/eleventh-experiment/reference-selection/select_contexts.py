"""Frozen label-only, prediction-independent context selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, maximum_filter

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'selection-plan.json').read_text())
for item in PLAN['source_bindings']:
    body = Path(item['path']).read_bytes()
    assert len(body) == item['bytes'] and hashlib.sha256(body).hexdigest() == item['sha256']
images = ROOT.parent / 'optical-acquisition'
labels = np.asarray(Image.open(images / 'inklabels.png'))
mask = np.asarray(Image.open(images / 'mask.png'))
assert labels.shape == mask.shape == (9791, 6064)
assert set(np.unique(labels)) == set(np.unique(mask)) == {0, 1}
inside = minimum_filter(mask, size=33, mode='constant', cval=0).astype(bool)
lo = minimum_filter(labels, size=33, mode='constant', cval=0)
hi = maximum_filter(labels, size=33, mode='constant', cval=0)
known = inside & (lo == hi)
del inside, lo, hi
border = np.zeros((384, 384), dtype=bool)
border[32:-32, 32:-32] = True
candidates = []
for y in range(0, 9791 - 1152 + 1, 1152):
    for x in range(0, 6064 - 1152 + 1, 1152):
        sl = (slice(y+1,y+1152,3), slice(x+1,x+1152,3))
        truth = labels[sl]
        primary = known[sl] & border
        fraction = float(mask[y:y+1152,x:x+1152].mean())
        positive = int(np.count_nonzero(primary & (truth == 1)))
        negative = int(np.count_nonzero(primary & (truth == 0)))
        eligible = fraction >= .90 and positive >= 512 and negative >= 8192
        candidates.append(dict(y=y,x=x,surface_fraction=fraction,primary_positive=positive,
                               primary_negative=negative,eligible=eligible))
eligible = [c for c in candidates if c['eligible']]
assert eligible, 'No eligible reference contexts; no relaxation'
first = eligible[0]
separated = [c for c in eligible if c['y'] >= first['y']+2304]
assert separated, 'No eligible separated second context; no relaxation'
selected = [first, separated[-1]]
for i, case in enumerate(selected):
    name = f'frag4-context{i+1}'
    case['id'] = name
    d = ROOT / name
    d.mkdir(exist_ok=False)
    y,x = case['y'],case['x']
    sl = (slice(y+1,y+1152,3),slice(x+1,x+1152,3))
    for filename,array in [('labels.npy',labels[sl]),('primary-mask.npy',known[sl]&border),
                           ('known-mask.npy',known[sl]),('surface-mask.npy',mask[sl].astype(bool))]:
        np.save(d / filename,array,allow_pickle=False)
    case['native_bbox_yxhw'] = [y,x,1152,1152]
    case['files'] = [{'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
                      for f in sorted(d.glob('*.npy'))]
result = {'completed_at_utc':datetime.now(timezone.utc).isoformat(),
          'plan_sha256':hashlib.sha256((ROOT/'selection-plan.json').read_bytes()).hexdigest(),
          'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'candidates':candidates,'selected':selected,'CT_pixel_decodes':0,'model_calls':0,
          'independence_limit':'Two contexts from one fragment, not independent specimen replicates; upstream training exposure unknown.'}
(ROOT/'selection-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'candidate_count':len(candidates),'eligible_count':len(eligible),'selected':selected}))
