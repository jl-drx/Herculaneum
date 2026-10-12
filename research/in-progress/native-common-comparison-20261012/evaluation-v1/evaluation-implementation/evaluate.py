"""Apply the pre-forward evaluation contract only after all jobs complete."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import runner as r

MODELS = ['reader_v2', 'dnative', 'seed42', 'seed43', 'hecate']
ALL = MODELS + ['mean5', 'constant_0.5']
SCALARS = ['ap', 'auc', 'prevalence', 'brier', 'bce', 'ece', 'mean_positive',
           'mean_negative', 'negative_ge_half']


def metrics(y, p):
    y = np.asarray(y, bool).ravel(); p = np.asarray(p, np.float64).ravel()
    assert len(y) == len(p) and len(y) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    positives = int(y.sum()); negatives = int((~y).sum())
    ap = auc = None
    if positives and negatives:
        order = np.argsort(-p, kind='stable'); sy = y[order]; sp = p[order]
        ends = np.r_[np.flatnonzero(sp[1:] != sp[:-1]), len(y)-1]
        tp = np.cumsum(sy)[ends]; fp = ends+1-tp
        ap = float(np.sum(np.diff(np.r_[0, tp])/positives * (tp/(tp+fp))))
        auc = float(np.trapezoid(np.r_[0, tp/positives], np.r_[0, fp/negatives]))
    clipped = np.clip(p, 1e-6, 1-1e-6)
    bins = np.minimum((p*10).astype(int), 9); calibration = []; ece = 0.
    for i in range(10):
        mask = bins == i; count = int(mask.sum())
        mean = float(p[mask].mean()) if count else None
        fraction = float(y[mask].mean()) if count else None
        if count:
            ece += count/len(y)*abs(mean-fraction)
        calibration.append(dict(bin=i, lower=i/10, upper=(i+1)/10, count=count,
                                mean_score=mean, positive_fraction=fraction))
    return dict(pixels=len(y), positives=positives, negatives=negatives,
        ap=ap, auc=auc, prevalence=positives/len(y),
        brier=float(np.mean((p-y)**2)), bce=float(np.mean(np.where(y, -np.log(clipped), -np.log1p(-clipped)))),
        ece=ece, mean_positive=float(p[y].mean()) if positives else None,
        mean_negative=float(p[~y].mean()) if negatives else None,
        negative_ge_half=float((p[~y] >= .5).mean()) if negatives else None, calibration=calibration)


def macro(rows):
    return {key:float(np.mean([m[key] for m in rows])) if all(m[key] is not None for m in rows) else None
            for key in SCALARS}


def load_panel(root, run):
    go_sha = r.sha(run/'GO.json'); complete = r.read(run/'INFERENCE_COMPLETE.json')
    assert complete['status'] == 'ALL_MAPS_COMPLETE' and complete['go_sha256'] == go_sha
    assert complete['completed_jobs'] == 20 and complete['core_maps'] == 120
    for row in complete['receipts']:
        r.check(run, row)
    go = r.read(run/'GO.json')
    for row in go['input_pins']:
        r.check(root, row)
    selection = r.read(root/'SELECTION.json'); panel = []
    for case in selection['cases']:
        for core in case['cores']:
            with np.load(root/'cores'/(core['id']+'.npz')) as ref:
                ink = ref['ink'].copy(); mask = ref['score_mask'].copy()
            maps = {}
            for model in MODELS:
                folder = run/'jobs'/(model+'--'+case['id']); receipt = r.read(folder/'COMPLETE.json')
                assert receipt['binding']['go_sha256'] == go_sha
                path = folder/(core['id']+'.npz')
                pin = next(row for row in receipt['outputs'] if row['path'] == path.name)
                r.check(folder, pin)
                with np.load(path) as z:
                    maps[model] = z['aligned'].copy()
            maps['mean5'] = np.mean(np.stack([maps[m] for m in MODELS]), axis=0, dtype=np.float32)
            maps['constant_0.5'] = np.full((128, 128), .5, np.float32)
            panel.append(dict(case=case['id'], parent=case['sample'], core=core, ink=ink, mask=mask, maps=maps))
    assert len(panel) == 24 and sum(int(x['mask'].sum()) for x in panel) == 232250
    assert sum(int((x['ink'] & x['mask']).sum()) for x in panel) == 20137
    return panel


def summarize(panel):
    output = {key:[] for key in ['core', 'segment', 'parent_pooled', 'parent_macro', 'panel_pooled', 'equal_parent_macro']}
    def pooled(rows, model):
        y = np.concatenate([r['ink'][r['mask']] for r in rows])
        p = np.concatenate([r['maps'][model][r['mask']] for r in rows])
        return metrics(y, p)
    cases = list(dict.fromkeys(x['case'] for x in panel)); parents = list(dict.fromkeys(x['parent'] for x in panel))
    for model in ALL:
        for row in panel:
            output['core'].append(dict(model=model, id=row['core']['id'], parent=row['parent'],
                case=row['case'], metrics=pooled([row], model)))
        for case in cases:
            rows = [x for x in panel if x['case'] == case]
            output['segment'].append(dict(model=model, id=case, parent=rows[0]['parent'], metrics=pooled(rows, model)))
        parent_means = []
        for parent in parents:
            rows = [x for x in panel if x['parent'] == parent]
            output['parent_pooled'].append(dict(model=model, id=parent, metrics=pooled(rows, model)))
            segments = [x['metrics'] for x in output['segment'] if x['model'] == model and x['parent'] == parent]
            m = macro(segments); parent_means.append(m)
            output['parent_macro'].append(dict(model=model, id=parent, metrics=m))
        output['equal_parent_macro'].append(dict(model=model, id='equal_parent_macro', metrics=macro(parent_means)))
        output['panel_pooled'].append(dict(model=model, id='panel_pooled', metrics=pooled(panel, model)))
    return output


def comparisons(summary):
    scores = {x['id']:{m:next(r['metrics']['ap'] for r in summary['parent_macro']
                             if r['id'] == x['id'] and r['model'] == m) for m in ALL}
              for x in summary['parent_macro']}
    primary = {x['model']:x['metrics']['ap'] for x in summary['equal_parent_macro']}
    gap = dict(equal_parent_ap=abs(primary['seed42']-primary['seed43']),
               parent_ap={p:abs(v['seed42']-v['seed43']) for p,v in scores.items()})
    rows = []
    for i, left in enumerate(ALL[:-1]):
        for right in ALL[i+1:-1]:
            deltas = {p:v[left]-v[right] for p,v in scores.items()}
            rows.append(dict(left=left, right=right, equal_parent_ap_delta=primary[left]-primary[right],
                parent_ap_delta=deltas, opposite_parent_order=min(deltas.values()) < 0 < max(deltas.values()),
                macro_difference_smaller_than_released_seed_gap=abs(primary[left]-primary[right]) < gap['equal_parent_ap'],
                parents_difference_smaller_than_seed_gap=[p for p,d in deltas.items() if abs(d) < gap['parent_ap'][p]]))
    return dict(released_checkpoint_variation=gap, pairs=rows,
                decision='Development comparison only; no general-winner, physical-accuracy or text claim',
                training_decision='Does not close an adequately powered coverage-training hypothesis')


def figures(root, out, panel):
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    columns = ['native CT', 'reference', 'score mask', *MODELS, 'mean5']
    paths = []
    for case in dict.fromkeys(x['case'] for x in panel):
        rows = [x for x in panel if x['case'] == case]
        image = Image.new('RGB', (len(columns)*138, 76+len(rows)*160), 'white'); draw = ImageDraw.Draw(image)
        draw.text((6, 4), case+' | Fixed display; all six selected cores', fill='black', font=font)
        draw.text((6, 22), 'CT: raw uint8. Reference: white=positive, black=negative, grey=outside score mask. Predictions: 0..1.', fill='black', font=font)
        for col, title in enumerate(columns):
            draw.text((col*138+4, 51), title, fill='black', font=font)
        for ri, row in enumerate(rows):
            core = row['core']; x, y = core['x'], core['y']
            patch = root/'model-patches'/case/f'hybrid-y{y:05d}-x{x:05d}.npy'
            ct = np.load(patch, allow_pickle=False)[8]
            reference = np.where(row['mask'], row['ink'].astype(np.uint8)*255, 100).astype(np.uint8)
            arrays = [ct, reference, row['mask'].astype(np.uint8)*255]
            arrays += [np.rint(row['maps'][m]*255).astype(np.uint8) for m in MODELS+['mean5']]
            yy = 76+ri*160
            draw.text((4, yy), f'x{x} y{y} | n={row["mask"].sum()}, positives={(row["mask"] & row["ink"]).sum()}', fill='black', font=font)
            for ci, array in enumerate(arrays):
                image.paste(Image.fromarray(array).convert('RGB'), (ci*138+4, yy+20))
        path = out/(case+'.png'); image.save(path); paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(); p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    out = a.run/'evaluation'; assert not (out/'METRICS.json').exists(), 'Do not overwrite evaluation'
    panel = load_panel(a.inputs, a.run); summary = summarize(panel); comparison = comparisons(summary)
    out.mkdir(exist_ok=True)
    r.atomic(out/'METRICS.json', dict(go_sha256=r.sha(a.run/'GO.json'), utc=r.now(),
             definitions='implementation/PROTOCOL.md', summary=summary, comparisons=comparison,
             implementation=r.pin(Path(__file__))))
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['level','model','id','pixels','positives','negatives',*SCALARS])
        writer.writeheader()
        for level, values in summary.items():
            for row in values:
                writer.writerow(dict(level=level,model=row['model'],id=row['id'],
                    **{k:v for k,v in row['metrics'].items() if k != 'calibration'}))
    figures(a.inputs, out, panel)
    print(__import__('json').dumps(dict(equal_parent_macro=summary['equal_parent_macro'], comparisons=comparison),indent=2))


if __name__ == '__main__':
    main()
