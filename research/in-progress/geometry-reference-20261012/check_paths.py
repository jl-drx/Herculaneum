"""Coordinate qualification and CT views of manually proposed lamina traces."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--anchors', type=Path, required=True)
    p.add_argument('--arrays', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text())
    anchors = json.loads(a.anchors.read_text())
    assert anchors['plan_sha256'] == sha(a.plan)
    a.output.mkdir(parents=True)
    metrics = []
    for case in plan['cases']:
        path = a.arrays / case['file']
        assert sha(path) == case['sha256']
        ct = np.load(path, allow_pickle=False)
        assert ct.dtype == np.uint8 and ct.shape == (192, 192, 192)
        chains = [r for r in anchors['paths'] if r['case'] == case['id']]
        canvas = Image.new('RGB', (1584, 844), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, overlay in enumerate([False, True]):
            left, top = 8+col*792, 32
            canvas.paste(Image.fromarray(ct[:,96,:]).resize((768,768),Image.Resampling.NEAREST).convert('RGB'), (left,top))
            draw.text((left,8), case['id'] + (' Y=96; candidate paths, not truth' if overlay else ' Y=96; raw CT; 0..255'), fill='black')
            if overlay:
                for chain, color in zip(chains, [(0,255,255),(255,160,0)]):
                    points = np.asarray(chain['points_zyx'],dtype=np.int64)
                    assert np.all((points>=32)&(points<160)) and np.all(points[:,1]==96)
                    coords=[(left+int(x)*4+2,top+int(z)*4+2) for z,y,x in points]
                    draw.line(coords, fill=color, width=2)
                    for i,(cx,cy) in enumerate(coords):
                        draw.ellipse((cx-6,cy-6,cx+6,cy+6),outline=color,width=2)
                        draw.text((cx+8,cy-5),str(i),fill=color)
                    draw.text(coords[0],chain['id'],fill=color)
        draw.text((8,812),'Manual visible-lamina candidates. Local continuity does not establish physical sheet, recto side, tear or global winding identity.',fill='black')
        canvas.save(a.output/(case['id']+'-paths.png'))
        for chain in chains:
            pts=np.asarray(chain['points_zyx'])
            length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
            origin=np.asarray(case['ct_context_box_zyx'][0])
            metrics.append({'id':chain['id'], 'case':case['id'],
                'points_zyx':pts.tolist(),'global_points_zyx':(pts+origin).tolist(),
                'length_voxels':length,'ct_at_anchors':ct[tuple(pts.T)].tolist(),
                'all_points_inside_original_core':True})
        # Three exact cross-cuts sample the proposed region in the other axes.
        cuts = [('Z',96,ct[96,:,:]),('Z',112,ct[112,:,:]),('Z',128,ct[128,:,:])]
        if case['id']=='PHerc0814-06':
            cuts=[('X',48,ct[:,:,48]),('X',64,ct[:,:,64]),('X',80,ct[:,:,80])]
        cross=Image.new('RGB',(1200,424),'white');d=ImageDraw.Draw(cross)
        for i,(axis,index,plane) in enumerate(cuts):
            left=8+400*i
            cross.paste(Image.fromarray(plane).resize((384,384),Image.Resampling.NEAREST).convert('RGB'),(left,32))
            d.text((left,8),f'{case["id"]} {axis}={index}; '+('rows Y, columns X' if axis=='Z' else 'rows Z, columns Y'),fill='black')
        cross.save(a.output/(case['id']+'-cross-cuts.png'))
    report={'status':'COORDINATES_VERIFIED_NOT_ADJUDICATED', 'plan_sha256':sha(a.plan),
        'anchors_sha256':sha(a.anchors),'script_sha256':sha(Path(__file__)), 'paths':metrics,
        'model_payloads_read':0,'scope':'Source/coordinate checks and raw CT views. CT intensity is descriptive, never an identity/ink score.'}
    (a.output/'COORDINATES.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'paths':len(metrics)}))


if __name__=='__main__':main()
