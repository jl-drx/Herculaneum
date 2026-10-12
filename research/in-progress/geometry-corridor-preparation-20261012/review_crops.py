"""Enlarge the fixed failed 0814 reference corridor, with no model fields."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    raw=a.input.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest=='52542fcd01069065819e6341e20d64f680c6007967f9d42d83f65b41e0e4ba55'
    ct=np.load(a.input,allow_pickle=False);assert ct.shape==(192,192,192) and ct.dtype==np.uint8
    a.output.mkdir(parents=True)
    ys=list(range(72,121,4));paths=[]
    for y in ys:
        panel=Image.new('RGB',(752,464),'white');draw=ImageDraw.Draw(panel)
        panel.paste(Image.fromarray(ct[72:120,y,24:104]).resize((640,384),Image.Resampling.NEAREST).convert('RGB'),(64,40))
        draw.text((64,5),f'PHerc0814-06 | Y={y} | rows Z=72:120, columns X=24:104 | raw 0..255',fill='black')
        for x in range(24,105,8):draw.text((64+(x-24)*8,24),str(x),fill='black')
        for z in range(72,121,4):draw.text((30,40+(z-72)*8),str(z),fill='black')
        draw.text((64,440),'CT-only selected development corridor. No recto or whole-sheet reference.',fill='black')
        path=a.output/f'Y{y:03}.png';panel.save(path);paths.append(path)
    result=dict(status='FIXED_CT_CORRIDOR_REVIEW',source_sha256=digest,original_case='PHerc0814-06',
        roi_zx=[72,120,24,104],serial_y=ys,scope='Enlarges the original failed-path neighbourhood; no new context or model payload',
        outputs=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'PLAN_AND_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
