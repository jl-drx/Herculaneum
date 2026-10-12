"""Render fixed CT-only serial planes; no surface or ink prediction input."""
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
    p.add_argument('--arrays', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text())
    a.output.mkdir(parents=True)
    outputs = []
    for case in plan['cases']:
        path = a.arrays / case['file']
        assert sha(path) == case['sha256']
        ct = np.load(path, allow_pickle=False)
        assert ct.dtype == np.uint8 and ct.shape == (192, 192, 192)
        planes = plan['serial_y_indices']
        for page, start in enumerate(range(0, len(planes), 6)):
            slab = planes[start:start + 6]
            canvas = Image.new('RGB', (1200, 836), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, y in enumerate(slab):
                left, top = 8 + (i % 3) * 400, 32 + (i // 3) * 416
                image = Image.fromarray(ct[:, y, :]).resize((384, 384), Image.Resampling.NEAREST).convert('RGB')
                canvas.paste(image, (left, top))
                draw.text((left, top-22), f'{case["id"]} Y={y}; columns X, rows Z; 0..255', fill='black')
                # Axes are outside the unmodified CT image.
                for tick in range(0, 192, 32):
                    draw.text((left + tick*2, top+384), str(tick), fill='black')
            out = a.output / f'{case["id"]}-serial-{page+1}.png'
            canvas.save(out)
            outputs.append(out)
        central = Image.fromarray(ct[:, 96, :]).resize((768, 768), Image.Resampling.NEAREST).convert('RGB')
        canvas = Image.new('RGB', (832, 832), 'white')
        canvas.paste(central, (48, 40))
        draw = ImageDraw.Draw(canvas)
        for tick in range(0, 193, 16):
            draw.line((48+tick*4,40,48+tick*4,808), fill=(100,35,35), width=1)
            draw.line((48,40+tick*4,816,40+tick*4), fill=(100,35,35), width=1)
            draw.text((48+tick*4,20), str(tick), fill='black')
            draw.text((16,40+tick*4), str(tick), fill='black')
        draw.text((48,812), 'Y=96; X increases right, Z down; reference candidates only', fill='black')
        out = a.output / f'{case["id"]}-coordinate-view.png'
        canvas.save(out)
        outputs.append(out)
    receipt = {'status': 'PASS', 'plan_sha256': sha(a.plan), 'script_sha256': sha(Path(__file__)),
               'serial_planes_per_case': len(plan['serial_y_indices']), 'model_payloads_read': 0,
               'inputs': plan['cases'], 'outputs': [{'file': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size}
                                                   for p in outputs],
               'scope': 'Source hash verification and prescribed CT display; no sheet identity adjudication.'}
    (a.output / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': receipt['status'], 'panels': len(outputs)}))


if __name__ == '__main__':
    main()
