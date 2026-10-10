from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=Path(__file__).resolve().parent
packet=root.parent/'packet-v1'
manifest=json.loads((packet/'COMPLETE.json').read_text())
provenance=[]
for rec in manifest['records']:
    identity=rec['id']
    arr_path=Path(rec['arrays']['path'])
    digest=hashlib.sha256(arr_path.read_bytes()).hexdigest()
    assert digest == rec['arrays']['sha256']
    with np.load(arr_path) as a:
        op=a['optical_RGBA']
        ct=a['CT_mean3']
        Image.fromarray(op).save(root/f'{identity}-optical-native.png')
        fig,axes=plt.subplots(2,3,figsize=(12,8.5),layout='constrained')
        axes.flat[0].imshow(op, interpolation='nearest',extent=(-.5,383.5,383.5,-.5))
        axes.flat[0].set_title('Optical: retained RGBA, 0–255')
        for ax,layer in zip(axes.flat[1:],[1,10,20,30,40]):
            ax.imshow(ct[layer-1],cmap='gray',vmin=0,vmax=65535,
                      interpolation='nearest',extent=(-.5,383.5,383.5,-.5))
            ax.set_title(f'CT layer {layer}: fixed 0–65535')
        for ax in axes.flat:
            ax.set_xticks([0,96,192,288,383]); ax.set_yticks([0,96,192,288,383])
            ax.set_xlabel('Local raw column'); ax.set_ylabel('Local raw row')
        fig.suptitle(f'{identity}: display comparison only; no accepted correspondences\n'
                     'CT pixels are 3×3 means; raster orientation retained; layer is a historical index')
        fig.savefig(root/f'{identity}-detail.png',dpi=140)
        plt.close(fig)
    provenance.append({'id':identity,'arrays_path':str(arr_path),'sha256':digest,
                       'native_optical_output':f'{identity}-optical-native.png',
                       'detail_output':f'{identity}-detail.png','detail_layers':[1,10,20,30,40]})
(root/'render-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Rendered 4 native optical PNGs and 4 fixed-scale detail panels; NPZ hashes match packet manifest.')
