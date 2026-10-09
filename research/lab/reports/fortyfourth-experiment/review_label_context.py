"""View already retained full labels; no acquisition, crop reselection or model."""
import os
os.environ['MPLCONFIGDIR']='/tmp/herculaneum-mpl';os.environ['XDG_CACHE_HOME']='/tmp/herculaneum-cache'
from pathlib import Path
import json,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
cases=json.loads((P/'CROP_SELECTION.json').read_text());fig,axs=plt.subplots(1,3,figsize=(12,5),dpi=120)
for ax,c in zip(axs,cases):
 a=np.array(Image.open(P/'pixel-responses'/f"{c['case']}-label-full-hop1.bin"));ax.imshow(a,cmap='gray',vmin=0,vmax=255);ax.add_patch(plt.Rectangle((c['x'],c['y']),c['size'],c['size'],fill=False,color='red'));ax.set_title(c['case']);ax.axis('off')
fig.suptitle('Original grayscale annotations — fixed acquired crops outlined in red')
fig.text(.5,.02,'Vesuvius Challenge (2026), doi:10.1038/s41598-026-58467-1 • CC BY-NC 4.0',ha='center');fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(P/'label-context.png')
