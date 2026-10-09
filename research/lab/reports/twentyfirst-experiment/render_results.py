"""Plot retained allocation metrics; no scientific recomputation or model calls."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'allocation-analysis/metrics.json'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '75ec95d6f03185d78088fd787512b2e5aabf9e23d845db195a9eef4a0986558c'
m = json.loads(SOURCE.read_text())
out = HERE / 'figures-v2'
out.mkdir(exist_ok=False)
arms = [('manual_double', 'Manual ×2', '#2864a5'),
        ('manual_retired', 'Manual + retired', '#d06a28')]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(left=.075, right=.98, top=.84, bottom=.18, hspace=.62, wspace=.28)
fig.suptitle('Plinius: preservation allocation study\nTraining mechanism passed; development screen failed', fontsize=17, y=.98)
steps = [0, 1, 8, 16, 32]
for key, label, color in arms:
    t = m['training'][key]
    points = [t['initial']] + [t['fixed_updates'][str(s)] for s in steps[1:]]
    for ax, field in zip(axes[0], ['base', 'kl_retired']):
        ax.plot(steps, [p[field] for p in points], 'o-', label=label, color=color)
for ax, title, ylabel in zip(axes[0], ['Shared manual training loss', 'Drift on retired-reference outputs'],
                            ['Smoothed binary cross-entropy', 'Mean KL from seed42']):
    ax.set(title=title, xlabel='Optimizer update (fixed observations)', ylabel=ylabel,
           xticks=[0, 8, 16, 24, 32])
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
cases = sorted(m['per_case'])
labels = ['Fr47 / 1', 'Fr47 / 2', 'Fr143 / 1', 'Fr143 / 2', 'Fr34 / 1', 'Fr34 / 2']
x = np.arange(6)
for ax, field, title in zip(axes[1], ['auc', 'ap'], ['AUROC change from seed42', 'Average precision change from seed42']):
    for i, (key, label, color) in enumerate(arms):
        values = [m['per_case'][c]['models'][key][field] - m['per_case'][c]['models']['frozen_seed42'][field] for c in cases]
        ax.bar(x+(i-.5)*.36, values, width=.36, color=color, label=label)
    ax.axhline(0, color='black', linewidth=.8)
    ax.axhline(-.02, color='#a52828', linestyle='--', linewidth=1, label='Candidate retention limit')
    ax.axvline(3.5, color='#888888', linewidth=.8)
    ax.set(title=title, ylabel='Difference', xticks=x, xticklabels=labels, ylim=(-.15, .075))
    ax.tick_params(axis='x', labelrotation=25)
    ax.grid(axis='y', alpha=.2)
    ax.set_xlabel('All six reused development regions: Paris2 (first four), Paris1 (last two)')
axes[1, 1].legend(frameon=False, fontsize=9, loc='lower right')
fig.text(.5, .018, '32 updates per arm • same 5,296-parameter adapter • frozen backbone • no independent confirmation • £0 GPU spend',
         ha='center', fontsize=10)
outputs=[]
for suffix in ['png', 'svg']:
    path = out / ('preservation-allocation.'+suffix)
    fig.savefig(path, dpi=170)
    outputs.append({'path':str(path),'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
receipt={'status':'PASS','source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'outputs':outputs,'new_model_calls':0,'all_six_regions_shown':True,
         'training_points':'Full fixed schedule at updates 0,1,8,16,32; lines connect observations',
         'not_shown_as_comparable':'Unlike own objectives are not used as a common risk scale'}
(out/'presentation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
