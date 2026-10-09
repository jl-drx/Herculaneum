"""Plot retained primary metrics without recomputing or changing any scores."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / 'benchmark/run/analysis/metrics.json'
assert hashlib.sha256(METRICS.read_bytes()).hexdigest() == 'b58b21192240ec89bd6821613b26a53e06201a0b76c3aa27f7f8229bb7254dc8'
os.environ['MPLCONFIGDIR'] = '/workspace/.cache/herculaneum-matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = json.loads(METRICS.read_text())
ids = ['pherc0139-w016-rank01', 'pherc0139-w016-rank02',
       'pherc0814-46527-rank01', 'pherc0814-46527-rank07']
rows = [data['contexts'][name]['domains']['primary'] for name in ids]
y = np.arange(4)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True, layout='constrained')
delta = np.array([r['paired_ex016_control']['ex016_minus_control_roc_auc'] for r in rows])
axes[0].barh(y, delta, height=.43, color=['#27827d' if v >= 0 else '#b35c48' for v in delta])
axes[0].axvline(0, color='#333333', linewidth=.8)
for i, value in enumerate(delta):
    axes[0].text(value + (.004 if value >= 0 else -.003), i, f'{value:+.3f}',
                 ha='left' if value >= 0 else 'right', va='center', fontsize=10)
axes[0].set(xlim=(-.035, .213), xlabel='Dense minus manual-control ROC AUC', title='Ranking changes across contexts')
axes[0].set_yticks(y, ['PHerc0139 · A', 'PHerc0139 · B', 'PHerc0814 · C', 'PHerc0814 · D'])
axes[0].invert_yaxis()
styles = [('russo_ex01675k', 'Dense ex016', '#176f80', -.09),
          ('russo_control75k', 'Manual control', '#bb673a', .09)]
for ax, field, title in [(axes[1], 'sensitivity', 'Supplied positives detected'),
                         (axes[2], 'specificity', 'Supplied negatives rejected')]:
    for role, label, color, offset in styles:
        values = [100*r['models'][role]['development_locked_operating_point'][field] for r in rows]
        ax.scatter(values, y + offset, label=label, color=color, s=46, zorder=3)
    ax.set(title=title, xlabel=f'{field.capitalize()} at frozen threshold (%)')
axes[1].set_xlim(-2, 80)
axes[2].set_xlim(90, 100.6)
axes[2].axvline(95, color='#666666', linestyle='--', linewidth=1, label='Development target: 95%')
axes[2].legend(loc='lower left', fontsize=8, frameon=False)
for ax in axes:
    ax.grid(axis='x', color='#dddddd', linewidth=.6)
    ax.set_axisbelow(True)
    ax.axhline(1.5, color='#dddddd', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Dense supervision: improved ranking, uneven threshold transfer\n'
             'Four fixed contexts, two specimens; thresholds selected on used PHerc1667 development data', fontsize=12)
fig.supxlabel('Descriptive agreement with supplied annotations; known model-development corpus. No reserved threshold tuning.', fontsize=9)
for suffix in ['png', 'svg']:
    output = ROOT / f'panel-summary.{suffix}'
    assert not output.exists()
    fig.savefig(output, dpi=170)
plt.close(fig)
