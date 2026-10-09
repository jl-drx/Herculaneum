"""Display frozen endpoints; no new model calls, metrics or threshold fitting."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / 'implementation/analysis/metrics.json'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'ce8354f3a023b105b5e02aeabd4fbf512915ec81d3b76e7b999b2e16a9ddb89b'
data = json.loads(source.read_text())
arms = ['frozen_seed42', 'A', 'B', 'C']
cases = ['pherc0814-46527-fresh-rank01', 'pherc0814-46527-fresh-rank14']
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, layout='constrained')
colors = ['#146d82', '#ba643e']
for index, case in enumerate(cases):
    offset = -.12 if index == 0 else .12
    y = np.arange(4) + offset
    rows = [data['fresh'][case]['models'][arm] for arm in arms]
    for axis, field in [(axes[0], 'auc'), (axes[1], 'ap')]:
        values = [row[field] for row in rows]
        axis.scatter(values, y, color=colors[index], s=42, zorder=3)
        for value, yy in zip(values, y):
            axis.annotate(f'{value:.4f}', (value, yy), xytext=(-7, 0), textcoords='offset points',
                          ha='right', va='center', fontsize=8, color=colors[index])
    primary = [100 * row['operating']['primary']['sensitivity'] for row in rows]
    secondary = [100 * row['operating']['secondary']['sensitivity'] for row in rows]
    axes[2].scatter(primary, y, color=colors[index], s=42, zorder=4, label=f'Fresh region {index+1}')
    for value, alternative, yy in zip(primary, secondary, y):
        axes[2].annotate(f'{value:.1f}', (value, yy), xytext=(-7, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=8, color=colors[index])
        if abs(value - alternative) > .001:
            axes[2].plot([value, alternative], [yy, yy], color=colors[index], linewidth=1, linestyle='--')
            axes[2].scatter([alternative], [yy], facecolors='white', edgecolors=colors[index], s=42, zorder=4)
axes[0].set_yticks(range(4), ['Frozen seed42', 'A · Manual', 'B · Hard pseudo', 'C · Balanced anchors'])
axes[0].invert_yaxis()
axes[0].set(xlim=(.96, 1.001), title='Ranking: ROC AUC', xlabel='AUC (higher is better)')
axes[1].set(xlim=(.925, 1.001), title='Ranking: average precision', xlabel='Average precision')
axes[2].set(xlim=(0, 100), title='Supplied positives detected', xlabel='Sensitivity at frozen cutoff (%)')
axes[2].scatter([], [], facecolors='white', edgecolors='#444444', s=42, label='Secondary cutoff, when different')
axes[2].legend(loc='upper left', fontsize=7, frameon=False)
for axis in axes:
    axis.grid(axis='x', color='#dddddd', linewidth=.6)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('The 32-step candidate failed the fixed improvement screen\n'
             'Two fresh PHerc0814 contexts; identical initialization, inputs and update count', fontsize=12)
fig.supxlabel('Primary specificity ≥99.44% in every row. Known historical model/annotation overlap; supplied-reference agreement only.', fontsize=8)
for suffix in ['png', 'svg']:
    output = ROOT / f'results-summary-v2.{suffix}'
    assert not output.exists()
    fig.savefig(output, dpi=180)
plt.close(fig)
