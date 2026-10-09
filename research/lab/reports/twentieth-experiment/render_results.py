"""Present complete fixed training/development metrics; no model execution or fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/herculaneum-phase20-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/herculaneum-phase20-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def record(path):
    p = Path(path).resolve()
    return dict(path=str(p), size_bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--expected-metrics-sha256', required=True)
args = parser.parse_args()
source = HERE / 'analysis/metrics.json'
assert record(source)['sha256'] == args.expected_metrics_sha256
data = json.loads(source.read_text())
assert data['status'] == 'PASS'
output = HERE / 'figures'
output.mkdir(exist_ok=False)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout='constrained')
colors = {'raw_control': '#b07535', 'explicit_depth': '#286d91'}
for arm, color in colors.items():
    training = data['training'][arm]
    steps = [0, 1, 8, 16, 32]
    for objective, style in [('full', '-'), ('base', '--')]:
        change = [0] + [training['fixed_updates'][str(step)][objective] - training['initial'][objective]
                        for step in steps[1:]]
        axes[0].plot(steps, change, style, marker='o', markersize=4, color=color,
                     label=arm.replace('_', ' ') + ' / ' + objective)
axes[0].axhline(0, color='gray', linewidth=.8)
axes[0].set(xlabel='Completed updates', ylabel='Fixed training-risk change from initializer',
            title='Full and base training objectives', xticks=[0, 1, 8, 16, 32])
axes[0].legend(fontsize=8)
ids = sorted(data['per_case'])
assert len(ids) == 6
positions = np.arange(len(ids))
all_differences = []
for axis, reference, title in [(axes[1], 'raw_control', 'Candidate versus matched raw control'),
                                (axes[2], 'frozen_seed42', 'Candidate versus unchanged seed42')]:
    for offset, metric, color in [(-.16, 'auc', '#286d91'), (.16, 'ap', '#4a8a63')]:
        differences = [data['per_case'][case]['models']['explicit_depth'][metric]
                       - data['per_case'][case]['models'][reference][metric] for case in ids]
        all_differences.extend(differences)
        axis.barh(positions + offset, differences, height=.29, label=metric.upper(), color=color)
    axis.axvline(0, color='gray', linewidth=.8)
    axis.set(yticks=positions,
             yticklabels=[case.replace('frag', 'F').replace('-context', ' / C') for case in ids],
             xlabel='Explicit depth minus reference', title=title, ylim=(6, -.6))
    axis.legend(loc='lower right', fontsize=8)
bound = max(max(abs(value) for value in all_differences) * 1.15, .001)
for axis in axes[1:]:
    axis.set_xlim(-bound, bound)
axes[0].text(0, -.2, 'Same PHerc0139 data, initialization and 32 updates.\nFull = smoothed manual BCE + preservation KL.',
             transform=axes[0].transAxes, fontsize=8)
axes[1].text(0, -.2, 'All six reused development regions shown.\nShared horizontal scale across both comparisons.',
             transform=axes[1].transAxes, fontsize=8)
axes[2].text(0, -.2, 'No fresh confirmation or cutoff fitting.\nFeature screen: ' +
             ('PASS' if data['feature_screen_all'] else 'FAIL') + '; no automatic promotion.',
             transform=axes[2].transAxes, fontsize=8)
fig.suptitle('Plinius: matched 5,296-parameter depth-fusion study', fontsize=14)
for suffix in ['png', 'svg']:
    fig.savefig(output / ('depth-fusion-study.' + suffix), dpi=180)
plt.close(fig)
receipt = dict(status='PASS', source=record(__file__), metrics=record(source),
               outputs=[record(output / ('depth-fusion-study.' + suffix)) for suffix in ['png', 'svg']],
               new_model_calls=0, threshold_fits=0)
with (output / 'presentation-receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status='PASS', receipt=record(output / 'presentation-receipt.json'))))
