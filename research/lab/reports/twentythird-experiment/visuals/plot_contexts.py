"""Render protocol-fixed whole-context panels after an explicit root release.

Does not load models, compute performance metrics, select thresholds or contact networks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
PHASE = BASE.parent
PROTOCOL_SHA256 = '5a8fbf4762ed8dc03c3c13b0f5bcc7acaddaf146cce5987164463161eb76dc02'
CASE_ORDER = ['frag3-context1', 'frag3-context2', 'frag1-context1', 'frag1-context2', 'frag2-context1', 'frag2-context2']
OUTPUT_BUDGET = 16 * 1024 * 1024
INPUT_CONTRACT_SHA256 = 'edb87179a3c8949bd43306f2c6e388c4de657ee9f52ea40f5e1192895c0ceafa'

def require(value, message):
    if not value:
        raise RuntimeError(message)

def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}

def check_pin(desc):
    actual = pin(desc['path'])
    require(actual['sha256'] == desc['sha256'] and actual['size_bytes'] == desc['size_bytes'], 'Source file pin mismatch')
    return actual

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-result', type=Path, required=True)
    parser.add_argument('--native-result-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root-release', action='store_true', required=True,
                        help='Caller confirms root separately released actual array plotting.')
    args = parser.parse_args()
    require(args.root_release, 'Root release is required')
    require(not args.output_dir.exists(), 'Fresh output directory required; no overwrite')
    protocol_pin = pin(PHASE/'PROTOCOL.md')
    require(protocol_pin['sha256'] == PROTOCOL_SHA256, 'Protocol pin mismatch')
    contract_path = PHASE/'input-audit/INPUT_CONTRACT.json'
    contract_pin = pin(contract_path)
    require(contract_pin['sha256'] == INPUT_CONTRACT_SHA256, 'Input contract pin mismatch')
    result_pin = pin(args.native_result)
    require(result_pin['sha256'] == args.native_result_sha256, 'Native result pin mismatch')
    contract = json.loads(contract_path.read_text())
    native_result = json.loads(args.native_result.read_text())
    require(native_result['status'] == 'PASS', 'Native worker did not pass')
    require([c['id'] for c in contract['cases']] == CASE_ORDER, 'Case order mismatch')
    require(set(native_result['runs']) == set(CASE_ORDER), 'Native cases missing or extra')
    code_pin = pin(__file__)

    # Imports occur only after explicit CLI release and metadata pin validation.
    os.environ.setdefault('MPLCONFIGDIR', str(BASE/'matplotlib-cache'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.patches import Patch

    source_pins = []
    def array(desc, dtype):
        source_pins.append(check_pin(desc))
        value = np.load(desc['path'], allow_pickle=False)
        require(value.shape == (384,384) and value.dtype == np.dtype(dtype), 'Array shape/dtype mismatch')
        require(list(value.shape) == desc['shape'] and value.dtype.str == desc['dtype'], 'Descriptor shape/dtype mismatch')
        require(hashlib.sha256(value.tobytes(order='C')).hexdigest() == desc['raw_c_order_sha256'], 'Raw array pin mismatch')
        value.flags.writeable = False
        return value

    cases = []
    for original in contract['cases']:
        case_id = original['id']
        native = native_result['runs'][case_id]
        require(native['id'] == case_id, 'Native metadata identity mismatch')
        primary = array(original['arrays']['primary_mask'], 'bool')
        labels = array(original['arrays']['labels'], 'uint8')
        baseline = array(original['seed42']['scores'], 'float32')
        baseline_evaluated = array(original['seed42']['evaluated'], 'bool')
        candidate = array(native['arrays']['raw_scores'], 'float32')
        candidate_evaluated = array(native['arrays']['evaluated_mask'], 'bool')
        require(np.all(baseline_evaluated) and np.all(candidate_evaluated), 'Incomplete evaluated context')
        require(np.all((labels == 0) | (labels == 1)), 'Unexpected supplied label value')
        require(int(primary.sum()) == original['counts']['primary'], 'Original support count mismatch')
        require(int(np.count_nonzero(primary & (labels == 1))) == original['counts']['primary_positive'], 'Positive support count mismatch')
        for scores in [baseline, candidate]:
            require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid raw scores')
        cases.append({'id':case_id,'fragment':original['fragment'],'parent':original['parent_papyrus'],
                      'mask':primary,'labels':labels,'seed42':baseline,'native':candidate})

    args.output_dir.mkdir(parents=True)
    gray = '#a8a8a8'
    labels_cmap = ListedColormap(['#ffffff','#181818'], name='supplied_binary')
    labels_cmap.set_bad(gray)
    probability_cmap = plt.get_cmap('viridis').copy()
    probability_cmap.set_bad(gray)
    difference_cmap = plt.get_cmap('coolwarm').copy()
    difference_cmap.set_bad(gray)
    label_norm = BoundaryNorm([-0.5,0.5,1.5], labels_cmap.N)
    probability_norm = Normalize(0,1,clip=False)
    difference_norm = Normalize(-1,1,clip=False)
    column_titles = ['Supplied reference label','Seed42 · raw score','Native checkpoint · raw score','Native minus seed42']
    outputs = []
    for parent, indices in [('Paris1',[0,1]),('Paris2',[2,3,4,5])]:
        group = [cases[i] for i in indices]
        rows = len(group)
        # Fixed physical panel size and DPI. No data-dependent layout or contrast.
        fig = plt.figure(figsize=(15.6,rows*3.35+1.75),dpi=160,facecolor='white')
        grid = fig.add_gridspec(rows+1,4,height_ratios=[1]*rows+[0.055],
                               left=0.105,right=0.985,top=0.89,bottom=0.085,wspace=0.10,hspace=0.12)
        images = []
        for row, case in enumerate(group):
            values = [case['labels'],case['seed42'],case['native'],case['native'].astype(np.float64)-case['seed42'].astype(np.float64)]
            for col,value in enumerate(values):
                ax = fig.add_subplot(grid[row,col])
                masked = np.ma.array(value,mask=~case['mask'],copy=False)
                im = ax.imshow(masked,cmap=[labels_cmap,probability_cmap,probability_cmap,difference_cmap][col],
                               norm=[label_norm,probability_norm,probability_norm,difference_norm][col],
                               origin='upper',interpolation='nearest',aspect='equal')
                ax.set_xlim(-0.5,383.5)
                ax.set_ylim(383.5,-0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(column_titles[col],fontsize=11,pad=9)
                if col == 0:
                    ax.text(-0.075,0.5,case['id']+'\n'+case['fragment'],transform=ax.transAxes,
                            ha='right',va='center',fontsize=10,rotation=90)
                if row == 0:
                    images.append(im)
        for col in range(4):
            cax = fig.add_subplot(grid[rows,col])
            bar = fig.colorbar(images[col],cax=cax,orientation='horizontal',
                               ticks=[0,1] if col==0 else ([0,0.5,1] if col in [1,2] else [-1,0,1]))
            if col == 0:
                bar.ax.set_xticklabels(['0 · reference background','1 · supplied ink'])
            bar.ax.tick_params(labelsize=8)
        fig.suptitle(parent+' · supplied reference and reused development',fontsize=17,y=0.975)
        fig.text(0.545,0.936,'Whole 384 × 384 contexts · fixed scales · original primary support',ha='center',fontsize=10)
        fig.legend(handles=[Patch(facecolor=gray,label='Gray: outside original primary mask')],loc='lower center',
                   bbox_to_anchor=(0.54,0.006),frameon=False,fontsize=10)
        path = args.output_dir/(parent.lower()+'-whole-contexts.png')
        fig.savefig(path,dpi=160,pil_kwargs={'compress_level':9},metadata={'Software':'Plinius phase23 fixed-context plotter'})
        plt.close(fig)
        outputs.append(pin(path))
        require(sum(f.stat().st_size for f in args.output_dir.iterdir() if f.is_file()) <= OUTPUT_BUDGET, 'Panel output budget exceeded')

    # Verify the exact bound source bytes remain unchanged after plotting.
    for source in source_pins+[protocol_pin,contract_pin,result_pin,code_pin]:
        check_pin(source)
    receipt = {
        'status':'PASS','protocol':protocol_pin,'input_contract':contract_pin,'native_result':result_pin,
        'plotter':code_pin,'source_arrays':source_pins,'outputs':outputs,
        'case_order':CASE_ORDER,'whole_context_shape':[384,384],
        'mask':'Original primary mask only, identical across all four panels; never narrowed by output values.',
        'scales':{'labels':[0,1],'seed42_raw':[0,1],'native_raw':[0,1],'native_minus_seed42':[-1,1]},
        'unsupported_color':gray,'interpolation':'nearest','dpi':160,
        'selection':'All six fixed contexts, all384² pixels; no crops, fitted thresholds, adaptive contrast or generated labels.',
        'evidence_role':'Supplied-reference, reused-development diagnostics; no scientific interpretation.',
        'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'max_output_bytes':OUTPUT_BUDGET,'model_calls':0,'network_calls':0,
    }
    write_json(args.output_dir/'PLOT_RECEIPT.json',receipt)
    require(sum(f.stat().st_size for f in args.output_dir.iterdir() if f.is_file()) <= OUTPUT_BUDGET, 'Receipt-inclusive output budget exceeded')
    print(json.dumps({'status':'PASS','pngs':len(outputs),'output_bytes':sum(f.stat().st_size for f in args.output_dir.iterdir() if f.is_file())}))

if __name__ == '__main__':
    main()
