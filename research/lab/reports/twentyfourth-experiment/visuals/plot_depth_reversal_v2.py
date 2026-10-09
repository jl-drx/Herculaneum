"""Render phase24 fixed depth-reversal panels after an explicit root release.

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
PROTOCOL_SHA256 = 'f9cc08b99144f55e5842b94e08cb92a8524badafd58d5bf7483254953641c3ee'
CASE_ORDER = ['frag3-context1', 'frag3-context2', 'frag1-context1', 'frag1-context2', 'frag2-context1', 'frag2-context2']
OUTPUT_BUDGET = 32 * 1024 * 1024
INPUT_CONTRACT_SHA256 = 'edb87179a3c8949bd43306f2c6e388c4de657ee9f52ea40f5e1192895c0ceafa'
FORWARD_NATIVE_RESULT_SHA256 = 'c38ad03a287af2c20f365eb1a5edd63f40639b2a0c7dd61922a60abd1493ecf9'
MODELS = ['seed42', 'native']

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
    parser.add_argument('--reverse-seed42-result', type=Path, required=True)
    parser.add_argument('--reverse-seed42-result-sha256', required=True)
    parser.add_argument('--reverse-native-result', type=Path, required=True)
    parser.add_argument('--reverse-native-result-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root-release', action='store_true', required=True,
                        help='Caller confirms root separately released actual array plotting.')
    args = parser.parse_args()
    require(args.root_release, 'Root release is required')
    require(args.output_dir.resolve().is_relative_to(BASE.resolve()), 'Output directory must remain within visual-stage root')
    require(not args.output_dir.exists(), 'Fresh output directory required; no overwrite')
    protocol_pin = pin(PHASE/'PROTOCOL.md')
    require(protocol_pin['sha256'] == PROTOCOL_SHA256, 'Protocol pin mismatch')
    prior_phase = PHASE.parent/'twentythird-experiment'
    contract_path = prior_phase/'input-audit/INPUT_CONTRACT.json'
    contract_pin = pin(contract_path)
    require(contract_pin['sha256'] == INPUT_CONTRACT_SHA256, 'Input contract pin mismatch')
    forward_native_path = prior_phase/'implementation/run-v1/native/result.json'
    forward_native_pin = pin(forward_native_path)
    require(forward_native_pin['sha256'] == FORWARD_NATIVE_RESULT_SHA256, 'Forward native result pin mismatch')
    reverse_paths = {'seed42':args.reverse_seed42_result,'native':args.reverse_native_result}
    reverse_expected = {'seed42':args.reverse_seed42_result_sha256,'native':args.reverse_native_result_sha256}
    reverse_pins = {model:pin(reverse_paths[model]) for model in MODELS}
    for model in MODELS:
        require(reverse_pins[model]['sha256'] == reverse_expected[model], 'Reverse result pin mismatch')
    contract = json.loads(contract_path.read_text())
    forward_native = json.loads(forward_native_path.read_text())
    reverse_results = {model:json.loads(reverse_paths[model].read_text()) for model in MODELS}
    require(forward_native['status'] == 'PASS', 'Forward native worker did not pass')
    require([c['id'] for c in contract['cases']] == CASE_ORDER, 'Case order mismatch')
    require(set(forward_native['runs']) == set(CASE_ORDER), 'Forward native cases missing or extra')
    for model in MODELS:
        require(reverse_results[model]['status'] == 'PASS', 'Reverse worker did not pass')
        require(set(reverse_results[model]['runs']) == set(CASE_ORDER), 'Reverse cases missing or extra')
    code_pin = pin(__file__)

    # Imports occur only after explicit CLI release and metadata pin validation.
    os.environ.setdefault('MPLCONFIGDIR', str(BASE/'matplotlib-cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(BASE/'xdg-cache'))
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

    cases = {model:[] for model in MODELS}
    for original in contract['cases']:
        case_id = original['id']
        primary = array(original['arrays']['primary_mask'], 'bool')
        labels = array(original['arrays']['labels'], 'uint8')
        require(np.all((labels == 0) | (labels == 1)), 'Unexpected supplied label value')
        require(int(primary.sum()) == original['counts']['primary'], 'Original support count mismatch')
        require(int(np.count_nonzero(primary & (labels == 1))) == original['counts']['primary_positive'], 'Positive support count mismatch')
        for model in MODELS:
            reversed_run = reverse_results[model]['runs'][case_id]
            require(reversed_run['id'] == case_id and reversed_run['model'] == model, 'Reverse metadata identity/model mismatch')
            require(reversed_run['adapter_read_direction'] == 'forward' and reversed_run['model_depth_permutation'] == list(range(16,-1,-1)), 'Actual normalized-depth intervention metadata mismatch')
            if model == 'seed42':
                forward_scores = array(original['seed42']['scores'], 'float32')
                forward_evaluated = array(original['seed42']['evaluated'], 'bool')
            else:
                forward_run = forward_native['runs'][case_id]
                require(forward_run['id'] == case_id, 'Forward native identity mismatch')
                forward_scores = array(forward_run['arrays']['raw_scores'], 'float32')
                forward_evaluated = array(forward_run['arrays']['evaluated_mask'], 'bool')
            reverse_scores = array(reversed_run['arrays']['raw_scores'], 'float32')
            reverse_evaluated = array(reversed_run['arrays']['evaluated_mask'], 'bool')
            require(np.all(forward_evaluated) and np.all(reverse_evaluated), 'Incomplete evaluated context')
            for scores in [forward_scores,reverse_scores]:
                require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid raw scores')
            cases[model].append({'id':case_id,'fragment':original['fragment'],'parent':original['parent_papyrus'],
                                'mask':primary,'labels':labels,'forward':forward_scores,'reverse':reverse_scores})

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
    column_titles = ['Supplied reference label','Forward order · raw score','Reversed order · raw score','Reverse minus forward']
    outputs = []
    for model,parent,indices in [('seed42','Paris1',[0,1]),('seed42','Paris2',[2,3,4,5]),('native','Paris1',[0,1]),('native','Paris2',[2,3,4,5])]:
        group = [cases[model][i] for i in indices]
        rows = len(group)
        # Fixed physical panel size and DPI. No data-dependent layout or contrast.
        fig = plt.figure(figsize=(15.6,rows*3.35+1.75),dpi=160,facecolor='white')
        grid = fig.add_gridspec(rows+1,4,height_ratios=[1]*rows+[0.055],
                               left=0.105,right=0.985,top=0.89,bottom=0.085,wspace=0.10,hspace=0.12)
        images = []
        for row, case in enumerate(group):
            values = [case['labels'],case['forward'],case['reverse'],case['reverse'].astype(np.float64)-case['forward'].astype(np.float64)]
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
        fig.suptitle(model+' · '+parent+' · reused development',fontsize=17,y=0.975)
        fig.text(0.545,0.936,'Normalized depth-order permutation · whole 384 × 384 contexts · original primary support',ha='center',fontsize=10)
        fig.legend(handles=[Patch(facecolor=gray,label='Gray: outside original primary mask')],loc='lower center',
                   bbox_to_anchor=(0.54,0.006),frameon=False,fontsize=10)
        path = args.output_dir/(model+'-'+parent.lower()+'-depth-reversal.png')
        fig.savefig(path,dpi=160,pil_kwargs={'compress_level':9},metadata={'Software':'Plinius phase24 fixed-context depth-reversal plotter'})
        plt.close(fig)
        outputs.append(pin(path))
        require(sum(f.stat().st_size for f in BASE.rglob('*') if f.is_file()) <= OUTPUT_BUDGET, 'Whole visual-stage budget exceeded')

    # Verify the exact bound source bytes remain unchanged after plotting.
    for source in source_pins+[protocol_pin,contract_pin,forward_native_pin,code_pin]+list(reverse_pins.values()):
        check_pin(source)
    receipt = {
        'status':'PASS','protocol':protocol_pin,'input_contract':contract_pin,'forward_native_result':forward_native_pin,'reverse_results':reverse_pins,
        'plotter':code_pin,'source_arrays':source_pins,'outputs':outputs,
        'model_order':MODELS,'case_order':CASE_ORDER,'whole_context_shape':[384,384],
        'adapter_read_direction':'forward','reversed_model_depth_permutation':list(range(16,-1,-1)),
        'mask':'Original primary mask only, identical across all four panels; never narrowed by output values.',
        'scales':{'labels':[0,1],'forward_raw':[0,1],'reverse_raw':[0,1],'reverse_minus_forward':[-1,1]},
        'unsupported_color':gray,'interpolation':'nearest','dpi':160,
        'selection':'All six fixed contexts for each model, all384² pixels; no crops, fitted thresholds, adaptive contrast or generated labels.',
        'evidence_role':'Supplied-reference, reused-development diagnostics; no scientific interpretation.',
        'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'max_output_bytes':OUTPUT_BUDGET,'model_calls':0,'network_calls':0,
    }
    write_json(args.output_dir/'PLOT_RECEIPT.json',receipt)
    require(sum(f.stat().st_size for f in BASE.rglob('*') if f.is_file()) <= OUTPUT_BUDGET, 'Receipt-inclusive whole-stage budget exceeded')
    print(json.dumps({'status':'PASS','pngs':len(outputs),'output_bytes':sum(f.stat().st_size for f in args.output_dir.iterdir() if f.is_file())}))

if __name__ == '__main__':
    main()
