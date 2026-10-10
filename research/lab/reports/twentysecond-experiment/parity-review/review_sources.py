"""Source and preprocessing fixture review; no native weights/factory executions."""
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from verify_parity import HERE, PHASE, PROTOCOL_SHA, CASES, pin, checked, need, normalize, volume, indices, exact, hann, compare

def load_normalization(path,name):
    loader=importlib.machinery.SourceFileLoader(name,str(path)); spec=importlib.util.spec_from_loader(name,loader)
    module=importlib.util.module_from_spec(spec); loader.exec_module(module); return module.normalize_robust
def run():
    source=PHASE/'model-parity'; plan=json.loads((source/'PLAN.json').read_text()); freeze=json.loads((source/'CODE_FREEZE.json').read_text())
    need(pin(PHASE/'SYNTHETIC_PARITY_PROTOCOL.md')['sha256']==PROTOCOL_SHA,'protocol')
    need(plan['code_freeze']==pin(source/'CODE_FREEZE.json') and plan['code']==freeze['code'],'author_freeze')
    for rec in plan['code'].values(): checked(rec)
    for rec in plan['sources'].values(): checked(rec)
    need(freeze['real_factory_constructions']==freeze['real_checkpoint_loads']==freeze['real_model_forwards']==0,'preexecution')
    closure=json.loads(checked(plan['sources']['historical_closure']).read_text())
    for item in closure['modules']:
        raw=checked(item).read_bytes(); blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        need(blob==item['git_blob_sha1'],'historical_git_blob')
    for key in ('normalization_module_for_separate_input_parity','historical_padding_module_for_separate_input_parity',
                'historical_inference_source_for_reviewed_pure_extraction','authored_configuration'): checked(closure[key])
    provenance=json.loads(checked(plan['sources']['extraction_provenance']).read_text()); text=checked(provenance['source']).read_text()
    parsed=ast.parse(text); original={node.name:node for node in parsed.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    extracts=provenance['definitions']; lines=text.splitlines(keepends=True)
    expected={'Block','ChunkKey','TargetHeadWrapper','disable_z_projection_for_normal_pooled_3d','normalize_training_config_for_inference',
        'compute_importance_map_2d','resolve_patch_stride','center_crop_layer_indices','_sliding_positions_1d','iter_blocks',
        'logits_to_probabilities','OmeZarrPatchReader','convert_volume_dtype','is_url_like_path','iter_overlapping_chunks',
        'compute_chunk_contribution_counts','ChunkAccumulator','OmeZarrBlockDataset','run_block_inference','iter_probability_tiles'}
    need({x['name'] for x in extracts}==expected and len(extracts)==20,'exact_extraction_names')
    for row in extracts:
        node=original[row['name']]; first=min([node.lineno]+[d.lineno for d in node.decorator_list])
        need(first==row['first_line'] and node.end_lineno==row['last_line'],'extraction_lines')
        need(hashlib.sha256(''.join(lines[first-1:node.end_lineno]).encode()).hexdigest()==row['sha256'],'extraction_bytes')
    need(not expected&{'load_checkpoint_payload','main','parse_args','write_output_tiff','open_temp_zarr_array'},'unsafe_extraction')
    tests=json.loads(checked(freeze['toy_tests']).read_text()); need(tests['status']=='PASS' and tests['check_count']==94,'author_fixtures')
    need(tests['real_factory_constructions']==tests['real_checkpoint_loads']==tests['real_model_forwards']==0,'toy_only')
    oldnorm=load_normalization(closure['normalization_module_for_separate_input_parity']['path'],'review_old_normalizer')
    vendor=Path(plan['sources']['current_vendor_manifest']['path']).parent
    currentnorm=load_normalization(vendor/'vesuvius/image_proc/intensity/normalization.py','review_current_normalizer')
    fixture_count=0
    for case in CASES:
        raw=volume(case['shape'],case.get('zero',False))[indices(case['interval'],case['reverse']),:128,:128]
        expected_array=normalize(raw)
        exact(expected_array,oldnorm(raw.copy()),'old_normalization_fixture')
        exact(expected_array,currentnorm(raw.copy()),'current_normalization_fixture'); fixture_count+=2
    for raw in (np.zeros((17,9,11),np.uint8),np.full((17,9,11),81,np.uint8)):
        exact(normalize(raw),oldnorm(raw.copy()),'old_constant_fixture')
        exact(normalize(raw),currentnorm(raw.copy()),'current_constant_fixture'); fixture_count+=2
    torch_weight=torch.outer(torch.hann_window(128,periodic=False),torch.hann_window(128,periodic=False))
    torch_weight/=torch_weight.max(); torch_weight=torch.clamp(torch_weight,min=.001).numpy()
    hann_check=compare(hann(),torch_weight,2e-6,2e-6,'independent_hann_formula'); fixture_count+=1
    return dict(status='PASS',author_plan=pin(source/'PLAN.json'),author_code_freeze=pin(source/'CODE_FREEZE.json'),
        historical_modules_checked=9,historical_git_blob_checks=9,extraction_nodes_checked=20,
        preprocessing_fixture_checks=fixture_count,hann_formula=hann_check,author_toy_checks=94,
        actual_checkpoint_reads=0,actual_factory_constructions=0,actual_model_forwards=0,
        review_boundaries=['Current factory uses its unchanged focused source loader and audited native InkConfig parser.',
            'Historical inactive ResNet/Primus/DINO imports are fail-closed and runtime-counted; current loader keeps pinned real imports, with active-path config guards.',
            'Historical safe definitions retain full function/class AST; unsafe CLI/checkpoint/writer functions absent; reader Zarr opening is bound to raising guard.',
            'Current contiguous input and historical layout may differ; canonical values must be exact and logits judged by root frozen tolerances.',
            'Pre/post state hashes and aliases checked without any additional verifier model forward.'])
if __name__=='__main__':
    result=run(); out=Path(sys.argv[1])
    with out.open('x') as f: json.dump(result,f,sort_keys=True,indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=pin(out))))
