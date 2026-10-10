"""Additive v2: verified unique storage/alias accounting and observed checkpoint step."""
from collections.abc import Mapping
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

LAB = Path('/workspace/herculaneum-lab')
HERE = Path(__file__).resolve().parent
REFERENCE = LAB / 'run_ink_reference.py'
REFERENCE_SHA = '16bc6e9fb080eb5c2fcc1f20975481881e513b36b7533c1987bdb422340fea9f'
VENDOR_SHA = '2165b5def9d0dede136dde8202993be800b121cb2feb13cf98e98cd0629d75df'
RUSSO_REVISION = '3e5e7ed44e04606cf4a3766cf83a016543a30d83'
RELEASED_REVISION = '7109667e2607db1b90c37c8b09cb876ea7fe7bb1'
PINS = {
    'russo_ex01675k': ('domenicor046/ink9um-dense', RUSSO_REVISION,
        'dense9um-w016excluded-step075000.pth', 138367629,
        '1d7b99bc641b459736a762b2a1d8c357f61ada4f0117c24c7327b6a1293ba1eb'),
    'russo_control75k': ('domenicor046/ink9um-dense', RUSSO_REVISION,
        'control-manuallabels-step075000.pth', 138366975,
        'e60cee6b2c08a021ddfa9683869dab91787531cef762d833cc1f21fdee7fa114'),
    'seed42': ('scrollprize/ink_9um', RELEASED_REVISION,
        'hybrid_3d2d-seed42/step-075000.pth', 138360039,
        'e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab'),
    'seed43': ('scrollprize/ink_9um', RELEASED_REVISION,
        'hybrid_3d2d-seed43/step-075000.pth', 138360231,
        '2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f'),
}
MODEL_CONFIG = dict(autoconfigure=True, basic_encoder_block='BasicBlockD',
                    basic_decoder_block='ConvBlock', stem_channels=16, z_projection_mode='none')
TARGETS = {'ink': dict(activation='none', out_channels=1, z_projection_mode='none')}
NORMALIZATION = dict(mode='robust_mad', percentile_lower=1.0, percentile_upper=99.0)
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_STATE_BYTES = 160 * 1024 * 1024
MAX_TENSORS = 4096
MAX_PARAMETERS = 40000000


class ContractError(ValueError):
    """A controlled diagnostic that contains no transport URLs or credentials."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_plan(path, expected_sha):
    """Verify a caller-pinned plan and every retained source/code input."""
    path = Path(path).resolve()
    require(path.parent == HERE and sha(path) == expected_sha, 'Plan location or SHA changed')
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1 and plan['paid_compute_authorized'] is False,
            'Unsupported plan or paid authorization')
    require({a['label'] for a in plan['artifacts']} == set(PINS), 'Four exact model roles required')
    for artifact in plan['artifacts']:
        require((artifact['repository'],artifact['revision'],artifact['repository_path'],
                 artifact['size_bytes'],artifact['sha256']) == PINS[artifact['label']], 'Plan artifact pin changed')
    for record in plan['verified_inputs']:
        input_path = Path(record['path'])
        require(input_path.stat().st_size == record['size_bytes'] and sha(input_path) == record['sha256'],
                'Retained code/metadata input changed: '+input_path.name)
    return plan


def verify_artifact(path, artifact):
    require(artifact['label'] in PINS, 'Unknown checkpoint role')
    pin = PINS[artifact['label']]
    require((artifact['repository'], artifact['revision'], artifact['repository_path'],
             artifact['size_bytes'], artifact['sha256']) == pin, 'Artifact differs from published immutable pin')
    require(Path(path).stat().st_size == pin[3] and sha(path) == pin[4],
            'Checkpoint size or SHA mismatch; preserve for diagnosis')


def load_runtime():
    """Return the verified unchanged adapter and focused upstream factory/normalizer once."""
    require(sha(REFERENCE) == REFERENCE_SHA, 'Reference adapter source changed')
    require(sha(LAB/'vendor/villa-ink-baseline/source-manifest.json') == VENDOR_SHA,
            'Focused source manifest changed')
    spec = importlib.util.spec_from_file_location('sixth_verified_reference', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    InkConfig, make_model, normalize_robust, manifest_sha = module.load_focused_sources()
    require(manifest_sha == VENDOR_SHA, 'Verified factory manifest differs')
    return dict(reference=module, InkConfig=InkConfig, make_model=make_model,
                normalize_robust=normalize_robust, source_manifest_sha256=manifest_sha)


def validate_authored_config(authored):
    """Reject resource/architecture changes before constructing the model."""
    require(authored.get('mode') == 'flat' and authored.get('model_type') == 'vesuvius_unet_3d_stem_2d',
            'Unexpected model or data mode')
    require(authored.get('patch_size') == [17,128,128]
            and authored.get('crop_size', [17,128,128]) == [17,128,128]
            and authored.get('in_channels') == 1, 'Wrong native input contract')
    require(authored.get('model_config') == MODEL_CONFIG and authored.get('targets') == TARGETS,
            'Architecture/target config differs from retained baseline')
    require(authored.get('image_normalization') == NORMALIZATION,
            'Normalization config differs from retained robust-MAD contract')
    require(authored.get('loss', {}).get('bce_label_smoothing') == .5
            and authored.get('loss', {}).get('dice_label_smoothing') == 0,
            'Score training semantics differ from smoothed baseline')
    require(not authored.get('use_stitched_forward', False)
            and not authored.get('ema', {}).get('enabled', False), 'Unexpected stitched or EMA training mode')


def storage_groups(state):
    groups = {}
    for key in sorted(state):
        identity = state[key].untyped_storage()._cdata
        groups.setdefault(identity, []).append(key)
    return sorted(groups.values(), key=lambda names: names[0])


def verify_storage_contract(state, expected_state, torch):
    groups, factory_groups = storage_groups(state), storage_groups(expected_state)
    named_bytes = sum(t.numel()*t.element_size() for t in state.values())
    unique_bytes = sum(state[names[0]].untyped_storage().nbytes() for names in groups)
    require(unique_bytes <= MAX_STATE_BYTES, 'Unique selected storage byte bound exceeded')
    require(groups == factory_groups, 'Checkpoint alias partition differs from factory')
    require(len(state) == 508 and len(groups) == 264 and named_bytes == 272701704
            and unique_bytes == 138185992, 'Observed fixed four-artifact storage contract differs')
    alias_records = []
    for names in groups:
        first = state[names[0]]
        factory_first = expected_state[names[0]]
        require(first.untyped_storage().nbytes() == factory_first.untyped_storage().nbytes(),
                'Storage allocation differs from factory')
        for key in names:
            require(state[key].shape == first.shape and state[key].stride() == first.stride()
                    and state[key].storage_offset() == first.storage_offset()
                    and state[key].dtype == first.dtype, 'Alias is not an exact duplicate tensor view')
            require(state[key].stride() == expected_state[key].stride()
                    and state[key].storage_offset() == expected_state[key].storage_offset(),
                    'Checkpoint tensor view layout differs from factory')
            require(torch.equal(first.contiguous().reshape(-1).view(torch.uint8),
                                state[key].contiguous().reshape(-1).view(torch.uint8)),
                    'Aliased tensor values differ bitwise')
        if len(names) > 1:
            alias_records.append(dict(keys=names,storage_bytes=first.untyped_storage().nbytes(),
                                      exact_same_view=True,bitwise_equal=True))
    require(len(alias_records) == 244, 'Fixed factory alias-group count differs')
    return dict(named_tensor_bytes=named_bytes,unique_storage_bytes=unique_bytes,
                duplicate_named_bytes=named_bytes-unique_bytes,tensor_count=len(state),
                unique_storage_count=len(groups),alias_group_count=len(alias_records),
                alias_groups=alias_records,alias_partition_matches_factory=True,
                bitwise_alias_values_consistent=True,unique_storage_cap_bytes=MAX_STATE_BYTES)


def build_model(path, artifact_record, runtime):
    """Strictly load a pinned checkpoint; caller controls CPU threads/time and inference."""
    import torch
    path = Path(path).resolve()
    verify_artifact(path, artifact_record)
    require(zipfile.is_zipfile(path), 'Expected PyTorch ZIP checkpoint; no legacy pickle fallback')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 8192 and sum(e.file_size for e in entries) <= MAX_ARCHIVE_BYTES,
                'Checkpoint expanded ZIP resource bound exceeded')
    # No unrestricted loading, custom safe globals, code downloads or retry fallback.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(payload, Mapping) and isinstance(payload.get('config'), Mapping),
            'Checkpoint lacks an embedded config mapping')
    authored = dict(payload['config'])
    validate_authored_config(authored)
    config = runtime['InkConfig'].from_mapping(authored)
    require(config.data.mode == 'flat' and tuple(config.model.crop_size) == (17,128,128)
            and config.model.pretrained_backbone is None and config.model.input_pad_depth_to is None,
            'Parsed factory input/backbone/depth contract mismatch')
    normalizer = config.data.normalization
    require(normalizer.mode == 'robust_mad' and
            (normalizer.percentile_lower,normalizer.percentile_upper) == (1.0,99.0),
            'Parsed normalizer differs')
    container, state = runtime['reference'].selected_state(payload)
    require(container == 'model', 'Unexpected selected weight container; do not substitute EMA or other weights')
    require(0 < len(state) <= MAX_TENSORS and all(isinstance(k,str) for k in state), 'Invalid tensor keys/count')
    require(all(isinstance(t,torch.Tensor) and t.layout == torch.strided and t.device.type == 'cpu'
                and not t.is_quantized and not t.is_complex() for t in state.values()),
            'Weights must be ordinary dense CPU tensors')
    state_bytes = sum(t.numel()*t.element_size() for t in state.values())
    unique_state_bytes = sum(state[names[0]].untyped_storage().nbytes() for names in storage_groups(state))
    require(unique_state_bytes <= MAX_STATE_BYTES, 'Unique selected storage byte bound exceeded')
    require(all(bool(torch.isfinite(t).all()) for t in state.values()), 'Nonfinite selected weights')
    model = runtime['make_model'](config)
    expected_state = model.state_dict()
    require(set(state) == set(expected_state), 'Missing or unexpected tensor keys')
    require(all(state[k].shape == expected_state[k].shape and state[k].dtype == expected_state[k].dtype
                for k in state), 'Tensor shape/dtype mismatch; no automatic conversion or repair')
    require(sum(p.numel() for p in model.parameters()) <= MAX_PARAMETERS, 'Model parameter bound exceeded')
    storage_contract = verify_storage_contract(state,expected_state,torch)
    result = model.load_state_dict(state, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, 'Strict model-state mismatch')
    model = model.float().to(torch.device('cpu')).eval()
    tensor_schema = [dict(key=k,shape=list(state[k].shape),dtype=str(state[k].dtype),
                          numel=state[k].numel(),bytes=state[k].numel()*state[k].element_size()) for k in sorted(state)]
    encoded_schema = json.dumps(tensor_schema,sort_keys=True,separators=(',',':')).encode()
    step = payload.get('step')
    require(artifact_record.get('embedded_step') == 74999 and isinstance(step,int)
            and not isinstance(step,bool) and step == 74999, 'Checkpoint stored step differs from observed pinned74999')
    metadata = dict(label=artifact_record['label'],path=str(path),size_bytes=artifact_record['size_bytes'],
        sha256=artifact_record['sha256'],model_repository=artifact_record['repository'],
        model_revision=artifact_record['revision'],repository_path=artifact_record['repository_path'],
        deserialization="torch.load(map_location='cpu', weights_only=True)",weight_container=container,
        strict_state_match=True,missing_keys=[],unexpected_keys=[],step=step,
        model_type=config.model.model_type,parameter_count=sum(p.numel() for p in model.parameters()),
        selected_state_bytes=state_bytes,selected_unique_storage_bytes=unique_state_bytes,
        storage_contract=storage_contract,filename_step_label=75000,tensor_schema=tensor_schema,
        tensor_schema_sha256=hashlib.sha256(encoded_schema).hexdigest(),embedded_config=authored,
        reference_adapter_sha256=REFERENCE_SHA,source_manifest_sha256=VENDOR_SHA,
        output_semantics='Unchanged reference infer_surface: [1,1,128,128] ink logits, sigmoid, raw float32 scores; no calibrated confidence or display rescaling.')
    verify_artifact(path,artifact_record)  # Guard local file stability across deserialization/construction.
    del payload,state,expected_state
    gc.collect()
    return model,metadata


def dataset_rows(config):
    rows = {}
    for dataset in config['datasets']:
        for segment in dataset['segments']:
            representation = dataset['sampling_representation_keys'][segment]
            require(representation not in rows, 'Duplicate training representation')
            rows[representation] = dict(segment=segment,label_root=dataset['segments_path'],
                physical_key=dataset['sampling_physical_segment_keys'][segment],
                scroll=dataset['sampling_scroll'],source_family=dataset['source_family'],
                volume_scale=dataset['volume_scale'],surface_path=dataset['surface_volume_paths'][segment])
    return rows


def config_differences(left,right,path=''):
    """Full authored JSON comparison for reporting, without treating path changes as accuracy."""
    if isinstance(left,Mapping) and isinstance(right,Mapping):
        result = []
        for key in sorted(set(left)|set(right)):
            child = path+'.'+str(key) if path else str(key)
            if key not in left or key not in right:
                result.append(dict(path=child,left=left.get(key),right=right.get(key),missing_side='left' if key not in left else 'right'))
            else:
                result.extend(config_differences(left[key],right[key],child))
        return result
    return [] if left==right else [dict(path=path,left=left,right=right)]


def audit_matched_pair(control,dense):
    globals_control = {k:v for k,v in control.items() if k not in ['datasets','out_dir']}
    globals_dense = {k:v for k,v in dense.items() if k not in ['datasets','out_dir']}
    global_diff = config_differences(globals_control,globals_dense)
    left,right = dataset_rows(control),dataset_rows(dense)
    require(not global_diff and set(left)==set(right) and len(left)==29,
            'Matched-pair training settings or29-representation membership differ')
    changes = []
    for key in sorted(left):
        before,after = dict(left[key]),dict(right[key])
        a,b = before.pop('label_root'),after.pop('label_root')
        require(before==after, 'Matched-pair scan/physical-key/sampling representation differs')
        if a!=b:
            changes.append(dict(representation=key,before_label_root=a,after_label_root=b))
    expected = {'public_2p4_level2_zmean4:'+x for x in
        ['pherc0139-w017','pherc0139-w028','pherc0139-w029','pherc1667-w028','pherc1667-w029','pherc0814-46527']}
    require({c['representation'] for c in changes}==expected,
            'Dense ex016 label-root treatment differs from published six-segment change')
    require(left['public_2p4_level2_zmean4:pherc0139-w016']['label_root']==
            right['public_2p4_level2_zmean4:pherc0139-w016']['label_root'], 'w016 was not excluded from pseudo labels')
    return dict(matched_settings=True,representation_count=29,changed_label_root_count=6,
                changed_label_roots=changes,w016_keeps_manual_label_root=True,
                out_dir_control=control.get('out_dir'),out_dir_dense=dense.get('out_dir'),
                global_config_differences=global_diff,
                limitation='Matching embedded settings does not establish exact historical training data/label byte identity or independent target accuracy.')
