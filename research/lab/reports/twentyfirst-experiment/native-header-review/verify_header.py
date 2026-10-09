"""Independent retained safetensors metadata review; no tensor/model/network IO."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'native-checkpoint-contract'
RUN=BASE/'query-run'


def pin(path):
    data=path.read_bytes()
    return dict(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def unique_pairs(items):
    result={}
    for key,value in items:
        assert key not in result,'Duplicate JSON key'
        result[key]=value
    return result


def strict_json(raw):
    return json.loads(raw.decode('utf-8'),object_pairs_hook=unique_pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(AssertionError('Nonfinite JSON token')))


def main():
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(dict(name=name,passed=True))
    def verify(record):
        actual=pin(Path(record['path']))
        check('pin:'+record['path'],actual['sha256']==record['sha256'] and actual['size_bytes']==record.get('size_bytes',record.get('bytes')))
    anchors=[(RUN/'result.json','c274c8e530d58a0c7a0743adfdc3b046b1cfef6dbfa0b319ca9b1a27f4434472'),
        (RUN/'MANIFEST.json','1376a04f0faebe109fa16791310c45cdbfc32e163c796013e25a39c6cf9267d9'),
        (BASE/'HEADER_QUERY_PLAN.json','cdd8be455ddb2fc4e8e5baa225a22120fbc17d178e9083be8a8f847a18ed441f'),
        (BASE/'REFERENCE_SCHEMA.json','ae6665633364e584a41b4729c6751ca9a880242e97f87a63925c1d162ad45c1f')]
    for path,digest in anchors:check('external_anchor:'+path.name,pin(path)['sha256']==digest)
    receipt=strict_json((RUN/'result.json').read_bytes());manifest=strict_json((RUN/'MANIFEST.json').read_bytes())
    plan=strict_json((BASE/'HEADER_QUERY_PLAN.json').read_bytes());reference=strict_json((BASE/'REFERENCE_SCHEMA.json').read_bytes())
    for item in manifest['files']+receipt['inputs']+[manifest['implementation_freeze'],manifest['root_GO'],
        reference['source_storage_diagnostic'],reference['source_strict_audit']]:verify(item)
    freeze=strict_json(Path(manifest['implementation_freeze']['path']).read_bytes())
    for item in freeze['files']:verify(item)
    prefix=(RUN/'prefix.bin').read_bytes();raw=(RUN/'header.json').read_bytes()
    check('prefix_is_exact_eight_bytes',len(prefix)==8)
    length=int.from_bytes(prefix,byteorder='little',signed=False)
    check('length_word_exact_header_extent',length==len(raw)==66192 and length%8==0 and 8<=length<=245752)
    extent=plan['source']['expected_file_extent_bytes'];data_start=8+length
    check('tensor_data_starts_after_retained_header',extent==272767904 and data_start==66200
          and extent-data_start==272701704)
    header=strict_json(raw)
    check('header_object_and_metadata_absence',isinstance(header,dict) and '__metadata__' not in header)
    expected={row['name']:row for row in reference['tensor_schema']}
    check('exact_508_names',len(header)==len(expected)==508 and set(header)==set(expected))
    intervals=[];named_numel=0
    for name in sorted(header):
        value=header[name];ref=expected[name]
        check('fields:'+name,isinstance(value,dict) and set(value)=={'dtype','shape','data_offsets'})
        shape=value['shape'];span=value['data_offsets']
        check('dtype_shape:'+name,value['dtype']==ref['dtype']=='F32' and isinstance(shape,list)
              and all(type(n) is int and n>=0 for n in shape) and shape==ref['shape'])
        count=math.prod(shape);named_numel+=count
        check('offset_extent:'+name,isinstance(span,list) and len(span)==2 and all(type(n) is int and n>=0 for n in span)
              and span[0]<=span[1]<=extent-data_start and span[1]-span[0]==count*4==ref['data_bytes'] and count==ref['numel'])
        intervals.append((span[0],span[1],name))
    cursor=0
    for begin,end,name in sorted(intervals):
        check('contiguous_nonoverlap:'+name,begin==cursor);cursor=end
    check('exact_named_numel_and_data_extent',named_numel==68175426 and cursor==272701704
          and cursor==extent-data_start and named_numel*4==cursor)
    # Independently project the original retained factory JSON, not just the new reference file.
    original=strict_json(Path(reference['source_storage_diagnostic']['path']).read_bytes())['models']['seed42']
    factory=original['factory'];factory_schema=factory['schema']
    projection=[dict(name=r['key'],dtype='F32',shape=r['shape'],numel=r['numel'],data_bytes=r['named_bytes']) for r in factory_schema]
    check('reference_exactly_projects_original_factory',all(r['dtype']=='torch.float32' for r in factory_schema)
          and sorted(projection,key=lambda x:x['name'])==sorted(reference['tensor_schema'],key=lambda x:x['name'])
          and reference['factory_full_schema_sha256']==factory['schema_sha256'])
    original_groups=factory['storage_groups']
    aliases=[dict(names=g['keys'],shape=expected[g['keys'][0]]['shape']) for g in original_groups if len(g['keys'])>1]
    check('alias_groups_exact_original_factory_projection',aliases==reference['expected_factory_alias_groups'] and len(aliases)==244)
    assigned=set();collapsed_numel=0;canonical_groups=[]
    for group in original_groups:
        names=group['keys'];first=names[0]
        check('group_partition:'+first,all(name in header and name not in assigned for name in names))
        assigned.update(names)
        shapes=[header[name]['shape'] for name in names]
        check('group_matching_schemas:'+first,all(shape==shapes[0] for shape in shapes)
              and all(header[name]['dtype']=='F32' for name in names)
              and len({tuple(header[name]['data_offsets']) for name in names})==len(names))
        numel=math.prod(shapes[0]);collapsed_numel+=numel
        canonical_groups.append(dict(names=names,shape=shapes[0],conditional_numel=numel,
                                     separately_serialized_intervals=[header[name]['data_offsets'] for name in names]))
    check('conditional_canonical_count_not_value_equality',assigned==set(header) and len(original_groups)==264
          and sum(len(g['keys'])==1 for g in original_groups)==20 and collapsed_numel==34546498
          and collapsed_numel*4==138185992 and original['unique_parameter_numel']==collapsed_numel)
    attempts=receipt['attempts'];check('actual_attempt_sequence',len(attempts)==receipt['https_attempt_count']==3
          and [a['attempt'] for a in attempts]==[1,2,3] and [a['status'] for a in attempts]==[302,206,206]
          and [a['logical_request'] for a in attempts]==['prefix','prefix','header'])
    check('allowed_hosts_and_redirect_count',[a['hostname'] for a in attempts]==['huggingface.co','us.aws.cdn.hf.co','us.aws.cdn.hf.co']
          and attempts[0]['redirect_target_hostname']=='us.aws.cdn.hf.co' and receipt['redirect_response_count']==1)
    check('declared_observed_body_accounting',[a['observed_body_bytes'] for a in attempts]==[1106,8,66192]
          and all(a['observed_body_bytes']==a['declared_body_bytes'] for a in attempts)
          and sum(a['observed_body_bytes'] for a in attempts)==receipt['observed_response_body_bytes']==67306
          and len(attempts)<=6 and attempts[0]['observed_body_bytes']<=4096 and receipt['observed_response_body_bytes']<=262144)
    check('only_prefix_header_ranges_requested',[a['range'] for a in attempts]==['bytes=0-7','bytes=0-7','bytes=8-66199']
          and all(a['method']=='GET' and a['elapsed_seconds']<30 for a in attempts)
          and max(int(a['range'].split('-')[-1]) for a in attempts)<data_start)
    for item,data,begin,end in [(attempts[1],prefix,0,7),(attempts[2],raw,8,66199)]:
        check('success_response_identity:'+str(begin),item['Content_Range']==f'bytes {begin}-{end}/{extent}'
              and item['Content_Length']==len(data) and item['Content_Encoding']=='identity'
              and item['observed_body_sha256']==hashlib.sha256(data).hexdigest())
    check('strong_same_etag',attempts[1]['ETag']==attempts[2]['ETag']
          and re.fullmatch(r'"[\x21\x23-\x7e]+"',attempts[1]['ETag']) is not None)
    check('zero_tensor_and_model_activity',receipt['tensor_value_bytes_requested']==receipt['model_calls']==receipt['retries']==0
          and receipt['redirect_error_raw_bodies_retained'] is False and receipt['signed_URLs_retained'] is False)
    comparison=receipt['comparison']
    wanted=dict(tensor_name_count=508,named_numel=68175426,named_data_bytes=272701704,header_JSON_bytes=66192,
        expected_factory_alias_groups_schema_matched=244,conditional_factory_unique_groups=264,
        conditional_factory_unique_parameters=34546498,conditional_factory_unique_storage_bytes=138185992,
        optional_metadata_present=False,author_no_metadata_claim_matches=True,full_file_sha256_verified=False,
        alias_value_equality_verified=False,shared_native_storage_verified=False,strict_model_load_verified=False)
    check('published_comparison_exact',all(comparison[key]==value for key,value in wanted.items()))
    check('claim_ceiling_preserved',receipt['status']==comparison['status']=='HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED'
          and receipt['full_file_sha256_verified'] is False and receipt['original_inputs_unchanged'])
    summary=dict(header_bytes=length,prefix_bytes=8,tensor_data_file_offset=data_start,total_advertised_extent=extent,
        tensor_names=508,named_F32_elements=named_numel,named_data_bytes=cursor,
        factory_declared_alias_groups=244,factory_singletons=20,conditional_factory_groups=264,
        conditional_canonical_parameters=collapsed_numel,conditional_canonical_bytes=collapsed_numel*4,
        exact_contiguous_nonoverlapping_serialized_intervals=True,optional_metadata_present=False,
        actual_https_attempts=3,observed_response_body_bytes=67306,retained_prefix_and_header_bytes=66200,
        redirect_body_observed_unretained_bytes=1106,tensor_data_bytes_requested_or_retained=0,
        full_SHA_verified=False,tensor_values_or_finiteness_verified=False,alias_value_equality_verified=False,
        shared_native_storage_verified=False,strict_model_load_verified=False)
    output=dict(kind='phase21-independent-native-header-review-v1',status='PASS',
        compatibility_status='HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),check_count=len(checks),checks=checks,summary=summary,
        retained_inputs=[pin(p) for p,_ in anchors]+[pin(RUN/'prefix.bin'),pin(RUN/'header.json'),pin(Path(reference['source_storage_diagnostic']['path'])),pin(HERE/'verify_header.py')],
        execution=dict(new_network_calls=0,model_calls=0,tensor_value_bytes_read=0),
        resources=dict(wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        limitations=['The full-file hash is an unverified registry/documentation claim; a strong matching ETag proves only transport continuity here.',
                    'Factory alias names describe conditional binding semantics; separate native intervals do not prove equal values or shared storage.',
                    'Redirect-body count/hash are checked against retained receipts; the deliberately discarded body cannot be rehashed.',
                    'Application-observed body accounting excludes headers, TLS framing and unread socket buffers.',
                    'No tensor values, finiteness, strict load, forward behavior, historical training or accuracy are verified.'])
    with (HERE/'verification.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    with (HERE/'conditional-alias-projection.json').open('x') as f:json.dump(dict(groups=canonical_groups,claim='Conditional factory name grouping only; values/storage equality unverified.'),f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks),summary=summary,resources=output['resources'])))


if __name__=='__main__':main()
