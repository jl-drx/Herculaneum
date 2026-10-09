"""Compare three prospectively frozen CT values to a previously unread layer."""
import json,struct
from pathlib import Path
from run_correspondence import Acquisition,pin,save,require

HERE=Path(__file__).resolve().parent


def main():
    go=json.loads((HERE/'FRESH_CONFIRMATION_GO.json').read_text())
    for item in go['files']:require(pin(Path(item['path']))==item,'Frozen fresh-check source changed')
    original=json.loads((HERE/'SAMPLING_PLAN.json').read_text())
    predictions=json.loads((HERE/'FRESH_PREDICTIONS.json').read_text())
    prior=[json.loads((HERE/p).read_text()) for p in ['run-v1/REQUEST_LEDGER.json','software-tags-v1/REQUEST_LEDGER.json','renderer-source-v1/LEDGER.json']]
    require(sum(len(p['requests']) for p in prior)==93,'Unexpected prior request count')
    require(sum(p['reserved_body_bytes'] for p in prior)+200<=original['limits']['max_reserved_bytes'],'Combined reservation exceeded')
    url=original['layers'][0]['object']['url'].rsplit('/',1)[0]+'/16.tif'
    plan=json.loads(json.dumps(original));plan['layers']=[dict(object=dict(url=url))]
    plan['limits'].update(max_requests=7,max_reserved_bytes=200,total_seconds=240)
    run=HERE/'fresh-confirmation-v1';run.mkdir(exist_ok=False)
    client=Acquisition(run,plan);total=109865915
    receipt=dict(status='RUNNING',predictions=pin(HERE/'FRESH_PREDICTIONS.json'),new_target_layer=16)
    try:
        entry=client.get(url,run/'header.bin',cap=8,span=[0,7],total=total)
        etag=entry['response_headers']['ETag']
        header=(run/'header.bin').read_bytes()
        require(header[:4]==b'II*\0' and struct.unpack('<I',header[4:])[0]==109865738,'Exact TIFF header layout')
        client.get(url,run/'count.bin',cap=2,etag=etag,span=[109865738,109865739],total=total)
        require(struct.unpack('<H',(run/'count.bin').read_bytes())[0]==12,'Expected twelve TIFF entries')
        client.get(url,run/'ifd.bin',cap=150,etag=etag,span=[109865738,109865887],total=total)
        body=(run/'ifd.bin').read_bytes();require(struct.unpack('<H',body[:2])[0]==12 and body[-4:]==b'\0'*4,'Single twelve-entry IFD')
        tags={}
        for i in range(12):
            tag,typ,count,value=struct.unpack('<HHII',body[2+12*i:14+12*i]);require(tag not in tags,'Duplicate TIFF tag')
            tags[tag]=(typ,count,value)
        expected={256:(3,1,6205),257:(3,1,8853),258:(3,1,16),259:(3,1,1),262:(3,1,1),
            273:(4,1,8),277:(3,1,1),278:(3,1,8853),279:(4,1,109865730),284:(3,1,1),
            305:(2,27,109865888),339:(3,1,1)}
        require(tags==expected,'Exact published layer TIFF schema required; no unsupported transforms')
        observed=[]
        for i,p in enumerate(predictions['predictions']):
            dest=run/f'pixel-{i}.bin'
            client.get(url,dest,cap=2,etag=etag,span=p['byte_range'],total=total)
            observed.append(int.from_bytes(dest.read_bytes(),'little'))
        client.get(url,run/'software.bin',cap=27,etag=etag,span=[109865888,109865914],total=total)
        require((run/'software.bin').read_bytes()==b'volume-cartographer 2.26.0\0','Software tag must agree')
        expected_values=[p['historical_uint16'] for p in predictions['predictions']]
        standard=[p['correct_trilinear_uint16'] for p in predictions['predictions']]
        exact=observed==expected_values;discriminating=sum(abs(a-b)>=2 for a,b in zip(observed,standard))>=2
        receipt.update(status='FRESH_LAYER_MATCH' if exact and discriminating else 'FRESH_LAYER_CHECK_FAILED',
            target_url=url,target_etag=etag,target_total_bytes=total,observed=observed,frozen_predictions=expected_values,
            correct_trilinear=standard,all_three_exact=exact,at_least_two_discriminate=discriminating,
            interpretation='Narrow artifact reproduction on three fresh-layer points; not ink accuracy or historical build/command authentication')
    except Exception as e:receipt.update(status='STOP',error=type(e).__name__+': '+str(e));raise
    finally:
        receipt.update(requests=len(client.ledger['requests']),returned_body_bytes=client.ledger['returned_body_bytes'],
            reserved_body_bytes=client.ledger['reserved_body_bytes'],ledger=pin(run/'REQUEST_LEDGER.json'))
        save(run/'CONFIRMATION_RESULT.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
