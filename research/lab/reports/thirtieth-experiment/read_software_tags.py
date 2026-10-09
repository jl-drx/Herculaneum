"""Exactly three pinned TIFF Software-tag payloads; no new scientific pixels."""
import json
from pathlib import Path
from run_correspondence import Acquisition,pin,save,require

HERE=Path(__file__).resolve().parent


def main():
    freeze=json.loads((HERE/'SOFTWARE_TAG_GO.json').read_text())
    for item in freeze['files']:require(pin(Path(item['path']))==item,'Changed follow-up source')
    original=json.loads((HERE/'SAMPLING_PLAN.json').read_text())
    prior=json.loads((HERE/'run-v1/REQUEST_LEDGER.json').read_text())
    require(len(prior['requests'])+3<=original['limits']['max_requests'],'Combined request cap')
    require(prior['reserved_body_bytes']+84<=original['limits']['max_reserved_bytes'],'Combined reservation cap')
    structural=json.loads(Path(original['structural']['path']).read_text())
    plan=json.loads(json.dumps(original));plan['limits'].update(max_requests=3,max_reserved_bytes=84,total_seconds=120)
    run=HERE/'software-tags-v1';run.mkdir(exist_ok=False)
    client=Acquisition(run,plan);items=[]
    try:
        for layer in structural['tiffs']:
            tags=[t for t in layer['parsed']['tags'] if t['tag']==305]
            require(len(tags)==1,'Exactly one Software tag')
            tag=tags[0]
            require(tag['type']==2 and tag['count']==27 and tag['payload_offset']==109865888,'Frozen Software tag declaration changed')
            obj=layer['object'];dest=run/f"layer-{layer['layer_index']:02d}.bin"
            client.get(obj['url'],dest,cap=27,etag=obj['etag'],span=[109865888,109865914],total=obj['total_bytes'])
            raw=dest.read_bytes();require(raw.endswith(b'\0'),'ASCII TIFF tag must terminate')
            value=raw.rstrip(b'\0').decode('ascii');require(all(32<=ord(c)<=126 for c in value),'Printable Software tag required')
            items.append(dict(layer=layer['layer_index'],software=value,payload=pin(dest)))
        result=dict(status='PASS',items=items,requests=len(client.ledger['requests']),
                    returned_body_bytes=client.ledger['returned_body_bytes'],reserved_body_bytes=client.ledger['reserved_body_bytes'],
                    scope='Embedded producer metadata only; original CT comparison unchanged')
        save(run/'SOFTWARE_TAGS.json',result);print(json.dumps(result,indent=2))
    except Exception as e:
        save(run/'STOP.json',dict(error=type(e).__name__+': '+str(e),items=items));raise


if __name__=='__main__':main()
