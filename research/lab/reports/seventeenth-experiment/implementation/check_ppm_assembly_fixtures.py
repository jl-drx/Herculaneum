"""Synthetic proof that physical multipart order cannot alter the frozen schedule."""
import copy
import hashlib
import json
from run_ppm_diagnostic import AXIS, assemble_request


def main():
    case = dict(id="frag1-context1", bbox_source_xywh=[120, 200, 1152, 1152], native_bbox_yxhw=[200, 120, 1152, 1152])
    contract = dict(dim=6, type="double", ordered=True, record_bytes=48, data_offset=73,
                    width=2000, height=2000, object_total_bytes=73 + 2000*2000*48,
                    object_id="frag1-result", url="https://fixture.invalid/result.ppm", etag='"fixture"')
    request = dict(request_id="fixture", case_id=case["id"], object_id=contract["object_id"], variant="result",
                   url=contract["url"], pin=dict(etag=contract["etag"], object_total_bytes=contract["object_total_bytes"]), points=[], ranges=[])
    for cy in AXIS:
        for cx in AXIS:
            y, x = 201+3*cy, 121+3*cx
            spans = []
            for dy in (-1, 0, 1):
                start = 73 + 48*((y+dy)*2000+x-1)
                spans.append([start, start+143])
            request["points"].append(dict(coarse_yx=[cy,cx],native_center_yx=[y,x],row_ranges=spans))
            request["ranges"].extend(spans)
    body = bytearray()
    parts = []
    payloads = []
    # Physically reverse the parts, including each payload's retained offsets.
    for index in reversed(range(243)):
        span = request["ranges"][index]
        payload = bytes([index])*144
        body.extend(b"synthetic multipart header\r\n\r\n")
        offset = len(body)
        body.extend(payload)
        body.extend(b"\r\n")
        parts.append(dict(range_inclusive=span,payload_offset=offset,payload_bytes=144,
                          payload_sha256=hashlib.sha256(payload).hexdigest(),
                          headers={"content-range":f"bytes {span[0]}-{span[1]}/{contract['object_total_bytes']}"}))
    receipt = {key:request[key] for key in ("request_id","case_id","object_id","variant","url")}
    receipt.update(parts=parts,etag=contract["etag"],object_total_bytes=contract["object_total_bytes"])
    raw, xy, points = assemble_request(bytes(body), receipt, request, contract, case)
    assert raw == b"".join(bytes([i])*144 for i in range(243))
    assert xy.shape == (729,2) and xy[0].tolist() == [216,296] and points == request["points"]
    checks = ["reordered-physical-parts-reassembled-exactly", "independent-native-xy-schedule"]
    for name, mutate in (
        ("corrupt-payload-hash", lambda r,q: r["parts"][0].update(payload_sha256="0"*64)),
        ("duplicate-part", lambda r,q: r["parts"].__setitem__(0,copy.deepcopy(r["parts"][1]))),
        ("shifted-center", lambda r,q: q["points"][0]["native_center_yx"].__setitem__(0,999)),
        ("changed-endian-independent-range", lambda r,q: q["ranges"][0].__setitem__(0,100)),
    ):
        changed, changed_request = copy.deepcopy(receipt), copy.deepcopy(request)
        mutate(changed,changed_request)
        try:
            assemble_request(bytes(body),changed,changed_request,contract,case)
        except ValueError:
            checks.append("reject-"+name)
        else:
            raise AssertionError(name)
    print(json.dumps(dict(status="PASS",synthetic_only=True,actual_input_arrays_decoded=0,count=len(checks),checks=checks)))


if __name__ == "__main__":
    main()
