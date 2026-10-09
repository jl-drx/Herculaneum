"""Deterministic Stage A traversal; no subsequent-stage or pixel reader exists."""
from common import LIMITS, now, pin, resources, save
from parsers import ascii_description, ppm_prefix, tiff_count, tiff_header, tiff_ifd
from transport import RangeClient


def run(root, objects, plan_record, opener=None, limits=None):
    client = RangeClient(root,objects,plan_record,opener=opener,limits=limits)
    result = dict(schema="herculaneum.frag6-structural-contract.v1",status="RUNNING",started_utc=now(),
                  plan=plan_record,scope="Stage A only; literal structure and declared text. No numerical PPM or image decode.",
                  ppm=None,tiffs=[],numerical_ppm_records_decoded=0,image_pixels_decoded=0,
                  model_calls=0,gpu_spend_gbp=0,automatic_stage_advance=False,
                  unknown_render_fields=dict(source_volume_uuid=None,ppm_to_native_transform=None,
                      center_index=None,depth_step_um=None,plane_offset_rule=None,normal_orientation=None,
                      interpolation=None,renderer_version_and_arguments=None,physical_ink_depth=None))
    save(client.root/"STRUCTURAL_CONTRACT.json",result)
    try:
        obj = objects[0]
        raw,receipt = client.get(obj["url"],0,1023,"ppm-prefix.bin","ppm_prefix")
        result["ppm"] = dict(object=obj,parsed=ppm_prefix(raw,obj["total_bytes"]),prefix=receipt["file"])
        save(client.root/"STRUCTURAL_CONTRACT.json",result)
        canvas = (result["ppm"]["parsed"]["width"],result["ppm"]["parsed"]["height"])
        headers = []
        # Preserve the four initial-request order in the sealed proposal.
        for obj in objects[1:]:
            layer = obj["url"].rsplit("/",1)[-1][:2]
            raw,receipt = client.get(obj["url"],0,7,"layer-"+layer+"-header.bin","tiff_header")
            parsed = tiff_header(raw,obj["total_bytes"])
            item = dict(layer_index=int(layer),object=obj,header=parsed,header_file=receipt["file"])
            result["tiffs"].append(item);headers.append(item)
            save(client.root/"STRUCTURAL_CONTRACT.json",result)
        for item in headers:
            obj,header = item["object"],item["header"]
            name = "layer-"+str(item["layer_index"]).zfill(2)
            start = header["ifd_offset"]
            raw,receipt = client.get(obj["url"],start,start+1,name+"-ifd-count.bin","ifd_count")
            count = tiff_count(raw,header,obj["total_bytes"])
            item.update(count=count,count_file=receipt["file"])
            save(client.root/"STRUCTURAL_CONTRACT.json",result)
            raw,receipt = client.get(obj["url"],start,start+count["table_bytes"]-1,name+"-ifd.bin","ifd_table")
            parsed = tiff_ifd(raw,header,count,obj["total_bytes"],canvas)
            item.update(parsed=parsed,ifd_file=receipt["file"])
            save(client.root/"STRUCTURAL_CONTRACT.json",result)
            if parsed["description_range"] is not None:
                start,end = parsed["description_range"]
                raw,receipt = client.get(obj["url"],start,end,name+"-description.bin","description")
                parsed["description"] = ascii_description(raw)
                parsed["description_file"] = receipt["file"]
                for tag in parsed["tags"]:
                    if tag["tag"] == 270:
                        tag["payload_acquired"] = True
                save(client.root/"STRUCTURAL_CONTRACT.json",result)
        result.update(status="PASS",canvas_shape_yx=[canvas[1],canvas[0]],
                      interpretation="Storage structure established for three TIFFs and PPM header only. Publisher text is not authenticated rendering history; no geometry, offset, anatomy or accuracy conclusion.")
    except Exception as error:
        result.update(status="STOP",failure_type=type(error).__name__,stop_reason=str(error))
        raise
    finally:
        result.update(completed_utc=now(),request_ledger=pin(client.root/"request-ledger.json"),
                      http_attempts=len(client.ledger["requests"]),
                      reserved_response_body_bytes=client.ledger["reserved_response_body_bytes"],
                      retained_response_body_bytes=client.ledger["retained_response_body_bytes"])
        save(client.root/"STRUCTURAL_CONTRACT.json",result)
        try:
            result["resources"] = resources(client.root,LIMITS if limits is None else limits)
            save(client.root/"STRUCTURAL_CONTRACT.json",result)
            resources(client.root,LIMITS if limits is None else limits)
        except Exception as error:
            result.update(status="STOP",failure_type=type(error).__name__,stop_reason=str(error))
            save(client.root/"STRUCTURAL_CONTRACT.json",result)
            raise
    return result
