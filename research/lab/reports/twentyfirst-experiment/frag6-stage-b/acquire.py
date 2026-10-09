"""One-request Stage B acquisition and conditional sparse PPM geometry only."""
import os
from common import LIMITS, ContractError, now, pin, require, resources, save
from geometry import assemble,candidates,describe
from transport import MultipartClient


def run(root,object,ranges,points,plan_record,opener=None,limits=None):
    client=MultipartClient(root,object,ranges,plan_record,opener=opener,limits=limits)
    result=dict(schema="herculaneum.frag6-stage-b-result.v1",status="RUNNING",started_utc=now(),plan=plan_record,
                sample_scope="Fixed sparse PPM geometry only",validation=None,geometry=None,
                image_pixels_decoded=0,labels_loaded=0,model_calls=0,gpu_spend_gbp=0,
                stage_c_authorized=False,source_volume_binding=None,plane_offset_rule=None,
                anatomical_normal=None,physical_ink_depth=None)
    save(client.root/"SPARSE_GEOMETRY.json",result)
    try:
        body,parts,entry=client.get()
        raw=assemble(body,parts,ranges)
        path=client.root/"raw/records.bin"
        with path.open("xb") as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        result["records"]=pin(path)
        result["validation"]=candidates(raw)
        save(client.root/"SPARSE_GEOMETRY.json",result)
        require(result["validation"]["status"]=="CONDITIONAL","PPM byte-order hypotheses do not identify exactly one invariant-consistent mapped interpretation")
        result["geometry"]=describe(raw,result["validation"],points)
        result.update(status="PASS",interpretation="Conditional serialization and sparse PPM consistency only; not authenticated source/render provenance, papyrus anatomy, physical ink depth or model accuracy.")
    except Exception as error:
        result.update(status="STOP",failure_type=type(error).__name__,stop_reason=str(error))
        raise
    finally:
        result.update(completed_utc=now(),request_ledger=pin(client.root/"request-ledger.json"),
                      http_attempts=len(client.ledger["requests"]),
                      reserved_response_body_bytes=client.ledger["reserved_response_body_bytes"],
                      retained_response_body_bytes=client.ledger["retained_response_body_bytes"])
        save(client.root/"SPARSE_GEOMETRY.json",result)
        try:
            result["resources"]=resources(client.root,LIMITS if limits is None else limits)
            save(client.root/"SPARSE_GEOMETRY.json",result)
            resources(client.root,LIMITS if limits is None else limits)
        except Exception as error:
            result.update(status="STOP",failure_type=type(error).__name__,stop_reason=str(error))
            save(client.root/"SPARSE_GEOMETRY.json",result)
            raise
    return result
