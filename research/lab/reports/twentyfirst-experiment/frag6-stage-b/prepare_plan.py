"""Freeze one exact Stage B multipart schedule and its implementation inputs."""
import argparse
import json
from pathlib import Path
from common import (LIMITS,PROPOSAL,PROPOSAL_SHA,ROOT,STAGE_A,STAGE_A_SHA,
                    STAGE_A_REVIEW,STAGE_A_REVIEW_SHA,now,pin,require,save,sha,verify)
from geometry import derive_points

CODE_NAMES=("common.py","geometry.py","multipart.py","transport.py","acquire.py",
            "prepare_plan.py","worker.py","controller.py","check_fixtures.py")


def source_records():
    return [pin(ROOT/name) for name in CODE_NAMES]


def dependencies():
    for path,expected in ((PROPOSAL,PROPOSAL_SHA),(STAGE_A,STAGE_A_SHA),(STAGE_A_REVIEW,STAGE_A_REVIEW_SHA)):
        require(sha(path)==expected,"Prior approved source changed")
    proposal=json.loads(PROPOSAL.read_text());a=json.loads(STAGE_A.read_text());review=json.loads(STAGE_A_REVIEW.read_text())
    require(a["status"]=="PASS" and review["status"]=="PASS" and review["structural_contract"]==pin(STAGE_A),"Verified successful Stage A required")
    require(a["ppm"]["object"]==proposal["present_evidence"]["objects"][0],"PPM object identity changed")
    adaptation=json.loads((ROOT/"SOURCE_ADAPTATION.json").read_text())
    for record in adaptation["sources"]:verify(record)
    return proposal,a,review


def fixed_schedule():
    proposal,a,review=dependencies();p=a["ppm"]["parsed"]
    points,ranges=derive_points(p["width"],p["height"],p["data_offset"],a["ppm"]["object"]["total_bytes"])
    return dict(object=a["ppm"]["object"],width=p["width"],height=p["height"],data_offset=p["data_offset"],
                points=points,ranges=ranges,expected_parts=243,expected_records=729,useful_bytes=34992,
                stage_rules=proposal["stages"][1])


def load_plan(path,expected_sha):
    path=Path(path).resolve(strict=True);require(sha(path)==expected_sha,"Execution plan hash mismatch")
    plan=json.loads(path.read_text())
    require(plan["schema"]=="herculaneum.frag6-stage-b-execution.v1" and plan["stage"]=="B_sparse_ppm","Only Stage B supported")
    require(plan["schedule"]==fixed_schedule(),"Fixed lattice/object/range derivation changed")
    require(plan["limits"]==LIMITS and plan["code"]==source_records(),"Implementation or limits changed")
    require(plan["proposal"]==pin(PROPOSAL) and plan["stage_a"]==pin(STAGE_A) and plan["stage_a_review"]==pin(STAGE_A_REVIEW),"Prior stage pins changed")
    require(plan["adaptation"]==pin(ROOT/"SOURCE_ADAPTATION.json"),"Adaptation source manifest changed")
    fixture=json.loads(verify(plan["synthetic_fixtures"]).read_text())
    require(fixture["status"]=="PASS" and fixture["network_calls"]==0 and fixture["code"]==plan["code"],"Current-source passing offline fixtures required")
    return plan


def load_go(path,plan_sha):
    go=json.loads(Path(path).read_text())
    require(go.get("status")=="GO" and go.get("stage")=="B_sparse_ppm" and go.get("plan_sha256")==plan_sha
            and go.get("allow_live_requests") is True and go.get("allow_conditional_ppm_geometry") is True,
            "Root GO must permit the exact Stage B request and conditional PPM geometry")
    return pin(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--fixtures",required=True);p.add_argument("--output",required=True);args=p.parse_args()
    output=Path(args.output).resolve();require(output.is_relative_to(ROOT) and not output.exists(),"Fresh owned plan required")
    fixture_pin=pin(args.fixtures);fixture=json.loads(verify(fixture_pin).read_text())
    require(fixture["status"]=="PASS" and fixture["network_calls"]==0 and fixture["code"]==source_records(),"Passing current-source fixtures required")
    plan=dict(schema="herculaneum.frag6-stage-b-execution.v1",stage="B_sparse_ppm",created_utc=now(),
              schedule=fixed_schedule(),limits=LIMITS,proposal=pin(PROPOSAL),stage_a=pin(STAGE_A),stage_a_review=pin(STAGE_A_REVIEW),
              adaptation=pin(ROOT/"SOURCE_ADAPTATION.json"),code=source_records(),synthetic_fixtures=fixture_pin,
              root_go_required=True,geometry_interpretation="Exactly little/big float64 invariant comparison; unique finite/unit-normal hypothesis required. No authenticated frame, source, anatomy or depth claim.",
              stage_c_authorized=False,images_labels_models_authorized=False)
    output.parent.mkdir(parents=True,exist_ok=True);save(output,plan);load_plan(output,sha(output));print(json.dumps(pin(output)))


if __name__=="__main__":main()
