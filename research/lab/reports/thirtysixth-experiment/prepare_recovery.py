from common import *
require(not (HERE/'RECOVERY_GO.json').exists(),'New recovery')
partial=HERE/'run-v1/SYNTHETIC_GATE.json.writing';require(partial.exists() and not (HERE/'run-v1/COMPLETE.json').exists(),'Preserve failed attempt')
save(HERE/'SETUP_FAILURE.json',dict(status='SERIALIZATION_FAILURE',error='TypeError: NumPy bool is not JSON serializable',partial=pin(partial),scope='All41synthetic cases computed in memory; gate writing failed before real input evaluation. No completed gate or scientific conclusion retained.',real_contexts_evaluated=0,network_requests=0,new_model_forwards=0))
s=(HERE/'run_study.py').read_text().replace("HERE/'GO.json'","HERE/'RECOVERY_GO.json'").replace("HERE/'run-v1'","HERE/'run-v2'").replace('    synthetic.append(record)','    record[\'passed\']=bool(record[\'passed\'])\n    synthetic.append(record)')
(HERE/'run_study_recovery.py').write_text(s)
files=[pin(HERE/'GO.json'),pin(HERE/'SETUP_FAILURE.json'),pin(HERE/'run_study_recovery.py'),pin(__file__)]+json.loads((HERE/'GO.json').read_text())['files']
for p in files:verify(p)
save(HERE/'RECOVERY_GO.json',dict(status='GO_UNCHANGED_RULE_NATIVE_BOOL_SERIALIZATION_ONLY',created_at_utc=now(),files=files,scientific_parameter_changes=0,original_partial_preserved=True))
print(json.dumps(pin(HERE/'RECOVERY_GO.json')))
