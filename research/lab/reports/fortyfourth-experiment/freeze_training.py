from pixel_http import *
need(json.loads((P/'LEARNING_FIXTURES.json').read_text())['status']=='PASS','Numerical fixtures')
need(json.loads((P/'REFERENCE_ADMISSION.json').read_text())['status']=='QUALIFIED_ANNOTATION_AGREEMENT_ADMISSION','Qualified scope')
files=[P/n for n in ['TRAINING_PROTOCOL.md','learn_resolution.py','test_learning_contract.py','LEARNING_FIXTURES.json','REFERENCE_ADMISSION.json','PIXEL_QC.json','CROP_SELECTION.json','HEADER_SELECTION.json','pixel_http.py','freeze_training.py']]
files+=sorted((P/'crops').glob('*.npy'))
save(P/'TRAINING_GO.json',dict(status='GO_CPU_ONLY_FOR_ANNOTATION_AGREEMENT',created_at_utc=now(),pins=[pin(p) for p in files],updates_per_arm=2048,arms=6,seed=20261009,held_out_parent_tuning=False,GPU_spend_GBP=0))
print('Frozen CPU training GO')
