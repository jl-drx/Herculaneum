from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys,resource
HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent
UPDATES=[32,128,256,384,512]
ARMS=['C_coverage']
def require(ok,why):
    if not ok:raise ValueError(why)
def pin(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha)
def verify(p):
    require(pin(p['path'])==p,'Changed pin: '+p['path']);return Path(p['path'])
def save(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def legacy():
    sys.path.insert(0,str(REPORTS/'thirteenth-experiment/implementation'))
    return module(REPORTS/'thirteenth-experiment/implementation/run_worker.py','phase38_legacy_training')
def plan():
    go=json.loads((HERE/'GO-V2.json').read_text());require(go['status']=='GO_APPROVED_CPU_TRAINING_COVERAGE','Frozen GO required')
    for p in go['files']:verify(p)
    p=json.loads((HERE/'PLAN-V2.json').read_text())
    require(p['arms']==['C_coverage'] and p['evaluation_updates']==[512] and p['new_model_forwards_allowed']==937 and p['new_optimizer_updates_allowed']==512,'Phase38 execution scope')
    require(p['success']==p['gate']==dict(Paris_AP_gain_vs_seed42=.01,Paris_AP_gain_vs_A512=.01,regional_AP_AUROC_min_vs_seed42=-.02),'Phase38 decision gate')
    return p
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def limits():
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(7200,)*2)

def load_case(p,c):
    import numpy as np
    if c['family']=='paris17':
        x=np.load(verify(c['arrays']['surface']),allow_pickle=False)
        y=np.load(verify(c['arrays']['labels']),allow_pickle=False).astype(bool)
        m=np.load(verify(c['arrays']['primary_mask']),allow_pickle=False)
        require(x.dtype==np.float32 and x.shape==(17,384,384),'Paris input')
    else:
        shared=module(verify(p['pooled_shared']),'phase38_pooled_shared')
        d,y,m=shared.evaluation(c['legacy_case']);x=d['surface']
        require(x.dtype==np.uint8 and x.shape==(21,384,384),'PHerc0814 input')
        require(np.array_equal(m,np.load(verify(c['primary_mask']),allow_pickle=False)),'Frozen pooled mask')
    require(y.dtype==m.dtype==np.bool_ and y.shape==m.shape==(384,384) and m.any(),'Label/support shape')
    require(int((y&m).sum())==c['positive'] and int((~y&m).sum())==c['negative'],'Unchanged class support')
    return x,y,m
