"""Small synthetic selector and exact rational recovery-screen checks only."""
import argparse
from fractions import Fraction
from pathlib import Path
import socket
import sys


def offline(*args,**kwargs):raise RuntimeError('Network prohibited in source fixtures')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

from calibration_common import LAB,SELECTOR_SHA,record,module,require,write_json,recovery_screen


def confusion(tp,positive=50,fp=0,negative=100):
    return dict(tp=tp,fn=positive-tp,fp=fp,tn=negative-fp)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    import numpy as np
    selector_record=record(LAB/'reports/seventh-experiment/benchmark/operating_point.py')
    require(selector_record['sha256']==SELECTOR_SHA,'Original selector changed')
    selector=module(selector_record,'fourteenth_synthetic_selector');checks=[]
    def check(name,condition):
        require(condition,name);checks.append(name)
    # N=19 excludes every crossing maximum; the nextafter1 cutoff is finite above1.
    values=np.ones(19,dtype=np.float32);row=selector.choose_development_threshold(values)
    check('19-negative maximum tie permits finite cutoff above1',row['maximum_false_positives']==0
        and row['development_false_positives']==0 and row['boundary_tie_count']==19
        and row['threshold']==float(np.nextafter(np.float32(1),np.float32(np.inf))) and np.isfinite(row['threshold']))
    values=np.asarray([1,.8,.8]+[.1]*17,dtype=np.float32);row=selector.choose_development_threshold(values)
    check('20-negative crossing tie fully excluded',row['maximum_false_positives']==1 and row['development_false_positives']==1
        and row['boundary_tie_count']==2 and np.float32(row['threshold']).view(np.uint32)==np.nextafter(np.float32(.8),np.float32(np.inf)).view(np.uint32))
    check('classification uses float64 score and exact stored cutoff',np.count_nonzero(values.astype(np.float64)>=row['threshold'])==1)
    for name,bad in [('float64',values.astype(np.float64)),('NaN',np.array([np.nan],np.float32)),('outside range',np.array([1.1],np.float32))]:
        rejected=False
        try:selector.choose_development_threshold(bad)
        except ValueError:rejected=True
        check('invalid selector vector rejected: '+name,rejected)
    baseline={'one':confusion(10),'two':confusion(10)}
    candidate={'one':confusion(9),'two':confusion(11)};screen=recovery_screen(candidate,baseline)
    check('inclusive exact sensitivity delta minus1/50',screen['passed'] and screen['per_context_sensitivity_checks']['one']['passed']
        and screen['per_context_sensitivity_checks']['one']['delta']==dict(numerator=-1,denominator=50,value=-.02))
    baseline19={k:confusion(10,negative=19) for k in ['one','two']};candidate19={k:confusion(10,fp=1,negative=19) for k in ['one','two']}
    screen19=recovery_screen(candidate19,baseline19)
    check('19-negative exact specificity rejects one false positive',not screen19['passed'] and not screen19['specificity_by_context']['one']['passed'])
    baseline20={k:confusion(10,negative=20) for k in ['one','two']};candidate20={k:confusion(10,fp=1,negative=20) for k in ['one','two']}
    screen20=recovery_screen(candidate20,baseline20)
    check('20-negative exact specificity permits one false positive',screen20['passed'])
    baseline={'one':confusion(5,positive=10),'two':confusion(80,positive=100)}
    candidate={'one':confusion(6,positive=10),'two':confusion(78,positive=100)};screen=recovery_screen(candidate,baseline)
    check('equal-context sensitivity replaces misleading pooled comparison',screen['passed']
        and screen['mean_sensitivity_comparison']['delta']['numerator']==1
        and screen['mean_sensitivity_comparison']['delta']['denominator']==25
        and Fraction(84,110)<Fraction(85,110))
    baseline={k:confusion(10) for k in ['one','two']};primary={k:confusion(10,fp=6) for k in ['one','two']};secondary={k:confusion(10) for k in ['one','two']}
    check('successful secondary screen cannot replace failed primary',not recovery_screen(primary,baseline)['passed'] and recovery_screen(secondary,baseline)['passed'])
    check('passing screen never promotes automatically',screen['automatic_promotion'] is False)
    require('torch' not in sys.modules,'Fixture imported model runtime')
    source=[record(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))]
    write_json(args.output_dir/'fixture-receipt.json',dict(status='PASS',checks=checks,source_files=source,selector=selector_record,
        arrays='Only explicitly constructed synthetic score vectors; no retained scientific array loaded',model_calls=0,
        actual_checkpoint_reads=0,actual_source_array_reads=0,network_requests=0,torch_imported=False))
    print(__import__('json').dumps(dict(status='PASS',checks=len(checks),receipt=record(args.output_dir/'fixture-receipt.json'))))


if __name__=='__main__':main()
