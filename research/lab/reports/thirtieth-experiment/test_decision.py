"""Focused checks that correspondence acceptance does not select on check outcomes."""
import json
from pathlib import Path
import unittest
import numpy as np
from compare_mapping import compare


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).resolve().parent/'SAMPLING_PLAN.json').read_text())
        self.observed=np.arange(189,dtype=float).reshape(7,3,9)*100
        self.candidates=self.observed[None]+np.arange(6)[:,None,None,None]*100
        self.controls=self.candidates+5000

    def test_exact_distinguishable_mapping_qualifies(self):
        r=compare(self.candidates,self.controls,self.observed,self.plan)
        self.assertEqual(r['status'],'MAPPING_REPRODUCED_ON_FIXED_PROBES')
        self.assertEqual(r['selected_index'],0)

    def test_check_best_cannot_replace_calibration_choice(self):
        self.candidates[0,3:]+=1000
        self.candidates[1,3:]=self.observed[3:]
        r=compare(self.candidates,self.controls,self.observed,self.plan)
        self.assertEqual(r['selected_index'],0)
        self.assertEqual(r['status'],'MAPPING_NOT_QUALIFIED')

    def test_calibration_tie_not_resolved_by_check(self):
        self.candidates[1,:3]=self.observed[:3]
        r=compare(self.candidates,self.controls,self.observed,self.plan)
        self.assertFalse(r['criteria']['identified_from_calibration'])

    def test_matching_wrong_coordinates_reject_identification(self):
        self.controls[0]=self.observed
        r=compare(self.candidates,self.controls,self.observed,self.plan)
        self.assertFalse(r['criteria']['wrong_coordinate_absolute'])

    def test_missing_or_uninformative_evidence_cannot_qualify(self):
        observed=np.full((7,3,9),50.)
        candidates=observed[None]+np.arange(6)[:,None,None,None]*100
        r=compare(candidates,candidates+5000,observed,self.plan)
        self.assertFalse(r['criteria']['informative_check_sites'])
        self.candidates[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):compare(self.candidates,self.controls,self.observed,self.plan)


if __name__=='__main__':unittest.main(verbosity=2)
