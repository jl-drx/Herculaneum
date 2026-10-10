import unittest
import numpy as np
from diagnostic import *

class Tests(unittest.TestCase):
    def test_translation_direction(self):
        labels=np.zeros((128,128),np.uint8);labels[42,43]=1
        shifted=shifted_labels(labels)
        for plane,(dy,dx) in zip(shifted,SHIFTS,strict=True):
            self.assertEqual(np.argwhere(plane==1).tolist(),[[10+dy,11+dx]])
    def test_larger_canvas_no_core_padding(self):
        labels=np.zeros((128,128),np.uint8);labels[31,33]=1;labels[127,33]=1
        shifted=shifted_labels(labels)
        self.assertEqual(int(shifted[7,0,1]),1)
        self.assertEqual(int(shifted[1].sum()),0)
    def test_unknown_intersection(self):
        labels=np.zeros((128,128),np.uint8);labels[:,64]=2
        shifted,common,stable=support(labels)
        self.assertEqual(int(common.sum()),61*64)
        self.assertFalse(common[:,31:34].any());self.assertTrue(np.array_equal(common,stable))
    def test_common_support_need_not_be_class_stable(self):
        labels=np.zeros((128,128),np.uint8);labels[:,64:]=1
        shifted,common,stable=support(labels)
        self.assertTrue(common.all());self.assertEqual(int(stable.sum()),62*64)
    def test_known_boundary_band_geometry(self):
        labels=np.zeros((128,128),np.uint8);labels[:,64:]=1
        d,b=boundary_bands(labels)
        self.assertEqual([int(b[k].sum()) for k in BANDS],[256,256,3584])
        self.assertEqual(d[20,31],1);self.assertEqual(d[20,30],2);self.assertEqual(d[20,27],5)
    def test_single_class_remains_undefined(self):
        labels=np.zeros((64,64),np.uint8);mask=np.ones_like(labels,bool)
        m=measure(np.ones_like(labels,float)*.4,labels,mask)
        self.assertIsNone(m['ap']);self.assertIsNone(m['auc']);self.assertEqual(m['status'],'undefined_requires_both_classes')

if __name__=='__main__':unittest.main()
