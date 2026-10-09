import unittest
import numpy as np
from PIL import Image
from reference_qc import classify, tile_table, select_tiles

class QC(unittest.TestCase):
    def test_gray_unknown(self):
        self.assertEqual(classify(Image.fromarray(np.array([[0,1,127,254,255]],np.uint8))).tolist(),[[0,2,2,2,1]])
    def test_alpha_and_color_are_unknown(self):
        a=np.array([[[0,0,0,255],[255,255,255,255],[255,255,255,0],[255,0,0,255]]],np.uint8)
        self.assertEqual(classify(Image.fromarray(a)).tolist(),[[0,1,2,2]])
    def test_palette_transparency(self):
        im=Image.new('P',(3,1)); im.putpalette([0,0,0,255,255,255,127,127,127]+[0]*759)
        im.putdata([0,1,2]);im.info['transparency']=1
        self.assertEqual(classify(im).tolist(),[[0,2,2]])
    def test_coverage_and_no_band_replacement(self):
        lab=np.zeros((2200,1100),np.uint8);mask=np.ones_like(lab)
        lab[32:64,32:64]=1; lab[544:576,32:64]=1
        lab[1056:1088,32:64]=1;lab[1568:1600,32:64]=1
        mask[544:1056,32:544]=2
        table=tile_table(lab,mask)
        chosen=select_tiles(list(reversed(table)))
        self.assertEqual([t['y'] for t in chosen],[32,1056,1568])
        self.assertTrue(all(t['x']==32 for t in chosen))
        failed=next(t for t in table if t['y']==544 and t['x']==32)
        self.assertFalse(failed['eligible']);self.assertEqual(failed['common_support'],0)

if __name__=='__main__':unittest.main()
