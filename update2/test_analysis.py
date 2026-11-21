"""Small software tests; not additional neural experiments."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('analyze_update2',Path(__file__).with_name('analyze_update2.py'))
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
class AnalysisTests(unittest.TestCase):
    def test_perfect_predictions(self):
        m=a.metrics(np.array([0,0,1,1]),np.array([0,0,1,1]))
        self.assertEqual(m['accuracy'],1); self.assertEqual(m['macro_f1'],1)
        self.assertEqual(m['fpr'],0); self.assertEqual(m['fnr'],0)
    def test_all_negative_predictions(self):
        m=a.metrics(np.array([0,0,1,1]),np.array([0,0,0,0]))
        self.assertEqual(m['fpr'],0); self.assertEqual(m['fnr'],1)
        self.assertEqual(m['precision_hate'],0); self.assertAlmostEqual(m['macro_f1'],1/3)
    def test_confusion_counts(self):
        m=a.metrics(np.array([0,0,0,1,1]),np.array([0,0,1,0,1]))
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(2,1,1,1))
    def test_rejected_condition(self):
        with self.assertRaises(ValueError): a.require(False,'bad input')
    def test_group_names(self):
        self.assertEqual(len(a.GROUPS),7); self.assertEqual(len(set(a.GROUPS)),7)
if __name__=='__main__': unittest.main(verbosity=2)
