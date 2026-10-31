"""Offline logic checks on SYNTHETIC fixtures, NOT research experiments."""
import tempfile
import unittest
from pathlib import Path
import sys
import zipfile
import json
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project as p

class LogicTests(unittest.TestCase):
    def test_metrics_known_confusion(self):
        m = p.binary_metrics([0, 0, 1, 1], [.1, .8, .2, .9])
        self.assertEqual([m[k] for k in ('tn', 'fp', 'fn', 'tp')], [1, 1, 1, 1])
        for k in ('accuracy', 'macro_f1', 'fpr', 'fnr'):
            self.assertAlmostEqual(m[k], .5)

    def test_absent_denominator_is_not_zero(self):
        self.assertIsNone(p.binary_metrics([1, 1], [.8, .2])['fpr'])
        self.assertIsNone(p.binary_metrics([0, 0], [.8, .2])['fnr'])

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError): p.binary_metrics([0], [float('nan')])
        with self.assertRaises(ValueError): p.binary_metrics([0], [1.2])
        with self.assertRaises(ValueError): p.binary_metrics([0, 1], [.2])

    def test_threshold_direction(self):
        low = p.binary_metrics([0, 0, 1, 1], [.2, .6, .4, .8], .3)
        high = p.binary_metrics([0, 0, 1, 1], [.2, .6, .4, .8], .7)
        self.assertLessEqual(high['fpr'], low['fpr'])
        self.assertGreaterEqual(high['fnr'], low['fnr'])

    def test_original_three_way_vote(self):
        self.assertIsNone(p.majority_label([{'label': 'normal'}, {'label': 'offensive'}, {'label': 'hatespeech'}]))
        self.assertEqual(p.majority_label([{'label': 'offensive'}, {'label': 'offensive'}, {'label': 'hatespeech'}]), 'offensive')
        self.assertEqual(p.LABEL_MAP['offensive'], 0)

    def test_unknown_annotation_rejected(self):
        with self.assertRaises(ValueError): p.majority_label([{'label': 'unknown'}])

    def test_normalization_is_for_matching(self):
        self.assertEqual(p.normal_key(' A  B\n'), p.normal_key('a b'))
        self.assertNotEqual(p.normal_key('a b!'), p.normal_key('a b'))

    def test_label_order_verified(self):
        self.assertEqual(p.positive_class_index({'0': 'NON_HATE', '1': 'HATE'}), 1)
        self.assertEqual(p.positive_class_index({0: 'HATE', 1: 'NOT-HATE'}), 0)
        with self.assertRaises(ValueError): p.positive_class_index({0: 'LABEL_0', 1: 'LABEL_1'})

    def test_data_split_filters(self):
        def frame(prefix, extra):
            return pd.DataFrame([{'id': prefix+'0', 'key': prefix+'0', 'label': 0},
                                 {'id': prefix+'1', 'key': prefix+'1', 'label': 1}] + extra)
        frames = {'test': frame('t', []), 'val': frame('v', [{'id': 'dup_t', 'key': 't0', 'label': 0}]),
                  'train': frame('r', [{'id': 'dup_v', 'key': 'v0', 'label': 0}, {'id': 'dup_h', 'key': 'hc', 'label': 1}])}
        out, counts = p.isolate_splits(frames, pd.DataFrame({'key': ['hc']}))
        self.assertEqual(counts, {'test': 0, 'val': 1, 'train': 2})
        self.assertFalse(set(out['train'].key) & set(out['val'].key))

    def test_parse_majority_no_binary_tie_shortcut(self):
        raw, split_ids = {}, {}
        for split in ('train', 'val', 'test'):
            split_ids[split] = [split+'a', split+'b']
            raw[split+'a'] = {'post_tokens': [split, 'example'], 'annotators': [{'label': 'normal'}, {'label': 'normal'}, {'label': 'hatespeech'}]}
            raw[split+'b'] = {'post_tokens': [split, 'tie'], 'annotators': [{'label': 'normal'}, {'label': 'offensive'}, {'label': 'hatespeech'}]}
        frames, audit = p.parse_hatexplain(raw, split_ids)
        self.assertEqual(len(frames['train']), 1)
        self.assertEqual(audit['train']['excluded_no_majority'], 1)

    def test_error_export_contains_ids_not_text(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            hc = pd.DataFrame({'id': ['1', '2', '3', '4'], 'label': [0, 0, 1, 1],
                'functionality': ['n', 'n', 'h', 'h'], 'target_ident': ['group']*4, 'templ_id': [1, 2, 3, 4],
                'text': ['synthetic example']*4})
            p.save_evaluation(root, 'dehatebert', 'released', hc, np.array([.1, .8, .2, .9]))
            df = pd.read_csv(root/'results/dehatebert_released_predictions.csv')
            self.assertNotIn('text', df.columns)
            self.assertEqual(len(df), 4)
            table = p.write_summary(root)
            self.assertEqual(len(table), 1)  # never invent a missing fine-tuned result

    def test_export_excludes_raw_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'project'; (root/'results').mkdir(parents=True)
            (root/'results/audit.json').write_text('{}')
            (root/'data').mkdir(); (root/'data/raw.txt').write_text('not for export')
            (root/'checkpoints').mkdir(); (root/'checkpoints/weights.bin').write_bytes(b'not for export')
            archive = p.export_results(root)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ['results/audit.json'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
