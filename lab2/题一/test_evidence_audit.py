import unittest

from run_evidence_audit import map_fingerprint, paired_median_interval, percentile


class EvidenceAuditTests(unittest.TestCase):
    def test_linear_percentile_and_edges(self):
        self.assertEqual(percentile([0, 10, 20, 30], .25), 7.5)
        self.assertEqual(percentile([4], .5), 4)
        self.assertEqual(percentile([3, 1], 0), 1)
        self.assertEqual(percentile([3, 1], 1), 3)
        with self.assertRaises(ValueError):
            percentile([], .5)

    def test_pair_resampling_is_reproducible_and_constant_safe(self):
        pairs = [(0.5, 2.0)] * 8
        self.assertEqual(paired_median_interval(pairs, 100, 1), (.5, .5, 2.0, 2.0))
        varying = [(i / 10, i + 1) for i in range(8)]
        first = paired_median_interval(varying, 100, 5)
        self.assertEqual(first, paired_median_interval(varying, 100, 5))
        self.assertTrue(0 <= first[0] <= first[1] <= .7)
        self.assertTrue(1 <= first[2] <= first[3] <= 8)
        with self.assertRaises(ValueError):
            paired_median_interval(varying[:4])

    def test_map_identity_ignores_seed_but_preserves_endpoints(self):
        a = {"grid": ["..", ".."], "start": [0, 0], "goal": [1, 1], "seed": 1}
        b = dict(a, seed=2)
        self.assertEqual(map_fingerprint(a), map_fingerprint(b))
        self.assertNotEqual(map_fingerprint(a), map_fingerprint(dict(b, goal=[1, 0])))


if __name__ == "__main__":
    unittest.main()
