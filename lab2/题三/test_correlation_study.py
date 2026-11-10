"""Properties of the correlation model and independent ordering diagnostics."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from alphabeta_tree import TreeNode
from run_experiments import encode, execute, oracle
from run_correlation_study import (distribution_metrics, first_leaf_mean, make_basis,
                                   make_tree, output_directory, pair_counts,
                                   quality_metrics, write_protocol)


class CorrelationStudyTests(unittest.TestCase):
    def test_basis_reproducibility_and_independent_noise_boundary(self):
        a=make_basis(812,3,3)
        self.assertEqual(a,make_basis(812,3,3))
        self.assertEqual(len(a['leaf_noise']),27)
        self.assertEqual(len(a['edge_noise']),39)
        _,values=make_tree(a,0)
        self.assertEqual(values,a['leaf_noise'])
        self.assertNotEqual(a,make_basis(813,3,3))

    def test_normalized_hierarchical_signal_and_shared_basis_mixture(self):
        basis=dict(seed=1,branching=2,depth=2,decay=.65,
                   leaf_noise=[1,2,3,4],edge_noise=[10,20,30,40,50,60])
        original=json.dumps(basis,sort_keys=True)
        _,signals=make_tree(basis,1)
        norm=math.sqrt(1+.65**2)
        expected=[(10+.65*20)/norm,(10+.65*30)/norm,
                  (40+.65*50)/norm,(40+.65*60)/norm]
        for a,b in zip(signals,expected): self.assertAlmostEqual(a,b)
        _,mixed=make_tree(basis,.25)
        for z,h,value in zip(basis['leaf_noise'],signals,mixed):
            self.assertAlmostEqual(value,math.sqrt(.75)*z+.5*h)
        self.assertNotEqual(signals[0],signals[1])
        self.assertEqual(json.dumps(basis,sort_keys=True),original)
        for invalid in (-.1,1.1,True,float('nan')):
            with self.assertRaises(ValueError): make_tree(basis,invalid)

    def test_rank_agreement_excludes_true_ties_and_half_credits_prediction_ties(self):
        tied=pair_counts([5,5],[0,1])
        self.assertEqual(tied['eligible_pairs'],1)
        self.assertEqual(tied['predicted_tied_pairs'],1)
        self.assertEqual(tied['true_tied_pairs'],0)
        excluded=pair_counts([0,1],[4,4])
        self.assertEqual(excluded['eligible_pairs'],0)
        self.assertEqual(excluded['true_tied_pairs'],1)
        self.assertEqual(pair_counts([2,1],[1,2])['discordant_pairs'],1)

    def test_mean_prediction_is_not_treated_as_minimax_truth(self):
        root=TreeNode(children=[TreeNode(children=[TreeNode(0),TreeNode(10)]),
                                TreeNode(children=[TreeNode(4),TreeNode(5)])])
        metrics=quality_metrics(root,2)
        # Means 5 and 4.5 are reverse-ranked relative to MIN truths 0 and 4.
        self.assertEqual(first_leaf_mean(root.children[0],2),5)
        self.assertEqual(oracle(root)[0],4)
        informative=metrics['nonterminal_children']
        self.assertEqual(informative['eligible_pairs'],1)
        self.assertEqual(informative['discordant_pairs'],1)
        self.assertEqual(informative['pairwise_agreement'],0)
        self.assertGreater(metrics['all']['pairwise_agreement'],0)

    def test_each_mixing_and_budget_preserves_independent_game_value(self):
        basis=make_basis(821,3,3)
        for mixing in (0,.25,.5,.75,1):
            root,values=make_tree(basis,mixing)
            case=dict(root=root,seed=821)
            expected=oracle(root)[0]
            before=json.dumps(encode(root),sort_keys=True)
            for q in (0,1,4,8):
                result,counters,_,_=execute(case,'alphabeta',f'sample_{q}',True)
                self.assertEqual(result.value,expected)
                if q==0: self.assertEqual(counters['ordering_probe_leaves'],0)
                else: self.assertGreater(counters['ordering_probe_leaves'],0)
            metrics=distribution_metrics(values,3)
            self.assertEqual(metrics['sibling_pair_count'],9)
            self.assertGreaterEqual(metrics['leaf_variance'],0)
            self.assertEqual(before,json.dumps(encode(root),sort_keys=True))

    def test_protocol_is_exclusive_and_existing_output_is_preserved(self):
        test_root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root) as tmp:
            self.assertEqual(Path(tmp).resolve().parent,test_root)
            root=Path(tmp)/'results'
            first=output_directory(root)
            protocol=first/'protocol.json'
            digest=write_protocol(protocol,dict(budgets=[0,1,4,8],seeds=[20261201]))
            original=protocol.read_bytes()
            with self.assertRaises(FileExistsError): write_protocol(protocol,dict(champion=1))
            self.assertEqual(protocol.read_bytes(),original)
            second=output_directory(root)
            self.assertNotEqual(first,second)
            self.assertEqual(second.parent,first)
            self.assertEqual(len(digest),64)


if __name__=='__main__': unittest.main()
