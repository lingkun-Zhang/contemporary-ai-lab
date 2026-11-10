"""Independent value, state, ordering and actual-distance-query checks."""
import dataclasses
import heapq
import math
import random
import unittest
from collections import Counter

from treasure_game import Problem, State, game_search, initial_state, sample_problem
from lazy_game_search import lazy_game_search


def independent_matrix(problem):
    result=[]
    for start in problem.points:
        distances={start:0};pending=[(0,start)]
        while pending:
            cost,(r,c)=heapq.heappop(pending)
            if cost!=distances[r,c]:continue
            for nr,nc in ((r,c-1),(r-1,c),(r,c+1),(r+1,c)):
                if 0<=nr<len(problem.grid) and 0<=nc<len(problem.grid[0]) and problem.grid[nr][nc] is not None:
                    candidate=cost+problem.grid[nr][nc]
                    if candidate<distances.get((nr,nc),math.inf):
                        distances[nr,nc]=candidate;heapq.heappush(pending,(candidate,(nr,nc)))
        result.append(tuple(distances.get(point,math.inf) for point in problem.points))
    return tuple(result)


def independent_step(values,matrix,state,action):
    pa,pb,mask,sa,sb,turn=state
    target=action+2
    gain=values[action]-matrix[pa if turn=='A' else pb][target]
    if turn=='A':return target,pb,mask^(1<<action),sa+gain,sb,'B'
    return pa,target,mask^(1<<action),sa,sb+gain,'A'


def independent_game(values,matrix,state,depth):
    pa,pb,mask,sa,sb,turn=state
    if not depth or not mask:return sa-sb
    source=pa if turn=='A' else pb
    possible=[a for a in range(len(values)) if mask&(1<<a) and math.isfinite(matrix[source][a+2])]
    if not possible:return sa-sb
    returned=[independent_game(values,matrix,independent_step(values,matrix,state,a),depth-1) for a in possible]
    return (max if turn=='A' else min)(returned)


def independent_optimal_targets(problem,matrix,state,depth):
    if not depth or not state[2]:return set()
    source=state[0] if state[5]=='A' else state[1]
    moves={a:independent_game(problem.values,matrix,independent_step(problem.values,matrix,state,a),depth-1)
           for a in range(len(problem.values)) if state[2]&(1<<a) and math.isfinite(matrix[source][a+2])}
    if not moves:return set()
    best=(max if state[5]=='A' else min)(moves.values())
    return {problem.labels[a+2] for a,value in moves.items() if value==best}


def tuple_state(state):
    return state.pos_a,state.pos_b,state.remaining,state.score_a,state.score_b,state.turn


class _CountedRow:
    def __init__(self,owner,source):self.owner,self.source=owner,source
    def __len__(self):return len(self.owner.values)
    def __getitem__(self,target):
        if not 0<=target<len(self):raise IndexError(target)
        self.owner.queries.append((self.source,target))
        return self.owner.values[self.source][target]


class CountedDistances:
    """Only retrieving a scalar is counted; reading dimensions is free."""
    def __init__(self,labels,matrix):
        self.labels,self.values=tuple(labels),matrix
        self.queries=[]
        self.matrix=tuple(_CountedRow(self,i) for i in range(len(matrix)))


class LazyGameSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem=sample_problem()
        cls.matrix=independent_matrix(cls.problem)

    def assert_same_search_tree(self,original,lazy):
        self.assertEqual(lazy.best_value,original.best_value)
        self.assertEqual(lazy.best_target,original.best_target)
        self.assertEqual(lazy.principal_variation,original.principal_variation)
        for field in ('visited_nodes','internal_nodes','leaf_evaluations','bound_crossings'):
            self.assertEqual(getattr(lazy.stats,field),getattr(original.stats,field),field)
        self.assertEqual(lazy.stats.visited_nodes,lazy.stats.internal_nodes+lazy.stats.leaf_evaluations)

    def test_pdf_depths_preserve_value_first_target_and_entire_visit_tree(self):
        for depth in range(7):
            truth=independent_game(self.problem.values,self.matrix,(0,1,15,0,0,'A'),depth)
            for algorithm in ('minimax','alphabeta'):
                for order in ('label','reverse','gain'):
                    with self.subTest(depth=depth,algorithm=algorithm,order=order):
                        old=game_search(self.problem,self.matrix,depth,algorithm,order)
                        new=lazy_game_search(self.problem,self.matrix,depth,algorithm,order)
                        self.assertEqual(new.best_value,truth)
                        self.assert_same_search_tree(old,new)
                        if depth in (2,4):
                            self.assertEqual((new.best_target,new.best_value),('T2',-1) if depth==2 else ('T3',3))

    def test_h2_h4_instrumented_queries_skip_unvisited_candidate_costs(self):
        for depth,old_count,new_count,visits in ((2,16,12,13),(4,52,46,47)):
            with self.subTest(depth=depth):
                eager_proxy=CountedDistances(self.problem.labels,self.matrix)
                lazy_proxy=CountedDistances(self.problem.labels,self.matrix)
                old=game_search(self.problem,eager_proxy,depth)
                new=lazy_game_search(self.problem,lazy_proxy,depth)
                self.assertEqual(len(eager_proxy.queries),old_count)
                self.assertEqual(len(lazy_proxy.queries),new_count)
                self.assertEqual(old.stats.distance_lookups,old_count)
                self.assertEqual(new.stats.distance_lookups,new_count)
                self.assertEqual(new.stats.visited_nodes,visits)
                self.assertEqual(new.stats.generated_actions,visits-1)
                self.assert_same_search_tree(old,new)
                self.assertEqual(new.stats.cutoff_events,old.stats.cutoff_events)
                self.assertEqual(new.stats.skipped_child_roots,old.stats.skipped_child_roots)

    def test_minimax_without_pruning_is_distance_query_negative_control(self):
        for depth in range(5):
            for order in ('label','reverse','gain'):
                left=CountedDistances(self.problem.labels,self.matrix)
                right=CountedDistances(self.problem.labels,self.matrix)
                old=game_search(self.problem,left,depth,'minimax',order)
                new=lazy_game_search(self.problem,right,depth,'minimax',order)
                self.assert_same_search_tree(old,new)
                # Laziness moves a query to immediately before its child visit;
                # without pruning the same pairs are read, in a different order.
                self.assertEqual(Counter(left.queries),Counter(right.queries))
                self.assertEqual(new.stats.distance_lookups,old.stats.distance_lookups)
                self.assertEqual(new.stats.cutoff_events,0)

    def test_gain_order_still_prefetches_distances_for_sorting(self):
        for depth in (2,3,4):
            old_proxy=CountedDistances(self.problem.labels,self.matrix)
            new_proxy=CountedDistances(self.problem.labels,self.matrix)
            old=game_search(self.problem,old_proxy,depth,'alphabeta','gain')
            new=lazy_game_search(self.problem,new_proxy,depth,'alphabeta','gain')
            self.assert_same_search_tree(old,new)
            self.assertEqual(old_proxy.queries,new_proxy.queries)
            self.assertEqual(new.stats.sorting_calls,old.stats.sorting_calls)
            self.assertEqual(new.stats.distance_lookups,old.stats.distance_lookups)

    def test_random_connected_games_match_independent_oracle_and_eager(self):
        rng=random.Random(20261113)
        for case in range(40):
            points=rng.sample([(r,c) for r in range(4) for c in range(4)],6)
            grid=[[rng.randrange(6) for _ in range(4)] for _ in range(4)]
            for r,c in points:grid[r][c]=1
            values=tuple(rng.randint(-3,15) for _ in range(4))
            problem=Problem(grid,('A','B','T1','T2','T3','T4'),points,values)
            matrix=independent_matrix(problem)
            for depth in range(5):
                state=(0,1,15,0,0,'A')
                truth=independent_game(values,matrix,state,depth)
                targets=independent_optimal_targets(problem,matrix,state,depth)
                for algorithm in ('minimax','alphabeta'):
                    for order in ('label','reverse','gain'):
                        with self.subTest(case=case,depth=depth,algorithm=algorithm,order=order):
                            old=game_search(problem,matrix,depth,algorithm,order)
                            new=lazy_game_search(problem,matrix,depth,algorithm,order)
                            self.assertEqual(new.best_value,truth)
                            self.assertTrue(new.best_target in targets if targets else new.best_target is None)
                            self.assert_same_search_tree(old,new)
                            if order!='gain':self.assertEqual(new.stats.distance_lookups,new.stats.visited_nodes-1)

    def test_b_turn_and_nonzero_previous_scores_match_independent_transition(self):
        rng=random.Random(20261114)
        for case in range(24):
            raw=(0,1,15,0,0,'A')
            history=rng.sample(range(4),rng.randrange(4))
            for action in history:raw=independent_step(self.problem.values,self.matrix,raw,action)
            raw=(raw[0],raw[1],raw[2],raw[3]+3,raw[4]-2,raw[5])
            state=State(*raw)
            for depth in range(5):
                truth=independent_game(self.problem.values,self.matrix,raw,depth)
                for order in ('label','reverse','gain'):
                    with self.subTest(case=case,depth=depth,order=order):
                        old=game_search(self.problem,self.matrix,depth,order=order,state=state)
                        new=lazy_game_search(self.problem,self.matrix,depth,order=order,state=state)
                        self.assertEqual(new.best_value,truth)
                        self.assert_same_search_tree(old,new)

    def test_negative_gain_is_not_an_implicit_pass(self):
        problem=dataclasses.replace(self.problem,values=(-20,-25,-30,-35))
        truth=independent_game(problem.values,self.matrix,(0,1,15,0,0,'A'),1)
        self.assertLess(truth,0)
        for algorithm in ('minimax','alphabeta'):
            for order in ('label','reverse','gain'):
                result=lazy_game_search(problem,self.matrix,1,algorithm,order)
                self.assertEqual(result.best_value,truth)
                self.assertIsNotNone(result.best_target)
                self.assertEqual(len(result.principal_variation),1)
                self.assertGreater(result.stats.visited_nodes,1)

    def test_zero_horizon_and_empty_remaining_do_not_query_distance(self):
        for depth,state in [(0,State(0,1,15,-2,5,'B')),(9,State(2,3,0,-2,5,'B'))]:
            for order in ('label','reverse','gain'):
                proxy=CountedDistances(self.problem.labels,self.matrix)
                result=lazy_game_search(self.problem,proxy,depth,order=order,state=state)
                self.assertEqual((result.best_target,result.best_value,result.principal_variation),(None,-7,()))
                self.assertEqual(proxy.queries,[])
                self.assertEqual((result.stats.visited_nodes,result.stats.leaf_evaluations,result.stats.internal_nodes),(1,1,0))

    def test_unreachable_candidates_are_skipped_but_current_player_does_not_pass(self):
        problem=Problem(((1,1,None,1,1,None,1,1),),('A','B','T1','T2','T3','T4'),
                        ((0,0),(0,3),(0,1),(0,4),(0,6),(0,7)),(4,5,6,7))
        matrix=independent_matrix(problem)
        for depth in range(5):
            for order in ('label','reverse','gain'):
                old=game_search(problem,matrix,depth,order=order)
                new=lazy_game_search(problem,matrix,depth,order=order)
                self.assertEqual(new.best_value,independent_game(problem.values,matrix,(0,1,15,0,0,'A'),depth))
                self.assert_same_search_tree(old,new)
        # A cannot reach any remaining target; B could reach T2 if a pass were invented.
        state=State(2,1,14,3,0,'A')
        for order in ('label','reverse','gain'):
            result=lazy_game_search(problem,matrix,4,order=order,state=state)
            self.assertEqual((result.best_target,result.best_value,result.principal_variation),(None,3,()))
            self.assertEqual((result.stats.visited_nodes,result.stats.leaf_evaluations),(1,1))

    def test_search_preserves_problem_state_and_distance_values(self):
        state=State(2,1,14,-4,7,'B')
        problem_before=dataclasses.asdict(self.problem)
        state_before=dataclasses.asdict(state)
        matrix=tuple(tuple(row) for row in self.matrix)
        for order in ('label','reverse','gain'):
            lazy_game_search(self.problem,matrix,4,order=order,state=state,record_trace=True)
        self.assertEqual(dataclasses.asdict(self.problem),problem_before)
        self.assertEqual(dataclasses.asdict(state),state_before)
        self.assertEqual(matrix,self.matrix)

    def test_invalid_parameters_and_distance_shape_are_rejected(self):
        for depth in (-1,True,None,1.5):
            with self.subTest(depth=depth):
                with self.assertRaises((ValueError,TypeError)):
                    lazy_game_search(self.problem,self.matrix,depth)
        for key,value in [('algorithm','greedy'),('order','oracle'),('generation','other')]:
            with self.subTest(parameter=key):
                with self.assertRaises((ValueError,TypeError)):
                    lazy_game_search(self.problem,self.matrix,2,**{key:value})
        with self.assertRaises((ValueError,TypeError)):
            lazy_game_search(self.problem,self.matrix[:-1],2)

    def test_trace_return_bounds_and_matched_eager_mode_are_sound(self):
        for order in ('label','reverse','gain'):
            original=game_search(self.problem,self.matrix,4,order=order)
            matched=lazy_game_search(self.problem,self.matrix,4,order=order,generation='eager')
            self.assert_same_search_tree(original,matched)
            for field in ('distance_lookups','generated_actions','sorting_calls','sorted_actions','cutoff_events','skipped_child_roots'):
                self.assertEqual(getattr(original.stats,field),getattr(matched.stats,field),field)
            result=lazy_game_search(self.problem,self.matrix,4,order=order,record_trace=True)
            returns=[event for event in result.trace if event['event']=='return']
            self.assertTrue(returns)
            for event in returns:
                state=event['state']
                raw=(state['pos_a'],state['pos_b'],state['remaining'],state['score_a'],state['score_b'],state['turn'])
                truth=independent_game(self.problem.values,self.matrix,raw,4-event['path'].count('/'))
                value,kind=event['value'],event['returned_bound']
                if kind=='exact':self.assertEqual(value,truth)
                elif kind=='lower':self.assertLessEqual(value,truth)
                elif kind=='upper':self.assertGreaterEqual(value,truth)
                else:self.fail(kind)


if __name__=='__main__':unittest.main()
