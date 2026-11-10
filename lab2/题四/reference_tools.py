"""Independent distance and game references used only for validation/analysis."""
import heapq
import math


def dijkstra_matrix(problem):
    result=[]
    for source in problem.points:
        costs={source:0};queue=[(0,source)]
        while queue:
            value,point=heapq.heappop(queue)
            if value!=costs[point]: continue
            row,col=point
            for r,c in ((row,col+1),(row+1,col),(row,col-1),(row-1,col)):
                if 0<=r<len(problem.grid) and 0<=c<len(problem.grid[0]) and problem.grid[r][c] is not None:
                    candidate=value+problem.grid[r][c]
                    if candidate<costs.get((r,c),math.inf):
                        costs[(r,c)]=candidate;heapq.heappush(queue,(candidate,(r,c)))
        result.append(tuple(costs.get(target,math.inf) for target in problem.points))
    return tuple(result)


def validate_path(problem,source,target,result,expected):
    assert result.cost==expected
    if not math.isfinite(expected):
        assert not result.path;return
    assert result.path[0]==problem.points[source] and result.path[-1]==problem.points[target]
    total=0
    for previous,current in zip(result.path,result.path[1:]):
        assert abs(previous[0]-current[0])+abs(previous[1]-current[1])==1
        fee=problem.grid[current[0]][current[1]];assert fee is not None
        total+=fee
    assert total==result.cost
    assert len(result.ledger)==len(result.path)
    assert result.ledger[0]['enter_cost']==0 and result.ledger[-1]['cumulative_cost']==total
    for row,point in zip(result.ledger,result.path):
        assert tuple(row['position'])==point


def enumerate_game(matrix,values,depth,pos_a=0,pos_b=1,remaining=None,score_a=0,score_b=0,turn='A'):
    """Full enumeration, without formal transitions, pruning, or caches."""
    remaining=tuple(range(len(values))) if remaining is None else tuple(remaining)
    if depth==0 or not remaining:return score_a-score_b,()
    candidates=[]
    position=pos_a if turn=='A' else pos_b
    for action in remaining:
        cost=matrix[position][action+2]
        if not math.isfinite(cost):continue
        gain=values[action]-cost
        rest=tuple(i for i in remaining if i!=action)
        if turn=='A':
            result,line=enumerate_game(matrix,values,depth-1,action+2,pos_b,rest,score_a+gain,score_b,'B')
        else:
            result,line=enumerate_game(matrix,values,depth-1,pos_a,action+2,rest,score_a,score_b+gain,'A')
        candidates.append((result,(action,*line)))
    if not candidates:return score_a-score_b,()
    best=(max if turn=='A' else min)(row[0] for row in candidates)
    return next(row for row in candidates if row[0]==best)


def root_values(matrix,values,depth):
    if not depth:return {}
    answers={}
    for action,value in enumerate(values):
        cost=matrix[0][action+2]
        if math.isfinite(cost):
            answers[action]=enumerate_game(matrix,values,depth-1,action+2,1,
                                          [i for i in range(len(values)) if i!=action],value-cost,0,'B')[0]
    return answers


def history_ledger(problem,matrix,labels):
    positions=[0,1];scores=[0,0];remaining=set(range(len(problem.values)));rows=[]
    for step,label in enumerate(labels):
        action=problem.labels[2:].index(label)
        assert action in remaining
        player=step%2;source=positions[player];cost=matrix[source][action+2]
        gain=problem.values[action]-cost;scores[player]+=gain
        positions[player]=action+2;remaining.remove(action)
        rows.append(dict(step=step+1,player='AB'[player],source=problem.labels[source],target=label,
                         value=problem.values[action],travel_cost=cost,gain=gain,
                         score_a=scores[0],score_b=scores[1],utility=scores[0]-scores[1],
                         remaining=','.join(problem.labels[i+2] for i in sorted(remaining))))
    return rows
