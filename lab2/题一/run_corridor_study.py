"""走廊压缩：固定新地图、同内核消融及包含预处理的多查询实验。

运行 python run_corridor_study.py；全部算法与统计只需标准库。
任何已有结果目录均不覆盖；计时内无oracle、文件写入或图表生成。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from bfs_baseline import bfs_search
from corridor_astar import build_corridor_index
from landmark_heuristic import build_landmark_index
from maze_astar import astar_search, SAMPLE_GRID, SAMPLE_START, SAMPLE_GOAL

VARIANTS = ("grid_fifo", "grid_low_h", "bfs", "landmark_low_h",
            "identity_low_h", "identity_landmark", "corridor_zero",
            "corridor_fifo", "corridor_low_h", "corridor_landmark")
COUNTS = (1, 4, 16, 64)
SOURCE_NAMES = ("maze_astar.py", "bfs_baseline.py", "landmark_heuristic.py",
                "corridor_astar.py", "run_corridor_study.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_maze(n, rng):
    grid = [["X"] * n for _ in range(n)]
    stack = [(1, 1)]
    grid[1][1] = "."
    while stack:
        r, c = stack[-1]
        options = [(r+dr, c+dc) for dr, dc in ((0, 2), (2, 0), (0, -2), (-2, 0))
                   if 0 < r+dr < n-1 and 0 < c+dc < n-1 and grid[r+dr][c+dc] == "X"]
        if not options:
            stack.pop()
        else:
            nr, nc = rng.choice(options)
            grid[(r+nr)//2][(c+nc)//2] = grid[nr][nc] = "."
            stack.append((nr, nc))
    return grid


def make_workloads(seed):
    workloads = []
    for n in (41, 61, 81):
        for family in ("open", "snake", "maze", "random"):
            for replica in range(2 if family in ("maze", "random") else 1):
                map_seed = seed + n*100 + (100000 if family == "random" else 0) + replica
                rng = random.Random(map_seed)
                if family == "open":
                    grid = [["."]*n for _ in range(n)]
                elif family == "snake":
                    grid = [["X"]*n for _ in range(n)]
                    for r in range(1, n-1, 2):
                        grid[r][1:n-1] = ["."]*(n-2)
                        if r+2 < n-1:
                            grid[r+1][n-2 if ((r-1)//2) % 2 == 0 else 1] = "."
                elif family == "maze":
                    grid = make_maze(n, rng)
                else:
                    grid = [["X" if rng.random() < .25 else "." for _ in range(n)]
                            for _ in range(n)]
                cells = [(r, c) for r in range(n) for c in range(n) if grid[r][c] != "X"]
                query_rng = random.Random(map_seed + 700000)
                queries, seen = [], set()
                while len(queries) < 64:
                    a, b = query_rng.sample(cells, 2)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        queries.append((a, b))
                workloads.append(dict(case=f"{family}_{n}_{replica}", family=family,
                    size=n, map_seed=map_seed, grid=["".join(row) for row in grid],
                    queries=queries))
    # 开阔图和蛇形图没有伪称“随机重复地图”；这里每张保存图均不同。
    assert len(workloads) == 18
    assert len({tuple(row["grid"]) for row in workloads}) == len(workloads)
    return workloads


def oracle_distance(grid, start, end):
    """独立BFS；不调用待测求解器及其辅助函数。"""
    queue, distances = deque([start]), {start: 0}
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            return distances[end]
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and
                    grid[nr][nc] != "X" and (nr, nc) not in distances):
                distances[(nr, nc)] = distances[(r, c)]+1
                queue.append((nr, nc))
    return -1


def validate_result(grid, pair, result, expected):
    if result.cost != expected:
        raise AssertionError((pair, expected, result.cost))
    if expected < 0:
        assert result.path == []
        return
    assert result.path[0] == pair[0] and result.path[-1] == pair[1]
    assert len(result.path)-1 == expected
    for r, c in result.path:
        assert 0 <= r < len(grid) and 0 <= c < len(grid[0])
        assert grid[r][c] != "X"
    for a, b in zip(result.path, result.path[1:]):
        assert abs(a[0]-b[0])+abs(a[1]-b[1]) == 1


def run_batch(workload, variant, query_count):
    """一次完整任务：独立建表→接入端点→搜索→生成全部逐格路径。"""
    grid, queries = workload["grid"], workload["queries"][:query_count]
    begin = perf_counter_ns()
    index = None
    estimate = "manhattan"
    if variant.startswith(("identity", "corridor")):
        index = build_corridor_index(grid, contract=variant.startswith("corridor"))
    if "landmark" in variant:
        estimate = build_landmark_index(grid, max_landmarks=1).heuristic(1)
    if variant == "corridor_zero":
        estimate = "zero"
    built = perf_counter_ns()
    if index is not None:
        tie = "fifo" if variant in ("corridor_zero", "corridor_fifo") else "low_h"
        results = [index.search(a, b, estimate, tie) for a, b in queries]
    elif variant == "bfs":
        results = [bfs_search(a, b, grid) for a, b in queries]
    else:
        results = [astar_search(a, b, grid, estimate,
                   "fifo" if variant == "grid_fifo" else "low_h") for a, b in queries]
    end = perf_counter_ns()
    return results, dict(build_ms=(built-begin)/1e6, query_ms=(end-built)/1e6,
                         total_ms=(end-begin)/1e6)


def describe(values):
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return dict(median=statistics.median(ordered), q1=quartiles[0], q3=quartiles[2])


def summarize(raw):
    groups = {}
    lookup = {}
    for row in raw:
        key = row["case"], row["queries"], row["variant"]
        groups.setdefault(key, []).append(row)
        lookup[(row["case"], row["queries"], row["block"], row["variant"])] = row
    rows = []
    for (case, q, variant), group in groups.items():
        row = dict(case=case, family=group[0]["family"], size=group[0]["size"],
                   queries=q, variant=variant, repeats=len(group))
        for metric in ("build_ms", "query_ms", "total_ms"):
            for name, value in describe([r[metric] for r in group]).items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    pairs = []
    comparisons = (("corridor_low_h", "grid_low_h"),
                   ("corridor_low_h", "identity_low_h"),
                   ("corridor_low_h", "bfs"),
                   ("corridor_landmark", "landmark_low_h"),
                   ("corridor_landmark", "identity_landmark"),
                   ("corridor_landmark", "corridor_low_h"))
    cases = list(dict.fromkeys(row["case"] for row in raw))
    for case in cases:
        sample = next(row for row in raw if row["case"] == case)
        for q in COUNTS:
            blocks = sorted({row["block"] for row in raw if row["case"] == case and row["queries"] == q})
            for candidate, reference in comparisons:
                deltas = [lookup[(case, q, block, candidate)]["total_ms"]-
                          lookup[(case, q, block, reference)]["total_ms"] for block in blocks]
                pairs.append(dict(case=case, family=sample["family"], size=sample["size"],
                    queries=q, candidate=candidate, reference=reference,
                    delta_median_ms=statistics.median(deltas),
                    delta_q1_ms=describe(deltas)["q1"], delta_q3_ms=describe(deltas)["q3"],
                    faster_blocks=sum(value < 0 for value in deltas), repeats=len(deltas)))
    return rows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/corridor_study")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("至少3个计时块")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        output = output / ("rerun_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    workloads = make_workloads(args.seed)
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_phase="before solver/oracle warmup and every timed run",
        seed=args.seed, repeats=args.repeats, variants=VARIANTS, query_counts=COUNTS,
        source_sha256={name: digest(root/name) for name in SOURCE_NAMES},
        workloads=workloads, design={
            "maps": "18 distinct maps: 3 open, 3 snakes, 6 independent DFS mazes, 6 independent Bernoulli p=.25 maps; sizes41/61/81",
            "queries": "64 distinct ordered pairs sampled from all passable cells, unfiltered by reachability; counts use shared prefixes",
            "primary": "per-map paired total delta corridor_low_h minus grid_low_h; matched identity_low_h isolates graph reduction",
            "strong_baseline": "existing k=1 landmark with low_h, plus compression x landmark 2x2 within the same macro-search kernel",
            "timing": "whole independent build + terminal insertion + search + full path reconstruction; split times descriptive, total is primary",
            "blocks": "seeded shuffled workload/count tasks and seeded shuffled variants within each block; serial execution; default interpreter GC",
            "warmup": "one complete 64-query batch per variant/map, after registration; discarded",
            "oracle": "independent BFS costs and full path validation outside timing; every timed result validated afterward",
            "interpretation": "7 timing repetitions are not 7 independent maps; IQR describes repeats, not a confidence interval; retain all losses",
            "break_even": "first tested query count with paired median below zero; descriptive sampled boundary, not exact threshold or guaranteed monotonicity",
            "scope": "static four-direction unit grid only; original formal solver remains unchanged"})
    protocol_path = output/"protocol.json"
    with protocol_path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    print("协议已保存：18张地图、10种实现、4个查询数；开始独立校验。", flush=True)
    structures, query_rows, oracle_rows = [], [], []
    expected_by_case = {}
    for work in workloads:
        grid, queries = work["grid"], work["queries"]
        index = build_corridor_index(grid)
        expected = [oracle_distance(grid, a, b) for a, b in queries]
        expected_by_case[work["case"]] = expected
        structures.append(dict(case=work["case"], family=work["family"], size=work["size"],
            **index.metadata, solved_queries=sum(cost >= 0 for cost in expected),
            internal_terminal_queries=sum(a in index.interiors or b in index.interiors for a, b in queries),
            both_internal_queries=sum(a in index.interiors and b in index.interiors for a, b in queries),
            same_corridor_queries=sum(a in index.interiors and b in index.interiors and
                index.interiors[a][0] == index.interiors[b][0] for a, b in queries)))
        for i, ((a, b), distance) in enumerate(zip(queries, expected)):
            oracle_rows.append(dict(case=work["case"], query=i, start=a, end=b, cost=distance))
        for variant in VARIANTS:
            results, _ = run_batch(work, variant, 64)
            for i, (pair, result, distance) in enumerate(zip(queries, results, expected)):
                validate_result(grid, pair, result, distance)
                query_rows.append(dict(case=work["case"], family=work["family"], size=work["size"],
                    variant=variant, query=i, cost=distance, expanded_nodes=result.stats["expanded_nodes"],
                    generated_nodes=result.stats["generated_nodes"], popped_nodes=result.stats["popped_nodes"],
                    max_frontier=result.stats["max_frontier"], output_steps=max(distance, 0)))
        print("已校验/预热 "+work["case"], flush=True)
    write_csv(output/"structures.csv", structures)
    write_csv(output/"query_stats.csv", query_rows)
    write_csv(output/"oracle.csv", oracle_rows)

    raw, rng = [], random.Random(args.seed+990001)
    tasks = [(work, q) for work in workloads for q in COUNTS]
    for block in range(args.repeats):
        rng.shuffle(tasks)
        for work, q in tasks:
            variants = list(VARIANTS)
            rng.shuffle(variants)
            expected = expected_by_case[work["case"]][:q]
            for variant in variants:
                results, times = run_batch(work, variant, q)
                for pair, result, distance in zip(work["queries"][:q], results, expected):
                    validate_result(work["grid"], pair, result, distance)
                raw.append(dict(case=work["case"], family=work["family"], size=work["size"],
                    queries=q, variant=variant, block=block, **times))
        write_csv(output/"timings_raw.csv", raw)
        print(f"正式计时块 {block+1}/{args.repeats} 完成", flush=True)
    summary, paired = summarize(raw)
    write_csv(output/"timing_summary.csv", summary)
    write_csv(output/"paired_deltas.csv", paired)
    sample = build_corridor_index(SAMPLE_GRID)
    sample_result = sample.search(SAMPLE_START, SAMPLE_GOAL)
    save_json(output/"sample.json", dict(metadata=sample.metadata, path=sample_result.path,
              cost=sample_result.cost, stats=sample_result.stats))
    save_json(output/"manifest.json", dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(protocol_path), source_sha256=protocol["source_sha256"],
        python=sys.version, platform=platform.platform(),
        distinct_maps=len(workloads), distinct_queries=len(oracle_rows),
        checked_untimed_results=len(query_rows), timed_batches=len(raw),
        checked_timed_results=sum(row["queries"] for row in raw),
        all_paths_and_costs_validated=True,
        sha256={path.name: digest(path) for path in output.iterdir() if path.is_file()}))
    print("实验完成：所有路径、最短代价已核验；原始耗时未删改。", flush=True)


if __name__ == "__main__":
    main()
