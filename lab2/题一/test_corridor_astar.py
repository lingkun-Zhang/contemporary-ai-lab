"""走廊压缩的独立正确性检验；仅使用标准库与独立BFS距离判据。"""

from collections import deque
import unittest

from corridor_astar import build_corridor_index


def bfs_distances(grid, start):
    """直接逐格BFS，不调用A*、索引或其邻接生成辅助函数。"""
    height, width = len(grid), len(grid[0]) if grid else 0
    if not height or not width or grid[start[0]][start[1]] == "X":
        return {}
    distances = {start: 0}
    pending = deque([start])
    while pending:
        row, column = pending.popleft()
        for nr, nc in ((row - 1, column), (row, column - 1),
                       (row + 1, column), (row, column + 1)):
            neighbor = (nr, nc)
            if (0 <= nr < height and 0 <= nc < width
                    and grid[nr][nc] != "X" and neighbor not in distances):
                distances[neighbor] = distances[(row, column)] + 1
                pending.append(neighbor)
    return distances


class CorridorCorrectnessTests(unittest.TestCase):
    def assert_legal_result(self, grid, start, end, result, distance):
        self.assertEqual(result.cost, distance, (grid, start, end, result.path))
        if distance < 0:
            self.assertEqual(result.path, [])
            return
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], end)
        self.assertEqual(len(result.path) - 1, distance)
        self.assertEqual(result.stats["output_steps"], distance)
        for row, column in result.path:
            self.assertTrue(0 <= row < len(grid) and 0 <= column < len(grid[0]))
            self.assertNotEqual(grid[row][column], "X")
        for first, second in zip(result.path, result.path[1:]):
            self.assertEqual(abs(first[0] - second[0]) + abs(first[1] - second[1]), 1)

    def check_all_pairs(self, grid, configurations=None):
        if configurations is None:
            configurations = ((True, "manhattan", "low_h"),
                              (True, "manhattan", "fifo"),
                              (True, "zero", "fifo"),
                              (False, "manhattan", "low_h"))
        vertices = [(r, c) for r, row in enumerate(grid)
                    for c, value in enumerate(row) if value != "X"]
        oracles = {start: bfs_distances(grid, start) for start in vertices}
        indices = {contract: build_corridor_index(grid, contract=contract)
                   for contract, _, _ in configurations}
        checked = 0
        for start in vertices:
            for end in vertices:
                expected = oracles[start].get(end, -1)
                for contract, heuristic, tie_break in configurations:
                    result = indices[contract].search(start, end, heuristic, tie_break)
                    self.assert_legal_result(grid, start, end, result, expected)
                    checked += 1
        return checked

    def test_exhaustive_three_by_three_obstacles_and_ordered_endpoints(self):
        # 包含全部512种障碍布局及每种布局中所有有序开放端点（含s=t）。
        # 四个配置共46,080次搜索；每一次同时验证距离和逐格输出。
        checked = 0
        for mask in range(1 << 9):
            grid = ["".join("X" if mask & (1 << (3 * r + c)) else "."
                            for c in range(3)) for r in range(3)]
            checked += self.check_all_pairs(grid)
        self.assertEqual(checked, 46080)

    def test_singleton_straight_and_bent_corridors(self):
        for grid in (["."], ["........."], ["."] * 9,
                     ["...XX", "XX.XX", "XX..."]):
            with self.subTest(grid=grid):
                self.check_all_pairs(grid)

    def test_interior_endpoints_are_reinserted(self):
        grid = ["........."]
        index = build_corridor_index(grid)
        self.assertNotIn((0, 2), index.adjacency)
        self.assertNotIn((0, 6), index.adjacency)
        for start, end in (((0, 2), (0, 6)), ((0, 6), (0, 2)),
                           ((0, 2), (0, 2)), ((0, 0), (0, 6))):
            result = index.search(start, end)
            self.assert_legal_result(grid, start, end, result, abs(start[1] - end[1]))
        # 重复查询应保持索引本体，不能遗留上一次切分产生的局部边。
        self.assertEqual(len(index.adjacency), 2)
        self.assertEqual(len(index.paths), 1)

    def test_pure_cycle_both_routes_and_two_interior_endpoints(self):
        grid = [".....", ".XXX.", ".XXX.", ".XXX.", "....."]
        index = build_corridor_index(grid)
        self.assertEqual(index.metadata["pure_cycle_anchors"], 1)
        self.assertEqual(len(index.adjacency), 1)
        self.assertEqual(len(index.paths), 1)
        self.assertEqual(index.paths[0][0], index.paths[0][-1])
        self.check_all_pairs(grid)

    def test_long_parallel_corridors_keep_shorter_route(self):
        grid = ["XXXXXXXXX", "X.......X", "X.XXXXX.X", "X.......X",
                "X.XXXXX.X", "X.XXXXX.X", "XXXXXXXXX"]
        index = build_corridor_index(grid)
        first, second = (3, 1), (3, 7)
        lengths = sorted(len(path) - 1 for path in index.paths
                         if {path[0], path[-1]} == {first, second})
        self.assertEqual(lengths, [6, 10])
        self.assert_legal_result(grid, first, second, index.search(first, second), 6)
        self.check_all_pairs(grid)

    def test_lollipop_attached_loop_and_dead_end(self):
        grid = ["XXXXXXXXX", "X.......X", "X.XXXXX.X", "X.......X",
                "X.XXXXXXX", "X.XXXXXXX", "XXXXXXXXX"]
        index = build_corridor_index(grid)
        self.assertTrue(any(path[0] == path[-1] for path in index.paths))
        self.check_all_pairs(grid)

    def test_two_disconnected_pure_cycles(self):
        grid = ["...X...", ".X.X.X.", "...X..."]
        index = build_corridor_index(grid)
        self.assertEqual(index.metadata["pure_cycle_anchors"], 2)
        self.check_all_pairs(grid)

    def test_isolated_and_disconnected_vertices(self):
        self.check_all_pairs([".X.", "XXX", ".X."])

    def test_grid_snapshot_survives_callers_mutation(self):
        mutable = [list("S....E")]
        original = ["S....E"]
        index = build_corridor_index(mutable)
        mutable[0][2] = "X"
        mutable.append(list("XXXXXX"))
        self.assertEqual(index.grid, tuple(original))
        self.assert_legal_result(original, (0, 0), (0, 5),
                                 index.search((0, 0), (0, 5)), 5)
        # 新建索引则应读取变动后的地图，不能隐式沿用旧答案。
        self.assertEqual(build_corridor_index(mutable).search((0, 0), (0, 5)).cost, -1)

    def test_callable_heuristic_and_trace_path(self):
        grid = ["....", ".XX.", "...."]
        index = build_corridor_index(grid)
        result = index.search([0, 1], [2, 2],
                              lambda node, goal: abs(node[0] - goal[0])
                              + abs(node[1] - goal[1]), record_trace=True)
        self.assert_legal_result(grid, (0, 1), (2, 2), result, 5)
        self.assertTrue(result.trace)
        self.assertEqual(tuple(result.trace[-1]["node"]), (2, 2))

    def test_empty_grid_blocked_endpoints_and_empty_width(self):
        for grid in ([], [""], ["", ""]):
            result = build_corridor_index(grid).search((0, 0), (0, 0))
            self.assertEqual((result.cost, result.path), (-1, []))
        grid = ["X.", ".X"]
        index = build_corridor_index(grid)
        for start, end in (((0, 0), (0, 1)), ((0, 1), (1, 1)), ((0, 0), (0, 0))):
            self.assertEqual(index.search(start, end).cost, -1)

    def test_invalid_grid_and_contraction_flag(self):
        for invalid in (None, "...", [1], ["..", "."], ["?"], [[1]]):
            with self.subTest(grid=invalid), self.assertRaises(ValueError):
                build_corridor_index(invalid)
        for flag in (0, 1, None, "yes"):
            with self.subTest(contract=flag), self.assertRaises(ValueError):
                build_corridor_index(["."], contract=flag)

    def test_invalid_query(self):
        index = build_corridor_index(["..."])
        for coordinate in (None, (0,), (0, 0, 0), "00", (False, 0), (0.0, 1)):
            for start, end in ((coordinate, (0, 0)), ((0, 0), coordinate)):
                with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                    index.search(start, end)
        for start, end in (((-1, 0), (0, 0)), ((0, 0), (0, 3))):
            with self.assertRaises(ValueError):
                index.search(start, end)
        with self.assertRaises(ValueError):
            index.search((0, 0), (0, 2), heuristic="euclidean")
        with self.assertRaises(ValueError):
            index.search((0, 0), (0, 2), tie_break="lifo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
