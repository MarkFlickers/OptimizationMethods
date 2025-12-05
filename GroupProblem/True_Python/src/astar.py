from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Iterable
import heapq
import itertools
import time
import math

from collections import defaultdict
from typing import List, Tuple
import numpy as np

# Типы
Bird = int
BranchT = Tuple[Bird, ...]
StateT = Tuple[BranchT, ...]


@dataclass(frozen=True)
class Move:
    src_branch: int
    src_pos: int
    dst_branch: int
    dst_pos: int
    bird: Bird


@dataclass(frozen=True)
class State:
    branches: StateT
    # кэшированные атрибуты для быстрого доступа
    tops: Tuple[int, ...]         # индекс верхней (последней ненулевой) позиции в каждой ветви (-1 если пустая)
    first_birds: Tuple[int, ...]  # значение первой клетки каждой ветви (0 если пустая)
    unperfectness: int            # эвристика (чем меньше — тем ближе к цели)

    @staticmethod
    def from_lists(branch_lists: List[List[int]]) -> "State":
        # преобразуем в tuple of tuples
        branches = tuple(tuple(row) for row in branch_lists)
        return State._create_cached(branches)

    @staticmethod
    def _compute_unperf_weight(branch_len) -> int:
        """
        Compute weight for unordered birds in unperfectness calculation.
        Heuristic: proportional to branch length.
        """
        if branch_len <= 0:
            return 1

        return max(1, int(math.log2(branch_len+1)*10))
    
    @staticmethod
    def _calculate_heuristics(branches, tops):
        result = 0
        branch_len = len(branches[0])
        for i, b in enumerate(branches):
            fb = b[0]
            top = tops[i]

            if top == -1:
                # пустая ветка
                continue

            # ---------- 1. Пустые позиции в конце ----------
            empty_positions = (branch_len - 1) - top

            # ---------- 2. Упорядоченные птицы ----------
            limit = top + 1
            ordered = 0
            while ordered < limit and b[ordered] == fb:
                ordered += 1

            # ---------- 3. Неупорядоченные ----------
            unordered = (limit - ordered)
            unorder_penalty = branch_len
            # ---------- 4. Увеличиваем unperfectness ----------
            result += unordered * unorder_penalty + empty_positions

        return result

    @staticmethod
    def _create_cached(branches: StateT) -> "State":
        if not branches:
            return State(branches, (), (), 0)

        bcount = len(branches)
        branch_len = len(branches[0])

        tops = [0] * bcount
        firsts = [0] * bcount
        unperf = 0

        for i, b in enumerate(branches):

            # ---------- 1. Первый элемент ----------
            fb = b[0]
            if fb == 0:
                firsts[i] = 0
            else:
                firsts[i] = fb

            # ---------- 2. Находим top ----------
            # самый быстрый способ: искать последний ненулевой с конца
            top = -1
            for k in range(branch_len - 1, -1, -1):
                if b[k] != 0:
                    top = k
                    break
            tops[i] = top

            if top == -1:
                # пустая ветка
                continue

        unperf = State._calculate_heuristics(branches, tops)

        return State(
            branches=branches,
            tops=tuple(tops),
            first_birds=tuple(firsts),
            unperfectness=unperf
        )


    def apply_move(self, move: Move) -> "State":
        # собираем новые ветви, меняя только две ветви — источник и приемник
        b_list = [list(b) for b in self.branches]  # shallow copy of rows
        # perform move
        bird = b_list[move.src_branch][move.src_pos]
        b_list[move.src_branch][move.src_pos] = 0
        b_list[move.dst_branch][move.dst_pos] = bird
        # важно: собрать immutable представление и кэшировать атрибуты
        new_branches = tuple(tuple(row) for row in b_list)
        return State._create_cached(new_branches)


@dataclass
class Node:
    state: State
    g: int
    f: int
    parent_state: Optional[State]
    parent_move: Optional[Move]


class AStarSolver:
    MAX_DEPTH = 2000  # можно регулировать

    def __init__(self, start: State):
        self.start = start
        self.counter = itertools.count()

    def find_possible_moves(self, state: State) -> List[Move]:
        moves: List[Move] = []
        branches = state.branches
        bcount = len(branches)
        branch_len = len(branches[0]) if bcount > 0 else 0

        # заранее: tops
        for src in range(bcount):
            src_top = state.tops[src]
            if src_top == -1:
                continue
            bird_to_move = branches[src][src_top]
            for dst in range(bcount):
                if dst == src:
                    continue
                # find first empty in dst (from left)
                dst_branch = branches[dst]
                dst_pos = state.tops[dst] + 1
                if dst_pos >= branch_len:
                    continue
                # rule: if dst_pos > 0 and cell before != bird_to_move -> can't
                if dst_pos > 0 and dst_branch[dst_pos - 1] != bird_to_move:
                    continue
                moves.append(Move(src, src_top, dst, dst_pos, bird_to_move))
        return moves

    def reconstruct_solution(self, came_from: Dict[State, (Optional[State], Optional[Move])],
                             end_state: State) -> List[Move]:
        moves: List[Move] = []
        cur = end_state
        while True:
            parent, move = came_from.get(cur, (None, None))
            if parent is None:
                break
            moves.append(move)
            cur = parent
        moves.reverse()
        return moves

    def solve(self) -> tuple[int, List[Move], State]:
        open_heap: List[tuple] = []
        g_scores: Dict[State, int] = {}
        came_from: Dict[State, (Optional[State], Optional[Move])] = {}

        start_node = Node(state=self.start, g=0, f=self.start.unperfectness, parent_state=None, parent_move=None)
        g_scores[self.start] = 0
        came_from[self.start] = (None, None)
        heapq.heappush(open_heap, (start_node.f, next(self.counter), start_node))

        closed = set()
        iterations = 0

        while open_heap:
            f, _, cur_node = heapq.heappop(open_heap)
            cur_state = cur_node.state

            # lazy skip if this node is outdated
            if g_scores.get(cur_state, 10**12) != cur_node.g:
                continue

            iterations += 1
            # goal?
            if cur_state.unperfectness == 0:
                moves = self.reconstruct_solution(came_from, cur_state)
                return len(moves), moves, cur_state

            if cur_node.g > self.MAX_DEPTH:
                continue

            closed.add(cur_state)

            # expand
            for move in self.find_possible_moves(cur_state):
                new_state = cur_state.apply_move(move)
                tentative_g = cur_node.g + 1

                # if we've found a strictly better path to new_state
                if tentative_g < g_scores.get(new_state, 10**12):
                    g_scores[new_state] = tentative_g
                    came_from[new_state] = (cur_state, move)
                    new_f = tentative_g + new_state.unperfectness
                    new_node = Node(state=new_state, g=tentative_g, f=new_f, parent_state=cur_state, parent_move=move)
                    heapq.heappush(open_heap, (new_node.f, next(self.counter), new_node))

        # no solution
        return 0, [], self.start
