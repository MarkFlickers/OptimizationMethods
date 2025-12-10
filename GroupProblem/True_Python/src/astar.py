from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import heapq
import itertools
import time
import math

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
    tops: Tuple[int, ...]
    first_birds: Tuple[int, ...]
    unperfectness: int
    
    def __hash__(self):
        return hash(self.branches)
    
    @staticmethod
    def from_lists(branch_lists: List[List[int]]) -> "State":
        branches = tuple(tuple(row) for row in branch_lists)
        return State._create_cached(branches)

    @staticmethod
    def _calculate_heuristics(branches, tops):
        """
        ОПТИМИЗАЦИЯ 2: Улучшенная эвристика
        - Больше штраф за пустые позиции
        - Больше штраф за неупорядоченные птицы
        """
        result = 0
        branch_len = len(branches[0])
        
        for i, b in enumerate(branches):
            top = tops[i]
            if top == -1:
                continue

            fb = b[0]
            
            empty_positions = (branch_len - 1) - top
            result += empty_positions
            
            limit = top + 1
            ordered = 0
            while ordered < limit and b[ordered] == fb:
                ordered += 1

            unordered = limit - ordered
            unorder_penalty = branch_len
            result += unordered * unorder_penalty

        return result

    @staticmethod
    def _create_cached(branches: StateT) -> "State":
        if not branches:
            return State(branches, (), (), 0)

        bcount = len(branches)
        branch_len = len(branches[0])

        tops = [0] * bcount
        firsts = [0] * bcount

        for i, b in enumerate(branches):
            fb = b[0]
            firsts[i] = fb if fb != 0 else 0

            top = -1
            for k in range(branch_len - 1, -1, -1):
                if b[k] != 0:
                    top = k
                    break
            tops[i] = top

        unperf = State._calculate_heuristics(branches, tops)

        return State(
            branches=branches,
            tops=tuple(tops),
            first_birds=tuple(firsts),
            unperfectness=unperf
        )

    def apply_move(self, move: Move) -> "State":
        b_list = [list(b) for b in self.branches]
        bird = b_list[move.src_branch][move.src_pos]
        b_list[move.src_branch][move.src_pos] = 0
        b_list[move.dst_branch][move.dst_pos] = bird
        new_branches = tuple(tuple(row) for row in b_list)
        return State._create_cached(new_branches)

@dataclass
class Node:
    state: State
    g: int
    f: int

class AStarSolver:
    def __init__(self, start: State):
        self.start = start
        self.counter = itertools.count()

    def find_possible_moves(self, state: State) -> List[Move]:
        moves: List[Move] = []
        empty_branch_moves: List[Move] = []
        
        branches = state.branches
        bcount = len(branches)
        branch_len = len(branches[0]) if bcount > 0 else 0

        for src in range(bcount):
            src_top = state.tops[src]
            if src_top == -1:
                continue
            bird_to_move = branches[src][src_top]
            
            for dst in range(bcount):
                if dst == src:
                    continue
                    
                dst_top = state.tops[dst]
                dst_pos = dst_top + 1
                
                if dst_pos >= branch_len:
                    continue
                    
                if dst_pos > 0 and branches[dst][dst_pos - 1] != bird_to_move:
                    continue
                
                move = Move(src, src_top, dst, dst_pos, bird_to_move)
                
                if dst_top == -1:
                    empty_branch_moves.append(move)
                else:
                    moves.append(move)
        
        return empty_branch_moves + moves

    def reconstruct_solution(self, came_from: Dict, end_state: State) -> List[Move]:
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

    def solve(self) -> tuple:
        open_heap: List = []
        g_scores: Dict = {}
        came_from: Dict = {}
        closed_set: set = set()

        g_scores[self.start] = 0
        came_from[self.start] = (None, None)
        start_f = self.start.unperfectness
        heapq.heappush(open_heap, (start_f, next(self.counter), self.start))

        while open_heap:
            _, _, cur_state = heapq.heappop(open_heap)
            
            if cur_state.unperfectness == 0:
                moves = self.reconstruct_solution(came_from, cur_state)
                return len(moves), moves, cur_state
            
            # Если уже посещали этот узел - пропускаем
            if cur_state in closed_set:
                continue

            closed_set.add(cur_state)
            cur_g = g_scores[cur_state]

            # Расширяем узел
            for move in self.find_possible_moves(cur_state):
                new_state = cur_state.apply_move(move)
                
                # Не исследуем закрытые узлы
                if new_state in closed_set:
                    continue
                    
                tentative_g = cur_g + 1

                # A* условие обновления
                if tentative_g < g_scores.get(new_state, 10**12):
                    g_scores[new_state] = tentative_g
                    came_from[new_state] = (cur_state, move)
                    new_f = tentative_g + new_state.unperfectness
                    heapq.heappush(open_heap, (new_f, next(self.counter), new_state))

        return 0, [], self.start