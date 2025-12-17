from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import heapq
import itertools
import time
import math

# TEMP = 6.247
# TEMP = 3.054
# TEMP = 1
TEMP = 1.04884

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
    _cached_hash: int = field(default=0, compare=False, hash=False)
    
    def __hash__(self):
        if self._cached_hash != 0:
            return self._cached_hash
        return hash(self.branches)
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.branches == other.branches
    
    @staticmethod
    def from_lists(branch_lists: List[List[int]]) -> "State":
        branches = tuple(tuple(row) for row in branch_lists)
        return State._create_cached(branches)

    @staticmethod
    def _calculate_heuristics(branches, tops):
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
            unorder_penalty = branch_len * TEMP
            result += unordered * unorder_penalty

        return result

    @staticmethod
    def _create_cached(branches: StateT) -> "State":
        if not branches:
            return State(branches, (), (), 0, hash(branches))
        
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
        branches_hash = hash(branches)

        return State(
            branches=branches,
            tops=tuple(tops),
            first_birds=tuple(firsts),
            unperfectness=unperf,
            _cached_hash=branches_hash
        )

    def apply_move(self, move: Move) -> "State":
        b_list = [list(b) for b in self.branches]
        src_branch = b_list[move.src_branch]
        dst_branch = b_list[move.dst_branch]
        
        bird = src_branch[move.src_pos]
        src_branch[move.src_pos] = 0
        dst_branch[move.dst_pos] = bird
        
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

    def solve(self, time_limit: float = 120.0) -> tuple:
        open_heap: List = []
        g_scores: Dict = {}
        came_from: Dict = {}
        closed_set: set = set()
        
        # Track best solution found
        best_solution: Optional[List[Move]] = None
        best_state: Optional[State] = self.start
        best_unperfectness: float = self.start.unperfectness
        best_g: int = 0
        
        # Start timer
        start_time = time.time()
        explored_count = 0
        
        g_scores[self.start] = 0
        came_from[self.start] = (None, None)
        start_f = self.start.unperfectness
        heapq.heappush(open_heap, (start_f, next(self.counter), self.start))
        
        # Update best if start state is better
        if self.start.unperfectness < best_unperfectness:
            best_unperfectness = self.start.unperfectness
            best_g = 0
            best_state = self.start
            best_solution = []
        
        iteration_count = 0
        CHECK_TIME_INTERVAL = 100  # Check time every N iterations

        while open_heap:
            # Periodically check elapsed time to avoid overhead
            iteration_count += 1
            if iteration_count % CHECK_TIME_INTERVAL == 0:
                elapsed = time.time() - start_time
                if elapsed >= time_limit:
                    break
            
            _, _, cur_state = heapq.heappop(open_heap)
            
            # Found goal state
            if cur_state.unperfectness == 0:
                cur_g = g_scores[cur_state]
                moves = self.reconstruct_solution(came_from, cur_state)
                return len(moves), moves, cur_state
            
            if cur_state in closed_set:
                continue

            closed_set.add(cur_state)
            explored_count += 1
            cur_g = g_scores[cur_state]
            
            # Update best solution if this state is better
            # Better = lower unperfectness, or same unperfectness but lower g_score
            is_better = (
                cur_state.unperfectness < best_unperfectness or
                (cur_state.unperfectness == best_unperfectness and cur_g < best_g)
            )
            
            if is_better:
                best_unperfectness = cur_state.unperfectness
                best_g = cur_g
                best_state = cur_state

            for move in self.find_possible_moves(cur_state):
                new_state = cur_state.apply_move(move)
                
                if new_state in closed_set:
                    continue
                    
                tentative_g = cur_g + 1

                current_g = g_scores.get(new_state, 10**12)
                if tentative_g < current_g:
                    g_scores[new_state] = tentative_g
                    came_from[new_state] = (cur_state, move)
                    new_f = tentative_g + new_state.unperfectness
                    heapq.heappush(open_heap, (new_f, next(self.counter), new_state))

        # Time limit reached - return best solution found
        best_solution = self.reconstruct_solution(came_from, best_state)
        num_moves = len(best_solution) if best_solution else 0

        return num_moves, best_solution or [], best_state or self.start

    # DATA = [
    #         [5, 7, 3, 3, 3, 7, 4, 2, 5, 2, 6, 6, 6, 1],
    #         [3, 5, 5, 1, 5, 1, 4, 4, 7, 7, 6, 5, 7, 5],
    #         [3, 3, 7, 2, 5, 6, 5, 7, 4, 5, 3, 2, 2, 5],
    #         [6, 4, 2, 6, 2, 6, 3, 7, 5, 4, 7, 5, 4, 6],
    #         [4, 5, 2, 6, 4, 4, 3, 1, 7, 7, 5, 3, 4, 1],
    #         [3, 1, 1, 6, 5, 1, 5, 3, 3, 1, 1, 1, 4, 4],
    #         [6, 2, 7, 2, 3, 3, 4, 3, 5, 7, 1, 2, 2, 4],
    #         [4, 7, 3, 4, 4, 5, 2, 1, 2, 4, 2, 4, 2, 7],
    #         [3, 6, 7, 2, 4, 6, 1, 3, 3, 4, 5, 1, 5, 3],
    #         [2, 4, 1, 7, 4, 1, 2, 5, 1, 2, 3, 6, 7, 7],
    #         [7, 3, 5, 7, 6, 7, 1, 6, 1, 3, 4, 2, 5, 4],
    #         [6, 5, 7, 7, 6, 5, 6, 7, 3, 4, 1, 2, 6, 1],
    #         [6, 2, 1, 6, 1, 2, 4, 5, 2, 6, 6, 7, 1, 5],
    #         [3, 2, 7, 3, 1, 1, 3, 2, 7, 2, 6, 6, 6, 1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     ]

# import random
# from collections import Counter

# def generate_data():
#     # Параметры: 14 строк с данными, 3 пустые в конце, всего 17 строк
#     NUM_ROWS = 14
#     NUM_COLS = 14
#     NUM_VALUES_PER_DIGIT = 28  # по 28 штук каждого от 1 до 7
    
#     # Создаем пул значений: 28 * 7 = 196 значений
#     values_pool = []
#     for digit in range(1, 8):
#         values_pool.extend([digit] * NUM_VALUES_PER_DIGIT)
    
#     # Перемешиваем пул
#     random.shuffle(values_pool)
    
#     # Генерируем 14 строк по 14 элементов
#     data = []
#     idx = 0
#     for _ in range(NUM_ROWS):
#         row = values_pool[idx:idx + NUM_COLS]
#         data.append(row)
#         idx += NUM_COLS
    
#     # Добавляем 3 пустые строки
#     for _ in range(3):
#         data.append([0] * NUM_COLS)
    
#     return data


if __name__ == "__main__":
    import argparse
    import json
    import time as _time

    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, required=True)
    parser.add_argument("--time_limit", type=float, default=10.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--jsonl", type=str, default="runs_7.jsonl")  # можно отключить: --jsonl ""
    args = parser.parse_args()

    # DATA = [
    #     [5, 7, 3, 3, 3, 7, 4, 2, 5, 2, 6, 6, 6, 1],
    #     [3, 5, 5, 1, 5, 1, 4, 4, 7, 7, 6, 5, 7, 5],
    #     [3, 3, 7, 2, 5, 6, 5, 7, 4, 5, 3, 2, 2, 5],
    #     [6, 4, 2, 6, 2, 6, 3, 7, 5, 4, 7, 5, 4, 6],
    #     [4, 5, 2, 6, 4, 4, 3, 1, 7, 7, 5, 3, 4, 1],
    #     [3, 1, 1, 6, 5, 1, 5, 3, 3, 1, 1, 1, 4, 4],
    #     [6, 2, 7, 2, 3, 3, 4, 3, 5, 7, 1, 2, 2, 4],
    #     [4, 7, 3, 4, 4, 5, 2, 1, 2, 4, 2, 4, 2, 7],
    #     [3, 6, 7, 2, 4, 6, 1, 3, 3, 4, 5, 1, 5, 3],
    #     [2, 4, 1, 7, 4, 1, 2, 5, 1, 2, 3, 6, 7, 7],
    #     [7, 3, 5, 7, 6, 7, 1, 6, 1, 3, 4, 2, 5, 4],
    #     [6, 5, 7, 7, 6, 5, 6, 7, 3, 4, 1, 2, 6, 1],
    #     [6, 2, 1, 6, 1, 2, 4, 5, 2, 6, 6, 7, 1, 5],
    #     [3, 2, 7, 3, 1, 1, 3, 2, 7, 2, 6, 6, 6, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]

    DATA = [
        [4, 4, 4, 3, 4, 5, 2, 3, 2, 3, 6, 5],
        [6, 4, 2, 4, 6, 2, 2, 3, 2, 3, 2, 4],
        [5, 4, 2, 3, 6, 1, 3, 1, 2, 4, 3, 4],
        [1, 5, 2, 3, 4, 1, 1, 1, 5, 1, 2, 1],
        [6, 6, 1, 5, 1, 5, 2, 5, 6, 3, 5, 6],
        [5, 2, 2, 4, 6, 5, 3, 3, 1, 3, 2, 5],
        [3, 4, 4, 6, 2, 1, 4, 4, 5, 6, 1, 5],
        [1, 1, 3, 4, 1, 1, 4, 5, 2, 6, 1, 1],
        [2, 4, 6, 6, 1, 6, 4, 5, 6, 3, 6, 6],
        [5, 4, 6, 1, 5, 6, 3, 5, 6, 2, 1, 6],
        [2, 4, 3, 5, 3, 2, 6, 3, 1, 1, 3, 2],
        [5, 2, 4, 5, 3, 3, 5, 2, 4, 6, 5, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # Один запуск (runs=1) — проще для тюнера; но оставим цикл
    last_result = None
    cur_temp = args.temp

    for run in range(args.runs):
        TEMP = cur_temp

        start_state = State.from_lists(DATA)
        solver = AStarSolver(start_state)

        t0 = _time.perf_counter()
        steps, moves, res_state = solver.solve(time_limit=args.time_limit)
        dt = _time.perf_counter() - t0

        last_result = {
            "temp": round(TEMP, 6),
            "run": run + 1,
            "steps": int(steps),
            "dt_sec": float(dt),
            "unperf": float(res_state.unperfectness),
            "time_limit": float(args.time_limit),
            "ts": float(_time.time()),
        }

        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(last_result, ensure_ascii=False) + "\n")

        cur_temp += 0.01

    # Тюнер будет читать именно ЭТУ строку
    print(json.dumps(last_result, ensure_ascii=False))
