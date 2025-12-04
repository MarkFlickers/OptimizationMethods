from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import heapq
import itertools

# -----------------------------------------------------------------------------
# This file was created and refactored with the assistance of ChatGPT (OpenAI).
# Original logic, algorithms and intent were preserved while improving structure,
# readability and adherence to SOLID principles.
#
# The author of the project retains all rights to the original idea, logic and
# specifications. ChatGPT is a tool and does not claim authorship or copyright.
#
# You are free to use, modify and distribute this file as part of your project.
# -----------------------------------------------------------------------------

# --- Types -----------------------------------------------------------------
Bird = int  # 0..255, аналог char в C++
Branch = List[Bird]
StateMatrix = List[Branch]

# --- DTOs ------------------------------------------------------------------
@dataclass(frozen=True)
class Move:
    src_branch: int
    src_pos: int
    dst_branch: int
    dst_pos: int
    bird: Bird

@dataclass
class SolvedTree:
    steps_amount: int = 0
    Moves: List[Move] = field(default_factory=list)
    Resultant_tree: StateMatrix = field(default_factory=list)

# --- TreeState: single responsibility for state representation and transforms
class TreeState:
    def __init__(self, state: Optional[StateMatrix] = None, old_birds_rename_table: Optional[List[int]] = None):
        # internal representation: list of list of ints (birds)
        self.branches_: StateMatrix = [list(row) for row in state] if state else []
        self.birds_rename_table: List[int] = [0] * 256  # mapping like in C++. indexed by byte
        self.total_branches_: int = len(self.branches_)
        self.branch_len_: int = len(self.branches_[0]) if self.total_branches_ > 0 else 0
        self._hash: int = 0
        self._hash_computed: bool = False

        if self.total_branches_ == 0:
            return

        # If old table provided we skip recreating rename table from old_state;
        # behavior in C++ second ctor: normalization + sort + computeHash (no createBirdsRenameTable)
        
        if old_birds_rename_table is None:
            self._normalize_birds()
            self._create_birds_rename_table(state)
        self._sort_branches()
        # hash will be computed lazily
        self._hash_computed = False

    # --- internal helpers -------------------------------------------------
    def _create_birds_rename_table(self, old_state: StateMatrix):
        # In original: map normalized values to original
        for i in range(self.total_branches_):
            for j in range(self.branch_len_):
                normalized = self.branches_[i][j]
                orig = old_state[i][j]
                self.birds_rename_table[normalized] = orig

    def _normalize_birds(self):
        type_map = [0] * 256
        next_type = 1
        # iterate and replace bird values with compact normalized numbers
        for bi in range(self.total_branches_):
            branch = self.branches_[bi]
            for j in range(len(branch)):
                bird = branch[j]
                if bird != 0:
                    ub = bird & 0xFF
                    if type_map[ub] == 0:
                        type_map[ub] = next_type
                        next_type += 1
                    new_val = type_map[ub]
                    # if there was already mapping from original (rare), propagate it
                    if self.birds_rename_table[ub] != 0:
                        self.birds_rename_table[new_val] = self.birds_rename_table[ub]
                    else:
                        self.birds_rename_table[new_val] = ub
                    branch[j] = new_val

    def _is_branch_empty_flag(self, branch: Branch) -> bool:
        return branch[0] == 0

    def _sort_branches(self):
        n = len(self.branches_)
        if n <= 1:
            return

        idx = list(range(n))
        branch_len = self.branch_len_

        is_empty = [0] * n

        if branch_len <= 8:
            # pack into integer keys
            keys = [0] * n
            for i in range(n):
                b = self.branches_[i]
                is_empty[i] = 1 if (not b or b[0] == 0) else 0
                k = 0
                for j in range(branch_len):
                    k = (k << 8) | (b[j] & 0xFF)
                keys[i] = k

            idx.sort(key=lambda i: (is_empty[i], keys[i]))  # Python sorts ascending; is_empty 0->non-empty first
        else:
            keys = [None] * n
            for i in range(n):
                b = self.branches_[i]
                is_empty[i] = 1 if (not b or b[0] == 0) else 0
                # use tuple for lexicographic comparison
                keys[i] = tuple(b)

            idx.sort(key=lambda i: (is_empty[i], keys[i]))

        # rebuild branches in sorted order (move semantics not needed in python)
        new_branches = [self.branches_[i] for i in idx]
        self.branches_ = new_branches

    # --- public API ------------------------------------------------------
    def get_branches(self) -> StateMatrix:
        return self.branches_

    def get_branches_with_original_birds(self) -> StateMatrix:
        ret = [list(branch) for branch in self.branches_]
        for bi in range(len(ret)):
            for j in range(len(ret[bi])):
                val = ret[bi][j]
                if val != 0:
                    ret[bi][j] = self.birds_rename_table[val]
        return ret

    def get_total_branches(self) -> int:
        return self.total_branches_

    def get_branch_len(self) -> int:
        return self.branch_len_

    def compute_hash(self):
        if not self._hash_computed:
            h = 0
            for branch in self.branches_:
                for c in branch:
                    h = h * 131 + (c & 0xFF)
            self._hash = h
            self._hash_computed = True

    def get_hash(self) -> int:
        self.compute_hash()
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeState):
            return False
        return self.branches_ == other.branches_

    def apply_move(self, move: Move) -> "TreeState":
        # create copy of current state (C++ used copy & modify)
        new_state = TreeState()  # empty
        new_state.branches_ = [list(branch) for branch in self.branches_]  # deep copy
        new_state.total_branches_ = self.total_branches_
        new_state.branch_len_ = self.branch_len_
        new_state.birds_rename_table = list(self.birds_rename_table)
        new_state._hash_computed = False
        # perform move
        src_branch = new_state.branches_[move.src_branch]
        dst_branch = new_state.branches_[move.dst_branch]
        bird_to_move = src_branch[move.src_pos]
        src_branch[move.src_pos] = 0
        dst_branch[move.dst_pos] = bird_to_move
        # branches come from an already-normalized TreeState: skip normalize (same as C++)
        new_state._sort_branches()
        return new_state

# --- Tree: responsibility for heuristic/unperfectness ------------------------
class Tree:
    def __init__(self, state: Optional[TreeState] = None, parent: Optional["Tree"] = None, move: Optional[Move] = None):
        if state is not None and parent is None:
            # construct from state
            self.state_ = state
            self.unperfectness_ = 0
            self._compute_unperfectness()
        elif state is not None and parent is not None and move is not None:
            # incremental constructor (not used in original code except declaration)
            self.state_ = state
            self.unperfectness_ = 0
            self._compute_unperfectness()
        else:
            self.state_ = TreeState()
            self.unperfectness_ = 0

    def get_state(self) -> TreeState:
        return self.state_

    def get_unperfectness(self) -> int:
        return self.unperfectness_

    def get_hash(self) -> int:
        return self.state_.get_hash()

    def is_target_state(self) -> bool:
        return self.unperfectness_ == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return False
        return self.state_ == other.state_

    def _compute_unperfectness(self):
        # Same algorithm as in C++ computeUnperfectness()
        self.unperfectness_ = 0
        branches = self.state_.get_branches()
        branch_len = self.state_.get_branch_len()
        for branch in branches:
            if branch[0] == 0:
                continue
            first_bird = branch[0]
            ordered_birds = 0
            empty_positions = 0
            # count empty positions from end
            j = branch_len - 1
            while j >= 0 and branch[j] == 0:
                empty_positions += 1
                j -= 1
            # count ordered birds from start (matching first_bird)
            limit = branch_len - empty_positions
            for j in range(limit):
                if branch[j] != first_bird:
                    break
                ordered_birds += 1
            unordered_birds = branch_len - empty_positions - ordered_birds
            self.unperfectness_ += unordered_birds * 26 + empty_positions

# --- Node: responsibility for node data and f/g calculation ----------------
class Node:
    def __init__(self, tree: Tree, parent: Optional["Node"], move: Optional[Move], g: int):
        self.tree_ = tree
        self.parent_ = parent
        self.move_ = move if move is not None else Move(0,0,0,0,0)
        self.g_ = g
        self.hash_ = self.tree_.get_hash()
        self.f_ = self.g_ + self.tree_.get_unperfectness()

    def get_tree(self) -> Tree:
        return self.tree_

    def get_g(self) -> int:
        return self.g_

    def get_f(self) -> int:
        return self.f_

    def get_parent(self) -> Optional["Node"]:
        return self.parent_

    def get_move(self) -> Move:
        return self.move_

    def get_hash(self) -> int:
        return self.hash_

    def update(self, parent: "Node", move: Move, g: int):
        self.parent_ = parent
        self.move_ = move
        self.g_ = g
        self.f_ = g + self.tree_.get_unperfectness()

# --- AStarSolver: responsibility for search --------------------------------
class AStarSolver:
    MAX_DEPTH = 1000

    def __init__(self, start_state: TreeState):
        self.start_state_ = start_state
        self.node_registry_: Dict[int, Node] = {}

        # counter for tie-breaking in heap
        self._counter = itertools.count()

    def should_prune(self, node: Node) -> bool:
        return node.get_g() > self.MAX_DEPTH

    def register_node(self, node: Node) -> bool:
        h = node.get_hash()
        existing = self.node_registry_.get(h)
        if existing is not None:
            if existing.get_g() <= node.get_g():
                return False
            # existing worse -> replace
            self.node_registry_[h] = node
            return True
        else:
            self.node_registry_[h] = node
            return True

    def find_possible_moves(self, tree: Tree) -> List[Move]:
        moves: List[Move] = []
        state = tree.get_state()
        branches = state.get_branches()
        branches_count = state.get_total_branches()
        branch_len = state.get_branch_len()

        for src in range(branches_count):
            src_branch = branches[src]
            if src_branch[0] == 0:
                continue
            # find topmost bird (highest index with non-zero)
            src_pos = -1
            for j in range(branch_len - 1, -1, -1):
                if src_branch[j] != 0:
                    src_pos = j
                    break
            if src_pos == -1:
                continue
            bird_to_move = src_branch[src_pos]
            for dst in range(branches_count):
                if dst == src:
                    continue
                dst_branch = branches[dst]
                # find first empty position
                dst_pos = -1
                for j in range(branch_len):
                    if dst_branch[j] == 0:
                        dst_pos = j
                        break
                if dst_pos == -1:
                    continue
                # movement rule
                if dst_pos > 0 and dst_branch[dst_pos - 1] != bird_to_move:
                    continue
                moves.append(Move(src, src_pos, dst, dst_pos, bird_to_move))
        return moves

    def solve(self) -> SolvedTree:
        self.node_registry_.clear()
        start_tree = Tree(self.start_state_)
        start_node = Node(start_tree, None, None, 0)
        self.node_registry_[start_node.get_hash()] = start_node

        # priority queue of (f, counter, node)
        heap = []
        heapq.heappush(heap, (start_node.get_f(), next(self._counter), start_node))

        closed_set = set()

        while heap:
            _, _, current_node = heapq.heappop(heap)
            if current_node.get_hash() in closed_set:
                continue

            if current_node.get_tree().is_target_state():
                solution = SolvedTree()
                solution.Resultant_tree = current_node.get_tree().get_state().get_branches_with_original_birds()
                # reconstruct path
                path_nodes = []
                cur = current_node
                while cur is not None and cur.get_parent() is not None:
                    path_nodes.append(cur.get_move())
                    cur = cur.get_parent()
                path_nodes.reverse()
                solution.Moves = path_nodes
                solution.steps_amount = len(path_nodes)
                return solution

            if self.should_prune(current_node):
                continue

            closed_set.add(current_node.get_hash())

            moves = self.find_possible_moves(current_node.get_tree())
            for move in moves:
                new_state = current_node.get_tree().get_state().apply_move(move)
                new_hash = new_state.get_hash()
                new_g = current_node.get_g() + 1

                if new_hash in closed_set:
                    continue

                existing = self.node_registry_.get(new_hash)
                if existing is not None:
                    # states equal?
                    if existing.get_tree().get_state() == new_state:
                        if new_g < existing.get_g():
                            existing.update(current_node, move, new_g)
                            heapq.heappush(heap, (existing.get_f(), next(self._counter), existing))
                    else:
                        # hash collision: replace existing node object (like in C++)
                        new_tree = Tree(new_state)
                        new_node = Node(new_tree, current_node, move, new_g)
                        self.node_registry_[new_hash] = new_node
                        heapq.heappush(heap, (new_node.get_f(), next(self._counter), new_node))
                else:
                    new_tree = Tree(new_state)
                    new_node = Node(new_tree, current_node, move, new_g)
                    self.node_registry_[new_hash] = new_node
                    heapq.heappush(heap, (new_node.get_f(), next(self._counter), new_node))

        return SolvedTree()  # empty (no solution)

# start = [
#     [2,3,3,1,2,1],
#     [3,2,1,3,1,2],
#     [2,1,2,1,2,1],
#     [1,3,3,3,1,3],
#     [1,3,1,2,1,2],
#     [3,2,2,2,3,3],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0]
# ]
# state = TreeState(start)
# solver = AStarSolver(state)
# solution = solver.solve()
# print(solution.steps_amount)
# print(solution.Moves)
# print(solution.Resultant_tree)
