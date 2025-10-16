import heapq  # Модуль для работы с очередью с приоритетами
import itertools
from functools import lru_cache
from collections import Counter
#from parser import test

class Branch:
    __slots__ = ('birds', 'unfillness', '_hash')  # Уменьшаем использование памяти
    def __init__(self, birds_configuration):
        self.birds = self.parse_birds(birds_configuration)
        self.unfillness = self.measure_unfillness()
        self._hash = hash(self.birds)

    def parse_birds(self, birds_configuration):
        return tuple(birds_configuration)
    
    @lru_cache(maxsize=1024)
    def measure_unfillness_cached(birds_tuple):
        """Кэшированная версия вычисления unfillness"""
        if not birds_tuple or birds_tuple[0] == 0:
            return 0
        counts = Counter(birds_tuple)
        non_zero_counts = {k: v for k, v in counts.items() if k != 0}
        #non_zero_counts = {k: v for k, v in counts.items()}
        if not non_zero_counts:
            return 0
        max_count = max(non_zero_counts.values())
        return len(birds_tuple) - max_count

    def measure_unfillness(self):
        return Branch.measure_unfillness_cached(self.birds)

    def __eq__(self, other):
        return self.birds == other.birds
    
    def __hash__(self):
        return self._hash
    
class Tree:
    __slots__ = ('branch_len', 'amount_of_empty_branches', 'branches', 'branches_avaliable', 'unperfectness', '_hash')

    def __init__(self, TreeState, empty_branches=0):
        self.branch_len = len(TreeState[0])
        self.amount_of_empty_branches = self.parse_empty_Branches(TreeState) + empty_branches
        self.branches = self.parse_non_empty_Branches(TreeState)
        self.branches_avaliable = len(self.branches) + (self.amount_of_empty_branches > 0)
        self.unperfectness = sum(branch.unfillness for branch in self.branches)
        # Предварительно вычисляем хэш для ускорения  
        # Можно предварительно отсортировать для отбрасывания состояний с одинаковой конфигурацией но разными номерами веток 
        self._hash = hash(tuple(branch.birds for branch in sorted(self.branches, key=lambda x: x.birds)))
       
    def __eq__(self, other):
        if len(self.branches) != len(other.branches):
            return False
        return set(self.branches) == set(other.branches)
    
    def __hash__(self):
        return self._hash 

    def parse_non_empty_Branches(self, TreeState):
        branches = []
        for branch_state in TreeState:
            branch = Branch(branch_state)
            if branch.unfillness == 0:
                continue
            branches.append(branch)
        if self.amount_of_empty_branches > 0:
            branches.append(Branch([0] * self.branch_len))
            self.amount_of_empty_branches -= 1 
        if self.amount_of_empty_branches < 0:
            raise("Ветки в долг не даем!")

        return tuple(branches)

    def parse_empty_Branches(self, TreeState):
        return sum(1 for branch_state in TreeState if branch_state[0] == 0)
    
    def get_TreeState(self):
        return tuple(branch.birds for branch in self.branches)

class Node:
    __slots__ = ('Tree', 'g', 'h', 'f', 'parent', 'parent_diff', 'id', '_hash')
    id_iter = itertools.count()

    def __init__(self, Tree, parent_diff, g=0, h=0):
        self.Tree = Tree    # Состояние этого узла
        self.g = g  # Расстояние от начального узла до текущего узла
        self.h = h  # Примерное расстояние от текущего узла до конечного узла
        self.f = g + h  # Сумма g и h
        self.parent = None  # Родительский узел, используется для восстановления пути
        self.parent_diff = parent_diff # Изменение в родительском узле для получения этого узла
        self.id = next(self.id_iter)
        self._hash = hash(Tree)

    # Переопределяем оператор сравнения для сравнения узлов
    def __lt__(self, other):
        return self.f < other.f or (self.f == other.f and self.id < other.id)

    # Переопределяем оператор равенства для сравнения узлов
    def __eq__(self, other):
        return self.Tree == other.Tree
    
    def __hash__(self):
        return self._hash    

@lru_cache(maxsize=None)
def move_bird_cached(TreeState_tuple, src_branch_number, dst_branch_number):
    src_branch = list(TreeState_tuple[src_branch_number])
    dst_branch = list(TreeState_tuple[dst_branch_number])
    # Если некого перемещать или Не трогаем одинаковые ветки или Если нет места для новой птицы
    if src_branch[0] == 0 or src_branch_number == dst_branch_number or 0 not in dst_branch:
        return None
    
    src_non_zero_idx = None
    for i in range(len(src_branch)-1, -1, -1):
        if src_branch[i] != 0:
            src_non_zero_idx = i
            break
    
    if src_non_zero_idx is None:
        return None
    
    dst_zero_idx = None
    for i in range(len(dst_branch)):
        if dst_branch[i] == 0:
            dst_zero_idx = i
            break
    
    if dst_zero_idx is None:
        return None
    
    if dst_zero_idx == 0 or dst_branch[dst_zero_idx - 1] == src_branch[src_non_zero_idx]:
        src_branch[src_non_zero_idx], dst_branch[dst_zero_idx] = dst_branch[dst_zero_idx], src_branch[src_non_zero_idx]
        TreeState = list(TreeState_tuple)
        TreeState[src_branch_number] = src_branch
        TreeState[dst_branch_number] = dst_branch
        return (TreeState, ((src_branch_number, src_branch), (dst_branch_number, dst_branch)))

    return None

def move_bird(TreeState, src_branch_number, dst_branch_number):
    # Используем кэшированную версию
    TreeState_tuple = TreeState
    result = move_bird_cached(TreeState_tuple, src_branch_number, dst_branch_number)
    if result is None:
        return None
    new_state_tuple, moves = result
    return [new_state_tuple, moves]

def get_neighbors(current_node):
    neighbors = []
    CurrentTree = current_node.Tree.get_TreeState()
    
    # Предварительно собираем информацию о ветках
    non_empty_branches = []
    empty_branches = []
    
    for i, branch in enumerate(CurrentTree):
        if branch[0] == 0:
            empty_branches.append(i)
        else:
            non_empty_branches.append(i)
    
    # Ограничиваем количество проверяемых перемещений
    for src_idx in non_empty_branches:
        # Пробуем сначала перемещения в пустые ветки (обычно более выгодны)
        for dst_idx in empty_branches:
            if src_idx == dst_idx:
                continue
            result = move_bird(CurrentTree, src_idx, dst_idx)
            if result is not None:
                NewTree, parent_diff = result
                neighbor = Node(Tree(NewTree, current_node.Tree.amount_of_empty_branches), parent_diff)
                neighbors.append(neighbor)
        
        # Затем перемещения между непустыми ветками
        for dst_idx in non_empty_branches:
            if src_idx == dst_idx:  # Избегаем дублирующих проверок
                continue
            result = move_bird(CurrentTree, src_idx, dst_idx)
            if result is not None:
                NewTree, parent_diff = result
                neighbor = Node(Tree(NewTree, current_node.Tree.amount_of_empty_branches), parent_diff)
                neighbors.append(neighbor)
    
    return neighbors

def astar(startTreeState):
    # Создаем начальный и конечный узлы
    start_node = Node(Tree(startTreeState), [])
    start_node.h = start_node.Tree.unperfectness
    start_node.f = start_node.g * 1 + start_node.h

    # Инициализируем очередь с приоритетами
    open_list = []
    heapq.heappush(open_list, start_node)


    # Используем словарь для быстрого поиска узлов по состоянию
    open_dict = {hash(start_node): start_node}
    # Инициализируем множество посещенных узлов
    closed_set = set()

    nodes_processed = 0
    # Пока очередь с приоритетами не пуста
    while open_list:
        # Извлекаем узел с наименьшей оценкой f
        current_node = heapq.heappop(open_list)
        current_hash = hash(current_node)
        nodes_processed += 1

        if current_hash in closed_set:
            continue

        # Если текущий узел является конечным
        if current_node.h == 0:
            # Восстанавливаем путь от конечного узла до начального
            path = []
            while current_node is not None:
                path.append((current_node.parent_diff))
                current_node = current_node.parent
            print(f"Nodes processed: {nodes_processed}")
            return [len(path[-2::-1]), path[-2::-1]]

        # Добавляем текущий узел в множество посещенных узлов
        closed_set.add(current_hash)
        del open_dict[current_hash]

        # Получаем соседние узлы
        neighbors = get_neighbors(current_node)

        # Для каждого соседнего узла
        for neighbor in neighbors:
            neighbor.g = current_node.g + 1
            neighbor.h = neighbor.Tree.unperfectness
            neighbor.f = neighbor.g * 1 + neighbor.h
            neighbor.parent = current_node
            
            neighbor_hash = hash(neighbor)
            # Если соседний узел уже был посещен, пропускаем его
            if neighbor_hash in closed_set:
                continue

            # Вычисляем расстояние от начального узла до соседнего узлa
            
            # Если соседний узел уже находится в очереди с приоритетами
            if neighbor_hash in open_dict:
                existing_node = open_dict[neighbor_hash]
                # Если новое расстояние до соседнего узла меньше, чем старое, обновляем значения g, h и f
                if neighbor.g < existing_node.g:
                    existing_node.g = neighbor.g
                    existing_node.f = existing_node.g * 1 + existing_node.h
                    existing_node.parent = current_node
                    # Раз поменялся parent, то должен поменяться и parent_diff. КАК?
                    # А вот так:
                    existing_node.parent_diff = neighbor.parent_diff
                    # Обновляем приоритет соседнего узла в очереди с приоритетами
                    heapq.heapify(open_list)
            else:
                # Иначе добавляем соседний узел в очередь с приоритетами
                heapq.heappush(open_list, neighbor)
                open_dict[neighbor_hash] = neighbor

    print(f"Nodes processed: {nodes_processed}")
    # Если конечный узел недостижим, возвращаем None
    return None

#DATA = [[1, 2, 3, 4], [4, 2, 3, 1], [1, 2, 4, 3], [3, 4, 1, 2]] + [[0, 0, 0, 0]] * 31
#DATA = [[1, 2, 3, 4, 5], [4, 2, 3, 1, 5], [1, 2, 4, 3, 5], [3, 4, 1, 2, 5], [3, 5, 1, 4, 2]] + [[0, 0, 0, 0, 0]] * 31
#DATA = [[1, 2, 1, 2], [2, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
#DATA = [[1, 2], [2, 1], [0, 0]]
#DATA = [[1, 2, 2, 2, 3, 1], [2, 1, 3, 3, 1, 1], [2, 1, 3, 2, 3, 3], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
DATA = [[2, 2, 1], [5, 4, 3], [5, 3, 6], [9, 8, 7], [9, 10, 4], [8, 11, 6], [13, 9, 12], [13, 3, 1], [13, 6, 5], [1, 14, 7], [10, 12, 4], [14, 8, 14], [12, 10, 11], [15, 2, 11], [7, 15, 15], [0, 0, 0], [0, 0, 0]]
#DATA = [[2, 3, 2, 1], [1, 3, 5, 4], [1, 6, 4, 2], [1, 7, 6, 7], [8, 8, 6, 5], [8, 3, 2, 4], [8, 7, 5, 5], [4, 3, 7, 6], [0, 0, 0, 0], [0, 0, 0, 0]]
#DATA = [[4, 3, 2, 1], [1, 6, 3, 5], [1, 8, 7, 7], [1, 10, 5, 9], [6, 10, 6, 8], [11, 2, 11, 2], [13, 6, 11, 4], [13, 12, 4, 8], [12, 11, 13, 10], [12, 10, 4, 13], [12, 9, 5, 7], [3, 8, 9, 2], [9, 3, 5, 7], [0, 0, 0, 0], [0, 0, 0, 0]]
#DATA = [[9, 15, 14, 11], [9, 6, 3, 14], [4, 8, 8, 12], [8, 11, 14, 12], [8, 6, 10, 12], [13, 7, 13, 6], [11, 4, 1, 4], [11, 13, 15, 14], [1, 10, 7, 10], [10, 9, 3, 15], [3, 4, 1, 1], [7, 15, 9, 12], [6, 7, 3, 13], [0, 0, 0, 0], [0, 0, 0, 0]]
#DATA = [[8, 7, 10, 7, 10], [13, 6, 1, 3, 11], [13, 6, 7, 15, 1], [10, 8, 11, 1, 8], [14, 13, 14, 13, 3], [14, 15, 8, 12, 1], [12, 3, 8, 15, 11], [15, 12, 12, 12, 10], [3, 14, 1, 6, 10], [7, 13, 3, 6, 11], [7, 14, 6, 15, 11], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
#DATA = [[4, 4, 2, 3, 2, 1], [4, 3, 3, 1, 5, 3], [1, 2, 4, 3, 5, 5], [1, 6, 3, 6, 6, 6], [3, 5, 6, 5, 6, 5], [2, 3, 2, 2, 3, 6], [2, 3, 5, 1, 5, 3], [2, 2, 5, 3, 2, 2], [2, 5, 4, 5, 6, 6], [6, 6, 4, 1, 5, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

steps_count, steps = astar(DATA)
#print(steps_count, steps)



def test(branch_list, steps_list):
    for step_ind, (from_branch, to_branch) in enumerate(steps_list):
        branch = from_branch[1].copy()
        if 0 in from_branch[1]:
            i = from_branch[1].index(0)
        else:
            i = -1
        if to_branch[1] == [0] * len(to_branch[1]):
            print(f'Error in step {step_ind}: branch, to which element was moved, is empty')
            break
        if 0 in to_branch[1]:
            j = to_branch[1].index(0) - 1
        else:
            j = -1
        branch[i] = to_branch[1][j]
        if branch in branch_list:
            branch_list[branch_list.index(branch)] = from_branch[1].copy()
            branch = to_branch[1].copy()
            branch[j] = 0
            if branch in branch_list:
                branch_list[branch_list.index(branch)] = to_branch[1].copy()
            else:
                print(f'Error in step {step_ind}: there was not branch, to which element was moved')
                break
        else:
            print(f'Error in step {step_ind}: there was not branch, from which element was moved')
            break
    print(f'{len(steps_list)} steps were spent to solve the tree: {branch_list}')

test(DATA, steps)