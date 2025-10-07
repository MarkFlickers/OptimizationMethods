import heapq  # Модуль для работы с очередью с приоритетами
import itertools
from functools import lru_cache
from collections import Counter

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
        return tuple(branches)

    def parse_empty_Branches(self, TreeState):
        return sum(1 for branch_state in TreeState if branch_state[0] == 0)
    
    def get_TreeState(self):
        return [list(branch.birds) for branch in self.branches]

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

@lru_cache(maxsize=4096)
def move_bird_cached(TreeState_tuple, src_branch_number, dst_branch_number):
    TreeState = [list(branch) for branch in TreeState_tuple]
    src_branch = TreeState[src_branch_number]
    dst_branch = TreeState[dst_branch_number]
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
        return (tuple(tuple(branch) for branch in TreeState), ((src_branch_number, src_branch), (dst_branch_number, dst_branch)))

    return None

def move_bird(TreeState, src_branch_number, dst_branch_number):
    # Используем кэшированную версию
    TreeState_tuple = tuple(tuple(branch) for branch in TreeState)
    result = move_bird_cached(TreeState_tuple, src_branch_number, dst_branch_number)
    if result is None:
        return None
    new_state_tuple, moves = result
    # Конвертируем обратно в список списков
    new_state = [list(branch) for branch in new_state_tuple]
    return [new_state, moves]

def get_neighbors(current_node):
    # neighbors = []
    # CurrentTree = current_node.Tree.get_TreeState()
    # #current_hash = hash(current_node.Tree)
    
    # # Предварительно собираем информацию о ветках
    # non_empty_branches = []
    # empty_branches = []
    
    # for i, branch in enumerate(CurrentTree):
    #     if branch[0] == 0:
    #         empty_branches.append(i)
    #     else:
    #         non_empty_branches.append(i)
    
    # # Ограничиваем количество проверяемых перемещений
    # for src_idx in non_empty_branches:
    #     # Пробуем сначала перемещения в пустые ветки (обычно более выгодны)
    #     for dst_idx in empty_branches:
    #         if src_idx == dst_idx:
    #             continue
    #         result = move_bird(CurrentTree, src_idx, dst_idx)
    #         if result is not None:
    #             NewTree, parent_diff = result
    #             neighbor = Node(Tree(NewTree, current_node.Tree.amount_of_empty_branches - 1), parent_diff)
    #             neighbors.append(neighbor)
        
    #     # Затем перемещения между непустыми ветками
    #     for dst_idx in non_empty_branches:
    #         if src_idx >= dst_idx:  # Избегаем дублирующих проверок
    #             continue
    #         result = move_bird(CurrentTree, src_idx, dst_idx)
    #         if result is not None:
    #             NewTree, parent_diff = result
    #             neighbor = Node(Tree(NewTree, current_node.Tree.amount_of_empty_branches - 1), parent_diff)
    #             neighbors.append(neighbor)
    
    # return neighbors
    neighbors = []
    CurrentTree = current_node.Tree.get_TreeState()
    for i in range(len(CurrentTree)):
        # Пропускаем полностью пустые ветки как источник
        if CurrentTree[i][0] == 0:
            continue
        for j in range(len(CurrentTree)):
            if i == j:
                continue
            result = move_bird(CurrentTree, i, j)
            if result is not None:
                NewTree, parent_diff = result
                neighbor = Node(Tree(NewTree, current_node.Tree.amount_of_empty_branches - 1), parent_diff)
                neighbors.append(neighbor)
    
    return neighbors

def astar(startTreeState):
    # Создаем начальный и конечный узлы
    start_node = Node(Tree(startTreeState), [])
    start_node.h = start_node.Tree.unperfectness
    start_node.f = start_node.g + start_node.h
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
            neighbor.f = neighbor.g + neighbor.h
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
                    existing_node.f = existing_node.g + existing_node.h
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

    # Если конечный узел недостижим, возвращаем None
    return None

# DATA = [[1, 2, 3, 4], [4, 2, 3, 1], [1, 2, 4, 3], [3, 4, 1, 2]] + [[0, 0, 0, 0]] * 31
 DATA = [[1, 2, 3, 4, 5], [4, 2, 3, 1, 5], [1, 2, 4, 3, 5], [3, 4, 1, 2, 5], [3, 5, 1, 4, 2]] + [[0, 0, 0, 0, 0]] * 31
# DATA = [[1, 2, 1, 2], [2, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
Orig = Tree(DATA)

print(astar(DATA))

        

