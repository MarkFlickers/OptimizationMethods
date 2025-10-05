import heapq  # Модуль для работы с очередью с приоритетами
import itertools

class Branch:
    def __init__(self, birds_configuration):
        self.birds = self.parse_birds(birds_configuration)
        self.unfillness = self.measure_unfillness()

    def parse_birds(self, birds_configuration):
        return tuple(birds_configuration)

    def measure_unfillness(self):
        bird_types = set(self.birds)
        #bird_types.discard(0)
        return len(self.birds) - max(self.birds.count(bird_type) for bird_type in bird_types)
    
    def __eq__(self, other):
        return self.birds == other.birds
    
class Tree:
    def __init__(self, TreeState, empty_branches = 0):
        self.branch_len = len(TreeState[0])
        self.amount_of_empty_branches = self.parse_empty_Branches(TreeState) + empty_branches
        self.branches = self.parse_non_empty_Branches(TreeState)
        self.branches_avaliable = len(self.branches) + (self.amount_of_empty_branches > 0)
        self.unperfectness = sum(branch.unfillness for branch in self.branches)

    def __eq__(self, other):
        return self.branches == other.branches
    
    def __hash__(self):
        return hash(tuple([branch.birds for branch in self.branches]))

    def parse_non_empty_Branches(self, TreeState):
        is_empty_added_flag = 0
        branches = list()
        for branch_state in TreeState:
            branch = Branch(branch_state)
            # if (is_empty_added_flag == 0) and (branch_state[0] == 0):
            #     is_empty_added_flag = 1
            if branch.unfillness == 0:
                continue
            branches.append(branch)
        #if is_empty_added_flag == 0 and self.amount_of_empty_branches > 0:
        if self.amount_of_empty_branches > 0:
            branches.append(Branch([0] * self.branch_len))
        return tuple(branches)

    def parse_empty_Branches(self, TreeState):
        count = 0
        for branch_state in TreeState:
            if branch_state[0] == 0:
                count += 1
        return count
    
    def get_TreeState(self):
        TreeState = [list(branch.birds) for branch in self.branches]
        return TreeState

class Node:

    id_iter = itertools.count()

    def __init__(self, Tree, parent_diff):
        #self.TreeState = tuple(TreeState)  # Дерево этого узла
        self.Tree = Tree
        self.g = 0  # Расстояние от начального узла до текущего узла
        self.h = 0  # Примерное расстояние от текущего узла до конечного узла
        self.f = 0  # Сумма g и h
        self.parent = None  # Родительский узел, используется для восстановления пути
        self.parent_diff = parent_diff # Изменение в родительском узле для получения этого узла
        self.id = next(self.id_iter)
        #self.parent_diff = (tuple(l) for l in parent_diff)

    # Переопределяем оператор сравнения для сравнения узлов
    def __lt__(self, other):
        return self.f < other.f

    # Переопределяем оператор равенства для сравнения узлов
    def __eq__(self, other):
        return self.Tree == other.Tree
    
    def __hash__(self):
        return hash(self.Tree)    

def move_bird(TreeState, src_branch_number, dst_branch_number):
        #TreeState = oldTree.get_TreeState
        src_branch = TreeState[src_branch_number]
        dst_branch = TreeState[dst_branch_number]
        if src_branch[0] == 0:          # Если некого перемещать
            return None
        if src_branch_number == dst_branch_number:                      # Не трогаем одинаковые ветки
            return None
        if 0 not in dst_branch:         # Если нет места для новой птицы
            return None
        if 0 in src_branch:
            if (dst_branch.index(0) == 0) or (dst_branch[dst_branch.index(0) - 1] == src_branch[src_branch.index(0) - 1]):
                src_branch[src_branch.index(0) - 1],  dst_branch[dst_branch.index(0)] = dst_branch[dst_branch.index(0)], src_branch[src_branch.index(0) - 1]
            else:
                return None
        else:
            if (dst_branch.index(0) == 0) or (dst_branch[dst_branch.index(0) - 1] == src_branch[-1]):
                src_branch[-1], dst_branch[dst_branch.index(0)] = dst_branch[dst_branch.index(0)], src_branch[-1]
            else:
                return None
            
        #newTree = Tree(TreeState)
        if src_branch == [0, 0, 0, 0] and dst_branch == [1, 1, 0, 0] and src_branch_number == 4 and dst_branch_number == 0:
            pass
        return [TreeState, [[src_branch_number, src_branch], [dst_branch_number, dst_branch]]]


# def get_final_tree(OriginTree):
#     Types = {}
#     branchlen = len(DATA[0])
#     branchamount = len(DATA)
#     for branch in DATA:
#         for bird in branch:
#             if bird in Types:
#                 Types[bird] += 1
#             else:
#                 Types[bird] = 1

#     FinalTree = []
#     for bird, amount in Types.items():
#         if bird == 0:
#             continue
#         while True:
#             FinalTree.append([bird] * (amount % (branchlen + 1)))
#             amount -= branchlen
#             if amount <= 0:
#                 FinalTree[-1] += ([0] * (-amount))
#                 break
#     FinalTree += [[0] * branchlen] * (branchamount - len(FinalTree))
#     return FinalTree

def get_neighbors(current_node):
    neighbors = []
    CurrentTree = current_node.Tree.get_TreeState()
    for i in range(len(CurrentTree)):
        for j in range(len(CurrentTree)):
            result = move_bird(CurrentTree, i, j)
            if result == None:
                continue
            [NewTree, parent_diff] = result
            #NewTree += [[0] * current_node.Tree.branch_len for _ in range(current_node.Tree.amount_of_empty_branches - 1)]
            neighbor = Node(Tree(NewTree, current_node.Tree.amount_of_empty_branches - 1), parent_diff)
            if neighbor.id in [469, 2844, 16040]:
                pass
            CurrentTree = current_node.Tree.get_TreeState()
            neighbors.append(neighbor)
    return neighbors

def astar(startTreeState):
    # Создаем начальный и конечный узлы
    start_node = Node(Tree(startTreeState), [])
    start_node.h = start_node.Tree.unperfectness
    # Инициализируем очередь с приоритетами
    open_list = []
    heapq.heappush(open_list, start_node)

    # Инициализируем множество посещенных узлов
    closed_set = set()

    # Пока очередь с приоритетами не пуста
    while open_list:
        # Извлекаем узел с наименьшей оценкой f
        current_node = heapq.heappop(open_list)

        # Если текущий узел является конечным
        if current_node.h == 0:
            # Восстанавливаем путь от конечного узла до начального
            path = []
            while current_node is not None:
                path.append((current_node.parent_diff))
                current_node = current_node.parent
            return [len(path[-2::-1]), path[-2::-1]]

        # Добавляем текущий узел в множество посещенных узлов
        closed_set.add(current_node)

        # Получаем соседние узлы
        neighbors = get_neighbors(current_node)

        # Для каждого соседнего узла
        for neighbor in neighbors:
            # Если соседний узел уже был посещен, пропускаем его
            if neighbor in closed_set:
                continue

            # Вычисляем расстояние от начального узла до соседнего узла
            new_g = current_node.g + 1

            # Если соседний узел уже находится в очереди с приоритетами
            if nfo := next((n for n in open_list if n == neighbor), None):
                # Если новое расстояние до соседнего узла меньше, чем старое, обновляем значения g, h и f
                if new_g < nfo.g:
                    nfo.g = new_g
                    nfo.h = nfo.Tree.unperfectness
                    nfo.f = nfo.g + nfo.h
                    nfo.parent = current_node
                    # Обновляем приоритет соседнего узла в очереди с приоритетами
                    heapq.heapify(open_list)
            else:
                # Иначе добавляем соседний узел в очередь с приоритетами и вычисляем значения g, h и f
                neighbor.g = new_g
                neighbor.h = neighbor.Tree.unperfectness
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node
                heapq.heappush(open_list, neighbor)
                #print([branch.birds for branch in neighbor.Tree.branches])

    # Если конечный узел недостижим, возвращаем None
    return None

DATA = [[1, 2, 3, 4], [4, 2, 3, 1], [1, 2, 4, 3], [3, 4, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#DATA = [[1, 2, 1, 2], [2, 1, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
Orig = Tree(DATA)
print(Orig.branches, Orig.unperfectness)

#TargetState = get_final_tree(DATA)

print(astar(DATA))
#IntermediateState = deepcopy(TargetState)
#print(IntermediateState)

#Assemble Origin DATA
# for branch in DATA:
#     for ind, intermediate_branch in enumerate(IntermediateState):
#         if intermediate_branch == branch[:len(intermediate_branch)]:
#             selected_branch_ind = ind
#             break
#         if intermediate_branch[0] == 0:
#             selected_branch_ind = ind
#             break
#     while branch != IntermediateState[selected_branch_ind]:
#         if IntermediateState[selected_branch_ind][0] == 0:
#             IntermediateState[selected_branch_ind] = [branch[0]]
#             a = [item[0] for item in IntermediateState]
#             ind = a.index(branch[0])
#             IntermediateState[ind].pop()
#             if not IntermediateState[ind]:
#                 IntermediateState[ind] = [0] * branchlen
#         else:
#             i = len(IntermediateState[selected_branch_ind])
#             IntermediateState[selected_branch_ind].append(branch[i])
#             a = [item[0] for item in IntermediateState]
#             ind = a.index(branch[i])
#             IntermediateState[ind].pop()
#             if not IntermediateState[ind]:
#                 IntermediateState[ind] = [0] * branchlen
#         print(IntermediateState)
        

