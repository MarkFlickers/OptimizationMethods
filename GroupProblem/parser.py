#                   0          1          2         3           4           5           6           7           8            9           10           11            12          13            14          15        16
branch_list = [[2, 2, 1], [5, 4, 3], [5, 3, 6], [9, 8, 7], [9, 10, 4], [8, 11, 6], [13, 9, 12], [13, 3, 1], [13, 6, 5], [1, 14, 7], [10, 12, 4], [14, 8, 14], [12, 10, 11], [15, 2, 11], [7, 15, 15], [0, 0, 0], [0, 0, 0]]
steps_list = [((4, [9, 10, 0]), (15, [4, 0, 0])), ((10, [10, 12, 0]), (15, [4, 4, 0])), ((5, [8, 11, 0]), (16, [6, 0, 0])), ((12, [12, 10, 0]), (5, [8, 11, 11])), ((12, [12, 0, 0]), (4, [9, 10, 10])),
    ((6, [13, 9, 0]), (12, [12, 12, 0])), ((10, [10, 0, 0]), (12, [12, 12, 12])), ((4, [9, 10, 0]), (10, [10, 10, 0])), ((4, [9, 0, 0]), (10, [10, 10, 10])), ((4, [0, 0, 0]), (6, [13, 9, 9])), ((2, [5, 3, 0]), (13, [6, 6, 0])),
    ((1, [5, 4, 0]), (14, [3, 0, 0])), ((1, [5, 0, 0]), (13, [4, 4, 4])), ((2, [5, 0, 0]), (13, [3, 3, 0])), ((1, [0, 0, 0]), (2, [5, 5, 0])), ((6, [13, 6, 0]), (1, [5, 5, 5])), ((5, [13, 0, 0]), (10, [6, 6, 6])),
    ((4, [13, 3, 0]), (11, [1, 0, 0])), ((4, [13, 0, 0]), (10, [3, 3, 3])), ((4, [0, 0, 0]), (5, [13, 13, 0])), ((0, [2, 2, 0]), (9, [1, 1, 0])), ((7, [15, 2, 0]), (10, [11, 0, 0])), ((7, [15, 0, 0]), (0, [2, 2, 2])),
    ((7, [7, 15, 0]), (6, [15, 15, 0])), ((7, [7, 0, 0]), (6, [15, 15, 15])), ((0, [9, 8, 0]), (6, [7, 7, 0])), ((1, [8, 11, 0]), (8, [11, 11, 0])), ((1, [8, 0, 0]), (8, [11, 11, 11])), ((0, [9, 0, 0]), (1, [8, 8, 0])),
    ((2, [13, 9, 0]), (0, [9, 9, 0])), ((2, [13, 0, 0]), (0, [9, 9, 9])), ((1, [0, 0, 0]), (2, [13, 13, 13])), ((1, [1, 14, 0]), (3, [7, 7, 7])), ((2, [14, 8, 0]), (1, [1, 14, 14])), ((2, [14, 0, 0]), (0, [8, 8, 8])),
    ((0, [1, 14, 0]), (1, [14, 14, 0])), ((0, [1, 0, 0]), (1, [14, 14, 14])), ((0, [0, 0, 0]), (1, [1, 1, 1]))
]

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