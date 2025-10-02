from copy import deepcopy

DATA = [[1, 2, 3, 3], [1, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
Types = {}
branchlen = len(DATA[0])
branchamount = len(DATA)
for branch in DATA:
    for bird in branch:
        if bird in Types:
            Types[bird] += 1
        else:
            Types[bird] = 1

TargetState = []
for bird, amount in Types.items():
    if bird == 0:
        continue
    while True:
        TargetState.append([bird] * (amount % branchlen))
        amount -= branchlen
        if amount <= 0:
            #TargetState[-1] += ([0] * (-amount))
            break
TargetState += [[0] * branchlen] * (branchamount - len(TargetState))

IntermediateState = deepcopy(TargetState)
print(IntermediateState)
#Assemble Origin DATA
for branch in DATA:
    for ind, intermediate_branch in enumerate(IntermediateState):
        if intermediate_branch == branch[:len(intermediate_branch)]:
            selected_branch_ind = ind
            break
        if intermediate_branch[0] == 0:
            selected_branch_ind = ind
            break
    while branch != IntermediateState[selected_branch_ind]:
        if IntermediateState[selected_branch_ind][0] == 0:
            IntermediateState[selected_branch_ind] = [branch[0]]
            a = [item[0] for item in IntermediateState]
            ind = a.index(branch[0])
            IntermediateState[ind].pop()
            if not IntermediateState[ind]:
                IntermediateState[ind] = [0] * branchlen
        else:
            i = len(IntermediateState[selected_branch_ind])
            IntermediateState[selected_branch_ind].append(branch[i])
            a = [item[0] for item in IntermediateState]
            ind = a.index(branch[i])
            IntermediateState[ind].pop()
            if not IntermediateState[ind]:
                IntermediateState[ind] = [0] * branchlen
        print(IntermediateState)
        

