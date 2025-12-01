#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', nargs='?', default='input.txt')
args = parser.parse_args()
print("info: filename provided:", args.filename)

f = open(args.filename, "r")
CTRLSYMBOLS = ["/", "DATA", "ORDER"]
MAXBRNCHLEN = 26
MAXBRNCHCNT = 1000
bcnt = {
    'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0,
    'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0,
}
def validdata(x):
    beg = -1
    end = -2
    if "DATA" in x:
        beg = x.index("DATA")
    else:
        print("error: no data section beginning")
    if "/" in x:
        end = x.index("/")
    else:
        print("error: no data section ending")
    cnt = end - beg - 1
    if cnt > MAXBRNCHCNT:
        print(f'error: too many branches {cnt}')
    return beg, end, cnt

def removecomments(l):
    com = l.find('--')
    if com >= 0:
        l = l[:com]
    return l.rstrip()

def validbranch(b):
    lb = len(b)
    if lb == 0 or lb > MAXBRNCHLEN:
        print(f'error: length of branch is faulty ({lb})')
        return 2
    for e in b:
        le = len(e)
        if le == 2 and e == "==":
            continue
        if not e in bcnt.keys():
            print(f'error: {e} is not a valid bird')
            return 3
    return 0    

def validbranchlen(bl, cbl):
    if bl == cbl:
        return 0
    else: 
        print(f'error: len(branch) = {bl}, BRNCHLEN = {cbl}')
        return 4

def parseline(l):
    branch = l.rstrip().split(' ')
    v = validbranch(branch)
    if v != 0:
        print(f'error: branch {branch} is not valid')
        return v
    if branch[0] != "==":
        v = validbranchlen(len(branch), BRNCHLEN)
        if v != 0:
            print(f'error: branchlen of {branch} is not valid')
            return v
    o = []
    for bird in branch:
        if bird == "==":
            for _ in range(BRNCHLEN):
                o.append(0)
            break
        bcnt[bird] += 1
        o.append(int(ord(bird) - ord('A') + 1))
    out.append(o)
    return 0

def find_order_section(lines):
    if "ORDER" in lines:
        start = lines.index("ORDER")
    else:
        start = -1
    if start != -1 and "/" in lines[start:]:
        end = lines[start:].index("/")
    else:
        end = -1
    return start, end

def print_matrix(m):
    for i in range(len(m)):
        print(i+1, ':', *m[i])
    print()

def movin(DATA, order_section):
    print('info: start movin\'')
    print_matrix(DATA)
    err = 0
    for step in range(len(order_section)):
        f = int(order_section[step][0]) - 1
        t = int(order_section[step][1]) - 1
        b = order_section[step][2]
        print('step = {}, from = {}, to = {}, bird = {}'.format(step, f, t, b))
        
        if b == DATA[f][-1] and len(DATA[t]) < BRNCHLEN and (len(DATA[t]) == 0 or DATA[t][-1] == b):
            DATA[t].append(DATA[f].pop())
            print_matrix(DATA)
        else:
            print('error: step = {}, from = {}, to = {}, alpha = {}'.format(step, f, t, b))
            err = 1
            print_matrix(DATA)
            break
    print('info: end movin\'')
    return DATA, err
    

def countin(DATA, ORDER, LEN):
    N = LEN
    L = 0
    for b in DATA:
        if len(b) != 0 and all(map(lambda x: x == b[0], b)):
            L += 1
    K = len(ORDER)
    F = 100 * N * L - K
    return F

# void main(void)
err = 0
x = f.read().split('\n')
#print(x)
databeg, dataend, brnchcnt = validdata(x)
out = []
if databeg < 0 or dataend < 0 or (brnchcnt < 0 or brnchcnt > MAXBRNCHCNT):
    print("error: that is my end, there is no valid data section")
else:
    print('info: first stage start')
    linenum = 1
    l = removecomments(x[databeg+1]).rstrip()
    if validbranch(l.split(' ')) == 0:
        print(f'info: branch {linenum:2} [{l}] is valid')
        BRNCHLEN = len(l.split(' '))
        parseline(l)
        for el in x[databeg+2: dataend]:
            linenum += 1
            l = removecomments(el)
            err = parseline(l)
            if err != 0:
                break
            print(f'info: branch {linenum:2} [{l}] is valid')
            print('info: second stage start')
        for b in bcnt.keys():
            if bcnt[b] > 0:
                if bcnt[b] % BRNCHLEN != 0:
                    print(f'error: {b} count is {bcnt[b]}')
                    print(f'error: bird count is not proportional to {BRNCHLEN}')
                    err = 7
                    break
        if err == 7:
            pass
        elif err != 0:
            print(f'branch {linenum} error : exit with code {err}')
        else:
            print('success: all good here')
            of = open(args.filename + ".out", 'w')
            for s in out:
                of.write(' '.join(map(str, s)))
                of.write('\n')
            of.close()
            print(f'written: {args.filename + ".out"}')
    else:
        print(f'error: first branch is not valid, will not continue')
        err = 9

#x = f.read().split('\n')
#print(x)
if err:
    print('info: movin is pointless, data is bad')
else:
    obeg, oend = find_order_section(x)

    oerr = 0
    if obeg >= 0 and oend >= 0:
        ORDER = x[obeg+1:obeg+oend]
        oerr = 0
        print("info: order_section found")
        print("info: obeg = {}, oend = {}".format(obeg, oend))
    else:
        print("warning: order_section not found")
        print("warning: obeg = {}, oend = {}".format(obeg, oend))
        oerr = 1

    if oerr:
        pass
    else:
        ORDER = [tuple(i.split()) for i in ORDER]
        DATA = x[databeg+1:databeg+dataend]
        for i in range(len(DATA)):
            DATA[i] = DATA[i].split()
            if DATA[i][0] == '==':
                DATA[i].clear()
        #print(DATA)
        finalDATA, merr = movin(DATA, ORDER)
        if merr:
            print('error: movin ended in error so no Target Function')
        else:
            print(f'Target function = {countin(finalDATA, ORDER, BRNCHLEN)}')
f.close()
