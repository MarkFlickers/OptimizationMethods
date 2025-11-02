f = open("input.txt", "r")
CTRLSYMBOLS = ["/", "DATA", "ORDER"]
MAXBRNCHLEN = 26
MAXBRNCHCNT = 1000
bcnt = {
    'A': 0,
    'B': 0,
    'C': 0,
    'D': 0,
    'E': 0,
    'F': 0,
    'G': 0,
    'H': 0,
    'I': 0,
    'J': 0,
    'K': 0,
    'L': 0,
    'M': 0,
    'N': 0,
    'O': 0,
    'P': 0,
    'Q': 0,
    'R': 0,
    'S': 0,
    'T': 0,
    'U': 0,
    'W': 0,
    'X': 0,
    'Y': 0,
    'Z': 0,
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
    print(f'info: branch {branch} is valid')
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

# void main(void)
err = 0
x = f.read().split('\n')
databeg, dataend, brnchcnt = validdata(x)
out = []
if databeg < 0 or dataend < 0 or (brnchcnt < 0 or brnchcnt > MAXBRNCHCNT):
    print("error: that is my end, there is no valid data section")
else:
    print('info: first stage start')
    l = removecomments(x[databeg+1]).rstrip()
    if validbranch(l.split(' ')) == 0:
        print(f'info: branch {l} is valid')
        BRNCHLEN = len(l.split(' '))
        parseline(l)
        for el in x[databeg+2: dataend]:
            l = removecomments(el)
            err = parseline(l)
            if err != 0:
                break
        print('info: second stage start')
        for b in bcnt.keys():
            if bcnt[b] > 0:
                if bcnt[b] % BRNCHLEN != 0:
                    print(f'error: {b} count is {bcnt[b]}')
                    print(f'error: bird count is not proportional to {BRNCHLEN}')
                    err = 7
                    break
        if err != 0:
            print(f'error: exit with code {err}')
        else:
            print('success: all good here is some data')
            print(bcnt)
            print(out)
    else:
        print(f'error: first branch is not valid, can not continue')

f.close()
of = open("output.txt", 'w')
of.write(str(out))
of.close()
