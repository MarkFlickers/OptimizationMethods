from datetime import datetime

CTRLSYMBOLS = ["/", "DATA", "ORDER"]
MAXBRNCHLEN = 1000
MAXBRNCHCNT = 1000

class BranchProcessor:
    def __init__(self, lines):
        self.lines = lines
        self.bcnt = {chr(ord('A')+i):0 for i in range(26)}
        self.out = []
        self.BRNCHLEN = 0

    def validdata(self):
        beg = -1
        end = -2
        if "DATA" in self.lines:
            beg = self.lines.index("DATA")
        else:
            print("error: no data section beginning")
        if "/" in self.lines:
            end = self.lines.index("/")
        else:
            print("error: no data section ending")
        cnt = end - beg - 1
        if cnt > MAXBRNCHCNT:
            print(f'error: too many branches {cnt}')
        return beg, end, cnt

    @staticmethod
    def removecomments(l):
        com = l.find('--')
        if com >= 0:
            l = l[:com]
        return l.rstrip()

    def validbranch(self, b):
        lb = len(b)
        if lb == 0 or lb > MAXBRNCHLEN:
            print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - length of branch is faulty ({lb})')
            return 2
        for e in b:
            le = len(e)
            if le == 2 and e == "==":
                continue
            if not e in self.bcnt.keys():
                print(f'error: {e} is not a valid bird')
                return 3
        return 0

    def validbranchlen(self, bl, cbl):
        if bl == cbl:
            return 0
        else: 
            print(f'error: len(branch) = {bl}, BRNCHLEN = {cbl}')
            return 4

    def parseline(self, l):
        branch = l.rstrip().split(' ')
        v = self.validbranch(branch)
        if v != 0:
            print(f'error: branch {branch} is not valid')
            return v
        if branch[0] != "==":
            v = self.validbranchlen(len(branch), self.BRNCHLEN)
            if v != 0:
                print(f'error: branchlen of {branch} is not valid')
                return v
        o = []
        for bird in branch:
            if bird == "==":
                for _ in range(self.BRNCHLEN):
                    o.append(0)
                break
            self.bcnt[bird] += 1
            o.append(int(ord(bird) - ord('A') + 1))
        self.out.append(o)
        return 0

    def process_branches(self, databeg, dataend, verbose: bool = True):
        linenum = 1
        err = 0
        l = self.removecomments(self.lines[databeg+1]).rstrip()
        if self.validbranch(l.split(' ')) == 0:
            if verbose:
                print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - branch [{linenum:02d}] - ({l}) is valid')
            self.BRNCHLEN = len(l.split(' '))
            self.parseline(l)
            for el in self.lines[databeg+2: dataend]:
                linenum += 1
                l = self.removecomments(el)
                err = self.parseline(l)
                if err != 0:
                    break
                if verbose:
                    print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - branch [{linenum:02d}] - ({l}) is valid')
            # Проверка на кратность
            for b in self.bcnt.keys():
                if self.bcnt[b] > 0 and self.bcnt[b] % self.BRNCHLEN != 0:
                    print(f'error: {b} count is {self.bcnt[b]}')
                    print(f'error: bird count is not proportional to {self.BRNCHLEN}')
                    err = 7
                    break
        else:
            print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - first branch is not valid, will not continue')
            err = 9
        return err, self.out, self.BRNCHLEN, self.bcnt

class OrderProcessor:
    def __init__(self, lines, BRNCHLEN):
        self.lines = lines
        self.BRNCHLEN = BRNCHLEN

    def find_order_section(self):
        start = -1
        end = -1
        if "ORDER" in self.lines:
            start = self.lines.index("ORDER")
        if start != -1 and "/" in self.lines[start:]:
            end = self.lines[start:].index("/")  # относительный индекс
        return start, end

    @staticmethod
    def print_matrix(m):
        for i in range(len(m)):
            print(i+1, ':', *m[i])
        print()

    def movin(self, DATA, order_section, verbose: bool = False):
        if verbose:
            print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - start movin')
        if verbose:
            self.print_matrix(DATA)
        err = 0
        for step in range(len(order_section)):
            f = int(order_section[step][0]) - 1
            t = int(order_section[step][1]) - 1
            b = order_section[step][2]
            if verbose:
                print('step = {}, from = {}, to = {}, bird = {}'.format(step + 1, f + 1, t + 1, b))
            if b == DATA[f][-1] and len(DATA[t]) < self.BRNCHLEN and (len(DATA[t]) == 0 or DATA[t][-1] == b):
                DATA[t].append(DATA[f].pop())
                if verbose:
                    self.print_matrix(DATA)
            else:
                if verbose:
                    print('error: step = {}, from = {}, to = {}, alpha = {}'.format(step + 1, f + 1, t + 1, b))
                err = 1
                if verbose:
                    self.print_matrix(DATA)
                break
        if verbose:
            print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - end movin')
        return DATA, err

    def countin(self, DATA, ORDER):
        N = self.BRNCHLEN
        L = 0
        for b in DATA:
            if len(b) != 0 and all(map(lambda x: x == b[0], b)):
                L += 1
        K = len(ORDER)
        F = 100 * N * L - K
        return F

class BranchIntegrity:
    """
    Branches integrity checker Interface
    
    args
    file: pathlike
    """
    def __init__(self, filename, verbose):
        self.filename = filename
        self.verbose = verbose
        self.lines = []
        self.branches = None
        self.order = None
        self.BRNCHLEN = 0
        self.err = 0

    def run(self):
        with open(self.filename, "r") as f:
            self.lines = f.read().split('\n')

        bp = BranchProcessor(self.lines)
        databeg, dataend, brnchcnt = bp.validdata()

        if databeg < 0 or dataend < 0 or brnchcnt < 0 or brnchcnt > MAXBRNCHCNT:
            print("error: that is my end, there is no valid data section")
            self.err = 1
            return

        self.err, self.branches, self.BRNCHLEN, _ = bp.process_branches(databeg, dataend, False)

        if self.err:
            print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - movin is pointless, data is bad')
            return

        op = OrderProcessor(self.lines, self.BRNCHLEN)
        obeg, oend = op.find_order_section()

        if obeg >= 0 and oend >= 0:
            ORDER = self.lines[obeg+1:obeg+oend]
            ORDER = [tuple(i.split()) for i in ORDER]
            DATA = self.lines[databeg+1:databeg+dataend]
            for i in range(len(DATA)):
                DATA[i] = DATA[i].split()
                if DATA[i][0] == '==':
                    DATA[i].clear()
            finalDATA, merr = op.movin(DATA, ORDER, self.verbose)
            if merr:
                print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - movin ended in error so no Target Function')
            else:
                print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Target function = {op.countin(finalDATA, ORDER)}')
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - order_section not found")
