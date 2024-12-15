import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime


class SetCoverProblem:

    def __init__(self, filename, prob_name="", best=0.0):
        self.prob_name = prob_name
        self.best = best
        f = open(filename, 'r')
        lines = [int(x) for line in f.readlines() for x in line.strip().split(' ')]
        f.close()

        self.m, self.n = lines[0], lines[1]
        self.costs = np.zeros(self.n)

        for i, v in enumerate(lines[2:self.n+2]):
            self.costs[i] = v
        
        self.U = np.zeros((self.m, self.n))
        idx = self.n + 2

        for i in range(self.m):
            x = lines[idx]
            indices = lines[idx+1:idx+x+1]
            for j in indices:
                self.U[i, j-1] = 1
            idx = idx + x + 1
    
    
    def f_obj(self, sol):
        return np.dot(sol, self.costs).item()
    

    def fitness(self, sol):
        return -self.f_obj(sol)
    

    def random_sol(self, p0=4/5):
        return np.random.choice([0, 1], size=self.n, p=[p0, 1-p0])


    def sol_submatrix(self, sol):
        return self.U[:, [sol[i] == 1 for i in range(len(sol))]]
    

    def make_correct(self, sol):
        if self.check_sol(sol):
            return sol

        alpha = []
        w = []
        V = []
        S = np.where(sol == 1)[0].tolist()
        for i in range(self.m):
            alpha.append(np.where(self.U[i, :] == 1)[0].tolist())
            w.append(len(set(S).intersection(set(alpha[i]))))
            if w[i] == 0:
                V.append(i)
        beta = []
        for j in range(self.n):
            beta.append(np.where(self.U[:, j] == 1)[0].tolist())
        
        for i in V:
            m_best = float("inf")
            best_col = 0
            for col in alpha[i]:
                m = self.costs[col] / len(set(V).intersection(set(beta[col])))
                if m < m_best:
                    m_best = m
                    best_col = col
            sol[best_col] = 1
            for ii in beta[best_col]:
                w[ii] += 1
                try:
                    if i != ii:
                        V.remove(ii)
                except:
                    pass
        
        for j in range(self.n - 1, -1, -1):
            to_del = True
            for i in beta[j]:
                if w[i] < 2:
                    to_del = False
                    break
            if to_del:
                sol[j] = 0
                for i in beta[j]:
                    w[i] -= 1
        
        return sol
    

    def random_correct_sol(self):
        sol = self.random_sol()
        return self.make_correct(sol)
    

    def check_sol(self, sol):
        A = self.sol_submatrix(sol)
        return np.all(np.any(A, axis=1)).item()
        

class GeneticAlg:

    def __init__(self, prob, N, M, mf=10, mg=200, mc=1.5):
        self.prob = prob
        self.N = N
        self.M = M
        self.mf = mf
        self.mg = mg
        self.mc = mc
    
    def crossover(self, p1, p2):
        c = np.zeros(self.prob.n)
        for i in range(self.prob.n):
            if p1[i] == p2[i]:
                c[i] = p1[i]
            else:
                v = self.prob.f_obj(p2) / (self.prob.f_obj(p1) + self.prob.f_obj(p2))
                if np.random.rand() < v:
                    c[i] = p1[i]
                else:
                    c[i] = p2[i]

        return c
    

    def mutate(self, s, t=0):
        e = float(-4 * self.mg * (t - self.mc) / self.mf)
        v = np.ceil( float(self.mf) / ( 1.0 + np.exp(e) ) )

        to_change = np.random.choice([i for i in range(len(s))], size=int(v))

        for i in to_change:
            s[i] = int(not s[i])

        return s
    

    def generate_initial_pop(self):
        pop = np.zeros((self.N, self.prob.n))
        for i in range(self.N):
            p = self.prob.random_correct_sol()
            while p.tolist() in pop.tolist():
                p = self.prob.random_correct_sol()
            pop[i] = p 
        return pop
    

    def binary_tournament(self, pop):
        ns = np.random.choice([x for x in range(len(pop))], size=4, replace=False)

        a1, a2 = (pop[ns[0]], pop[ns[1]])
        b1, b2 = (pop[ns[2]], pop[ns[3]])

        # print(f'a1: {self.prob.f_obj(a1)} - a2: {self.prob.f_obj(a2)} -- b1: {self.prob.f_obj(b1)} -- b2: {self.prob.f_obj(b2)}')

        best1 = a1 if self.prob.f_obj(a1) < self.prob.f_obj(a2) else a2
        best2 = b1 if self.prob.f_obj(b1) < self.prob.f_obj(b2) else b2
        
        return (best1, best2)
    

    def get_best(self, s1, s2):
        f1 = self.prob.f_obj(s1)
        f2 = self.prob.f_obj(s2)

        if f1 < f2:
            return (s1, f1)
        else:
            return (s2, f2)
    

    def create_offspring(self, pop, t=0):
        p1, p2 = self.binary_tournament(pop)
        c = self.crossover(p1, p2)
        # print(f'dopo crossover: {self.prob.f_obj(c)}')
        c = self.mutate(c, t)
        # print(f'dopo mutazione: {self.prob.f_obj(c)}')
        c = self.prob.make_correct(c)
        # print(f'dopo correzione: {self.prob.f_obj(c)}')
        return c
    

    def get_above_avg_sol(self, pop):
        pop_avg = 0

        for s in pop:
            pop_avg += self.prob.f_obj(s)
        
        pop_avg /= len(pop)

        n = np.random.choice([x for x in range(len(pop))])
        while self.prob.f_obj(pop[n]) < pop_avg:
            n = np.random.choice([x for x in range(len(pop))])

        return n

    
    def solve(self, out_folder=""):
        pop = self.generate_initial_pop()
        s_best = pop[0]
        f_best = self.prob.f_obj(s_best)
        all_best = []

        prob_name = self.prob.prob_name
        best = self.prob.best

        if out_folder != "":
            dt = datetime.now().strftime("%d-%m-%y_%H-%M")
            fold = os.path.join(out_folder, f'{prob_name}')
            os.makedirs(fold, exist_ok=True)
            pth = os.path.join(fold, f"{dt}_log.txt")
            log_file = open(pth, 'w')
            log_file.write(f"problem size (n): {self.prob.n} - set size (m): {self.prob.m} - population size (N): {self.N} - loops (M): {self.M} - mf: {self.mf} - mc: {self.mc} - mg: {self.mg}\n\n")

        for s in pop:
            s_best, f_best = self.get_best(s_best, s)
        
        print(f'best: {f_best}')
        
        for t in tqdm(range(self.M)):
            c = self.create_offspring(pop, t)
            while c.tolist() in pop.tolist():
                c = self.create_offspring(pop, t)
            
            n = self.get_above_avg_sol(pop)
            pop[n] = c

            # print(f'{self.prob.f_obj(c)}')

            f_bp = f_best

            s_best, f_best = self.get_best(s_best, c)

            if out_folder != "":
                if f_bp != f_best:
                    log_file.write(f"{t}: {f_bp} --> {f_best}\n")

            all_best.append(f_best)
        
        plt.plot(all_best)
        
        if out_folder != "":
            plt.savefig(os.path.join(fold, f'{dt}_img.png'))
            if best > 0.0:
                log_file.write(f"\n\n best found: {f_best} - best: {best} - ratio: {(f_best - best) / best}")
            log_file.close()

        return (s_best, f_best)


## ================== TEST ===========================##

FILE41 = "problems/scp41.txt"
FILE42 = "problems/scp42.txt"
FILE43 = "problems/scp43.txt"
FILE44 = "problems/scp44.txt"
FILE45 = "problems/scp45.txt"
FILE46 = "problems/scp46.txt"
FILE47 = "problems/scp47.txt"
FILE48 = "problems/scp48.txt"
FILE49 = "problems/scp49.txt"
FILE410 = "problems/scp410.txt"

FILES = {
    "4.1": FILE41,
    "4.2": FILE42,
    "4.3": FILE43,
    "4.4": FILE44,
    "4.5": FILE45,
    "4.6": FILE46,
    "4.7": FILE47,
    "4.8": FILE48,
    "4.9": FILE49,
    "4.10": FILE410,
}

BEST = {
    "4.1": 429.0,
    "4.2": 512.0,
    "4.3": 516.0,
    "4.4": 494.0,
    "4.5": 512.0,
    "4.6": 560.0,
    "4.7": 430.0,
    "4.8": 492.0,
    "4.9": 641.0,
    "4.10": 514.0,
}

NLOOPS = 10

for (k, v) in FILES.items():
    print(f'problem {k}: ')
    problem = SetCoverProblem(v, prob_name=f'{k}', best=BEST[k])
    ga = GeneticAlg(problem, 100, 3000, mc=2000, mf=20, mg=2.2)
    s = 0.0
    for _ in range(NLOOPS):
        _, f = ga.solve(out_folder="logs/")
        s += f
    s /= NLOOPS
    f_best = BEST[k]
    ratio = (s - f_best) / f_best
    print(f'problem: {k} - {s} / {f_best} -- {ratio}')
    print('\n')

