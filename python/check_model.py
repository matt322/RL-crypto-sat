#debugging instance generation yielding unsat instances
from instance_generation import Instance
from solver_interface import SolverController
import subprocess
import os

def check(model, cnf):
    satisfiedclauses = 0
    with open(cnf, 'r') as f:
        for i, clause in enumerate(f):
            if clause.startswith('c') or clause.startswith('p'):
                continue
            try:
                lits = list(map(int, clause.strip().split()))[:-1]
            except:
                print(clause, i)
            for lit in lits:
                if (int(model[abs(lit)-1]) > 0) == (lit > 0):
                    satisfiedclauses += 1
                    break
            else:
                print(f"model does not satisfy clause {i}: {clause.strip()}")
    print(f"model satisfied {satisfiedclauses} clauses")


i = Instance(seed=42, rounds=21)
solver = SolverController()
cnf = i.generate(p=0.17) #should be sat
check(cnf[1], cnf[0])
print(os.getcwd())
solver.start(cnf, 100000, args=["certified", "certified-output=cnf/proof.txt"])
while not solver.is_finished():
    solver.step()
os.chdir("..")
subprocess.run(f"drat-trim/drat-trim thesisrepo/{cnf[0]} thesisrepo/cnf/proof.txt", shell=True)

