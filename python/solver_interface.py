import subprocess
import os


#pass activity scores to file, run solver for n conflicts/decisions, write results to file
#model if solved, learnts and reward if not


class Solver:
    def __init__(self, model=None, solver_path="glucose_modified/simp/glucose"):
        self.solver_path = solver_path

    def solve(self, cnf_path, variable_embeddings=None, options=None):
        return subprocess.run(f"{self.solver_path} {cnf_path} -model", text=True, shell=True, stdout=subprocess.PIPE).stdout.split(' ')
    
    def 