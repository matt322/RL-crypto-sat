from solver_interface import SolverController
from instance_generation import Instance
import numpy as np
import json




if __name__ == "__main__":
    inst = Instance(rounds=21)
    solver = SolverController()
    cnf = inst.generate(seed=43)
    es_scores = list(json.load(open("logs/0/es_model_0.jsonl"))["model"])
    print(es_scores)
    
    solver.start(cnf, 100000000, timeout_secs=40, verb=0)
    print("started solver ES scores")
    print(solver.step([f"{i} {v}" for i,v in enumerate(es_scores)])[-1])

    solver.start(cnf, 100000000, timeout_secs=40, verb=0)
    print("started solver default")
    print(solver.step(solver.zero_scores())[-1])