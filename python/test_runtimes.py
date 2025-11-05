from solver_interface import SolverController
from instance_generation import Instance
import numpy as np
import json
from solver_eval import generate_dataset



if __name__ == "__main__":
    data = generate_dataset()
    vanilla_solver_log = "logs/es_logs/vanilla_runtimes.jsonl"
    es_solver_log = "logs/es_logs/es_initialization_runtimes.jsonl"
    es_scores = list(json.load(open("logs/0/es_model_0.jsonl"))["model"])
    solver = SolverController()
    for i, cnf in enumerate(data):        
        solver.start(cnf, 100000000, timeout_secs=200, verb=0)
        time = solver.step([f"{i} {v}" for i,v in enumerate(es_scores)])[-1]
        print(f"Instance {i} ES time: {time}")
        with open(es_solver_log, "a") as f:
            f.write(json.dumps({"instance": i, "time": time}) + "\n")

        solver.start(cnf, 100000000, timeout_secs=200, verb=0)
        time = solver.step(solver.zero_scores())[-1]
        print(f"Instance {i} Vanilla time: {time}")
        with open(vanilla_solver_log, "a") as f:
            f.write(json.dumps({"instance": i, "time": time}) + "\n")
