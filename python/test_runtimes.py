from solver_interface import SolverController
from instance_generation import Instance
import numpy as np
import json
from solver_eval import generate_dataset



if __name__ == "__main__":
    data = generate_dataset()
    vanilla_solver_log = "logs/es_logs/vanilla_runtimes.jsonl"
    es_solver_log = "logs/es_logs/es_initialization_runtimes_3.jsonl"
    es_scores = list(json.load(open("logs/logs/es_logs/1/es_model_0.jsonl"))["model"])
    scores = [f"{i} {v * 1e4}" for i,v in enumerate(es_scores)]
    solver = SolverController()
    for i, cnf in enumerate(data):
        if i < 10:
            continue     
        time = solver.start(cnf, 5000000, timeout_secs=200, verb=0)[-1]
        while not solver.is_finished():
            time += solver.step(scores)[-1]
        time = min(time, 200)
        print(f"Instance {i} ES time: {time}")
        with open(es_solver_log, "a") as f:
            f.write(json.dumps({"instance": i, "time": time}) + "\n")

        time = solver.start(cnf, 0, timeout_secs=200, verb=0)[-1]
        time = min(time, 200)
        print(f"Instance {i} Vanilla time: {time}")
        with open(vanilla_solver_log, "a") as f:
            f.write(json.dumps({"instance": i, "time": time}) + "\n")
