from solver_interface import SolverController
from instance_generation import Instance
import numpy as np
import json


if __name__ == "__main__":
    i = Instance(rounds=21)
    cnf = i.generate(seed=104, guarantee_soln=True)
    vanilla_solver_log = "logs/es_logs/vanilla_runtimes.jsonl"
    es_solver_log = "logs/es_logs/refocus_es_initialized_runtimes_0.jsonl"
    es_scores = list(json.load(open("logs/logs/es_logs/1/es_model_0.jsonl"))["model"])
    scores = [f"{i} {v * 1e4}" for i,v in enumerate(es_scores)]
    solver = SolverController()
    # if i < 10:
    #     continue     
    time = solver.start(cnf, 50000, timeout_secs=200, verb=2)[-1]
    iters = 0
    while not solver.is_finished():
        time += solver.step([], heuristic_type="refocus", get_csr=False)[-1]
        iters += 1
    time = min(time, 200)
    print(f"Instance {i} ES time: {time} Iters: {iters}")
    with open(es_solver_log, "a") as f:
        f.write(json.dumps({"instance": i, "time": time}) + "\n")

        # time = solver.start(cnf, 0, timeout_secs=200, verb=0)[-1]
        # time = min(time, 200)
        # print(f"Instance {i} Vanilla time: {time}")
        # with open(vanilla_solver_log, "a") as f:
        #     f.write(json.dumps({"instance": i, "time": time}) + "\n")
