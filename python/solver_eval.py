from solver_interface import SolverController
from solver_environment import SolverEnv
from instance_generation import Instance
import datetime
import torch
import json
import viz



def generate_dataset():
    i = Instance()
    for seed in range(100, 150):
        yield i.generate(seed=seed, guarantee_soln=True) #to keep i alive so the instance file doesnt get deleted

def evaluate_model(model, data, log, decisions_per_callback=500000):
    solver = SolverController()
    for i, cnf in enumerate(data):
        obs, _, _, m, time_taken = solver.start(cnf, decisions_per_callback=decisions_per_callback, timeout_secs=200) 
        while not solver.is_finished():
            scores = model(obs)
            obs, _, _, m, time_taken  = solver.step([f"{i} {score}" for i, score in enumerate(scores)])
           
        with open(log, "a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().strftime("%m/%d, %H:%M:%S"), "timed_out":m is None, "time": time_taken if m else timeout}) + "\n")
        print(f"Instance {i+1} solved: {not m is None}, time taken: {time_taken if m else timeout}s")
    return


if __name__ == "__main__":
    data = generate_dataset()
    model = torch.load("logs/ppo_branching_heuristic_model.pt")
    solver = SolverController()
    for i, cnf in enumerate(data):
        solver.start(cnf, 500000, timeout_secs=200, verb=1)
        done = False
        while not done:
            _, _, done, _, _, _ = solver.step([score.item() for score in scores])
        print(f"Finished instance {i}")