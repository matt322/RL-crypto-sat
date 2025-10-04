from solver_interface import SolverController
from instance_generation import Instance
import json


def average_solve_time(logfile, rounds=21, num_instances=10, timeout=1000, decisions_per_callback=0):
    instance = Instance(rounds=rounds, seed=41)
    solver = SolverController()

    for i in range(num_instances):
        cnf = instance.generate()
        solver.start(cnf, decisions_per_callback=decisions_per_callback, timeout_secs=timeout, verb=0)
        time_taken = 0
        model = None
        while not solver.is_finished():
            _, _, _, model, time_taken = solver.step(activity_scores=solver.zero_scores())
        with open(logfile, "a") as f:
            f.write(json.dumps({"rounds": rounds, "decisions/callback":decisions_per_callback, "timed_out":model is None, "time": time_taken if model else timeout}) + "\n")
        print(f"Instance {i+1} solved: {not model is None}, time taken: {time_taken if model else timeout}s")
        
        

if __name__ == "__main__":
    average_solve_time("logs/avg_solvetime_log.jsonl", rounds=21, num_instances=5, timeout=20, decisions_per_callback=0)