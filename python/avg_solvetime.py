from solver_interface import SolverController
from instance_generation import Instance
import json
import datetime


def average_solve_time(logfile, rounds=21, num_instances=10, timeout=1000, decisions_per_callback=0, free_outputs=0):
    instance = Instance(rounds=rounds)
    solver = SolverController()

    for i in range(num_instances):
        cnf = instance.generate(free_outputs, seed=i)
        _, _, _, model, time_taken = solver.start(cnf, decisions_per_callback=decisions_per_callback, timeout_secs=timeout)
        while not solver.is_finished():
            _, _, _, _, model, time_taken = solver.step(activity_scores=solver.zero_scores())
        with open(logfile, "a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().strftime("%m/%d, %H:%M:%S"), "rounds": rounds, "decisions/callback":decisions_per_callback, "free_outputs":free_outputs, "timed_out":model is None, "time": time_taken if model else timeout}) + "\n")
        print(f"Instance {i+1} solved: {not model is None}, time taken: {time_taken if model else timeout}s")
        
        

if __name__ == "__main__":
    #average_solve_time("logs/avg_solvetime_log_1.jsonl", rounds=21, num_instances=100, timeout=200, decisions_per_callback=0, free_outputs=0)
    average_solve_time("logs/avg_solvetime_log_1.jsonl", rounds=21, num_instances=100, timeout=200, decisions_per_callback=0, free_outputs=64)
    #average_solve_time("logs/avg_solvetime_log_1.jsonl", rounds=21, num_instances=100, timeout=200, decisions_per_callback=0, free_outputs=32)
    #average_solve_time("logs/avg_solvetime_log_1.jsonl", rounds=21, num_instances=100, timeout=200, decisions_per_callback=0, free_outputs=64)
    #average_solve_time("logs/avg_solvetime_log_1.jsonl", rounds=21, num_instances=100, timeout=200, decisions_per_callback=0, free_outputs=128)
    #average_solve_time("logs/avg_solvetime_log_1.jsonl", rounds=21, num_instances=100, timeout=200, decisions_per_callback=0, free_outputs=150)

