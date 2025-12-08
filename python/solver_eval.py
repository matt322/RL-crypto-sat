from solver_interface import SolverController
from util import GNNWrapper, get_gnn_config
from instance_generation import Instance
import multiprocessing as mp
import numpy as np
import torch
import json

_worker_instance = None

def worker_init():
    global _worker_instance
    _worker_instance = Instance()

def run_test(seed, model = None, timeout=200, heuristic="vanilla", decision_period=0, hybrid_period=0):
        cnf = _worker_instance.generate(seed=seed, guarantee_soln=True)
        assert heuristic in ["vanilla", "refocus", "hybrid"], "Invalid heuristic type"
        assert model is not None or heuristic == "vanilla"
        if model is None:
            modelcall = lambda x: []
        else:
            if isinstance(model, GNNWrapper):
                modelcall = lambda x: model.forward_direct_on_obs(x)
            else:
                modelcall = lambda x: model #assume static model

        solver = SolverController()
        obs, _, done, _, _, time = solver.start(cnf, decision_period, timeout_secs=timeout, verb=0, reward_pow=0)
        iters = 0
        llr_list = []
        while not solver.is_finished():
            obs, llr, done, _, _, steptime = solver.step([f"{i} {v * 1e4}" for i,v in enumerate(modelcall(obs))], heuristic_type=heuristic if heuristic != "vanilla" else "refocus", get_csr=heuristic == "refocus", static_score_decisions=hybrid_period)
            time += steptime
            llr_list.append(llr)
            iters += 1
        time = min(time, timeout)
        print(f"Instance {seed} time: {time} Iters: {iters}")
        return json.dumps({"instance": seed, "time": time, "llr":llr_list})

def run_tests(n_workers, num_datapoints, logfile, model = None, timeout=200, heuristic="vanilla", decision_period=20000, hybrid_period=0):
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_workers, initializer=worker_init)
    seeds = list(range(100, 100+num_datapoints))
    args = [(i, model, timeout, heuristic, decision_period, hybrid_period) for i in seeds]
    results = pool.starmap(run_test, args)
    with open(logfile, 'a') as f:
        for result in results:
            f.write(result + '\n')
            f.flush()
    pool.close()
    pool.join()

if __name__ == "__main__":
    static_scores_a2 = np.array(json.load(open("logs/es_logs/1/es_model_0.jsonl"))["model"]).squeeze()
    #run_tests(16, 32, "logs/test_vanilla_eval.jsonl", model=None, timeout=200, heuristic="vanilla", decision_period=50000, hybrid_period=0)
    run_tests(16, 100, "logs/refocus_eval_static_a=2.jsonl", model=static_scores_a2, timeout=200, heuristic="refocus", decision_period=50000, hybrid_period=0)
    
   