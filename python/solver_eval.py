from solver_interface import SolverController
from util import GNNWrapper, get_gnn_config
from instance_generation import Instance
import multiprocessing as mp
import torch
import json
import os

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
        obs, _, done, _, _, time = solver.start(cnf, decision_period, timeout_secs=timeout, verb=0, reward_pow=0, args=["-no-adapt", "-luby", "-var-decay=0.999", "-max-var-decay=0.999"])
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
    LOGSPATH = "logs/es_logs/"
    params = [
        [   
            {"title":"eval_static_noadvance_a=0_refocus200k",
            "num_datapoints":100,
            "n_workers":16,
            "timeout":300,
            "heuristic":"refocus",
            "decision_period":200000,
            "hybrid_period":0},
            {"static":True, "embed_dim":0, "nlits":3968 * 2, "model_path":f"{LOGSPATH}static_noadvance_alpha=0_9/model.pt"}
        ],
        [   
            {"title":"eval_gnn_embed_advance_a=0_refocus200k",
            "num_datapoints":100,
            "n_workers":16,
            "timeout":300,
            "heuristic":"refocus",
            "decision_period":200000,
            "hybrid_period":0},
            {"static":False, "embed_dim":4, "nlits":3968 * 2, "model_path":f"{LOGSPATH}embed_advance_alpha=0_1/model.pt"}
        ],
        [   
            {"title":"eval_static_noadvance_a=0_refocus50k",
            "num_datapoints":100,
            "n_workers":16,
            "timeout":300,
            "heuristic":"refocus",
            "decision_period":50000,
            "hybrid_period":0},
            {"static":True, "embed_dim":0, "nlits":3968 * 2, "model_path":f"{LOGSPATH}static_noadvance_alpha=0_9/model.pt"}
        ],
        [   
            {"title":"eval_gnn_embed_advance_a=0_refocus50k",
            "num_datapoints":100,
            "n_workers":16,
            "timeout":300,
            "heuristic":"refocus",
            "decision_period":50000,
            "hybrid_period":0},
            {"static":False, "embed_dim":4, "nlits":3968 * 2, "model_path":f"{LOGSPATH}embed_advance_alpha=0_1/model.pt"}
        ]
        
    ]

    id = os.getenv("SLURM_ARRAY_TASK_ID")
   
    if id is not None:
        id = int(id)
        p = params[id]
        p[0]["logfile"] = f"logs/{p[0]['title']}.jsonl"
        with open(p[0]["logfile"], 'a') as f:
            f.write(json.dumps(p[0] | p[1]) + '\n')
        print(f"Running experiment {id}: {p[0]['title']}")
        model_path = p[1].pop("model_path")
        p[0].pop("title")
        cfg = get_gnn_config(**p[1])
        model = GNNWrapper(cfg)
        model.load_state_dict(torch.load(model_path))
        run_tests(model = model, **p[0])
    
   