from solver_environment import SolverEnv
from instance_generation import Instance
from viz import var_viz
from es_optimizer import OpenAIESOptimizer
import os
import json
import torch
from torch import nn


def make_env():
    return SolverEnv(rounds=21, decisions_per_callback=50000, free_outputs=0, single_inst=False, verb=0, normalize_actions = False)
   
def fitness_fn(env, model, params, seed):
    obs, info = env.reset(seed)
    if obs is None:
        return 0
    done = False
    reward = env.step(params)[1]
    return reward
    
class ConstantVector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vec = nn.Parameter(torch.zeros(dim))  

    def forward(self, x):
        return self.vec

if __name__ == "__main__":
    print("starting job")

    LOGDIR = "logs/es_logs/"
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
        i = 0
    else:
        i = len(os.listdir(LOGDIR))
    LOGDIR = f"logs/es_logs/{i}/"
    os.makedirs(LOGDIR, exist_ok=True)
    LOGPATH = LOGDIR + f"es_log_{len(os.listdir(LOGDIR))}.jsonl"
    MODELPATH = LOGDIR + f"es_model_{len(os.listdir(LOGDIR))}.jsonl"
    FIGPATH = LOGDIR + f"es_fig_{len(os.listdir(LOGDIR))}/"
    os.makedirs(FIGPATH, exist_ok=True)


    e = make_env()
    optimizer = OpenAIESOptimizer(ConstantVector(e.action_space.shape[0]), make_env_fn=make_env, fitness_fn=fitness_fn, sigma=0.5, lr=0.005, popsize=128, n_workers=8)
    info = {}
    for i in range(4000):
        info = optimizer.step()   
        if i % 25 == 0:
            pass
            #var_viz("cnf/sha1_21round.cnf", , FIGPATH + f"es_model_gen_{i}.png", title="")
            #var_viz("cnf/sha1_21round.cnf", info["grad"], FIGPATH + f"es_grad_gen_{i}.png", title="")
           # with open(MODELPATH, "w") as f:
            #    f.write(json.dumps({"gen": info["gen"],
            #                     "model": info["model"].tolist(),
             #                    }) + "\n")
        with open(LOGPATH, "a") as f:
            f.write(json.dumps({"gen": info["gen"],
                             "fitness_mean": info["fitness_mean"],
                             "fitness_std": info["fitness_std"],
                             "fitness_max": info["fitness_max"],
                             }) + "\n")
        print(
                f"Gen {info['gen']:03d} | mean_fitness={info['fitness_mean']:.5e} | std_fitness={info['fitness_std']:.5e}" +
                f"| max_fitness={info['fitness_max']:.5e}",
                flush=True
            )
        
    print(info["model"])
        

        
    

