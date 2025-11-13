import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from solver_environment import SolverEnv
from instance_generation import Instance
from viz import var_viz
from es_optimizer import OpenAIESOptimizer
from util import GNNWrapper, construct_sparse_tensor

from torch.func import functional_call
import numpy as np
import json
import torch
import yappi


def make_env():
    return SolverEnv(
        rounds=21, 
        decisions_per_callback=20000, 
        free_outputs=0, 
        simplify_graph=True, 
        single_inst=False, 
        verb=0, 
        guarantee_soln=False, 
        normalize_actions = False,
        filter_scores=True,
    )
   
def fitness_fn(env, model, params, seed):
    obs, info = env.reset(seed)
    if obs is None:
        return 0
    
    param_dict = {}
    idx = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        param_dict[name] = params[idx:idx+numel].view_as(p)
        idx += numel

    pred = functional_call(model, param_dict, construct_sparse_tensor(obs))[0].squeeze()
    reward = env.step(pred)[1]
    return reward

def fitness_fn_stepforward(env, model, params, seed):
    env.reset(seed)
    
    np.random.seed(seed)
    fast_forward = np.random.geometric(p=1/100000)

    obs, info = env.advance(fast_forward)
    if obs is None:
        return 0, 0, fast_forward
    param_dict = {}
    idx = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        param_dict[name] = params[idx:idx+numel].view_as(p)
        idx += numel

    pred = functional_call(model, param_dict, construct_sparse_tensor(obs))[0].squeeze()
    reward = env.step(pred)[1]
    if reward == 1:
        return 0.01
    return reward

def embedding_variabletime_experiment():
    e = make_env()
    example_obs = e.reset()[0]
    nlits = example_obs["nlits"]
    example_obs = construct_sparse_tensor(example_obs)

    config = {
            "clause_dim":32,
            "lit_dim":16,
            "n_hops":2,
            "n_layers_C_update":3,
            "n_layers_L_update":3,
            "n_layers_score":1,
            "use_embeddings": True,
            "embed_dim":4,
            "nlits":nlits,
            "discrete":False,
            "normalize":False,
            "activation":"relu"
        }

    model = GNNWrapper(config)
    model.latent_dim_pi = nlits // 2
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = OpenAIESOptimizer(model, make_env_fn=make_env, fitness_fn=fitness_fn_stepforward, sigma=0.05, lr=0.001, popsize=512, n_workers=32)
    #yappi.set_clock_type("wall")
    #yappi.clear_stats()
    #yappi.start()
    

    info = {}
    for i in range(4000):
        info = optimizer.step()   
        if i % 50 == 0:
            pred = model(example_obs)[0]
            var_viz("cnf/sha1_21round.cnf", pred.detach().cpu().numpy().squeeze(), path=FIGPATH + f"es_nn_model_gen_{i}.png", title="")
            #var_viz("cnf/sha1_21round.cnf", info["grad"], FIGPATH + f"es_grad_gen_{i}.png", title="")
            torch.save(model.state_dict(), MODELPATH)
                
                
        with open(LOGPATH, "a") as f:
            f.write(json.dumps({"gen": info["gen"],
                             "fitness_mean": info["fitness_mean"],
                             "fitness_std": info["fitness_std"],
                             "fitness_max": info["fitness_max"],
                             "fast_forward": info["ff"],
                             }) + "\n")
        print(
                f"Gen {info['gen']:03d} | mean_fitness={info['fitness_mean']:.5e} | std_fitness={info['fitness_std']:.5e}" +
                f"| max_fitness={info['fitness_max']:.5e} | fast-forward: {info["ff"]}",
                flush=True
            )
        
    #yappi.stop()
    #yappi.get_func_stats(filter_callback=lambda x: x.ttot > 0.1).print_all()
    optimizer.close()

def no_embedding_startsonly_experiment():
    e = make_env()
    example_obs = e.reset()[0]
    nlits = example_obs["nlits"]
    example_obs = construct_sparse_tensor(example_obs)

    config = {
            "clause_dim":32,
            "lit_dim":16,
            "n_hops":2,
            "n_layers_C_update":3,
            "n_layers_L_update":3,
            "n_layers_score":1,
            "use_embeddings": False,
            "nlits":nlits,
            "discrete":False,
            "normalize":False,
            "activation":"relu"
        }

    model = GNNWrapper(config)
    model.latent_dim_pi = nlits // 2
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = OpenAIESOptimizer(model, make_env_fn=make_env, fitness_fn=fitness_fn, sigma=0.07, lr=0.001, popsize=512, n_workers=64)
    #yappi.set_clock_type("wall")
    #yappi.clear_stats()
    #yappi.start()
    

    info = {}
    for i in range(4000):
        info = optimizer.step()   
        if i % 50 == 0:
            pred = model(example_obs)[0]
            var_viz("cnf/sha1_21round.cnf", pred.detach().cpu().numpy().squeeze(), path=FIGPATH + f"es_nn_model_gen_{i}.png", title="")
            #var_viz("cnf/sha1_21round.cnf", info["grad"], FIGPATH + f"es_grad_gen_{i}.png", title="")
            torch.save(model.state_dict(), MODELPATH)
                
                
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
        
    #yappi.stop()
    #yappi.get_func_stats(filter_callback=lambda x: x.ttot > 0.1).print_all()
    optimizer.close()

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
    LOGPATH = LOGDIR + f"es_nn_log_{i}.jsonl"
    MODELPATH = LOGDIR + f"es_nn_model_{i}.pt"
    FIGPATH = LOGDIR + f"es_nn_fig_{i}/"
    os.makedirs(FIGPATH, exist_ok=True)

    embedding_variabletime_experiment()