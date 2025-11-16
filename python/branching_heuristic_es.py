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


def make_env(decisions_per_callback, simplify, reward_pow):
    return SolverEnv(
        rounds=21, 
        decisions_per_callback=decisions_per_callback, 
        free_outputs=0, 
        simplify_graph=simplify, 
        single_inst=False, 
        verb=0, 
        guarantee_soln=False, 
        normalize_actions = False,
        filter_scores=True,
        reward_pow=reward_pow,
    )
   
def fitness_fn(env, model, params, seed, step_forward):   
    obs, info = env.reset(seed)
    if step_forward > 0:
        np.random.seed(seed)
        fast_forward = np.random.geometric(p=1/step_forward)
        obs, info = env.advance(fast_forward)
    
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
    if reward == 1:
        return 0.01
    return reward

def run_experiment(title, steps, popsize, reward_pow, step_forward, embed_dim, static, decision_period, simplify_graph, n_workers):
    r"""
    Runs Evolution Strategies experiment
    
    Args
        title:
        steps: (adam) optimizer updates
        popsize: number of samples for gradient estimate (uses antithetic sampling; must be even)
        reward_pow: each clause learned in the decision_period contributes LBD^-reward_pow to the return
        step_forward: sample from geometric dist with mean step_forward, offset solver by that many decisions
        embed_dim: per-literal learned embedding. set to 0 for a single embedding for all lits
        static: learn 1 dimensional literal embeddings and affine transformation
        decision_period: decisions per step
        simplify_graph: eliminate assigned variables and satisfied clauses
        n_workers: for parallel computing
    """

    print(f"starting experiment {title}")

    LOGDIR = "logs/es_logs/"
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
        i = 1
    else:
        i = len(list(filter(lambda x: x.startswith(title), os.listdir(LOGDIR)))) + 1
    LOGDIR = f"logs/es_logs/{title}_{i}/"
    os.makedirs(LOGDIR, exist_ok=True)
    LOGPATH = LOGDIR + f"log.jsonl"
    MODELPATH = LOGDIR + f"model.pt"
    FIGPATH = LOGDIR + f"fig/"
    os.makedirs(FIGPATH, exist_ok=True)

    _make_env_config = (decision_period, simplify_graph, reward_pow)
    _fitness_fn_config = (step_forward)

    e = make_env(*_make_env_config)
    examples = []
    example_obs = e.reset()[0]
    nlits = example_obs["nlits"]
    examples.append(construct_sparse_tensor(example_obs))
    for i in range(4):
        example_obs = e.step([])[0]
        examples.append(construct_sparse_tensor(example_obs))
        
    if static:
            config = {
                "clause_dim":1,
                "lit_dim":1,
                "n_hops":0,
                "n_layers_C_update":1,
                "n_layers_L_update":1,
                "n_layers_score":1,
                "use_embeddings":True,
                "embed_dim":1,
                "nlits":nlits,
                "discrete":False,
                "normalize":False,
                "activation":"relu"
            }
    else:
        config = {
                "clause_dim":32,
                "lit_dim":16,
                "n_hops":2,
                "n_layers_C_update":3,
                "n_layers_L_update":3,
                "n_layers_score":1,
                "use_embeddings": embed_dim > 0,
                "nlits":nlits,
                "discrete":False,
                "normalize":False,
                "activation":"relu"
            }
        if embed_dim > 0:
            config["embed_dim"] = embed_dim

    model = GNNWrapper(config)
    model.latent_dim_pi = nlits // 2
    print(f"Paramters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = OpenAIESOptimizer(
        model, 
        make_env_fn=make_env, 
        make_env_args=_make_env_config, 
        fitness_fn=fitness_fn, 
        fitness_args=_fitness_fn_config, 
        sigma=0.05, 
        lr=0.001, 
        popsize=popsize, 
        n_workers=n_workers
    )

    info = {}
    for i in range(steps):
        info = optimizer.step()   
        if i % 50 == 0:
            for j, example_obs in enumerate(examples):
                pred = model(example_obs)[0]
                var_viz("cnf/sha1_21round.cnf", pred.detach().cpu().numpy().squeeze(), path=FIGPATH + f"es_{title}_gen_{i}_example_{j}.png", title="")
            torch.save(model.state_dict(), MODELPATH)
                
                
        with open(LOGPATH, "a") as f:
            f.write(json.dumps({"gen": info["gen"],
                             "return_mean": info["return_mean"],
                             "return_std": info["return_std"],
                             "return_max": info["return_max"],
                             "fast_forward": info["ff"],
                             }) + "\n")
        print(
                f"Gen {info['gen']:03d} | mean_return={info['return_mean']:.5e} | std_return={info['return_std']:.5e}" +
                f"| max_return={info['return_max']:.5e} | fast-forward: {info["ff"]}",
                flush=True
            )
        
    optimizer.close()


if __name__ == "__main__":
    n_workers = 64
    run_experiment(
        title="static_noadvance_alpha=0",
        steps=1000,
        popsize=128,
        reward_pow=0,
        step_forward=0,
        embed_dim=0,
        static=True,
        decision_period=10000,
        simplify_graph=True,
        n_workers=n_workers,
    )
    
    run_experiment(
        title="static_advance_alpha=0",
        steps=1000,
        popsize=128,
        reward_pow=0,
        step_forward=20000,
        embed_dim=0,
        static=True,
        decision_period=10000,
        simplify_graph=True,
        n_workers=n_workers,
    )


