from solver_environment import SolverEnv
from solver_interface import SolverController
from util import GNNWrapper, get_gnn_config
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import norm
import json

def sample_return_at_step(env, model, step):
    env.reset()
    obs = env.advance(step)[0]
    if obs is None:
        return None
    if isinstance(model, GNNWrapper):
        pred = model.forward_direct_on_obs(obs)
    else:
        pred = model(obs)
    return env.step(pred)[1]

def sample_periodically(env, model, nsteps):
    res = []
    obs, info = env.reset()
    if isinstance(model, GNNWrapper):
        pred = model.forward_direct_on_obs(obs)
    else:
        pred = model(obs)
    for i in range(nsteps):
        obs, reward, done, trunc, _ = env.step(pred)
        if done or trunc:
            break
        if isinstance(model, GNNWrapper):
            pred = model.forward_direct_on_obs(obs)
        else:
            pred = model(obs)
        res.append(reward)
        
    return res

def sample_returns_within_period(env, model, model_steps, samples_per_step):
    res = []
    obs, info = env.reset()
    if isinstance(model, GNNWrapper):
        pred = model.forward_direct_on_obs(obs)
    else:
        pred = model(obs)
    for i in range(model_steps):
        obs, reward, done, trunc, _ = env.step(pred)
        if done or trunc:
            break
        if isinstance(model, GNNWrapper):
            pred = model.forward_direct_on_obs(obs)
        else:
            pred = model(obs)
        res.append(reward)

        for j in range(samples_per_step-1):
            obs, reward, done, trunc, _ = env.step([])
            if done or trunc:
                break
            res.append(reward)

        
    return res

def mean_and_conf(trajectories, confidence = 0.95):
    maxlen = max(map(lambda x: len(x), trajectories))
    data = [[x for x in t if x is not None] for t in zip(*map(lambda x: x + [None] * (maxlen - len(x)), trajectories))]
    mres, cres = [], []
    z = norm.ppf(1 - (1 - confidence) / 2)
    for i in data:
        mres.append(np.mean(i))
        cres.append(z * np.std(i, ddof=1) / np.sqrt(len(i)))
    return np.array(mres), np.array(cres)



if __name__ == "__main__":
    decisions = 10000
    env = SolverEnv(simplify_graph=True, decisions_per_callback=decisions, filter_scores=True, normalize_actions=False, reward_pow=0)
    hybrid_env = SolverEnv(simplify_graph=True, decisions_per_callback=decisions, filter_scores=True, normalize_actions=False, reward_pow=2, heuristic_type="hybrid", hybrid_period=4000)
    gnn_config = get_gnn_config(static=True, nlits=env.nvars * 2, embed_dim=4)
    
    model = GNNWrapper(gnn_config)
    model.load_state_dict(torch.load("logs/static_noadvance_alpha=0_6/model.pt"))
    obs, _ = env.reset()
    model_all = model.forward_direct_on_obs(obs)

    static_scores_a2 = lambda x: np.array(json.load(open("logs/logs/es_logs/1/es_model_0.jsonl"))["model"]).squeeze() 
    static_scores_a0 = lambda x: model_all

    vanilla_baseline = lambda x: []
    random_baseline = lambda x: np.random.uniform(0, 1, env.nvars)
    zero_baseline = lambda x: np.ones(env.nvars) * 1e-8

    nsteps = 20

    model_samples_x, model_samples_y = [], []
    vbase_samples_x, vbase_samples_y = [], []
    model_samples_x, model_samples_y = [], []
    for i in range(10):
        print("getting model samples")
        samples = sample_returns_within_period(env, static_scores_a2, nsteps, 1)
        model_samples_y.append(samples)
        
        print("getting static samples")
        samples = sample_returns_within_period(env, vanilla_baseline, nsteps, 1)
        vbase_samples_y.append(samples)


    model_mean, model_conf = mean_and_conf(model_samples_y)
    plt.plot(range(0, decisions*len(samples), decisions), model_mean, color="blue", label = "periodic refocusing")
    plt.fill_between(range(0, decisions*len(samples), decisions), model_mean-model_conf, model_mean+model_conf, color='blue', alpha=0.2, label='95% CI')

    baseline_mean, baseline_conf = mean_and_conf(vbase_samples_y)
    plt.plot(range(0, decisions*len(samples), decisions), baseline_mean, color="green", label = "vanilla solver")
    plt.fill_between(range(0, decisions*len(samples), decisions), baseline_mean-baseline_conf, baseline_mean+baseline_conf, color='green', alpha=0.2)

    # plt.plot(range(0, decisions*len(samples), decisions), samples, color="blue", label = "starting with model scores" if  i == 0 else "")
    # plt.plot(range(0, decisions*len(samples), decisions), samples, color="blue", label = "starting with model scores" if  i == 0 else "")


    plt.legend()
    plt.xlabel("Decisions")
    plt.ylabel("Return (a=2)")
    plt.title("Average return refocusing every 20k decisions, 20 samples")
    plt.show()

    






    