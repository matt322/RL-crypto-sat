from solver_environment import SolverEnv
from solver_interface import SolverController
from util import GNNWrapper
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
    env = SolverEnv(simplify_graph=True, decisions_per_callback=decisions, filter_scores=True, normalize_actions=False)
    config = {
            "clause_dim":32,
            "lit_dim":16,
            "n_hops":2,
            "n_layers_C_update":3,
            "n_layers_L_update":3,
            "n_layers_score":1,
            "use_embeddings": True,
            "embed_dim":4,
            "nlits":env.nvars * 2,
            "discrete":False,
            "normalize":False,
            "activation":"relu"
        }
    
    model = GNNWrapper(config)
    model.latent_dim_pi = env.nvars
    model.load_state_dict(torch.load("logs/20/es_nn_model_20.pt"))

    static_scores = np.array(json.load(open("logs/logs/es_logs/1/es_model_0.jsonl"))["model"]).squeeze()

    vanilla_baseline = lambda x: []
    random_baseline = lambda x: np.random.uniform(0, 1, env.nvars)
    zero_baseline = lambda x: np.ones(env.nvars) * 1e-8
    static_model = lambda x: static_scores

    nsteps = 200

    model_samples_x, model_samples_y = [], []
    vbase_samples_x, vbase_samples_y = [], []
    model_samples_x, model_samples_y = [], []
    for i in range(10):
        print("getting model samples")
        #samples = sample_periodically(env, static_model, nsteps)
        samples = sample_returns_within_period(env, static_model, nsteps, 1)
        model_samples_y.append(samples)
        
        print("getting base samples")
        samples = sample_returns_within_period(env, vanilla_baseline, nsteps, 1)
        vbase_samples_y.append(samples)

        # vbase_samples_x += list(range(0, decisions*len(samples), decisions))
        # plt.plot(range(0, decisions*len(samples), decisions), samples, color="orange", label = "vanilla solver" if  i == 0 else "")

        # print("getting random samples")
        # samples = sample_returns_within_period(env, random_baseline, nsteps, 10)
        # vbase_samples_y += samples
        # vbase_samples_x += list(range(0, decisions*len(samples), decisions))
        # plt.plot(range(len(samples)), samples, color="red", label = "random baseline" if  i == 0 else "")

    model_mean, model_conf = mean_and_conf(model_samples_y)
    plt.plot(range(0, decisions*len(samples), decisions), model_mean, color="blue", label = "periodic refocusing average return")
    plt.fill_between(range(0, decisions*len(samples), decisions), model_mean-model_conf, model_mean+model_conf, color='blue', alpha=0.2, label='95% CI')

    baseline_mean, baseline_conf = mean_and_conf(vbase_samples_y)
    plt.plot(range(0, decisions*len(samples), decisions), baseline_mean, color="orange", label = "vanilla solver average return")
    plt.fill_between(range(0, decisions*len(samples), decisions), baseline_mean-baseline_conf, baseline_mean+baseline_conf, color='orange', alpha=0.2)

    # plt.plot(range(0, decisions*len(samples), decisions), samples, color="blue", label = "starting with model scores" if  i == 0 else "")
    # plt.plot(range(0, decisions*len(samples), decisions), samples, color="blue", label = "starting with model scores" if  i == 0 else "")


    plt.legend()
    plt.xlabel("number of decisions")
    plt.ylabel("return")
    plt.title("Average return every 10,000 decisions, 10 sample trajectories")
    plt.show()

    






    