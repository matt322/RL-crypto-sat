import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
from solver_interface import SolverController
from instance_generation import Instance
import json
import gc

class SolverEnv(gym.Env):
    def __init__(self, rounds=21, decisions_per_callback=50000, free_outputs=0, single_inst=False, simplify_graph=False, verb=0, logfile="logs/episode_log.jsonl"):
        super().__init__()
        self.free_outputs = free_outputs
        self.decisions_per_callback = decisions_per_callback
        self.simplify_graph = simplify_graph
        self.verb = verb
        self.single_inst = single_inst
        self.sha1_instance = Instance(rounds=rounds)
        self.nvars = self.sha1_instance.nvars
        self.max_clauses = 200_000 #most ive observed is around 150k
        self.solver = SolverController()
        self.action_space = Box(low=-1, high=1, shape=(self.nvars,), dtype=np.float32) #neurocore outputs a prob dist over variables and scales by nvars * 10e4
        self.max_nnz = 20_000_000
        self.episode_log_dict = {}
        self.episode_log_file = logfile
        self.global_stepcount = 0
        self.score_multiplier = 1e4 * self.nvars
         #these will need to be tuned in the future since they directly affect memory usage
        # SB3 needs finite bounds, for now we will let it clip without changing the gnn. 
        #Observation space: in the GNN we have G @ V where G is Nclauses x Nlits "adjacency matrix" and V is Nlits x features
        #C_update(G @ V) is aggregating variable embeddings for each clause and passing through nn C_update. 
        #Since the embedding is what will diff l and ~l I am now thinking the embedding should be learned or that we should use vcg then aggregate at the end
        #also, it shouldn't be too hard to modify glucose to store a list of preferred polarities rather than doing random
        self.observation_space = self.observation_space = Dict({
            "crow_indices": Box(low=0, high=self.max_nnz, shape=(self.max_clauses+1,), dtype=np.int32),
            "col_indices": Box(low=0, high=2*self.nvars-1, shape=(self.max_nnz,), dtype=np.int16),
            "values": Box(low=0, high=1, shape=(self.max_nnz,), dtype=np.int8),
            "nlits": Box(low=0, high=20000, shape=(1,), dtype=np.int32),
            "n_clauses": Box(low=0, high=self.max_clauses, shape=(1,), dtype=np.int32),
            "nnz": Box(low=0, high=self.max_nnz, shape=(1,), dtype=np.int32) #graph shape
        }) #SB3 needs fixed size space but it doesnt matter since we are already using sparse tensors



    def reset(self, seed=None):
        if len(self.episode_log_dict) > 0:
            with open(self.episode_log_file, "a") as f:
                f.write(json.dumps(self.episode_log_dict) + "\n")
        self.episode_log_dict = {}
        self.episode_log_dict["seed"] = seed
        self.episode_log_dict["steps"] = 0
        self.episode_log_dict["first_action"] = None
        super().reset(seed=seed)
        self.cnf = self.sha1_instance.generate(self.free_outputs, seed=42 if self.single_inst else seed)

        obs, reward, done, _, init_time = self.solver.start(
            self.cnf, 
            decisions_per_callback=self.decisions_per_callback, 
            timeout_secs=200, 
            verb=self.verb,
            simplify_clauses=self.simplify_graph,
        )

        self.episode_log_dict["rewards"] = [reward]
        self.episode_log_dict["truncated"] = False
        self.episode_log_dict["max_nnz"] = 0

        return self.get_obs(obs), {"init time":init_time} #initial observation has no learnt clauses, fixed clauses added in method (whatever)



    def step(self, action):
        self.global_stepcount += 1
        print(self.global_stepcount)
        learnt, reward, done, truncated, model, time = self.solver.step(activity_scores=[f"{i} {score * self.score_multiplier}" for i, score in enumerate(action)])
        self.episode_log_dict["steps"] += 1
        self.episode_log_dict["rewards"].append(reward)
        self.episode_log_dict["truncated"] = truncated
        if self.episode_log_dict["first_action"] is None:
            self.episode_log_dict["first_action"] = list(map(lambda x: round(float(x), 4), action))
        obs = self.get_obs(learnt)
        self.episode_log_dict["max_nnz"] = max(self.episode_log_dict["max_nnz"], obs["nnz"])
        return obs, reward, done, truncated, {"step time": time}
    
    def get_obs(self, obs): #obs comes in as dict of np arrays, need to handle truncating
        if obs["n_clauses"] > self.max_clauses or obs["nnz"] > self.max_nnz:
            raise ValueError(f"Observation exceeds maximum size limits: {obs['n_clauses']} clauses, {obs['nnz']} nnz")
        return obs
    
 
if __name__ == "__main__":
    env = SolverEnv(free_outputs=128, decisions_per_callback=10000)
    for i in range(100):
        obs, info = env.reset(seed=42)
        print(info)
        done = False
        while not done:
            obs, reward, done, truncated, info = env.step(np.zeros(env.nvars))
            print(f"time: {info}")
    
   
    





