import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
from solver_interface import SolverController
from instance_generation import Instance
import json

class SolverEnv(gym.Env):
    def __init__(self, rounds=21, free_outputs=0, single_inst=False):
        super().__init__()
        self.free_outputs = free_outputs
        self.single_inst = single_inst
        self.sha1_instance = Instance(rounds=rounds)
        self.nvars = self.sha1_instance.nvars
        self.max_clauses = 200_000 #most ive observed is around 150k
        self.solver = SolverController()
        self.action_space = Box(low=-1, high=1, shape=(self.nvars,), dtype=np.float32) #neurocore outputs a prob dist over variables and scales by nvars * 10e4
        self.max_nnz = 5_000_000
        self.episode_log_dict = {}
        self.episode_log_file = "logs/episode_log.jsonl"
        
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
        super().reset(seed=seed)
        if seed:
            self.sha1_instance = Instance(rounds=self.sha1_instance.rounds, seed=seed)
        if self.single_inst:
            self.sha1_instance = Instance(rounds=self.sha1_instance.rounds, seed=41)
        self.cnf = self.sha1_instance.generate(self.free_outputs)
        self.fixed_obj, reward, done, _, init_time = self.solver.start(self.cnf, decisions_per_callback=50000, timeout_secs=200, verb=4)
        self.episode_log_dict["sum_rewards"] = reward
        self.episode_log_dict["truncated"] = False
        self.episode_log_dict["max_nnz"] = 0

        return self.get_obs(), {"init time":init_time} #initial observation has no learnt clauses, fixed clauses added in method (whatever)



    def step(self, action):
        learnt, reward, done, truncated, model, time = self.solver.step(activity_scores=[f"{i+1} {score}" for i, score in enumerate(action)])
        self.episode_log_dict["steps"] += 1
        self.episode_log_dict["sum_rewards"] += reward
        self.episode_log_dict["truncated"] = truncated
        obs = self.get_obs(learnt)
        self.episode_log_dict["max_nnz"] = max(self.episode_log_dict["max_nnz"], obs["nnz"])
        return obs, reward, done, truncated, {"step time": time}
    
    def get_obs(self, learnt_obj=None): #obs comes in as dict of np arrays
        obs = {}
        if learnt_obj is None:
            for key in ["crow_indices", "col_indices", "values"]:
                obs[key] = self.fixed_obj[key].copy()
            for key in ["nlits", "n_clauses", "nnz"]:
                obs[key] = int(self.fixed_obj[key])
        else:
            a = self.fixed_obj
            b = learnt_obj
            
            if a["nlits"] != b["nlits"]:
                raise ValueError("Incompatible Observations")  
            obs["crow_indices"] = np.concatenate([a["crow_indices"][:int(a["n_clauses"])+1], b["crow_indices"][1:] + int(a["nnz"])])
            obs["col_indices"] = np.concatenate([a["col_indices"][:int(a["nnz"])], b["col_indices"]])
            obs["values"] = np.concatenate([a["values"][:int(a["nnz"])], b["values"]])
            obs["nlits"] = int(a["nlits"])
            obs["n_clauses"] = int(a["n_clauses"]) + int(b["n_clauses"])
            obs["nnz"] = int(a["nnz"]) + int(b["nnz"])
        
        if obs["n_clauses"] > self.max_clauses or obs["nnz"] > self.max_nnz:
            raise ValueError("Observation exceeds maximum size limits")
        obs["crow_indices"] = np.pad(obs["crow_indices"], (0, self.max_clauses + 1 - len(obs["crow_indices"])), 'constant') #doesnt matter what we pad with it will be removed
        obs["col_indices"] = np.pad(obs["col_indices"], (0, self.max_nnz - len(obs["col_indices"])), 'constant')
        obs["values"] = np.pad(obs["values"], (0, self.max_nnz - len(obs["values"])), 'constant') 
        return obs
    
 
if __name__ == "__main__":
    env = SolverEnv()
    obs, info = env.reset(seed=42)
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(np.zeros(env.nvars))
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
    
   
    





