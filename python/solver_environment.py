import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
from solver_interface import SolverController
from instance_generation import Instance
import torch

class SolverEnv(gym.Env):
    def __init__(self, rounds=21, ):
        super().__init__()
        self.sha1_instance = Instance(rounds=rounds)
        self.nvars = self.sha1_instance.nvars
        self.max_clauses = 200_000 #most ive observed is around 150k
        self.solver = SolverController()
        self.action_space = Box(low=-1, high=1, shape=(self.nvars,), dtype=np.float32) #neurocore outputs a prob dist over variables and scales by nvars * 10e4
        self.max_nnz = 5_000_000 #these will need to be tuned in the future since they directly affect memory usage
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
            "nclauses": Box(low=0, high=self.max_clauses, shape=(1,), dtype=np.int32),
            "nnz": Box(low=0, high=self.max_nnz, shape=(1,), dtype=np.int32) #graph shape
        }) #SB3 needs fixed size space but it doesnt matter since we are already using sparse tensors



    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed:
            self.sha1_instance = Instance(rounds=self.sha1_instance.rounds, seed=seed)
        self.cnf = self.sha1_instance.generate()
        self.fixed_clauses = self.solver.start(self.cnf)
        return self.clauses_to_tensor([]), {} #initial observation has no learnt clauses, fixed clauses added in method (whatever)



    def step(self, action):
        learnt, reward, done, model, time = self.solver.step(activity_scores=[f"{i+1} {score}" for i, score in enumerate(action)])
        truncated = done and model is None
        clauses = self.fixed_clauses + learnt
        obs = self.clauses_to_tensor(clauses)
        return obs, reward, done, truncated, {"step time": time}
    

class TestEnv(SolverEnv):
    def __init__(self):
        super().__init__(rounds=21)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.fixed_clauses = [[1,2], [-1], [-2]]
        return self.clauses_to_tensor([]), {}



