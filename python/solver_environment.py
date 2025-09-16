import gymnasium as gym
import numpy as np
from solver_interface import SolverController
from instance_generation import Instance

class SolverEnv(gym.Env):
    def __init__(self, rounds=21, ):
        super().__init__()
        self.sha1_instance = Instance(rounds=rounds)
        self.nvars = self.sha1_instance.nvars
        self.max_clauses = 1000000 #max in neurocore paper
        self.solver = SolverController()
        self.fixed_clauses = []
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.nvars,), dtype=np.float32) #neurocore outputs a prob dist over variables and scales by nvars * 10e4
        #Observation space: in the GNN we have G @ V where G is Nclauses x Nvars "adjacency matrix" and V is Nvars x features
        #C_update(G @ V) is aggregating variable embeddings for each clause and passing through nn C_update. 
        #Since the embedding is what will diff l and ~l I am now thinking the embedding should be learned or that we should use vcg then aggregate at the end
        #also, it shouldn't be too hard to modify glucose to store a list of preferred polarities rather than doing random
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.nvars, self.max_clauses), dtype=np.float32)

    def update_clauses(self):


    def reset(self, seed=None):
        super().reset(seed=seed)


    def step(self, action):
        pass
