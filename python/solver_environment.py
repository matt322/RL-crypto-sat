import gymnasium as gym
import numpy as np
from solver_interface import SolverController
from instance_generation import Instance
import torch

class SolverEnv(gym.Env):
    def __init__(self, rounds=21, ):
        super().__init__()
        self.sha1_instance = Instance(rounds=rounds)
        self.nvars = self.sha1_instance.nvars
        self.max_clauses = 1000000 #max in neurocore paper
        self.solver = SolverController()
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.nvars,), dtype=np.float32) #neurocore outputs a prob dist over variables and scales by nvars * 10e4
        #Observation space: in the GNN we have G @ V where G is Nclauses x Nvars "adjacency matrix" and V is Nvars x features
        #C_update(G @ V) is aggregating variable embeddings for each clause and passing through nn C_update. 
        #Since the embedding is what will diff l and ~l I am now thinking the embedding should be learned or that we should use vcg then aggregate at the end
        #also, it shouldn't be too hard to modify glucose to store a list of preferred polarities rather than doing random
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.nvars, self.max_clauses), dtype=np.float32)

    def clauses_to_tensor(self, clauses): #keeping it modular since the clause passing will change
        crow_indices = [0]
        col_indices = []
        for clause in clauses:
            crow_indices.append(len(clause) + crow_indices[-1])
            for lit in clause:
                idx = lit - 1 if lit > 0 else self.nvars + (-lit - 1)
                col_indices.append(idx)
        values = [1] * len(col_indices)
        return torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor(col_indices), torch.tensor(values))


    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed:
            self.sha1_instance = Instance(rounds=self.sha1_instance.rounds, seed=seed)
        self.cnf = self.sha1_instance.generate()
        self.fixed_clauses = self.clauses_to_adj(self.solver.start(self.cnf))
        return self.fixed_clauses, {}



    def step(self, action):
        learnt, reward, done, model, time = self.solver.step(activity_scores=[f"{i+1} {score}" for i, score in enumerate(action)])
        truncated = done and model is None
        return learnt, reward, done, truncated, {"step time": time}
        
if __name__ == "__main__":
    env = SolverEnv(rounds=21)
    env.nvars = 4
    clauses = [[1, -3, 4], [-1, 2], [-2, 3], [-4]]
    tensor = env.clauses_to_tensor(clauses)
    print(tensor)