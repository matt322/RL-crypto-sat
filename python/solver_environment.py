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



    def clauses_to_tensor(self, clauses): #keeping it modular since the clause passing will change. Here we return the arguments to sparse_csr_tensor because the observation space needs fixed size dense tensors.
        if self.fixed_clauses:
            clauses.extend(self.fixed_clauses)
        crow_indices = [0]
        col_indices = []
        nnz = 0
        for clause in clauses:
            if nnz + len(clause) > self.max_nnz:
                break
            else:
                nnz += len(clause)
            crow_indices.append(len(clause) + crow_indices[-1])
            for lit in clause:
                idx = lit - 1 if lit > 0 else self.nvars + (-lit - 1)
                col_indices.append(idx)
        values = [1] * nnz
        n_clauses = len(clauses)
        if n_clauses < self.max_clauses:
            last_nnz = crow_indices[-1]
            crow_indices.extend([last_nnz] * (self.max_clauses - n_clauses))
        if nnz < self.max_nnz:
            col_indices.extend([0] * (self.max_nnz - nnz))
            values.extend([0] * (self.max_nnz - len(values)))
        
        return {"crow_indices":torch.tensor(crow_indices), 
                "col_indices":torch.tensor(col_indices),
                "values":torch.tensor(values),
                "nlits":2*self.nvars,
                "nclauses":n_clauses,
                "nnz":nnz
            } #shape is needed for sparse_csr_tensor

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
        #simple unsat instance
        self.fixed_clauses = [[1,2], [-1], [-2]]
        return self.clauses_to_tensor([]), {}


def construct_sparse_tensor(obs):
        return torch.sparse_csr_tensor(
            crow_indices=obs["crow_indices"][:, :int(obs["nclauses"])+1].to(torch.int32).squeeze(), 
            col_indices=obs["col_indices"][:, :int(obs["nnz"])].to(torch.int32).squeeze(), 
            values=obs["values"][:, :int(obs["nnz"])].to(torch.float32).squeeze(), 
            size=(int(obs["nclauses"]), int(obs["nlits"]))
        )
        

if __name__ == "__main__":
    env = SolverEnv(rounds=21)
    env.reset(seed=41)
    obs, _, _, _, info = env.step(np.random.rand(env.nvars) * 100000)
    torch.sparse.mm(construct_sparse_tensor(obs), torch.rand((2*env.nvars, 1)))