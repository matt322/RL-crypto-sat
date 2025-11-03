import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
from solver_interface import SolverController
from instance_generation import Instance
import json
import gc

class SolverEnv(gym.Env):
    def __init__(self, 
                 cnf = None, 
                 rounds=21, 
                 decisions_per_callback=50000, 
                 free_outputs=0, 
                 single_inst=False, 
                 simplify_graph=False, 
                 verb=0, 
                 normalize_actions = False,
                 step_limit=None,
                 logfile="logs/episode_log.jsonl"):
        super().__init__()
        self.free_outputs = free_outputs
        self.decisions_per_callback = decisions_per_callback
        self.simplify_graph = simplify_graph
        self.verb = verb
        self.normalize_actions = normalize_actions
        self.step_limit = step_limit
        self.single_inst = single_inst
        self.cnf = cnf
        if self.cnf is None:
            self.sha1_instance = Instance(rounds=rounds)
        if single_inst and self.cnf is None:
            self.single_cnf = self.sha1_instance.generate(free_outputs, guarantee_soln=True, seed=42)
        self.nvars = self.sha1_instance.nvars if self.cnf is None else self.cnf[2]
        self.max_clauses = 200_000 #most ive observed is around 150k
        self.solver = SolverController()
        self.action_space = Box(low=-1, high=1, shape=(self.nvars,), dtype=np.float32) #neurocore outputs a prob dist over variables and scales by nvars * 10e4
        self.max_nnz = 20_000_000
        self.episode_log_dict = {}
        self.episode_log_file = logfile
        self.episode_stepcount = 0
        self.score_multiplier = 1e4
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

    def get_instance(self, seed):
        if self.cnf is not None:
            return self.cnf
        if self.single_inst:
            return self.single_cnf
        if self.sha1_instance is not None:
            return self.sha1_instance.generate(self.free_outputs, seed=seed)
        raise ValueError("No instance or instance generator available")

    def reset(self, seed=None):
        if self.verb > 1:
            print("Resetting environment")
        if len(self.episode_log_dict) > 0:
            with open(self.episode_log_file, "a") as f:
                f.write(json.dumps(self.episode_log_dict) + "\n")
            if self.verb > 0:
                print(self.episode_log_dict)
            
        self.episode_log_dict = {}
        self.episode_log_dict["seed"] = seed
        self.episode_log_dict["steps"] = 0
        super().reset(seed=seed)
        cnf = self.get_instance(seed)

        obs, reward, done, truncated, model, init_time = self.solver.start(
            cnf, 
            decisions_per_callback=self.decisions_per_callback, 
            timeout_secs=200, 
            verb=self.verb - 2,
            simplify_clauses=self.simplify_graph,
        )
        if obs is None:
            print(reward, model)

        self.episode_log_dict["cumulative_reward"] = reward
        self.episode_log_dict["truncated"] = False
        self.episode_log_dict["max_nnz"] = 0
        self.episode_stepcount = 0

        return self.get_obs(obs), {"init time":init_time} #initial observation has no learnt clauses, fixed clauses added in method (whatever)



    def step(self, action):
        step_truncated = False
        if self.step_limit is not None and self.episode_stepcount >= self.step_limit:
            step_truncated = True

        self.episode_stepcount += 1

        if self.verb > 1:
            print(self.episode_stepcount)
        
        mult = self.nvars if self.normalize_actions else 1
        learnt, reward, done, truncated, model, time = self.solver.step(activity_scores=[f"{i} {score * mult * self.score_multiplier}" for i, score in enumerate(action)])
        truncated = truncated or step_truncated

        self.episode_log_dict["steps"] += 1
        self.episode_log_dict["cumulative_reward"] += reward
        self.episode_log_dict["truncated"] = truncated
      
        
        obs = self.get_obs(learnt)
        if not (done or truncated):
            self.episode_log_dict["max_nnz"] = max(self.episode_log_dict["max_nnz"], obs["nnz"])

        
        return obs, reward, done, truncated, {"step time": time}
    
    def get_obs(self, obs): #obs comes in as dict of np arrays, need to handle truncating
        if obs is not None and (obs["n_clauses"] > self.max_clauses or obs["nnz"] > self.max_nnz):
            raise ValueError(f"Observation exceeds maximum size limits: {obs['n_clauses']} clauses, {obs['nnz']} nnz")
        return obs
    


class BranchingHeuristicEnv(SolverEnv):
    def __init__(self, *args, **kwargs):
        kwargs["decisions_per_callback"], kwargs["simplify_graph"] = 1, True
        super().__init__(*args, **kwargs)
        self.action_space = Discrete(self.nvars)


    def step(self, action): #action is variable index
        a = np.zeros(self.nvars)
        a[action] = 1.0
        res = list(super().step(a))
        reward = res[1]
        if reward == 0: #neuroglue reward
            reward = -1/self.nvars
        elif reward == 1:
            reward = 0 
        self.episode_log_dict["cumulative_reward"] += reward - res[1]
        res[1] = reward
        return tuple(res)





 
if __name__ == "__main__":
    env = BranchingHeuristicEnv(free_outputs=128, verb=2)
    obs, info = env.reset(seed=42)
    print(info)
    done = False
    for i in range(300):
        obs, reward, done, truncated, info = env.step(np.random.randint(0, env.nvars))
       
        print(f"reward: {reward}, step: {env.episode_stepcount}, obs nclauses: {obs['n_clauses']}, nnz: {obs['nnz']}")
    
   
    





