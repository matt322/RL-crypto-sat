from stable_baselines3 import PPO
from gymnasium import spaces
from solver_environment import SolverEnv
from gnn import rl_GNN1
from util import LoggingCallback, GNNPolicy, VariableRolloutBuffer
import torch.nn as nn
import torch
import gc

    

if __name__ == "__main__":
    callback = LoggingCallback("ppo_test.jsonl", save_freq=100, verbose=1)
    env = SolverEnv(rounds=21, free_outputs=150, single_inst=True, verb=0, logfile="logs/ppo_log_test.jsonl")
    model = PPO(GNNPolicy, 
                env, 
                verbose=2, 
                n_steps=4, 
                rollout_buffer_class=VariableRolloutBuffer,
            )
    model.learn(total_timesteps=1000, progress_bar=True, callback=callback)





