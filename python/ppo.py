from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from solver_environment import SolverEnv
from gnn import rl_GNN1
from util import LoggingCallback, GNNPolicy, VariableRolloutBuffer, ObjectVecEnv
import torch.nn as nn
import torch
import gc

    

if __name__ == "__main__":
    callback = LoggingCallback("ppo_test.jsonl", save_freq=100, verbose=1)
    env = ObjectVecEnv([
        lambda: SolverEnv(rounds=21, free_outputs=150, single_inst=True, verb=0, logfile="logs/ppo_log_test.jsonl")
        ])

    model = PPO(GNNPolicy, 
                env, 
                verbose=2, 
                n_steps=2,
                rollout_buffer_class=VariableRolloutBuffer,
            )
    print(sum(p.numel() for p in model.policy.mlp_extractor.parameters() if p.requires_grad))
    model.learn(total_timesteps=1000, progress_bar=True, callback=callback)





