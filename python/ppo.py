from stable_baselines3 import PPO
from solver_environment import SolverEnv, BranchingHeuristicEnv
from util import LoggingCallback, GNNPolicy, VariableRolloutBuffer, ObjectVecEnv
import torch

def branching_heuristic_experiment():

    #callback = LoggingCallback("ppo_test.jsonl", save_freq=100, verbose=1)
    env = ObjectVecEnv([
        lambda: BranchingHeuristicEnv(rounds=21, free_outputs=64, single_inst=False, verb=1, step_limit=4096, logfile="logs/ppo_log_test.jsonl")
        ])

    model = PPO(GNNPolicy, 
                env, 
                verbose=1, 
                n_steps=2048,
                batch_size=16,
                n_epochs=1,
                rollout_buffer_class=VariableRolloutBuffer,
            )
    print(sum(p.numel() for p in model.policy.mlp_extractor.parameters() if p.requires_grad))
    for i in range(100):
        torch.save(model.policy.state_dict(), "logs/ppo_branching_heuristic_model.pt")
        model.learn(total_timesteps=4096*2, progress_bar=False)


    

if __name__ == "__main__":
    branching_heuristic_experiment()




