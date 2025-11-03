from stable_baselines3 import PPO
from solver_environment import SolverEnv, BranchingHeuristicEnv
from util import LoggingCallback, GNNPolicy, VariableRolloutBuffer, ObjectVecEnv

def branching_heuristic_experiment():

    #callback = LoggingCallback("ppo_test.jsonl", save_freq=100, verbose=1)
    env = ObjectVecEnv([
        lambda: BranchingHeuristicEnv(rounds=21, free_outputs=64, single_inst=False, verb=1, step_limit=250, logfile="logs/ppo_log_test.jsonl")
        ])

    model = PPO(GNNPolicy, 
                env, 
                verbose=1, 
                n_steps=64,
                batch_size=16,
                rollout_buffer_class=VariableRolloutBuffer,
            )
    print(sum(p.numel() for p in model.policy.mlp_extractor.parameters() if p.requires_grad))
    model.learn(total_timesteps=200000, progress_bar=False)


    

if __name__ == "__main__":
    branching_heuristic_experiment()




