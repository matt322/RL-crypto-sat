from stable_baselines3 import PPO
from solver_environment import SolverEnv, BranchingHeuristicEnv, BranchingHeuristicTestEnv
from util import LoggingCallback, GNNPolicy, VariableRolloutBuffer, ObjectVecEnv
import torch

def branching_heuristic_test():

    #callback = LoggingCallback("ppo_test.jsonl", save_freq=100, verbose=1)
    env = ObjectVecEnv([
        lambda: BranchingHeuristicTestEnv(logfile="logs/ppo_log_test_withemb.jsonl")
        ])

    model = PPO(GNNPolicy, 
                env, 
                verbose=1, 
                n_steps=1024,
                batch_size=16,
                n_epochs=2,
                rollout_buffer_class=VariableRolloutBuffer,
                policy_kwargs={"use_embeddings": True},
            )
    print(sum(p.numel() for p in model.policy.mlp_extractor.parameters() if p.requires_grad))
    #loaded = torch.load("logs/ppo_branching_heuristic_model_emb.pt")
    #model.policy.load_state_dict(loaded)
    for i in range(16):
        torch.save(model.policy.state_dict(), "logs/ppo_branching_heuristic_model_emb.pt")
        model.learn(total_timesteps=4096*8, progress_bar=True)

def branching_heuristic_experiment():

    #callback = LoggingCallback("ppo_test.jsonl", save_freq=100, verbose=1)
    env = ObjectVecEnv([
        lambda: BranchingHeuristicEnv(logfile="logs/ppo_log_branching_test_withemb.jsonl", step_limit=512)
        ])

    model = PPO(GNNPolicy, 
                env, 
                verbose=1, 
                n_steps=2048,
                batch_size=16,
                n_epochs=2,
                rollout_buffer_class=VariableRolloutBuffer,
                policy_kwargs={"use_embeddings": True},
            )
    print(sum(p.numel() for p in model.policy.mlp_extractor.parameters() if p.requires_grad))
    #loaded = torch.load("logs/ppo_branching_heuristic_model_emb.pt")
    #model.policy.load_state_dict(loaded)
    for i in range(32):
        torch.save(model.policy.state_dict(), "logs/ppo_branching_heuristic_model_emb.pt")
        model.learn(total_timesteps=4096*4, progress_bar=True)

    

if __name__ == "__main__":
    branching_heuristic_experiment()




