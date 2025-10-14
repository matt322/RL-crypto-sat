from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from solver_environment import SolverEnv
from gnn import rl_GNN1
import torch.nn as nn
import torch

class GNNWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gnn = rl_GNN1(**config)
        self.latent_dim_pi = None 
        self.latent_dim_vf = None
        self.value_net = None
        self.action_net = None

    def forward(self, x):
        x = construct_sparse_tensor(x)
        policy, value = self.gnn(x)
        return policy.squeeze(), torch.mean(value).unsqueeze(0)

class GNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["ortho_init"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        config = {
            "clause_dim":16,
            "lit_dim":64,
            "n_hops":4,
            "n_layers_C_update":3,
            "n_layers_L_update":3,
            "n_layers_score":1,
            "activation":"relu"
        }
        self.features_extractor = nn.Identity()
        self.mlp_extractor = GNNWrapper(config)
        self.mlp_extractor.latent_dim_pi = self.action_space.shape[0]
        self.mlp_extractor.latent_dim_vf = 1

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.value_net = nn.Identity()        
        self.action_net = nn.Identity()

def construct_sparse_tensor(obs):
        """
        converts SB3 internal representation of observation into pt sparse tensor
        """
        return torch.sparse_csr_tensor(
            crow_indices=obs["crow_indices"][:, :int(obs["nclauses"])+1].to(torch.int32).squeeze(), 
            col_indices=obs["col_indices"][:, :int(obs["nnz"])].to(torch.int32).squeeze(), 
            values=obs["values"][:, :int(obs["nnz"])].to(torch.float32).squeeze(), 
            size=(int(obs["nclauses"]), int(obs["nlits"]))
        )
    



if __name__ == "__main__":
    model = PPO(GNNPolicy, SolverEnv(), verbose=2)
    model.learn(total_timesteps=10, progress_bar=True)





