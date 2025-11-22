import os
import json
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from gymnasium import spaces
import torch
import torch.nn as nn
from gnn import rl_GNN1, GNNwithEmbeddings
import yappi
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Sparse CSR tensor support is in beta state.*",   # regex match
    category=UserWarning
)

def get_gnn_config(static, nlits, embed_dim, censor=2):
    assert censor in [0, 1, 2]
    if static:
            config = {
                "clause_dim":1,
                "lit_dim":1,
                "n_hops":0,
                "n_layers_C_update":1,
                "n_layers_L_update":1,
                "n_layers_score":-1,
                "use_embeddings":True,
                "embed_dim":1,
                "nlits":nlits,
                "discrete":False,
                "normalize":False,
                "activation":"relu",
                "censor":censor
            }
    else:
        config = {
                "clause_dim":32,
                "lit_dim":32,
                "n_hops":4,
                "n_layers_C_update":2,
                "n_layers_L_update":2,
                "n_layers_score":2,
                "use_embeddings": embed_dim > 0,
                "nlits":nlits,
                "discrete":False,
                "normalize":False,
                "activation":"relu",
                "censor":censor
            }
        if embed_dim > 0:
            config["embed_dim"] = embed_dim
    return config

def construct_sparse_tensor(obs):
        """
        converts SB3 internal representation of observation into pt sparse tensor, unpadding as necessary
        """
        if isinstance(obs, dict):
            obs = np.array([obs], dtype=object)
        res = np.empty(obs.shape[0], dtype=object)
        for i in range(obs.shape[0]):
            res[i] = torch.sparse_csr_tensor(
                # crow_indices=torch.tensor(obs[i]["crow_indices"][:, :int(obs[i]["n_clauses"])+1], dtype=torch.int32).squeeze(), 
                # col_indices=torch.tensor(obs[i]["col_indices"][:, :int(obs[i]["nnz"])], dtype=torch.int32).squeeze(), 
                # values=torch.tensor(obs[i]["values"][:, :int(obs[i]["nnz"])], dtype=torch.float32).squeeze(), 
                # size=(int(obs[i]["n_clauses"]), int(obs[i]["nlits"]))
                crow_indices=torch.tensor(obs[i]["crow_indices"], dtype=torch.int32), 
                col_indices=torch.tensor(obs[i]["col_indices"], dtype=torch.int32), 
                values=torch.tensor(obs[i]["values"], dtype=torch.float32), 
                size=(int(obs[i]["n_clauses"]), int(obs[i]["nlits"]))
            )
        return res




class VariableRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device = "cuda" if torch.cuda.is_available() else "cpu", gae_lambda = 1, gamma = 0.99, n_envs = 1):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs), dtype=object) # pointer based instead of np array
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(self, obs, action, reward, done, value, log_prob):
        placeholder = np.zeros(1) 
        super().add(placeholder, action, reward, done, value, log_prob)
        self.observations[self.pos-1] = obs # pos incremented in super().add()

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env = None,
    ):
        data = (
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(self.observations[batch_inds].squeeze(), *tuple(map(self.to_torch, data))) #observations dict still on cpu
    


class ObjectVecEnv(DummyVecEnv):
    """
    A version of DummyVecEnv that supports arbitrary (object-type)
    observations — for example, dicts containing variable-sized arrays
    or sparse tensors.
    """
    def __init__(self, env_fns):
        super().__init__(env_fns)
        # Replace the numeric buffers with object buffers
        self.buf_obs = np.empty((self.num_envs,), dtype=object)

    def reset(self):
        for i, env in enumerate(self.envs):
            obs_i, info = env.reset()
            self._save_obs(i, obs_i)  
        return self.buf_obs[0]

    def _save_obs(self, env_idx, obs):
        # override to avoid type mismatch in DummyVecEnv
        self.buf_obs[env_idx] = obs

    def step_wait(self):
        results = [env.step(a) for env, a in zip(self.envs, self.actions)]
        obs, rews, dones, truncs, infos = zip(*results)
        for i in range(self.num_envs):
            self.buf_obs[i] = obs[i]
            if dones[i] or truncs[i]:
                maybe_options = {"options": self._options[i]} if self._options[i] else {}
                obs_i, self.reset_infos[i] = self.envs[i].reset(seed=self._seeds[i], **maybe_options)
                self.buf_obs[i] = obs_i
        return self.buf_obs[0], np.array(rews), np.array(dones), list(infos)


class GNNWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        e = config.pop("use_embeddings")
        self.censor = config.pop("censor")
        nlits = None
        self.discrete = config.pop("discrete", False)
        if e:
            self.gnn = GNNwithEmbeddings(**config)
            nlits = config["nlits"]
        else:
            nlits = config.pop("nlits", None)
            self.gnn = rl_GNN1(**config)
        
        self.latent_dim_pi = nlits // 2 if nlits is not None else None
        self.latent_dim_vf = None
        self.value_net = None
        self.action_net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GNN using device: {self.device}")
        self.gnn = self.gnn.to(self.device)

    def get_assigned_vars(self, col_indices, nvars):
        unique_col_indices = set(col_indices)
        return set(range(nvars)) - (set(filter(lambda x: x < nvars, unique_col_indices)) | set(map(lambda x: x - nvars, filter(lambda x: x >= nvars, unique_col_indices))))

    def forward(self, x):
        #x = construct_sparse_tensor(x)
       
        policy, value = torch.zeros(x.shape[0], self.latent_dim_pi, device=self.device), torch.zeros(x.shape[0], 1, device=self.device)
        for i in range(x.shape[0]):
            G = x[i].to(self.device)
            nvars = x[i].shape[1] // 2
            p, v = self.gnn(G)

            col_indices = G.col_indices().squeeze().cpu().numpy().tolist()

            empty_vars = self.get_assigned_vars(col_indices, nvars)

            mask = torch.zeros(nvars, dtype=torch.bool, device=self.device)
            mask[list(empty_vars)] = True
            p = p.squeeze()
            p_initial = p.clone()
            if self.discrete:
                p.masked_fill_(mask, float('-inf'))
            else:
                p.masked_fill_(mask, 0.0)
            if len(empty_vars) == nvars:
                p[0] = 1.0
                #print(G)
            # if torch.all(torch.isneginf(p) | torch.isnan(p)):
            #     print("detected all -inf")
            #     print(G)
            #     print(empty_vars)
            #     print(p_initial)
              
            policy[i], value[i] = p, torch.mean(v).unsqueeze(0)
            
        return policy, value
    
    def forward_actor(self, x):
        policy, _ = self.forward(x)
        return policy
    
    def forward_critic(self, x):
        _, value = self.forward(x)
        return value
    
    def forward_direct_on_obs(self, x):
        return self.forward(construct_sparse_tensor(x))[0].detach().cpu().numpy().squeeze()
    
    
class ConstructModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return construct_sparse_tensor(x)
    



class GNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["ortho_init"] = False
        self.use_embeddings = kwargs.pop("use_embeddings", False)       
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        config = {
            "clause_dim":64,
            "lit_dim":32,
            "n_hops":4,
            "n_layers_C_update":3,
            "n_layers_L_update":3,
            "n_layers_score":2,
            "use_embeddings": self.use_embeddings,
            "normalize":False,
            "activation":"relu"
        }
        self.features_extractor = ConstructModule() #extract_features and these attributes are used inconsistently
        self.vf_features_extractor = ConstructModule()
        
        self.is_discrete = type(self.action_space) == spaces.Discrete
        config["discrete"] = self.is_discrete
        nvars = self.action_space.n if self.is_discrete else self.action_space.shape[0]
        if config["use_embeddings"]:
            config["nlits"] = nvars * 2
        self.mlp_extractor = GNNWrapper(config)
        self.mlp_extractor.latent_dim_pi = nvars
        self.mlp_extractor.latent_dim_vf = 1


    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.value_net = nn.Identity()        
        self.action_net = nn.Identity()

    def extract_features(self, obs, features_extractor = None):
        return construct_sparse_tensor(obs)
    



class LoggingCallback(BaseCallback):
    """
    A generic Stable Baselines3 callback that:
    - Logs rewards and step times to a JSON file
    - Saves the model every 1000 steps
    - Saves a sample policy output every 1000 steps
    """

    def __init__(self, log_name: str, save_freq: int = 1000, verbose: int = 1, stepfns = []):
        super().__init__(verbose)
        self.stepfns = stepfns
        self.log_dir = "logs"
        self.save_freq = save_freq
        self.start_time = None
        self.episode_rewards = []
        self.last_logged_step = 0
        self.log_file = os.path.join(self.log_dir, log_name)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump({"steps": []}, f)

    def _on_training_start(self):
        self.start_time = time.time()
        yappi.start()

    def _on_step(self) -> bool:
        # Record reward if available
        for i in self.stepfns:
            print(i())
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.episode_rewards.append(float(np.mean(rewards)))

        # Log step time
        step_time = time.time() - self.start_time
        self.start_time = time.time()

        # Periodic saving and logging
        if self.num_timesteps - self.last_logged_step >= self.save_freq:
            self._log_to_json(step_time)
            self._save_model_and_policy()
            self.last_logged_step = self.num_timesteps

        return True

    def _log_to_json(self, step_time: float):
        data = {
            "timestep": self.num_timesteps,
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else None,
            "step_time": step_time,
        }

        # Append to JSON
        with open(self.log_file, "r+") as f:
            log_data = json.load(f)
            log_data["steps"].append(data)
            f.seek(0)
            json.dump(log_data, f, indent=2)

        if self.verbose > 0:
            print(f"[LoggingCallback] Step {self.num_timesteps}: mean_reward={data['mean_reward']:.3f}, time={step_time:.3f}s")

        self.episode_rewards.clear()

    
    def _on_rollout_start(self):
        #yappi.get_func_stats().print_all()
        pass

    def _save_model_and_policy(self):
        pass
        # Save model
        #model_path = os.path.join(self.log_dir, f"model_{self.num_timesteps}.zip")
        #self.model.save(model_path)

        # Sample policy output
        # try:
        #     obs = self.training_env.observation_space.sample()
        #     action, _ = self.model.predict(obs, deterministic=True)
        #     policy_path = os.path.join(self.log_dir, f"policy_{self.num_timesteps}.json")
        #     with open(policy_path, "w") as f:
        #         json.dump({"obs": obs.tolist(), "action": action.tolist()}, f, indent=2)
        # except Exception as e:
        #     if self.verbose > 0:
        #         print(f"[LoggingCallback] Policy output save failed at step {self.num_timesteps}: {e}")

        # if self.verbose > 0:
        #     print(f"[LoggingCallback] Saved model and policy at step {self.num_timesteps}")
