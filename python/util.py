import os
import json
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch

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

class LoggingCallback(BaseCallback):
    """
    A generic Stable Baselines3 callback that:
    - Logs rewards and step times to a JSON file
    - Saves the model every 1000 steps
    - Saves a sample policy output every 1000 steps
    """

    def __init__(self, log_dir: str, save_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.start_time = None
        self.episode_rewards = []
        self.last_logged_step = 0
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "training_log.json")
        # Initialize JSON log
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump({"steps": []}, f)

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Record reward if available
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

    def _save_model_and_policy(self):
        # Save model
        model_path = os.path.join(self.log_dir, f"model_{self.num_timesteps}.zip")
        self.model.save(model_path)

        # Sample policy output
        try:
            obs = self.training_env.observation_space.sample()
            action, _ = self.model.predict(obs, deterministic=True)
            policy_path = os.path.join(self.log_dir, f"policy_{self.num_timesteps}.json")
            with open(policy_path, "w") as f:
                json.dump({"obs": obs.tolist(), "action": action.tolist()}, f, indent=2)
        except Exception as e:
            if self.verbose > 0:
                print(f"[LoggingCallback] Policy output save failed at step {self.num_timesteps}: {e}")

        if self.verbose > 0:
            print(f"[LoggingCallback] Saved model and policy at step {self.num_timesteps}")
