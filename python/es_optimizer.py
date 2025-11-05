import numpy as np
import multiprocessing as mp
from torch.optim import Adam
import os, socket, torch

class OpenAIESOptimizer:
    """
    OpenAI Evolution Strategies (ES) optimizer that keeps separate envs per worker.

    Args:
        dim (int): Dimension of the parameter vector.
        make_env_fn (callable): A function that returns a new environment instance.
        fitness_fn (callable): f(env, x) -> scalar fitness (to maximize).
        sigma (float): Standard deviation of the Gaussian noise.
        lr (float): Learning rate.
        popsize (int): Number of perturbations (will be mirrored for antithetic sampling).
        n_workers (int): Number of worker processes.
    """

    def __init__(self, dim, make_env_fn, fitness_fn,
                 sigma=0.1, lr=0.01, popsize=128, n_workers=None):

        assert popsize % 2 == 0, "popsize must be even for antithetic sampling"
        self.dim = dim
        self.make_env_fn = make_env_fn
        self.fitness_fn = fitness_fn
        self.sigma = sigma
        self.lr = lr
        self.popsize = popsize
        self.model = torch.zeros(dim, requires_grad=True)
        self.n_workers = n_workers
        self.optimizer = Adam([self.model], lr=lr)

        # Shared global setup for worker envs
        if self.n_workers is not None:
            print(f"Starting ES with {self.n_workers} workers.")
            self._ctx = mp.get_context("spawn")
            self._pool = self._ctx.Pool(
                self.n_workers,
                initializer=self._worker_init,
                initargs=(make_env_fn, fitness_fn),
            )
            statuses = self._pool.map(self.worker_info, range(self.n_workers))
            print("\n".join(statuses))
        else:
            self._pool = None
        self._gen = 0

    @staticmethod
    def worker_info(i):
        return f"Worker {i} initialized on host {socket.gethostname()}, PID {os.getpid()}."

    @staticmethod
    def _worker_init(make_env_fn, fitness_fn):
        global _worker_env, _worker_fitness_fn
        _worker_env = make_env_fn()
        _worker_fitness_fn = fitness_fn
        

    @staticmethod
    def _worker_eval(x):
        # Called inside worker, where _worker_env is local
        cand, seed = x
        return _worker_fitness_fn(_worker_env, cand, seed)

    # --- ES main loop ---

    def get_grad_est(self):
        """Perform one ES update step."""
        eps_half = torch.randn(self.popsize // 2, self.dim)
        eps = torch.cat([eps_half, -eps_half], axis=0)
        candidates = self.model[None, :].detach() + self.sigma * eps
        stepseed = np.random.randint(0, 2**32 - 1)

        if self._pool is not None:
            fitness = self._pool.map(OpenAIESOptimizer._worker_eval, zip(candidates, [stepseed] * self.popsize))
        else:
            fitness = [self.fitness_fn(self.make_env_fn(), x) for x in candidates]

        fitness = torch.tensor(fitness) 

        fit_norm = -(fitness - fitness.mean()) / (fitness.std() + 1e-8) #negating here since using adam
        grad_est = (eps.T @ fit_norm) / (self.popsize * self.sigma)

        self._gen += 1

        return {
            "gen": self._gen,
            "grad": grad_est,
            "fitness_mean": float(fitness.mean()),
            "fitness_max": float(fitness.max()),
            "fitness_std": float(fitness.std()),
        }
    
    def step(self):
        info = self.get_grad_est()
        g = info["grad"]
        self.optimizer.zero_grad()
        self.model.grad = g
        self.optimizer.step()
        info["model"] = self.model.detach().cpu().numpy()

        return info
        

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


