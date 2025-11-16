import numpy as np
import multiprocessing as mp
from torch.optim import Adam
from torch.func import functional_call
import os, socket, torch, copy
from solver_environment import SolverEnv
from util import construct_sparse_tensor

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

    def __init__(self, model, make_env_fn, make_env_args, fitness_fn, fitness_args,
                 sigma=0.1, lr=0.01, popsize=128, n_workers=None):

        assert popsize % 2 == 0, "popsize must be even for antithetic sampling"
        self.make_env_fn = make_env_fn
        self.fitness_fn = fitness_fn
        self.evalargs = fitness_args
        self.step_forward = fitness_args
        self.envargs = make_env_args
        self.sigma = sigma
        self.lr = lr
        self.popsize = popsize
        self.model = model
        self.n_workers = n_workers
        self.optimizer = Adam(model.parameters(), lr=lr)

        theta = self.flatten_params(self.model)
        self.dim = theta.numel()

        # Shared global setup for worker envs
        if self.n_workers is not None:
            print(f"Starting ES with {self.n_workers} workers.")
            self._ctx = mp.get_context("spawn")
            self._pool = self._ctx.Pool(
                self.n_workers,
                initializer=self._worker_init,
                initargs=(make_env_fn, fitness_fn, self.model, self.evalargs, self.envargs),
            )
            statuses = self._pool.map(self.worker_info, range(self.n_workers))
            print("\n".join(statuses))
        else:
            self._pool = None
            self.env = self.make_env()
        self._gen = 0
    
    def flatten_params(self, model: torch.nn.Module):
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    @staticmethod
    def worker_info(i):
        return f"Worker {i} initialized on host {socket.gethostname()}, PID {os.getpid()}."

    @staticmethod
    def _worker_init(make_env_fn, fitness_fn, m, evalargs, envargs):
        global _worker_env, _worker_fitness_fn, _model, _make_env_config, _fitness_fn_config
        _make_env_config = envargs
        _fitness_fn_config = evalargs
        _worker_env = make_env_fn(*_make_env_config)
        _worker_fitness_fn = lambda *x: fitness_fn(*x, _fitness_fn_config)
        _model = m
        _model.eval()
        

    @staticmethod
    def _worker_eval(x):
        model, params, seed = x
        return _worker_fitness_fn(_worker_env, model, params, seed)
    
    def fitness_fn(env, model, params, seed):
        obs, info = env.reset(seed)
        if obs is None:
            return 0
        
        param_dict = {}
        idx = 0
        for name, p in model.named_parameters():
            numel = p.numel()
            param_dict[name] = params[idx:idx+numel].view_as(p)
            idx += numel

        pred = functional_call(model, param_dict, construct_sparse_tensor(obs))[0].squeeze()
        reward = env.step(pred)[1]
        return reward

    def make_env(self):
        return SolverEnv(rounds=21, decisions_per_callback=10000, free_outputs=0, simplify_graph=True, single_inst=False, verb=0, normalize_actions = False)

    def get_grad_est(self):
        """Perform one ES update step."""
        theta = self.flatten_params(self.model)

        eps_half = torch.randn(self.popsize // 2, self.dim, device=theta.device)
        eps = torch.cat([eps_half, -eps_half], dim=0)

        candidates = theta[None, :] + self.sigma * eps

        stepseed = np.random.randint(0, 2**32 - 1)
        if self.step_forward > 0:
            np.random.seed(stepseed)
            fast_forward = np.random.geometric(p=1/self.step_forward)
        else:
            fast_forward = 0

        argslist = zip([self.model]*len(candidates), candidates, [stepseed] * self.popsize)
        if self._pool is not None:
            fitness = self._pool.map(OpenAIESOptimizer._worker_eval, argslist)
        else:
            fitness = [self.fitness_fn(self.env, *x) for x in argslist]

        fitness = torch.tensor(fitness) 
        
        fit_norm = -(fitness - fitness.mean()) / (fitness.std() + 1e-8) #negating here since using adam
        grad_est = (eps.T @ fit_norm) / (self.popsize * self.sigma)
        

        self._gen += 1

        return {
            "gen": self._gen,
            "grad": grad_est,
            "ff":fast_forward,
            "return_mean": float(fitness.mean()),
            "return_max": float(fitness.max()),
            "return_std": float(fitness.std()),
        }
    
    def step(self):
        info = self.get_grad_est()
        g = info["grad"]
        self.optimizer.zero_grad()
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.grad = g[idx:idx+numel].view_as(p).clone()
            idx += numel
        self.optimizer.step()
        #info["model"] = self.model.detach().cpu().numpy()

        return info
        

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


