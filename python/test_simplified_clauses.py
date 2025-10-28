from solver_environment import SolverController
import torch

def tensor_to_clauses(t):
    nvars = t.shape[1] // 2
    clauses = []
    for clause in t:
        s = []
        for i, lit in enumerate(clause):
            if lit == 1:
                s.append(i+1 if i//nvars < 1 else -(i-nvars+1))
        clauses.append(s)
    return clauses





if __name__ == "__main__":
    solver = SolverController()
    cnf = ["cnf/test.cnf", [], 15, 67]
    print(solver.start(cnf, 1, verb=0, simplify_clauses=True))
    for i in range(100):
        obs = solver.step(solver.zero_scores())[0]
        obs_tensor = torch.sparse_csr_tensor(obs["crow_indices"], obs["col_indices"], torch.tensor(obs["values"]), size=(obs["n_clauses"], obs["nlits"]))
        print(f"step {i+1}\n\n")
        c = tensor_to_clauses(obs_tensor.to_dense())
        print(c)

