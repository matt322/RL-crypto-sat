from gnn import rl_GNN1


config = {
  "clause_dim":64,
  "lit_dim":16,
  "n_hops":4,
  "n_layers_C_update":3,
  "n_layers_L_update":3,
  "n_layers_score":1,
  "activation":"relu"
}

cnf = 


model = rl_GNN1(clause_dim=64, var_dim=16, n_layers=3, n_rounds=21)

