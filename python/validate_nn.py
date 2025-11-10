from util import GNNWrapper, GNNPolicy
import torch
import json
from instance_generation import Instance
from solver_environment import SolverEnv
from util import construct_sparse_tensor
from viz import var_viz






if __name__ == "__main__":
	config = {
		"clause_dim":16,
		"lit_dim":16,
		"n_hops":0,
		"n_layers_C_update":3 ,
		"n_layers_L_update":3,
		"n_layers_score":1,
		"activation":"relu",
		"use_embeddings": True,
	}
	config1 = {
            "clause_dim":64,
            "lit_dim":32,
            "n_hops":4,
            "n_layers_C_update":3,
            "n_layers_L_update":3,
            "n_layers_score":2,
            "use_embeddings": False,
            "activation":"relu"
        }
	env = SolverEnv(single_inst=True)
	config["nlits"] = env.nvars * 2
	G = construct_sparse_tensor(env.reset(seed=42)[0])
	model = GNNWrapper(config)
	basemodel = GNNWrapper(config1)
	model.latent_dim_pi = G[0].shape[1] // 2
	basemodel.latent_dim_pi = G[0].shape[1] // 2
	loaded_model = torch.load("logs/logs/ppo_branching_heuristic_model.pt", map_location=torch.device('cpu'))
	for key in list(loaded_model.keys()):
		if "mlp_extractor." in key:
			new_key = key.replace("mlp_extractor.", "")
			loaded_model[new_key] = loaded_model.pop(key)
	basemodel.load_state_dict(loaded_model)
	y1 = basemodel.forward(G)[0].detach().numpy().squeeze()

	y = torch.tensor(json.load(open("logs/logs/es_logs/1/es_model_0.jsonl"))["model"]).unsqueeze(0)
	var_viz("cnf/sha1_21round.cnf", y.squeeze().numpy())
	var_viz("cnf/sha1_21round.cnf", y1)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	for i in range(100):
		pred = model.forward(G)[0]
		loss = torch.nn.functional.mse_loss(pred, y)
		print(loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	res = model.forward(G)[0].detach().numpy().squeeze()
	var_viz("cnf/sha1_21round.cnf", res)

