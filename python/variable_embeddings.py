#autoencoder to learn variable embeddings
import numpy as np
import torch.nn as nn
import torch

def autoencoder(n=3968, dim=8):
    embeddings = nn.Linear(n, dim, bias=False)
    model = nn.Sequential(
        embeddings,
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, n)
    )
    
    def onehot(i):
        v = np.zeros(n)
        v[i] = 1
        return v
    
    def acc(true, pred):
        return np.mean((np.argmax(true, axis=1) == np.argmax(pred, axis=1)).astype(np.int32))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(500):
        data = np.array([onehot(i) for i in range(n)])
        np.random.shuffle(data)
        data = torch.tensor(data, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(torch.tensor(data))
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(epoch, acc(data.detach().numpy(), outputs.detach().numpy()), loss.item())
    res = embeddings.weight.detach().numpy()
    print(res)
    np.save("embeddings.npy", res)


if __name__ == "__main__":
    autoencoder()

