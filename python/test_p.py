from instance_generation import Instance
import random
import matplotlib.pyplot as plt

instance = Instance(rounds=21)

x, y = [], []

for i in range(1000):
    p = random.uniform(0, 0.5)
    _, _, nvars, _ = instance.generate(p)
    y.append(nvars)
    x.append(p)

plt.scatter(x, y)
plt.show()