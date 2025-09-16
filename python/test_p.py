from instance_generation import Instance
import random
import matplotlib.pyplot as plt

instance = Instance(rounds=23)

x, y = [], []

for i in range(200):
    p = random.uniform(0, 0.40)
    _, _, nvars, _ = instance.generate(p, simplify=True)
    y.append(nvars)
    x.append(p)

plt.scatter(x, y)
plt.title("23 round")
plt.xlabel("p")
plt.ylabel("number of variables after simp")
plt.show()