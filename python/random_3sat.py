import random
nvars= 100
nc = int(nvars * 4)

def lit(*exc):
    l = random.randint(-nvars, nvars)
    while abs(l) in exc:
       l = random.randint(-nvars, nvars)
    return l
print(f"p cnf {nvars} {nc}")
for i in range(nc):
    a = lit(0)
    b = lit(0, a)
    c = lit(0, a, b)
    print(f"{a} {b} {c} 0")