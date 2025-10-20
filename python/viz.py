import matplotlib.pyplot as plt
import json
import numpy as np

def extract(file, condition):
    res = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            if condition(line):
                res.append(line["time"])
    return res

def cumulative_plot(times, max=200):
    times = sorted(times)
    n = len(times)
    timeout_count = times.count(max)
    non_timeout_count = n - timeout_count

    cumulative_percentages = [(i / non_timeout_count) * 100 for i in range(non_timeout_count + 1)]
    cumulative_times = [t for t in times if t < max] + [max]

    plt.plot(cumulative_times, cumulative_percentages, marker='o')
    plt.axhline(y=(non_timeout_count / n) * 100, color='r', linestyle='--', label='Non-timeout percentage')
    plt.xlim(0, max)
    plt.ylim(0, 100)
    plt.xlabel('Time')
    plt.ylabel('% of Instances Solved')
    plt.title('Cumulative Solve Time Plot')
    plt.legend()
    plt.grid()
    plt.show()

def var_viz(cnf, values):
    cnf = open(cnf, 'r').read()
    lines = cnf.splitlines()
    w = []
    h_in = []
    h_out = []
    a = []

    getval = lambda x: values[x]

    for line in lines:
        if line.startswith('c var'):
            parts = line.split(' ')
            start = int(parts[2].split('/')[0])
            if parts[3].startswith('w'):
                w.append(list(map(getval, range(start, start + 32))))
            elif parts[3].startswith('h_in'):
                h_in.append(list(map(getval, range(start, start + 32))))
            elif  parts[3].startswith('h_out'):
                h_out.append(list(map(getval, range(start, start + 32))))
            elif  parts[3].startswith('a'):
                a.append(list(map(getval, range(start, start + 32))))
            else:
                print(parts)
        if line.startswith('c xor4'):
            break


    # Create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [len(w), len(h_in), len(h_out), len(a)]})

    # Plot w
    axs[0].imshow(np.array(w), cmap='inferno')
    axs[0].set_title('Message Schedule (w)')
    axs[0].set_xticks(range(0, 32, 2))
    axs[0].set_yticks(range(len(w)))

    # Plot h_in
    axs[1].imshow(np.array(h_in), cmap='inferno')
    axs[1].set_title('Input Hash (h_in)')
    axs[1].set_xticks(range(0, 32, 2))
    axs[1].set_yticks(range(len(h_in)))

    # Plot h_out
    axs[2].imshow(np.array(h_out), cmap='inferno')
    axs[2].set_title('Output Hash (h_out)')
    axs[2].set_xticks(range(0, 32, 2))
    axs[2].set_yticks(range(len(h_out)))

    # Plot a
    axs[3].imshow(np.array(a), cmap='inferno')
    axs[3].set_title('Auxiliary Variables (a)')
    axs[3].set_xticks(range(0, 32, 2))
    axs[3].set_yticks(range(len(a)))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    times = extract("logs/avg_solvetime_log.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 0)
    plt.hist(times, bins=20, range=(0, 200))
    plt.xlabel('Time')
    plt.ylabel('% of Instances Solved')
    plt.show()
    times = extract("logs/avg_solvetime_log.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 64)
    plt.hist(times, bins=20, range=(0, 100))
    plt.xlabel('Time')
    plt.ylabel('% of Instances Solved')
    plt.show()
    