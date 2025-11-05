import matplotlib.pyplot as plt
import json
import numpy as np

def extract(file, condition, key="time"):
    res = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            if condition(line):
                res.append(line[key])
    return res

def cumulative_plot(data, xmax=200):
    plt.figure()
    
    for label, times, max in data:
        times = sorted(times)
        n = len(times)
        timeout_count = times.count(max)
        non_timeout_count = n - timeout_count

        cumulative_percentages = [(i / n) * 100 for i in range(n)]

        plt.plot(times[:non_timeout_count], cumulative_percentages[:non_timeout_count], marker='o', label=label, markersize=3)

    plt.xlim(0, xmax)
    plt.ylim(0, 100)
    plt.xlabel('Time')
    plt.ylabel('% of Instances Solved')
    plt.title('Cumulative Solve Time Plot')
    plt.legend()
    plt.grid()
    plt.show()

def var_viz(cnf, values, path, title):
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

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [len(w), len(h_in), len(h_out), len(a)]})
    
    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.set_position([0.05, 0.1, 0.02, 0.8])  # Adjust the position of the colorbar
    cbar.set_label('Value Range')
   
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
    plt.suptitle(title)
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    rewards = np.array(extract("logs/logs/ppo_log_test.jsonl", lambda x: True, key="cumulative_reward"))
    stepcounts = np.array(extract("logs/logs/ppo_log_test.jsonl", lambda x: True, key="steps"))
    fitness = np.array(extract("logs/es_logs/4/es_log_0.jsonl", lambda x: True, key="fitness_mean"))
    plt.plot(fitness)
    plt.xlabel("Step")
    plt.ylabel("Mean Fitness")
    plt.title("Evolution strategies mean reward over time")
    plt.show()

    plt.plot(rewards/stepcounts)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Step")
    plt.title("PPO Branching Heuristic Average Reward per Step")
    plt.show()

    exit()
    basetimes = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 0)
   # reset_times = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] != 0 and x["free_outputs"] == 0)
    times_16 = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 16)
    times_32 = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 32)
    times_64 = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 64)
    times_128 = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 128)
    times_96 = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 96)
   # times_64 = extract("logs/avg_solvetime_log_1.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 64)

    cumulative_plot([
        ("0 Free Outputs", basetimes, 200),
        ("16 Free Outputs", times_16, 200),
        ("32 Free Outputs", times_32, 200),
       ("64 Free Outputs", times_64, 200),
        ("128 Free Outputs", times_128, 200),
       ("96 Free Outputs", times_96, 200),

    ], xmax=200)

    with open("logs/episode_log.jsonl", 'r') as f:
        reward_plots = [json.loads(i)["rewards"] for i in f.readlines()]
    
    
    for i in reward_plots:
        plt.plot(range(1,len(i)-1), i[1:-1])
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Step rewards for several episodes")
    plt.show()
    
    with open("logs/episode_log.jsonl", 'r') as f:
        example_action = json.loads(f.readlines()[1])["first_action"]
    var_viz("cnf/sha1_21round.cnf", example_action)

    