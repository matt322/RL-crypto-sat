import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import t

def extract(file, condition, key="time"):
    res = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            if condition(line):
                res.append(line[key])
    return res

def cumulative_plot(data):
    plt.figure()
    xmax = 0
    for label, times, timeout in data:
        times = sorted(times)
        xmax = max(xmax, max(times))
        n = len(times)
        timeout_count = times.count(timeout)
        non_timeout_count = n - timeout_count

        cumulative_percentages = [(i / n) * 100 for i in range(n)]

        plt.plot(times[:non_timeout_count], cumulative_percentages[:non_timeout_count], marker='o', label=label, markersize=3)

    #plt.xlim(0, xmax)
    plt.xscale('log')
    plt.ylim(0, 100)
    plt.xlabel('Time')
    plt.ylabel('% of Instances Solved')
    plt.title('Cumulative Solve Time Plot')
    plt.legend()
    plt.grid()
    plt.show()

def var_viz(cnf, values, path=None, title=None, null_vals=[-float("inf"), 0.0]):
    cnf = open(cnf, 'r').read()
    lines = cnf.splitlines()
    w = []
    h_in = []
    h_out = []
    a = []

    

    getval = lambda x: values[x]
    filtered_vals = list(filter(lambda x: x not in null_vals, values))
    values = [i if i not in null_vals else np.nan for i in values]

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

    fig, axs = plt.subplots(2, 2, figsize=(14, 6), gridspec_kw={'height_ratios': [len(w), len(h_in)]})
    plt.subplots_adjust(left=-0.3)
    
    norm = plt.Normalize(vmin=min(filtered_vals), vmax=max(filtered_vals))
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.05)
    cbar.ax.set_position([0.05, 0.1, 0.50, 0.8])  # Adjust the position of the colorbar
    cbar.set_label('Value Range')
   
    # Plot w
    axs[0][0].imshow(np.array(w), cmap='inferno')
    axs[0][0].set_title('Message Schedule (w)')
    axs[0][0].set_xticks(range(0, 32, 2))
    axs[0][0].set_yticks(range(len(w)))


    # Plot h_in
    axs[1][0].imshow(np.array(h_in), cmap='inferno')
    axs[1][0].set_title('Input Hash (h_in)')
    axs[1][0].set_xticks(range(0, 32, 2))
    axs[1][0].set_yticks(range(len(h_in)))

    # Plot h_out
    axs[1][1].imshow(np.array(h_out), cmap='inferno')
    axs[1][1].set_title('Output Hash (h_out)')
    axs[1][1].set_xticks(range(0, 32, 2))
    axs[1][1].set_yticks(range(len(h_out)))

    # Plot a
    axs[0][1].imshow(np.array(a), cmap='inferno')
    axs[0][1].set_title('Auxiliary Variables (a)')
    axs[0][1].set_xticks(range(0, 32, 2))
    axs[0][1].set_yticks(range(len(a)))

    plt.tight_layout()
    if title:
        plt.suptitle(title)
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def EMA(data, d=0.9):
    res = [data[0]]
    for i in data[1:]:
        res.append(res[-1] * d + i * (1 - d))
    return res

def mean_and_conf(trajectories, confidence = 0.95):
    maxlen = max(map(lambda x: len(x), trajectories))
    data = [[x for x in t if x is not None] for t in zip(*map(lambda x: x + [None] * (maxlen - len(x)), trajectories))]
    mres, cres = [], []
    for i in data:
        z = t.ppf(1 - (1 - confidence) / 2, df=len(i)-1)
        mres.append(np.mean(i))
        cres.append(z * np.std(i, ddof=1) / np.sqrt(len(i)))
    return np.array(mres), np.array(cres)

def mean_and_std(trajectories):
    maxlen = max(map(lambda x: len(x), trajectories))
    data = [[x for x in t if x is not None] for t in zip(*map(lambda x: x + [None] * (maxlen - len(x)), trajectories))]
    mres, cres = [], []
    for i in data:
        mres.append(np.mean(i))
        cres.append(np.std(i, ddof=1) if len(i) > 1 else 0.0)
    return np.array(mres), np.array(cres)

def decision_efficiency(llrs, period=50000, title="Decision Efficiency", file=None):
    maxlen = max([len(i) for i in llrs])
    meanplot, std = mean_and_std(llrs)
    ax = plt.gca()
    times = range(0, period*maxlen, period)
    plt.plot(times, meanplot, color="green", label = "mean llr")
    plt.fill_between(times, meanplot-std, meanplot+std, color='green', alpha=0.2, label="std")
    total = len(llrs)
    unsolved_pct = [100.0 * sum(1 for traj in llrs if len(traj) > i) / total for i in range(maxlen)]

    # secondary axis for percentage unsolved
    ax2 = ax.twinx()
    ax2.plot(times, unsolved_pct, color='tab:blue', label='% unsolved', linewidth=2)
    ax2.set_ylabel('% Instances Unsolved')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y')

    # style primary axis (mean LLR)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Decisions')
    ax.set_ylabel('Mean LLR')
    ax.tick_params(axis='y')

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(title)
    plt.grid(True)
    if file:
        plt.savefig(file)
    else:
        plt.show()
    plt.close()



if __name__ == "__main__":
    llrs = list(map(lambda x: x[:-1], extract("logs/noadapt_vanilla_eval.jsonl", lambda x: True, key="llr")))
    llrs_refocused = list(map(lambda x: x[:-1], extract("logs/noadapt_refocus_eval_static_a=2.jsonl", lambda x: True, key="llr")))
    decision_efficiency(llrs, 50000, "No adaptation Vanilla Solver, 100 instances", file="vanilla_dec_eff.png")
    decision_efficiency(llrs_refocused, 50000, "No adaptation Periodic Refocusing, 100 instances", file="refocus_dec_eff.png")
    #exit()
    # vals = np.array(json.load(open("logs/logs/es_logs/1/es_model_0.jsonl"))["model"]).squeeze() 
    # #var_viz("cnf/sha1_21round.cnf", vals, title="")

    # rewards = np.array(extract("logs/logs/ppo_log_branching_test_withemb.jsonl", lambda x: True, key="cumulative_reward"))
    # stepcounts = np.array(extract("logs/logs/ppo_log_branching_test_withemb.jsonl", lambda x: True, key="steps"))
    # fitness = np.array(extract("logs/test_pop512s=0.05_ranktrans_1/log.jsonl", lambda x: True, key="return_mean"))
    # fitness = EMA(fitness)
    # plt.plot(fitness)
    # plt.xlabel("Step")
    # plt.ylabel("Mean Fitness")
    # plt.title("Evolution strategies mean reward over time")
    # plt.show()
    # exit()
    # plt.plot(rewards/stepcounts)
    # plt.xlabel("Episode")
    # plt.ylabel("Average Reward per Step")
    # plt.title("PPO Branching Heuristic Average Reward per Step")
    # plt.show()

    es_times = extract("logs/noadapt_refocus_eval_static_a=2.jsonl", lambda x: True)
    vanilla_times = extract("logs/noadapt_vanilla_eval.jsonl", lambda x: True)

    cumulative_plot([
        ("Vanilla Solver", vanilla_times, 200),
        ("ES Initialized Solver one instance", es_times, 200),
    ])
    plt.scatter(vanilla_times, es_times)
    plt.xscale('log')
    plt.yscale('log')
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

    ])

    with open("logs/ppo_log_test.jsonl", 'r') as f:
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

    