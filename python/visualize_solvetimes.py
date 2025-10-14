import matplotlib.pyplot as plt
import json

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



if __name__ == "__main__":
    times = extract("logs/avg_solvetime_log.jsonl", lambda x: x["rounds"] == 21 and x["decisions/callback"] == 0 and x["free_outputs"] == 0)
    plt.hist(times, bins=20, range=(0, 200))
    plt.xlabel('Time')
    plt.ylabel('% of Instances Solved')
    plt.show()