# visualize.py
import matplotlib.pyplot as plt
import csv
import numpy as np

def plot_episode_curve(summary_csv="logs_episode_summary.csv"):
    eps = []
    rewards = []
    with open(summary_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
    plt.figure()
    plt.plot(eps, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward Curve")
    plt.grid(True)
    plt.show()

def plot_step_heatmap(step_csv="logs_step_logs.csv", max_state=10):
    # compute average reward per state-action
    import pandas as pd
    df = pd.read_csv(step_csv)
    # discretize state as people_count (already discrete)
    table = np.zeros((max_state+1, 3))
    counts = np.zeros((max_state+1, 3))
    for _, row in df.iterrows():
        s = int(min(int(row["people_count"]), max_state))
        a = int(row["action"])
        r = float(row["reward"])
        table[s, a] += r
        counts[s, a] += 1
    avg = np.divide(table, counts, out=np.zeros_like(table), where=counts>0)
    plt.figure(figsize=(8,6))
    plt.imshow(avg, cmap="viridis", aspect="auto")
    plt.colorbar(label="avg reward for state-action")
    plt.xlabel("Action (0=no-op,1=warn,2=emergency)")
    plt.ylabel("People count (state)")
    plt.title("Avg Reward per State-Action")
    plt.show()

if __name__ == "__main__":
    # adjust filenames if you passed save_prefix
    plot_episode_curve(summary_csv="logs_episode_summary.csv")
    plot_step_heatmap(step_csv="logs_step_logs.csv", max_state=10)
