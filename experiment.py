import os
from Bandit import EpsilonGreedy, ThompsonSampling, Visualization
import numpy as np
import csv
# Parameters
n_bandits = 4
bandit_rewards = [1, 2, 3, 4]
n_trials = 20000
epsilon_decay = 0.99

# Create results folder
output_folder = "results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def run_experiment(algorithm, n_trials):
    rewards = []
    cumulative_rewards = 0
    regret = []
    max_reward = max(bandit_rewards) * n_trials

    for _ in range(n_trials):
        bandit = algorithm.pull()
        reward = bandit_rewards[bandit] if np.random.rand() < 0.5 else 0
        algorithm.update(bandit, reward)

        rewards.append(reward)
        cumulative_rewards += reward
        regret.append(max_reward - cumulative_rewards)

    return rewards, regret, cumulative_rewards

# Initialize algorithms
epsilon_greedy = EpsilonGreedy(n_bandits=n_bandits, epsilon_decay=epsilon_decay)
thompson_sampling = ThompsonSampling(n_bandits=n_bandits)

# Run experiments
eg_rewards, eg_regret, eg_cumulative_reward = run_experiment(epsilon_greedy, n_trials)
ts_rewards, ts_regret, ts_cumulative_reward = run_experiment(thompson_sampling, n_trials)

# Visualization and saving images
Visualization.plot_cumulative_rewards(eg_rewards, ts_rewards, output_folder)
Visualization.plot_cumulative_regret(eg_regret, ts_regret, output_folder)

# Save results to CSV
csv_path = os.path.join(output_folder, "bandit_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "Trial", "Bandit", "Reward"])
    for i, reward in enumerate(eg_rewards):
        writer.writerow(["Epsilon-Greedy", i + 1, "Bandit", reward])
    for i, reward in enumerate(ts_rewards):
        writer.writerow(["Thompson Sampling", i + 1, "Bandit", reward])

# Print final cumulative rewards
print(f"Epsilon-Greedy Cumulative Reward: {eg_cumulative_reward}")
print(f"Thompson Sampling Cumulative Reward: {ts_cumulative_reward}")
print(f"Results saved in {output_folder}")
