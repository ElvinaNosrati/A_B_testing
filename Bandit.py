"""
Run this file at first, in order to see what it is printing. Instead of print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import os

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.counts = [0] * n_bandits
        self.values = [0.0] * n_bandits

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__} with {self.n_bandits} bandits"

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass

#--------------------------------------#

class Visualization:
    @staticmethod
    def plot_cumulative_rewards(eg_rewards, ts_rewards, output_folder):
        """Plot and save cumulative rewards."""
        plt.figure()
        plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Cumulative Rewards for Epsilon-Greedy and Thompson Sampling")
        
        # Save the plot
        output_path = os.path.join(output_folder, "cumulative_rewards.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved cumulative rewards plot: {output_path}")

    @staticmethod
    def plot_cumulative_regret(eg_regret, ts_regret, output_folder):
        """Plot and save cumulative regret."""
        plt.figure()
        plt.plot(eg_regret, label="Epsilon-Greedy")
        plt.plot(ts_regret, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.title("Cumulative Regret for Epsilon-Greedy and Thompson Sampling")
        
        # Save the plot
        output_path = os.path.join(output_folder, "cumulative_regret.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved cumulative regret plot: {output_path}")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, n_bandits, epsilon_decay=0.99):
        super().__init__(n_bandits)
        self.epsilon = 1.0  
        self.epsilon_decay = epsilon_decay
    
    def pull(self):
        """Select a bandit using epsilon-greedy."""
        if random.random() < self.epsilon:
            bandit = random.randint(0, self.n_bandits - 1)
            logger.debug(f"Exploring: selected bandit {bandit}")
        else:
            bandit = max(range(self.n_bandits), key=lambda x: self.values[x])
            logger.debug(f"Exploiting: selected bandit {bandit}")
        return bandit

    def update(self, bandit, reward):
        """Update the values and decay epsilon."""
        self.counts[bandit] += 1
        self.values[bandit] += (reward - self.values[bandit]) / self.counts[bandit]
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
        logger.info(f"Updated bandit {bandit} with reward {reward}. New value: {self.values[bandit]}")

    def experiment(self, n_trials):
        rewards = []
        for _ in range(n_trials):
            bandit = self.pull()
            reward = random.choice([0, 1, 2, 3])  
            self.update(bandit, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = sum(self.values) / self.n_bandits
        logger.info(f"Average Reward: {avg_reward}")
    
    def __repr__(self):
        return f"EpsilonGreedy with {self.n_bandits} bandits"

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, n_bandits):
        super().__init__(n_bandits)
        self.successes = [1] * n_bandits
        self.failures = [1] * n_bandits

    def pull(self):
        """Select a bandit based on Thompson Sampling."""
        samples = [np.random.beta(self.successes[i], self.failures[i]) for i in range(self.n_bandits)]
        bandit = np.argmax(samples)
        logger.debug(f"Selected bandit {bandit} based on Thompson Sampling")
        return bandit

    def update(self, bandit, reward):
        """Update the success/failure counts."""
        if reward > 0:
            self.successes[bandit] += 1
        else:
            self.failures[bandit] += 1
        logger.info(f"Updated bandit {bandit} with reward {reward}. Successes: {self.successes[bandit]}, Failures: {self.failures[bandit]}")

    def experiment(self, n_trials):
        rewards = []
        for _ in range(n_trials):
            bandit = self.pull()
            reward = random.choice([0, 1, 2, 3])  # Simulated reward for simplicity
            self.update(bandit, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = sum(self.values) / self.n_bandits
        logger.info(f"Average Reward: {avg_reward}")

    def __repr__(self):
        return f"ThompsonSampling with {self.n_bandits} bandits"

#--------------------------------------#

def comparison(eg_rewards, ts_rewards, eg_regret, ts_regret):
    """Compare the performances of Epsilon-Greedy and Thompson Sampling visually."""
    
    # Plot cumulative rewards for both algorithms
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy Cumulative Reward")
    plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling Cumulative Reward")
    plt.xlabel("Trials")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.title("Comparison of Cumulative Rewards for Epsilon-Greedy and Thompson Sampling")
    plt.show()

    # Plot cumulative regret for both algorithms
    plt.figure(figsize=(12, 6))
    plt.plot(eg_regret, label="Epsilon-Greedy Cumulative Regret")
    plt.plot(ts_regret, label="Thompson Sampling Cumulative Regret")
    plt.xlabel("Trials")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.title("Comparison of Cumulative Regrets for Epsilon-Greedy and Thompson Sampling")
    plt.show()

#--------------------------------------#

if __name__=='__main__':
    # Parameters
    n_bandits = 4
    n_trials = 20000
    epsilon_decay = 0.99
    bandit_rewards = [1, 2, 3, 4]

    # Initialize algorithms
    epsilon_greedy = EpsilonGreedy(n_bandits=n_bandits, epsilon_decay=epsilon_decay)
    thompson_sampling = ThompsonSampling(n_bandits=n_bandits)

    # Run experiments
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

    eg_rewards, eg_regret, eg_cumulative_reward = run_experiment(epsilon_greedy, n_trials)
    ts_rewards, ts_regret, ts_cumulative_reward = run_experiment(thompson_sampling, n_trials)

    # Visualize comparison
    comparison(eg_rewards, ts_rewards, eg_regret, ts_regret)

    # Write results to CSV
    with open("bandit_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Trial", "Bandit", "Reward"])
        for i, reward in enumerate(eg_rewards):
            writer.writerow(["Epsilon-Greedy", i + 1, "Bandit", reward])
        for i, reward in enumerate(ts_rewards):
            writer.writerow(["Thompson Sampling", i + 1, "Bandit", reward])

    print(f"Epsilon-Greedy Cumulative Reward: {eg_cumulative_reward}")
    print(f"Thompson Sampling Cumulative Reward: {ts_cumulative_reward}")
