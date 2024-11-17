# A/B Testing: Multi-Armed Bandit Experiment

## Structure of the Project

The project consists of the following files:

- **`Bandit.py`**:  
  Contains the implementation of the **Epsilon Greedy** and **Thompson Sampling** algorithms. These classes provide methods for selecting bandits, updating estimates, and conducting experiments. Additionally, the file includes visualization methods to compare cumulative rewards and regrets between the algorithms.

- **`experiment.py`**:  
  The main script to execute the experiments. It runs the Epsilon-Greedy and Thompson Sampling algorithms over a defined number of trials and generates visual outputs (cumulative rewards and regrets) along with saving the results in a CSV file.

- **`results/`**:  
  A directory where experiment outputs are saved:
  - **`bandit_results.csv`**: A CSV file storing detailed results of the experiment, including the selected bandit, rewards, and algorithm used.
  - **`cumulative_rewards.png`**: Visualization of cumulative rewards for both algorithms.
  - **`cumulative_regret.png`**: Visualization of cumulative regrets for both algorithms.

- **`requirements.txt`**:  
  A list of required Python dependencies for the project.

## Usage

- **`Step 1`**: Clone or Download the Repository
- **`Step 2`**: Set Up the Environment
- **`Step 3`**: Run the experiments.py

## Saving Experiment Output

The experiment outputs are saved for further analysis:

- **`CSV File`**: Includes details such as:
  Algorithm used (Epsilon-Greedy or Thompson Sampling)
  Selected bandit for each trial
  Reward obtained for each trial
  Trial number

- **`Visualizations`**:

  Cumulative Rewards: Tracks how rewards accumulate over time for each algorithm.
  Cumulative Regrets: Tracks the regret (difference between optimal and chosen rewards) over time for each algorithm.

## About

This repository is designed for implementing and experimenting with multi-armed bandit algorithms, specifically Epsilon-Greedy and Thompson Sampling. It provides:
  A foundation to explore the trade-offs between exploration and exploitation.
  Insights into how these algorithms perform in a simulated environment with defined bandit rewards.



