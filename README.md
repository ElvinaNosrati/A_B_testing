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
