# AlphaZero Tic Tac Toe

This project implements an AI agent for playing Tic Tac Toe using the AlphaZero algorithm, which combines Monte Carlo Tree Search (MCTS) with a deep neural network.

**DISCLAIMER**: I implemented it to learn about reinforcement learning and it seems to work well enough but please take this implementation with a grain of salt :)

## Features

*   **AlphaZero Algorithm:** Trains a neural network through self-play and MCTS.
*   **MCTS Agent:** Uses Monte Carlo Tree Search guided by a neural network (policy and value heads).
*   **Multiple Modes:** Supports training the agent, testing it against a human player, and evaluating its performance against a predefined strategy.
*   **Configurable:** Offers numerous command-line arguments to control training parameters, network architecture, MCTS settings, and opponent strategies.
*   **Opponent Annealing:** Supports gradually changing the opponent mix during training (e.g., from random moves to more strategic ones) to potentially improve learning robustness.

## Installation

Ensure you have Python 3 installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```
*(You might need to create a `requirements.txt` file based on your imports: `gym`, `numpy`, `torch`, `matplotlib`)*

## Usage

The main script `main.py` is used for all operations. You select the operation using the `--mode` argument.

### 1. Training the Agent (`--mode train`)

This mode trains the AlphaZero neural network using self-play and optionally playing against fixed opponents (Random, StrategicRandom).

**Example:**

```sh
python main.py --mode train \
    --run_tag "initial_run" \
    --az_max_iterations 50 \
    --az_games_per_iteration 100 \
    --az_simulations_per_move 50 \
    --az_buffer_size 20000 \
    --az_batch_size 64 \
    --az_learning_rate 0.001 \
    --az_train_epochs_per_iteration 1 \
    --az_initial_strategic_ratio 0.0 --az_final_strategic_ratio 0.25 \
    --az_initial_random_ratio 0.0 --az_final_random_ratio 0.25 \
    --save_interval 10 \
    --eval_interval 5 \
    --eval_episodes 100 \
    --plot_path_base "plots/training_run" \
    --az_model_path_base "models/az_model"
```

*   This command trains for 50 iterations.
*   In each iteration, 100 games are played to generate data.
*   Each move uses 50 MCTS simulations.
*   The opponent mix gradually shifts from 0% Strategic/0% Random (i.e., 100% Self-Play) towards 25% Strategic/25% Random (50% Self-Play).
*   The model (`.pth`) is saved every 10 iterations to the `models/` directory.
*   Evaluation against the `StrategicRandomAgent` happens every 5 iterations.
*   Plots are saved to the `plots/` directory.

### 2. Testing vs. Human (`--mode test`)

This mode allows you to play against the trained agent. The agent will load the latest model found based on `--az_model_path_base` and `--run_tag`. Player roles (X or O) are assigned randomly.

**Example:**

```sh
python main.py --mode test \
    --az_model_path_base "models/az_model" \
    --run_tag "initial_run" \
    --az_simulations_per_move 100 # Use more simulations for stronger play during test
```

### 3. Evaluating Agent Performance (`--mode eval_mcts`)

This mode evaluates the performance of the loaded AlphaZero agent against the `StrategicRandomAgent` over a specified number of episodes.

**Example:**

```sh
python main.py --mode eval_mcts \
    --az_model_path_base "models/az_model" \
    --run_tag "initial_run" \
    --eval_episodes 200 \
    --az_simulations_per_move 100
```

## Configuration Options (Command-Line Arguments)

All arguments are passed to `main.py`.

**General:**

*   `--mode`: Operating mode: `train`, `test`, `eval_mcts`. (Default: `train`)
*   `--plot_path_base`: Base path for saving training plots. Run tag and iteration/episode count will be appended. (Default: `"plot"`)
*   `--run_tag`: A unique tag for this run, appended to output filenames (plots, models). (Default: `""`)

**Evaluation & Saving:**

*   `--save_interval`: (AZ Training) How many iterations between saving the network model. (Default: 10000, but set lower like 10 or 50 for practical training)
*   `--eval_interval`: (AZ Training) How many iterations between evaluating the agent against the strategic opponent. (Default: 1000, but set lower like 5 or 10)
*   `--eval_episodes`: Number of games to play during evaluation phases (both in AZ training and `eval_mcts` mode). (Default: 100)