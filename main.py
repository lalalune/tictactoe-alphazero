import gym
import gym_TicTacToe
import numpy as np
import argparse
import torch
import time
import os
from typing import List, Optional, Dict, Tuple, Union # Added Union
import random
import matplotlib.pyplot as plt # For plotting
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict

from agent import MCTSAgent_AZ, AlphaZeroNet, get_board_representation
from utils import RandomAgent, StrategicRandomAgent

# --- Configuration --- #
# Base filenames, will be modified by run_tag
BASE_PLOT_FILENAME = "plot"

# Default number of training episodes
DEFAULT_TRAINING_EPISODES = 200000 # Can be adjusted
# MCTS Defaults
DEFAULT_MCTS_SIMULATIONS = 100
DEFAULT_MCTS_C_PARAM = 1.414
# Q-Learning Replay Buffer Defaults
DEFAULT_QL_REPLAY_BUFFER_SIZE = 10000
DEFAULT_QL_BATCH_SIZE = 64

DEFAULT_AZ_BATCH_SIZE=64
DEFAULT_AZ_MODEL_PATH_BASE="model.pth"

# Tabular Q-Learning with Replay Defaults
DEFAULT_QL_LEARNING_RATE = 0.1
DEFAULT_QL_DISCOUNT_FACTOR = 0.9
DEFAULT_QL_EPSILON = 1.0
DEFAULT_QL_EPSILON_DECAY = 0.99995 # Slower decay can be good with replay
DEFAULT_QL_MIN_EPSILON = 0.05

# Defaults for opponent mix annealing
DEFAULT_INITIAL_OPPONENT_MIX_RATIO = 0.0 # Start with 100% RandomAgent
DEFAULT_FINAL_OPPONENT_MIX_RATIO = 1.0   # End with 100% StrategicRandomAgent

# AlphaZero Training Defaults
DEFAULT_AZ_MAX_ITERATIONS = 50
DEFAULT_AZ_GAMES_PER_ITERATION = 100 
DEFAULT_AZ_SIMULATIONS_PER_MOVE = 25 # MCTS sims during self-play and choosing action
DEFAULT_AZ_C_PUCT = 1.0
DEFAULT_AZ_LEARNING_RATE = 0.001
DEFAULT_AZ_BUFFER_SIZE = 20000 
DEFAULT_AZ_TRAIN_EPOCHS_PER_ITERATION = 1 
DEFAULT_AZ_POLICY_LOSS_WEIGHT = 1.0
DEFAULT_AZ_TEMPERATURE_INITIAL = 1.0
DEFAULT_AZ_TEMPERATURE_FINAL = 0.01 
DEFAULT_AZ_TEMPERATURE_DECAY_ITERATIONS = 30 # Iterations over which temp decays to final

# New opponent mix annealing args
DEFAULT_AZ_INITIAL_STRATEGIC_RATIO = 0.0 # Start with 0% vs Strategic
DEFAULT_AZ_FINAL_STRATEGIC_RATIO = 0.25  # End with 25% vs Strategic
DEFAULT_AZ_INITIAL_RANDOM_RATIO = 0.0  # Start with 0% vs Random
DEFAULT_AZ_FINAL_RANDOM_RATIO = 0.25   # End with 25% vs Random

# --- Helper Functions --- #

def get_human_action(legal_actions: List[int]) -> int:
    """
    Prompts the human player for their move and validates it.

    Args:
        legal_actions (List[int]): The list of valid actions (0-8).

    Returns:
        int: The validated action chosen by the human.
    """
    while True:
        try:
            print(f"Available actions: {legal_actions}")
            action_str = input(f"Enter your move (0-8): ")
            action = int(action_str)
            if action in legal_actions:
                return action
            else:
                print(f"Invalid move. Action {action} is not in {legal_actions}. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {e}")

# --- Evaluation Function --- #

def evaluate_agent(env: gym.Env,
                     learning_agent: MCTSAgent_AZ,
                     opponent: Union[RandomAgent, StrategicRandomAgent],
                     num_episodes: int = 100,
                     # MCTS might need player ID if it were to play as P2
                     # but here it always plays as P1 for eval consistency.
                     ) -> Dict[str, float]:
    """
    Evaluates the learning_agent's performance against a fixed opponent.
    """
    stats = {"agent_wins": 0, "opponent_wins": 0, "draws": 0, "illegal": 0}
    original_epsilon = -1.0 # Placeholder

    agent_player_id = 1
    opponent_player_id = 2

    for _ in range(num_episodes):
        obs_array, info = env.reset()
        done = False

        while not done:
            current_player = info['current_player']
            legal_actions = info['legal_actions']

            if not legal_actions: break
            action = -1

            if current_player == agent_player_id:
                if isinstance(learning_agent, MCTSAgent_AZ):
                    action = learning_agent.choose_action(obs_array, current_player, legal_actions, temperature=0.01)
                else:
                    raise TypeError("Unsupported learning agent type in evaluate_agent.")
            else: # Opponent's turn
                if isinstance(opponent, StrategicRandomAgent):
                    action = opponent.choose_action(obs_array, legal_actions, opponent_player_id)
                elif isinstance(opponent, RandomAgent):
                    action = opponent.choose_action(legal_actions)
                else:
                    raise TypeError("Unsupported opponent type during evaluation.")

            if action == -1: break
            next_obs_array, _, done, info = env.step(action)
            obs_array = next_obs_array

        outcome = info.get("outcome", "N/A")
        if outcome == f"player_{agent_player_id}_wins": stats["agent_wins"] += 1
        elif outcome == "draw": stats["draws"] += 1
        elif outcome != "illegal_move": stats["opponent_wins"] += 1
        if outcome == "illegal_move" : stats["illegal"] +=1

    total_games = num_episodes
    rates = {
        "win_rate": stats["agent_wins"] / total_games if total_games else 0,
        "loss_rate": stats["opponent_wins"] / total_games if total_games else 0,
        "draw_rate": stats["draws"] / total_games if total_games else 0,
    }
    return rates

# --- MCTS "Training" (Evaluation against Strategic Agent) --- #
def train_mcts_evaluation_mode(env: gym.Env,
                               agent: MCTSAgent_AZ,
                               num_episodes: int,
                               opponent: StrategicRandomAgent):
    """
    "Trains" (evaluates) the MCTS agent by playing games against a strategic opponent.
    MCTS learns during its choose_action computation, not via explicit learn() calls between episodes.
    """
    print(f"--- MCTS Evaluation Mode ({num_episodes} episodes vs StrategicRandomAgent) ---")
    print(f"MCTS Params: Simulations={agent.num_simulations}, C={agent.c_puct}")
    
    stats = {"mcts_wins": 0, "opponent_wins": 0, "draws": 0, "illegal_moves": 0, "total_steps":0}
    start_time = time.time()

    mcts_player_id = 1 # MCTS plays as Player 1 for consistency
    opponent_player_id = 2

    for episode in range(1, num_episodes + 1):
        obs_array, info = env.reset()
        current_player = info['current_player']
        legal_actions = info['legal_actions']
        done = False
        episode_steps = 0

        while not done:
            if not legal_actions: break
            action = -1

            if current_player == mcts_player_id:
                action = agent.choose_action(obs_array, current_player, legal_actions)
            else: # Opponent's turn
                action = opponent.choose_action(obs_array, legal_actions, opponent_player_id)
            
            if action == -1: break # Should not happen
            obs_array, _, done, info = env.step(action)
            current_player = info['current_player']
            legal_actions = info.get('legal_actions', [])
            episode_steps +=1

        # Record outcome
        outcome = info.get("outcome", "N/A")
        stats["total_steps"] += episode_steps
        if outcome == f"player_{mcts_player_id}_wins": stats["mcts_wins"] += 1
        elif outcome == "draw": stats["draws"] += 1
        elif outcome == "illegal_move": stats["illegal_moves"] +=1
        else: stats["opponent_wins"] += 1

        if episode % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"  Episode {episode}/{num_episodes} completed...")

    end_time = time.time()
    total_games = num_episodes
    win_rate = stats["mcts_wins"] / total_games if total_games else 0
    loss_rate = stats["opponent_wins"] / total_games if total_games else 0
    draw_rate = stats["draws"] / total_games if total_games else 0
    avg_steps = stats["total_steps"] / total_games if total_games else 0

    print(f"--- MCTS Evaluation Finished ({time.time() - start_time:.2f}s) ---")
    print(f"  Results (vs Strategic): MCTS Wins: {win_rate:.2%}, Opponent Wins: {loss_rate:.2%}, Draws: {draw_rate:.2%}")
    print(f"  Avg steps/game: {avg_steps:.2f}. Illegal moves by MCTS/Opponent: {stats['illegal_moves']}")

# --- Testing Function (Human vs Agent) --- #

def test_human_vs_agent(env: gym.Env, agent: MCTSAgent_AZ, agent_player_choice: Optional[int] = None):
    """
    Allows a human player to play against the trained agent.
    If agent_player_choice is not specified, the human player is randomly
    assigned to be Player 1 (X, goes first) or Player 2 (O, goes second).
    """
    human_player_id = -1
    agent_player_id = -1

    if agent_player_choice is not None and agent_player_choice in [1, 2]:
        agent_player_id = agent_player_choice
        human_player_id = 1 if agent_player_id == 2 else 2
        print(f"Agent assigned Player {agent_player_id} by argument.")
    else:
        # Randomly assign human player if no specific assignment is given
        human_player_id = random.choice([1, 2])
        agent_player_id = 1 if human_player_id == 2 else 2
        print("Player roles assigned randomly.")

    print(f"--- Starting Human vs Agent Test ---")
    print(f"Human is Player {human_player_id} ({'X' if human_player_id == 1 else 'O'})" + (" (goes first)" if human_player_id == 1 else " (goes second)"))
    print(f"Agent is Player {agent_player_id} ({'X' if agent_player_id == 1 else 'O'})" + (" (goes first)" if agent_player_id == 1 else " (goes second)"))

    obs_array, info = env.reset()
    done = False
    env.render(mode='human')

    while not done:
        current_player = info['current_player']
        legal_actions = info['legal_actions']
        if not legal_actions: print("No legal actions available."); break

        print(f"\nPlayer {current_player}'s turn ({'X' if current_player == 1 else 'O'})")
        action = -1

        if current_player == agent_player_id:
            print("Agent is thinking...")
            # Ensure obs_array and current_player are correctly passed to agent.choose_action
            if isinstance(agent, MCTSAgent_AZ):
                action = agent.choose_action(obs_array, current_player, legal_actions, temperature=0.01)
            else: raise TypeError("Unsupported agent type for testing.")
            print(f"Agent (Player {agent_player_id}) chose action: {action}")
        else: # Human player's turn
            action = get_human_action(legal_actions)
            print(f"Human (Player {human_player_id}) chose action: {action}")
        
        if action == -1: print("Error: Action not set."); break
        
        next_obs_array, _, done, info = env.step(action)
        obs_array = next_obs_array
        print("\nBoard after move:")
        env.render(mode='human')

        if done:
            print("\n--- Game Over ---")
            outcome = info.get('outcome', 'Unknown outcome')
            print(f"Outcome: {outcome}")
            if "wins" in outcome:
                winner_id = int(outcome.split('_')[1])
                if winner_id == agent_player_id: print("Agent wins!")
                elif winner_id == human_player_id: print("Human wins!")
                else: print("Error determining winner from outcome string.")
            elif outcome == "draw": print("It's a draw!")
            elif outcome == "illegal_move": print("Game ended due to illegal move.")
            break 

def plot_training_history(history: Dict[str, List], episodes: int, run_tag: str, plot_basename: str = "training_plots"):
    fig, axs = plt.subplots(5, 1, figsize=(12, 26), sharex=True) # Increased to 5 subplots
    axs[0].plot(history['episodes'], history['epsilon'], label='Epsilon', color='cyan')
    axs[0].set_ylabel("Epsilon"); axs[0].set_title("Epsilon Decay"); axs[0].grid(True)
    axs[1].plot(history['episodes'], history['eval_win_rate'], label='Eval Win Rate', color='green', marker='o', linestyle='--')
    axs[1].plot(history['episodes'], history['eval_loss_rate'], label='Eval Loss Rate', color='red', marker='o', linestyle='--')
    axs[1].plot(history['episodes'], history['eval_draw_rate'], label='Eval Draw Rate', color='blue', marker='o', linestyle='--')
    axs[1].set_ylabel("Rate"); axs[1].set_title("Evaluation vs Strategic Agent"); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(history['episodes'], history['avg_training_reward'], label='Avg Training Reward per Interval', color='purple', marker='.')
    axs[2].set_ylabel("Avg Reward"); axs[2].set_title("Avg Training Reward (Agent vs Strategic)"); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(history['episodes'], history['buffer_size'], label='Replay Buffer Size', color='orange')
    axs[3].set_ylabel("Buffer Size"); axs[3].set_title("Replay Buffer Size"); axs[3].grid(True)
    axs[4].plot(history['episodes'], history['current_opponent_mix_ratio'], label='Strategic Opponent Mix Ratio', color='magenta')
    axs[4].set_xlabel("Episodes"); axs[4].set_ylabel("Mix Ratio (Strategic)"); axs[4].set_title("Annealed Opponent Mix Ratio"); axs[4].grid(True); axs[4].set_ylim(0, 1.1)
    plt.tight_layout()
    plot_save_path = f"{plot_basename}_e{episodes}{'_' + run_tag if run_tag else ''}.png"
    plt.savefig(plot_save_path); print(f"Training plots saved to {plot_save_path}")

# --- AlphaZero Training Function --- #
def train_alphazero(
    env: gym.Env, 
    network: AlphaZeroNet, 
    args: argparse.Namespace, 
    device: torch.device
):
    print(f"--- Starting AlphaZero Training ({args.az_max_iterations} iterations) ---")
    print(f"  Pure Self-Play. MCTS sims/move: {args.az_simulations_per_move}")
    print(f"  Strategic Opponent Mix: {args.az_initial_strategic_ratio*100:.0f}% -> {args.az_final_strategic_ratio*100:.0f}%")
    print(f"  Random Opponent Mix: {args.az_initial_random_ratio*100:.0f}% -> {args.az_final_random_ratio*100:.0f}%")
    optimizer = optim.AdamW(network.parameters(), lr=args.az_learning_rate, weight_decay=1e-4)
    value_loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=args.az_buffer_size)
    history = {
        'iteration': [], 'avg_policy_loss': [], 'avg_value_loss': [], 'total_loss': [],
        'eval_win_rate': [], 'eval_loss_rate': [], 'eval_draw_rate': [],
        'buffer_fill_count': [], 'game_length_avg': [],
        'self_play_p1_wins': [], 'self_play_p2_wins': [], 'self_play_draws': [], 
        'vs_strategic_wins': [], 'vs_strategic_losses': [], 'vs_strategic_draws': [],
        'vs_random_wins': [], 'vs_random_losses': [], 'vs_random_draws': [],
        'current_strategic_ratio': [], 'current_random_ratio': []
    }

    az_player = MCTSAgent_AZ(network, num_simulations=args.az_simulations_per_move, c_puct=args.az_c_puct, device=device)
    strategic_opponent = StrategicRandomAgent()
    random_opponent = RandomAgent()

    # Temperature annealing for self-play action selection
    if args.az_temperature_decay_iterations <= 0: temp_decay_rate = 1.0
    else: temp_decay_rate = (args.az_temperature_final / args.az_temperature_initial) ** (1.0 / args.az_temperature_decay_iterations)
    current_temp = args.az_temperature_initial

    for iteration in range(1, args.az_max_iterations + 1):
        iteration_start_time = time.time()
        print(f"\n--- Iteration {iteration}/{args.az_max_iterations} --- Self-Play Temp: {current_temp:.3f} ---")

        # Anneal opponent mix ratios
        progress = (iteration -1) / max(1, args.az_max_iterations - 1) # from 0 to 1
        current_strategic_ratio = args.az_initial_strategic_ratio + (args.az_final_strategic_ratio - args.az_initial_strategic_ratio) * progress
        current_random_ratio = args.az_initial_random_ratio + (args.az_final_random_ratio - args.az_initial_random_ratio) * progress
        history['current_strategic_ratio'].append(current_strategic_ratio)
        history['current_random_ratio'].append(current_random_ratio)

        print(f"  Data Gen: Strategic Mix={current_strategic_ratio:.2%}, Random Mix={current_random_ratio:.2%}, Self-Play Mix={(1-current_strategic_ratio-current_random_ratio):.2%}")
        network.eval()
        iteration_game_data = []
        iter_stats = defaultdict(int) # e.g. iter_stats['self_p1_wins'], iter_stats['az_vs_strat_wins']

        for game_num in range(1, args.az_games_per_iteration + 1):
            current_game_history = []
            obs_array, info = env.reset()
            done = False
            game_steps = 0
            first_move_of_game_by_p1_for_random_opening = True 

            # Determine game type for this game
            rand_val = random.random()
            game_opponent = None # None for self-play
            game_type_log = "SelfPlay"
            az_agent_is_p1 = True # In mixed games, AZ agent is always P1

            if rand_val < current_strategic_ratio:
                game_opponent = strategic_opponent
                game_type_log = "AZ_vs_Strategic"
            elif rand_val < current_strategic_ratio + current_random_ratio:
                game_opponent = random_opponent
                game_type_log = "AZ_vs_Random"
            # Else: game_opponent remains None for self-play

            while not done:
                board_2d = obs_array.reshape(3, 3)
                player_turn = info['current_player']
                legal_actions = info.get('legal_actions', [])
                if not legal_actions: break
                action = -1

                if game_opponent is None: # Self-play
                    mcts_policy, _ = az_player.get_action_probs(board_2d, player_turn, temperature=current_temp)
                    current_game_history.append((board_2d, player_turn, mcts_policy))
                    if player_turn == 1 and first_move_of_game_by_p1_for_random_opening:
                        action = random.choice(legal_actions)
                        first_move_of_game_by_p1_for_random_opening = False
                    else:
                        action = az_player.choose_action(obs_array, player_turn, legal_actions, temperature=current_temp)
                else: # AZ vs fixed opponent (AZ is P1)
                    if player_turn == 1: # AZ Agent's turn
                        mcts_policy, _ = az_player.get_action_probs(board_2d, player_turn, temperature=current_temp)
                        current_game_history.append((board_2d, player_turn, mcts_policy))
                        if first_move_of_game_by_p1_for_random_opening: # Still apply random first move for AZ agent
                            action = random.choice(legal_actions)
                            first_move_of_game_by_p1_for_random_opening = False
                        else:
                            action = az_player.choose_action(obs_array, player_turn, legal_actions, temperature=current_temp)
                    else: # Fixed opponent's turn (P2)
                        if isinstance(game_opponent, StrategicRandomAgent):
                            action = game_opponent.choose_action(obs_array, legal_actions, player_id=2)
                        else: # RandomAgent
                            action = game_opponent.choose_action(legal_actions)
                
                if action == -1: info['outcome'] = 'error_action_not_taken'; break
                next_obs_array, _, done, info = env.step(action)
                obs_array = next_obs_array; game_steps += 1
            
            # Update iter_stats based on final_outcome and game_type_log
            final_outcome = info.get('outcome', 'unknown')
            game_result_p1 = 0.0
            if final_outcome == 'player_1_wins': game_result_p1 = 1.0
            elif final_outcome == 'player_2_wins': game_result_p1 = -1.0

            if game_opponent is None: # Self-play
                if final_outcome == 'player_1_wins': iter_stats['self_play_p1_wins'] +=1
                elif final_outcome == 'player_2_wins': iter_stats['self_play_p2_wins'] +=1
                elif final_outcome == 'draw': iter_stats['self_play_draws'] +=1
            elif isinstance(game_opponent, StrategicRandomAgent):
                if final_outcome == 'player_1_wins': iter_stats['vs_strategic_wins'] +=1
                elif final_outcome == 'player_2_wins': iter_stats['vs_strategic_losses'] +=1
                elif final_outcome == 'draw': iter_stats['vs_strategic_draws'] +=1
            elif isinstance(game_opponent, RandomAgent):
                if final_outcome == 'player_1_wins': iter_stats['vs_random_wins'] +=1
                elif final_outcome == 'player_2_wins': iter_stats['vs_random_losses'] +=1
                elif final_outcome == 'draw': iter_stats['vs_random_draws'] +=1

            for board_s, player_s_at_turn, policy_s in current_game_history:
                value_target = game_result_p1 if player_s_at_turn == 1 else -game_result_p1
                board_repr_s = get_board_representation(board_s, player_s_at_turn)
                iteration_game_data.append((board_repr_s, policy_s, value_target))

        replay_buffer.extend(iteration_game_data)
        history['buffer_fill_count'].append(len(replay_buffer))
        avg_game_len_iter = game_steps / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0
        history['game_length_avg'].append(avg_game_len_iter)
        history['self_play_p1_wins'].append(iter_stats['self_play_p1_wins'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['self_play_p2_wins'].append(iter_stats['self_play_p2_wins'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['self_play_draws'].append(iter_stats['self_play_draws'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['vs_strategic_wins'].append(iter_stats['vs_strategic_wins'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['vs_strategic_losses'].append(iter_stats['vs_strategic_losses'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['vs_strategic_draws'].append(iter_stats['vs_strategic_draws'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['vs_random_wins'].append(iter_stats['vs_random_wins'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['vs_random_losses'].append(iter_stats['vs_random_losses'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        history['vs_random_draws'].append(iter_stats['vs_random_draws'] / args.az_games_per_iteration if args.az_games_per_iteration > 0 else 0)
        print(f"  Self-play finished. {len(iteration_game_data)} states added. Buffer: {len(replay_buffer)}. Avg game len: {avg_game_len_iter:.2f}")
        print(f"  Self-play outcomes this iter: P1 Wins: {iter_stats['self_play_p1_wins']}, P2 Wins: {iter_stats['self_play_p2_wins']}, Draws: {iter_stats['self_play_draws']}")

        # --- Network Training Phase (remains largely the same) --- # 
        if len(replay_buffer) >= args.az_batch_size:
            print(f"  Training network for {args.az_train_epochs_per_iteration} epochs on {len(replay_buffer)} samples...")
            network.train() 
            current_iter_policy_losses = []
            current_iter_value_losses = []
            current_iter_total_losses = []
            for epoch in range(args.az_train_epochs_per_iteration):
                num_batches = len(replay_buffer) // args.az_batch_size
                epoch_p_loss, epoch_v_loss, epoch_tot_loss = 0,0,0
                num_batches_to_run = num_batches

                for batch_idx in range(num_batches_to_run):
                    mini_batch = random.sample(replay_buffer, args.az_batch_size)
                    batch_states_np = np.array([exp[0] for exp in mini_batch])
                    batch_policy_targets_np = np.array([exp[1] for exp in mini_batch])
                    batch_value_targets_np = np.array([exp[2] for exp in mini_batch])

                    states_tensor = torch.from_numpy(batch_states_np).float().to(device)
                    policy_targets_tensor = torch.from_numpy(batch_policy_targets_np).float().to(device)
                    value_targets_tensor = torch.from_numpy(batch_value_targets_np).float().unsqueeze(1).to(device)

                    optimizer.zero_grad()
                    policy_logits_pred, value_pred = network(states_tensor)
                    
                    # Policy loss: Changed to MSE between softmax of logits and target probabilities
                    loss_p = F.mse_loss(F.softmax(policy_logits_pred, dim=1), policy_targets_tensor)
                    
                    loss_v = value_loss_fn(value_pred, value_targets_tensor)
                    total_loss = args.az_policy_loss_weight * loss_p + loss_v
                    
                    total_loss.backward()
                    optimizer.step()
                    epoch_p_loss += loss_p.item()
                    epoch_v_loss += loss_v.item()
                    epoch_tot_loss += total_loss.item()
                
                if num_batches_to_run > 0:
                    avg_p_loss = epoch_p_loss/num_batches_to_run
                    avg_v_loss = epoch_v_loss/num_batches_to_run
                    avg_total_loss = epoch_tot_loss/num_batches_to_run
                    print(f"    Epoch {epoch+1}/{args.az_train_epochs_per_iteration} | Avg P Loss: {avg_p_loss:.4f}, Avg V Loss: {avg_v_loss:.4f}, Avg Tot Loss: {avg_total_loss:.4f}")
                    current_iter_policy_losses.append(avg_p_loss)
                    current_iter_value_losses.append(avg_v_loss)
                    current_iter_total_losses.append(avg_total_loss)
            if current_iter_policy_losses: history['avg_policy_loss'].append(np.mean(current_iter_policy_losses))
            else: history['avg_policy_loss'].append(float('nan'))
            if current_iter_value_losses: history['avg_value_loss'].append(np.mean(current_iter_value_losses))
            else: history['avg_value_loss'].append(float('nan'))
            if current_iter_total_losses: history['total_loss'].append(np.mean(current_iter_total_losses))
            else: history['total_loss'].append(float('nan'))
        else:
            print("  Skipping network training: replay buffer too small.")
            history['avg_policy_loss'].append(float('nan')); history['avg_value_loss'].append(float('nan')); history['total_loss'].append(float('nan'))

        # --- Evaluation Phase (remains the same, vs StrategicRandomAgent) --- #
        if iteration % args.eval_interval == 0 or iteration == args.az_max_iterations:
            print("  Evaluating current network...")
            eval_rates = evaluate_agent(env, az_player, strategic_opponent, args.eval_episodes)
            history['iteration'].append(iteration)
            history['eval_win_rate'].append(eval_rates['win_rate'])
            history['eval_loss_rate'].append(eval_rates['loss_rate'])
            history['eval_draw_rate'].append(eval_rates['draw_rate'])
            print(f"  Evaluation vs Strategic => Wins: {eval_rates['win_rate']:.2%}, Losses: {eval_rates['loss_rate']:.2%}, Draws: {eval_rates['draw_rate']:.2%}")
        else: # Keep history arrays aligned if no eval this iteration
            if iteration not in history['iteration']: history['iteration'].append(iteration) # Should always append iteration number
            history['eval_win_rate'].append(float('nan'))
            history['eval_loss_rate'].append(float('nan'))
            history['eval_draw_rate'].append(float('nan'))

        # --- Save Model & Plot (remains the same) --- #
        if iteration % args.save_interval == 0 or iteration == args.az_max_iterations:
            model_save_path = f"{args.az_model_path_base}_iter{iteration}{'_' + args.run_tag if args.run_tag else ''}.pth"
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(network.state_dict(), model_save_path)
            print(f"AlphaZeroNet saved to {model_save_path}")
        
        # Anneal temperature for self-play for next iteration
        if iteration < args.az_temperature_decay_iterations:
             current_temp *= temp_decay_rate
        else:
             current_temp = args.az_temperature_final # Keep at final temp
        current_temp = max(current_temp, args.az_temperature_final)

        print(f"--- Iteration {iteration} finished in {time.time() - iteration_start_time:.2f}s ---")

    # Final plot after all iterations
    plot_alphazero_training_history(history, args.az_max_iterations, args.run_tag, args.plot_path_base)
    return history

# --- Placeholder for AlphaZero Plotting --- #
def plot_alphazero_training_history(history: Dict[str, List], total_iterations: int, run_tag: str, plot_basename: str):
    """Plots key metrics from the AlphaZero training history."""
    if not history or not history.get('iteration'):
        print("No data in history to plot for AlphaZero.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)
    iterations = history['iteration']

    # Plot Losses (Policy and Value)
    axs[0].plot(iterations, history['avg_policy_loss'], label='Avg Policy Loss', color='blue', marker='o', linestyle='-')
    axs[0].plot(iterations, history['avg_value_loss'], label='Avg Value Loss', color='red', marker='o', linestyle='-')
    if 'total_loss' in history and any(not np.isnan(x) for x in history['total_loss']): # Plot if not all NaN
        axs[0].plot(iterations, history['total_loss'], label='Avg Total Loss', color='purple', marker='x', linestyle=':')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Network Training Losses per Iteration")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Evaluation Performance (Win/Loss/Draw vs Strategic Agent)
    axs[1].plot(iterations, history['eval_win_rate'], label='Eval Win Rate (vs Strat)', color='green', marker='s', linestyle='--')
    axs[1].plot(iterations, history['eval_loss_rate'], label='Eval Loss Rate (vs Strat)', color='orange', marker='s', linestyle='--')
    axs[1].plot(iterations, history['eval_draw_rate'], label='Eval Draw Rate (vs Strat)', color='teal', marker='s', linestyle='--')
    axs[1].set_ylabel("Rate")
    axs[1].set_title("Evaluation Performance vs Strategic Agent")
    axs[1].legend()
    axs[1].grid(True)

    # Plot Replay Buffer Fill Count
    axs[2].plot(iterations, history['buffer_fill_count'], label='Replay Buffer Fill Count', color='brown', marker='.')
    axs[2].set_ylabel("Number of Samples")
    axs[2].set_title("Replay Buffer Size Over Iterations")
    axs[2].legend()
    axs[2].grid(True)

    # Plot Average Game Length during Self-Play
    axs[3].plot(iterations, history['game_length_avg'], label='Avg Game Length (Self-Play)', color='magenta', marker='.')
    axs[3].set_xlabel("Training Iteration")
    axs[3].set_ylabel("Average Steps")
    axs[3].set_title("Average Game Length in Self-Play")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plot_save_path = f"{plot_basename}_az_iters{total_iterations}{'_' + run_tag if run_tag else ''}.png"
    try:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path)
        print(f"AlphaZero training plots saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving AlphaZero plots to {plot_save_path}: {e}")
    # plt.show() # Uncomment to display plots interactively

# --- Main Execution Logic --- #

def main():
    parser = argparse.ArgumentParser(description="Train or test a Tic Tac Toe RL agent.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'eval_mcts'], # Added 'eval_mcts'
                        help='Operating mode: train, test (human vs agent), eval_mcts (MCTS vs StrategicRandomAgent).')
    parser.add_argument('--episodes', type=int, default=DEFAULT_TRAINING_EPISODES)
    parser.add_argument('--plot_path_base', type=str, default=BASE_PLOT_FILENAME, 
                        help='Base path for saving training plots. Run tag and episode count will be appended.')
    parser.add_argument('--run_tag', type=str, default="", 
                        help='A unique tag for this run, appended to output filenames (e.g., for hyperparameter sweeps).')
    
    # Tabular Q-Learning specific args - Re-add replay buffer args
    parser.add_argument('--lr', type=float, default=DEFAULT_QL_LEARNING_RATE, help='Learning rate (alpha).')
    parser.add_argument('--gamma', type=float, default=DEFAULT_QL_DISCOUNT_FACTOR, help='Discount factor.')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_QL_EPSILON, help='Initial exploration rate.')
    parser.add_argument('--epsilon_decay', type=float, default=DEFAULT_QL_EPSILON_DECAY, help='Exploration rate decay.')
    parser.add_argument('--min_epsilon', type=float, default=DEFAULT_QL_MIN_EPSILON, help='Minimum exploration rate.')
    parser.add_argument('--replay_buffer_size', type=int, default=DEFAULT_QL_REPLAY_BUFFER_SIZE,
                        help='Size of the experience replay buffer.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_QL_BATCH_SIZE,
                        help='Batch size for sampling from replay buffer.')

    # MCTS specific args
    parser.add_argument('--num_simulations', type=int, default=DEFAULT_MCTS_SIMULATIONS)
    parser.add_argument('--mcts_c_param', type=float, default=DEFAULT_MCTS_C_PARAM)
    
    # Training control args
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--eval_episodes', type=int, default=100)

    # New opponent mix annealing args
    parser.add_argument('--initial_opponent_mix_ratio', type=float, default=DEFAULT_INITIAL_OPPONENT_MIX_RATIO,
                        help='Initial probability (0-1) of playing StrategicRandomAgent (vs RandomAgent).')
    parser.add_argument('--final_opponent_mix_ratio', type=float, default=DEFAULT_FINAL_OPPONENT_MIX_RATIO,
                        help='Final probability (0-1) of playing StrategicRandomAgent at the end of training.')

    # AlphaZero (MCTS with NN) specific args
    parser.add_argument('--az_model_path_base', type=str, default=DEFAULT_AZ_MODEL_PATH_BASE, help="Base path for AlphaZero network model (.pth).")
    parser.add_argument('--az_max_iterations', type=int, default=DEFAULT_AZ_MAX_ITERATIONS)
    parser.add_argument('--az_games_per_iteration', type=int, default=DEFAULT_AZ_GAMES_PER_ITERATION)
    parser.add_argument('--az_simulations_per_move', type=int, default=DEFAULT_AZ_SIMULATIONS_PER_MOVE)
    parser.add_argument('--az_c_puct', type=float, default=DEFAULT_AZ_C_PUCT, help="Exploration constant c_puct for PUCT in MCTS-AZ.")
    parser.add_argument('--az_learning_rate', type=float, default=DEFAULT_AZ_LEARNING_RATE, help="Learning rate for AlphaZero network.")
    parser.add_argument('--az_buffer_size', type=int, default=DEFAULT_AZ_BUFFER_SIZE, help="Max size of replay buffer for (s,pi,z) tuples.")
    parser.add_argument('--az_batch_size', type=int, default=DEFAULT_AZ_BATCH_SIZE, help="Minibatch size for training AlphaZero network.")
    parser.add_argument('--az_train_epochs_per_iteration', type=int, default=DEFAULT_AZ_TRAIN_EPOCHS_PER_ITERATION, help="Epochs to train NN per AZ iteration.")
    parser.add_argument('--az_policy_loss_weight', type=float, default=DEFAULT_AZ_POLICY_LOSS_WEIGHT, help="Weight for policy loss in AlphaZero total loss.")
    parser.add_argument('--az_temperature_initial', type=float, default=DEFAULT_AZ_TEMPERATURE_INITIAL, help="Initial temperature for MCTS policy sampling during self-play.")
    parser.add_argument('--az_temperature_final', type=float, default=DEFAULT_AZ_TEMPERATURE_FINAL, help="Final temperature for MCTS policy sampling.")
    parser.add_argument('--az_temperature_decay_iterations', type=int, default=DEFAULT_AZ_TEMPERATURE_DECAY_ITERATIONS, help="Iterations to decay temperature to its final value.")
    parser.add_argument('--az_initial_strategic_ratio', type=float, default=DEFAULT_AZ_INITIAL_STRATEGIC_RATIO)
    parser.add_argument('--az_final_strategic_ratio', type=float, default=DEFAULT_AZ_FINAL_STRATEGIC_RATIO)
    parser.add_argument('--az_initial_random_ratio', type=float, default=DEFAULT_AZ_INITIAL_RANDOM_RATIO)
    parser.add_argument('--az_final_random_ratio', type=float, default=DEFAULT_AZ_FINAL_RANDOM_RATIO)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("TTT-v0")
    ACTION_SIZE = env.action_space.n

    agent_for_eval_or_test: Optional[MCTSAgent_AZ] = None
    az_network_for_train_or_test: Optional[AlphaZeroNet] = None 

    print("Initializing AlphaZero Agent (MCTS with NN)...")
    az_network_for_train_or_test = AlphaZeroNet(input_channels=2, board_height=3, board_width=3, num_actions=ACTION_SIZE).to(device)
    
    # Construct model path for loading.
    # If run_tag is specified, it forms part of the primary name. Otherwise, it's just the base name.
    # Iteration number is expected to be part of az_model_path_base if loading a specific iteration outside training loop.
    model_load_path = f"{args.az_model_path_base}{'_' + args.run_tag if args.run_tag else ''}.pth"
    
    if os.path.exists(model_load_path):
        print(f"Loading existing AlphaZeroNet model from: {model_load_path}")
        try:
            az_network_for_train_or_test.load_state_dict(torch.load(model_load_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {model_load_path}: {e}. Using fresh network.")
    else:
        print(f"No AlphaZeroNet model found at {model_load_path}. Using fresh (untrained) network.")
    
    agent_for_eval_or_test = MCTSAgent_AZ(network=az_network_for_train_or_test, 
                                            num_simulations=args.az_simulations_per_move, 
                                            c_puct=args.az_c_puct, 
                                            device=device)
    if args.mode == 'train':

        if az_network_for_train_or_test is None:
            print("Error: AlphaZero Network not initialized for training."); env.close(); return
        az_training_history = train_alphazero(env, az_network_for_train_or_test, args, device)
        if az_training_history:
                plot_alphazero_training_history(az_training_history, args.az_max_iterations, args.run_tag, args.plot_path_base)

    elif args.mode == 'test':
        if not agent_for_eval_or_test: print("Error: Agent could not be initialized for testing."); env.close(); return

        if agent_for_eval_or_test: test_human_vs_agent(env, agent_for_eval_or_test)
        else: print("Cannot start test: Agent is not initialized.")

    elif args.mode == 'eval_mcts':
        print(f"\n--- Evaluating AlphaZero MCTS Agent vs StrategicRandomAgent ({args.eval_episodes} games) ---")
        strategic_opponent = StrategicRandomAgent()
        eval_results = evaluate_agent(env, agent_for_eval_or_test, strategic_opponent, args.eval_episodes)
        print(f"  MCTS Eval vs Strategic => Wins: {eval_results['win_rate']:.2%}, Losses: {eval_results['loss_rate']:.2%}, Draws: {eval_results['draw_rate']:.2%}")

    env.close()
    print("\nProgram finished.")

if __name__ == "__main__":
    main() 