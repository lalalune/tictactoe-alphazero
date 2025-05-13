import gym
import numpy as np
import math

from typing import Tuple, List, Dict, Any


class TicTacToeEnv(gym.Env):
    """
    TicTacToe Environment based on OpenAI Gym standards.

    The game state is represented as a flattened numpy array of size 9, where:
    - 0 represents an empty cell
    - 1 represents Player 1's marker (X)
    - 2 represents Player 2's marker (O)

    Valid actions are integers 0-8 mapping to board positions:
    0 | 1 | 2
    ---------
    3 | 4 | 5  
    ---------
    6 | 7 | 8

    The reward structure prioritizes winning over drawing over losing:
    win_reward: Reward for winning (positive)
    loss_reward: Penalty for losing (negative) 
    draw_reward: Small reward for drawing (positive)
    illegal_move_reward: Penalty for invalid moves (negative)
    step_reward: Per-step reward (usually 0)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, win_reward: int = 1, loss_reward: int = -1, draw_reward: float = 0.0, 
                 illegal_move_reward: int = -10, step_reward: float = 0) -> None:
        """Initializes the TicTacToe environment with Win > Draw > Loss reward structure.

        Args:
            win_reward (int): Reward for winning (default: +10).
            loss_reward (int): Reward (penalty) for losing (default: -10).
            draw_reward (int): Reward for a draw game (default: +1).
            illegal_move_reward (int): Reward for attempting an illegal move (default: -20).
            step_reward (float): Reward for taking a step (default: 0).
        """
        super().__init__()

        self.n_actions = 9
        self.max_steps = self.n_actions
        self.board_size = int(math.sqrt(self.n_actions))
        if self.board_size * self.board_size != self.n_actions:
            raise ValueError("n_actions must be a perfect square.")

        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self.n_actions,), dtype=int)
        self.players = [1, 2]

        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.illegal_move_reward = illegal_move_reward
        self.step_reward = step_reward

        self._board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=int)
        self._current_player: int = self.players[0]
        self._step_count: int = 0
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state (empty board).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - The initial observation (flattened empty board).
                - An info dictionary containing legal actions and the current player.
        """
        self._board = np.zeros((self.board_size, self.board_size), dtype=int)
        self._current_player = self.players[0]
        self._step_count = 0
        legal_actions = self._get_legal_actions()
        info = {
            'current_player': self._current_player,
            'legal_actions': legal_actions
        }
        return self._board.flatten(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step. Rewards wins, penalizes losses, draws are neutral.

        Args:
            action (int): The action chosen by the current player (0-8).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                - observation (np.ndarray): The new state of the board (flattened).
                - reward (float): The reward obtained by the current player for this action.
                - done (bool): Whether the game has ended (win, draw, or illegal move).
                - info (Dict[str, Any]): Dictionary containing auxiliary information like
                  the next player and legal actions for the next state.
        """
        if self._get_winner() is not None or self._is_draw():
             print("Warning: Step called on a terminal state.")
             current_info = {
                'current_player': self._current_player,
                'legal_actions': [],
                'outcome': 'terminal_state_reached'
             }
             return self._board.flatten(), 0.0, True, current_info

        self._step_count += 1
        done = False
        if self._step_count < 5:
            reward = 0
        else:
            reward = self.step_reward # Default to 0
        outcome = 'game_ongoing'
        player_who_moves = self._current_player

        legal_actions = self._get_legal_actions()
        if action not in legal_actions:
            reward = self.illegal_move_reward
            done = True
            outcome = 'illegal_move'
            info = {
                'current_player': player_who_moves,
                'legal_actions': legal_actions,
                'outcome': outcome,
                'steps': self._step_count
            }
            return self._board.flatten(), reward, done, info

        row, col = self._decode_action(action)
        self._board[row, col] = player_who_moves

        winner = self._get_winner()
        is_draw = self._is_draw()

        if winner is not None:
            done = True
            outcome = f'player_{winner}_wins'
            if winner == player_who_moves:
                reward = self.win_reward
            else:
                reward = self.loss_reward
        elif is_draw:
            done = True
            reward = self.draw_reward
            outcome = 'draw'
        
        next_player = self._current_player
        if not done:
            self._current_player = self.players[1] if player_who_moves == self.players[0] else self.players[0]
            next_player = self._current_player

        next_legal_actions = self._get_legal_actions() if not done else []
        info = {
            'current_player': next_player,
            'legal_actions': next_legal_actions,
            'outcome': outcome,
            'steps': self._step_count
        }
        return self._board.flatten(), reward, done, info

    def _get_legal_actions(self) -> List[int]:
        """
        Calculates the legal actions for the current board state.

        Returns:
            List[int]: A list of integers representing the indices (0-8)
                       of the empty cells on the board.
        """
        # Flatten the board to easily find indices of empty cells (where value is 0)
        flattened_board = self._board.flatten()
        # Find the indices where the cell is empty (equal to 0)
        legal_actions = np.where(flattened_board == 0)[0].tolist()
        return legal_actions

    def _get_winner(self) -> int | None:
        """
        Checks if there is a winner on the current board.

        Returns:
            int | None: The ID of the winning player (1 or 2) if there is one,
                        otherwise None.
        """
        # Check rows, columns, and diagonals for a win for either player
        for player in self.players:
            # Check rows: Does any row contain three of the player's markers?
            if np.any(np.all(self._board == player, axis=1)):
                return player
            # Check columns: Does any column contain three of the player's markers?
            if np.any(np.all(self._board == player, axis=0)):
                return player
            # Check main diagonal (top-left to bottom-right)
            if np.all(np.diag(self._board) == player):
                return player
            # Check anti-diagonal (top-right to bottom-left)
            if np.all(np.diag(np.fliplr(self._board)) == player):
                return player
        # If no winner is found after checking all conditions
        return None

    def _is_draw(self) -> bool:
        """
        Checks if the game is a draw (board is full, and there is no winner).

        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        # A draw occurs if there are no empty cells (all cells are non-zero)
        # AND there is no winner. The winner check happens before this in step().
        return np.all(self._board != 0) and self._get_winner() is None

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """
        Converts a flattened action index (0-8) into board coordinates (row, col).

        Args:
            action (int): The action index (0-8).

        Returns:
            Tuple[int, int]: The corresponding (row, col) tuple.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions - 1}")
        # Row is the integer division of action by board size
        row = action // self.board_size
        # Column is the remainder of action divided by board size
        col = action % self.board_size
        return row, col

    def render(self, mode="human") -> str | None:
        """
        Renders the current state of the board as a simple 3x3 grid.
        Empty cells display their action index (0-8) in gray.
        """
        ESC = chr(27)
        GRAY = f"{ESC}[90m"
        RESET = f"{ESC}[0m"

        grid_lines = []
        for r in range(self.board_size):
            row_cells = []
            for c in range(self.board_size):
                action_index = r * self.board_size + c
                if self._board[r, c] == self.players[0]: # Player 1
                    row_cells.append('X')
                elif self._board[r, c] == self.players[1]: # Player 2
                    row_cells.append('O')
                else: # Empty cell
                    row_cells.append(f"{GRAY}{action_index}{RESET}")
            grid_lines.append(" | ".join(row_cells))

        grid = "\n---------\n".join(grid_lines)

        if mode == "human":
            print(grid)
            return None
        elif mode == "ansi":
            return grid
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # You might need to adjust the import if running this file directly
    # depending on your project structure.
    # This assumes gym_TicTacToe is installed or in the Python path.
    try:
        env = gym.make("TTT-v0") # Use the registered ID
    except gym.error.NameNotFound:
        print("Environment TTT-v0 not registered. Running with direct class instantiation.")
        env = TicTacToeEnv()

    obs, info = env.reset()
    print("Initial Board:")
    env.render()
    print(f"Initial Observation: {obs}")
    print(f"Initial Info: {info}")
    done = False

    # Example game loop (Player 1 vs Player 2 - simple alternating turns)
    while not done:
        current_player = info['current_player']
        legal_actions = info['legal_actions']
        print(f"\nPlayer {current_player}'s turn. Legal actions: {legal_actions}")

        # Simple strategy: choose the first legal action
        if not legal_actions:
            print("No legal actions left? This shouldn't happen unless game ended.")
            break
        action = legal_actions[0]
        print(f"Player {current_player} chooses action: {action}")

        # Take the step
        obs, reward, done, info = env.step(action)

        print("\nBoard after move:")
        env.render()
        print(f"Observation: {obs}")
        print(f"Reward received: {reward}")
        print(f"Game Done: {done}")
        print(f"Info: {info}")

    print(f"\nGame finished! Outcome: {info.get('outcome', 'N/A')}")
