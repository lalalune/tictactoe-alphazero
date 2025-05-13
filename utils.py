import numpy as np
from typing import Tuple, List
import random

def state_to_tuple(state: np.ndarray) -> Tuple[int, ...]:
    """
    Converts a numpy array representing the board state into a hashable tuple.

    Reinforcement learning algorithms like Q-learning often store learned values
    (like Q-values) in a dictionary or hash map. Dictionary keys must be
    hashable. NumPy arrays are mutable and therefore not hashable by default.
    Converting the array to an immutable tuple allows us to use the board state
    directly as a key.

    Args:
        state (np.ndarray): The board state, typically a flattened 1D NumPy array.
                           Example: np.array([0, 1, 0, 2, 0, 0, 1, 0, 0])

    Returns:
        Tuple[int, ...]: A hashable tuple representation of the state.
                         Example: (0, 1, 0, 2, 0, 0, 1, 0, 0)
    """
    # Ensure the input is a NumPy array
    if not isinstance(state, np.ndarray):
        raise TypeError(f"Input state must be a NumPy array, got {type(state)}")

    # Convert the NumPy array to a tuple of integers
    return tuple(state.astype(int))

# --- Example Usage ---
if __name__ == '__main__':
    example_state_array = np.array([0, 1, 0, 2, 0, 0, 1, 0, 0])
    example_state_tuple = state_to_tuple(example_state_array)

    print(f"Original NumPy array: {example_state_array} (Type: {type(example_state_array)})")
    print(f"Converted Tuple:      {example_state_tuple} (Type: {type(example_state_tuple)})")

    # Demonstrate hashability (can be used as a dictionary key)
    try:
        my_dict = {}
        my_dict[example_state_tuple] = 123
        print(f"Successfully used tuple as dictionary key: {my_dict}")
    except TypeError as e:
        print(f"Error using tuple as key: {e}")

    # Demonstrate non-hashability of NumPy array
    try:
        my_dict = {}
        my_dict[example_state_array] = 456
        print(f"Successfully used NumPy array as dictionary key (This should not happen!)")
    except TypeError as e:
        print(f"Error using NumPy array as key (expected): {e}")

# --- Random Agent for Evaluation --- #

class RandomAgent:
    """
    A simple agent that chooses a random action from the legal actions.
    Used as a baseline opponent for evaluating the learning agent.
    """
    def choose_action(self, legal_actions: List[int]) -> int:
        """
        Selects a random action from the provided list of legal actions.

        Args:
            legal_actions (List[int]): The list of valid actions.

        Returns:
            int: A randomly chosen legal action.

        Raises:
            ValueError: If `legal_actions` is empty.
        """
        if not legal_actions:
            raise ValueError("Cannot choose action: legal_actions list is empty.")
        return random.choice(legal_actions)

# --- Strategic Random Agent for Evaluation --- #
class StrategicRandomAgent:
    """
    A more strategic baseline agent for Tic Tac Toe.
    It prioritizes:
    1. Winning moves.
    2. Blocking opponent's winning moves.
    3. Taking the center square if available.
    4. Taking a corner square if available.
    5. Taking a side square if available.
    6. Making a move to extend its own line if possible.
    7. Random otherwise.
    This agent needs access to the board state to make decisions.
    """
    def __init__(self, board_size: int = 3):
        self.board_size = board_size
        self.center_action = board_size * board_size // 2 if board_size % 2 == 1 else -1
        self.corner_actions = [0, board_size - 1, board_size * (board_size - 1), board_size * board_size - 1]
        self.side_actions = []
        if board_size == 3: # Specific for 3x3 for simplicity in identifying sides
            self.side_actions = [1, 3, 5, 7]


    def _check_winner_on_board(self, board: np.ndarray, player: int) -> bool:
        """Checks if the given player has won on the given board."""
        # Check rows
        if np.any(np.all(board == player, axis=1)):
            return True
        # Check columns
        if np.any(np.all(board == player, axis=0)):
            return True
        # Check main diagonal
        if np.all(np.diag(board) == player):
            return True
        # Check anti-diagonal
        if np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def _get_board_from_flat(self, flat_board_array: np.ndarray) -> np.ndarray:
        """Converts a flattened board array to a 2D board."""
        return flat_board_array.reshape((self.board_size, self.board_size))

    def choose_action(self, flat_board_array: np.ndarray, legal_actions: List[int], player_id: int) -> int:
        """
        Chooses an action based on a set of strategic priorities.

        Args:
            flat_board_array (np.ndarray): The current state of the board (flattened).
            legal_actions (List[int]): A list of legal actions (indices of empty cells).
            player_id (int): The ID of this agent (1 or 2).

        Returns:
            int: The chosen action.
        """
        if not legal_actions:
            raise ValueError("Cannot choose action: legal_actions list is empty.")

        opponent_id = 1 if player_id == 2 else 2
        current_board_2d = self._get_board_from_flat(flat_board_array.copy())

        # 1. Check for winning move for self
        for action in legal_actions:
            temp_board = current_board_2d.copy()
            row, col = action // self.board_size, action % self.board_size
            temp_board[row, col] = player_id
            if self._check_winner_on_board(temp_board, player_id):
                return action # Take winning move

        # 2. Check for blocking opponent's winning move
        for action in legal_actions:
            temp_board = current_board_2d.copy()
            row, col = action // self.board_size, action % self.board_size
            temp_board[row, col] = opponent_id # Simulate opponent taking this spot
            if self._check_winner_on_board(temp_board, opponent_id):
                return action # Block opponent's win

        # 3. Try to take the center if available (for 3x3)
        if self.board_size == 3 and self.center_action in legal_actions:
            return self.center_action

        # 4. Try to take a corner if available
        available_corners = [act for act in self.corner_actions if act in legal_actions]
        if available_corners:
            return random.choice(available_corners)
        
        # 5. Try to take a side square if available (for 3x3)
        if self.board_size == 3:
            available_sides = [act for act in self.side_actions if act in legal_actions]
            if available_sides:
                return random.choice(available_sides)

        # 6. Fallback to random legal move if no strategic move found above
        # (More sophisticated line-building logic could be added here)
        return random.choice(legal_actions)

# --- Example Usage of state_to_tuple (if __name__ == '__main__') ---
if __name__ == '__main__':
    example_state_array = np.array([0, 1, 0, 2, 0, 0, 1, 0, 0])
    example_state_tuple = state_to_tuple(example_state_array)
    print(f"Original NumPy array: {example_state_array} (Type: {type(example_state_array)})")
    print(f"Converted Tuple:      {example_state_tuple} (Type: {type(example_state_tuple)})")
    try:
        my_dict = {}
        my_dict[example_state_tuple] = 123
        print(f"Successfully used tuple as dictionary key: {my_dict}")
    except TypeError as e:
        print(f"Error using tuple as key: {e}")

    # Test StrategicRandomAgent (simple test)
    print("\n--- StrategicRandomAgent Test ---")
    strategic_agent = StrategicRandomAgent()
    # Board where player 1 (X) can win at action 2
    # X | X | _ 
    # O | O | _
    # _ | _ | _
    test_board_can_win = np.array([1, 1, 0, 2, 2, 0, 0, 0, 0])
    legal_moves_can_win = [2, 5, 6, 7, 8]
    chosen_action_win = strategic_agent.choose_action(test_board_can_win, legal_moves_can_win, 1)
    print(f"Board: {test_board_can_win}, Player 1, Legal: {legal_moves_can_win}, Chosen (win): {chosen_action_win} (Expected: 2)")
    assert chosen_action_win == 2

    # Board where player 1 (X) must block player 2 (O) at action 2
    # O | O | _ 
    # X | X | _
    # _ | _ | _
    test_board_must_block = np.array([2, 2, 0, 1, 1, 0, 0, 0, 0])
    legal_moves_must_block = [2, 5, 6, 7, 8]
    chosen_action_block = strategic_agent.choose_action(test_board_must_block, legal_moves_must_block, 1)
    print(f"Board: {test_board_must_block}, Player 1, Legal: {legal_moves_must_block}, Chosen (block): {chosen_action_block} (Expected: 2)")
    assert chosen_action_block == 2
    
    # Board where player 2 (O) must block player 1 (X) at action 6
    # X | O | X 
    # O | X | O
    # _ | X | _
    test_board_p2_block = np.array([1,2,1, 2,1,2, 0,1,0])
    legal_moves_p2_block = [6,8]
    chosen_action_p2_block = strategic_agent.choose_action(test_board_p2_block, legal_moves_p2_block, 2) # Player 2's turn
    print(f"Board: {test_board_p2_block}, Player 2, Legal: {legal_moves_p2_block}, Chosen (block): {chosen_action_p2_block} (Expected: 6)")
    assert chosen_action_p2_block == 6

    # Board where center is best move
    test_board_center = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2])
    legal_moves_center = [1,2,3,4,5,6,7]
    chosen_action_center = strategic_agent.choose_action(test_board_center, legal_moves_center, 1)
    print(f"Board: {test_board_center}, Player 1, Legal: {legal_moves_center}, Chosen (center): {chosen_action_center} (Expected: 4)")
    assert chosen_action_center == 4 