# Placeholder for MCTS Agent
import numpy as np
import math
import random
import time
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

# Assuming state_to_tuple is available if needed, though we might use numpy arrays internally
# from utils import state_to_tuple

# --- Game Logic Helpers (Operating on NumPy 3x3 arrays) ---

def _get_legal_actions_from_state(board_array: np.ndarray) -> List[int]:
    """Gets legal actions (indices of 0s) from a 3x3 numpy board."""
    return list(np.where(board_array.flatten() == 0)[0])

def _check_terminal_and_winner(board_array: np.ndarray) -> Tuple[bool, Optional[int]]:
    """Checks if the game has ended and returns the winner (1 or 2) or None.

    Returns: (is_terminal, winner)
    """
    for player in [1, 2]:
        # Check rows
        if np.any(np.all(board_array == player, axis=1)):
            return True, player
        # Check columns
        if np.any(np.all(board_array == player, axis=0)):
            return True, player
        # Check diagonals
        if np.all(np.diag(board_array) == player) or np.all(np.diag(np.fliplr(board_array)) == player):
            return True, player

    # Check for draw (no 0s left)
    if not (board_array == 0).any():
        return True, 0 # Use 0 to signify draw winner

    # Game not finished
    return False, None

def _apply_move(board_array: np.ndarray, action: int, player: int) -> np.ndarray:
    """Applies a move to a copy of the 3x3 numpy board."""
    if not 0 <= action <= 8:
        raise ValueError("Action out of bounds")
    if board_array.flatten()[action] != 0:
        # This indicates an issue, either illegal action passed or state mismatch
        raise ValueError(f"Cannot apply action {action} to non-empty cell.")
    new_board = board_array.copy()
    row, col = action // 3, action % 3
    new_board[row, col] = player
    return new_board

# --- MCTS Node --- #

class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree.

    Stores statistics for a specific game state.
    """
    def __init__(self, board_array: np.ndarray, player_turn: int, parent: 'MCTSNode' = None, action_taken: Optional[int] = None):
        """Initializes a new MCTS node.

        Args:
            board_array (np.ndarray): The 3x3 numpy array representing the board state.
            player_turn (int): The player whose turn it is in this state (1 or 2).
            parent (MCTSNode, optional): The parent node in the tree. Defaults to None (for root).
            action_taken (Optional[int], optional): The action taken from the parent to reach this node.
        """
        self.board_array = board_array # Store the numpy array state
        self.player_turn = player_turn
        self.parent = parent
        self.action_taken = action_taken # Action that led to this node

        self.children: Dict[int, MCTSNode] = {} # Maps action to child node

        # Statistics
        self._wins = 0.0 # Tracks wins for the player whose turn it ISN'T (i.e., the player who just moved)
                        # Easier for UCT calculation where we want the value from the parent's perspective.
        self._visits = 0

        # MCTS Expansion Control
        self.is_terminal, self.winner = _check_terminal_and_winner(self.board_array)
        self._untried_actions = None if self.is_terminal else _get_legal_actions_from_state(self.board_array)

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def wins(self) -> float:
        return self._wins

    @property
    def untried_actions(self) -> Optional[List[int]]:
        # Ensure the list can be modified externally if needed (though expansion handles it)
        return self._untried_actions

    def is_fully_expanded(self) -> bool:
        """Checks if all legal actions from this node have been explored."""
        return self.is_terminal or (self._untried_actions is not None and len(self._untried_actions) == 0)

    def select_child(self, exploration_constant: float) -> 'MCTSNode':
        """Selects the best child node based on the UCT formula.

        UCT = (wins / visits) + C * sqrt(log(parent_visits) / visits)
        Assumes the node is fully expanded.

        Returns:
            MCTSNode: The child node with the highest UCT score.
        """
        if not self.children:
             # This should not happen if called correctly after expansion
             raise ValueError("Cannot select child from a node with no children.")

        log_parent_visits = math.log(self.visits)
        best_score = -float('inf')
        best_children = []

        for child in self.children.values():
            if child.visits == 0:
                # Prefer unvisited children if any exist (shouldn't strictly happen if fully expanded)
                # Assign infinite score to guarantee selection
                uct_score = float('inf')
            else:
                win_rate_for_current_player = (child.visits - child.wins) / child.visits
                explore_term = exploration_constant * math.sqrt(log_parent_visits / child.visits)
                uct_score = win_rate_for_current_player + explore_term

            if uct_score > best_score:
                best_score = uct_score
                best_children = [child]
            elif uct_score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def expand(self) -> 'MCTSNode':
        """Expands the tree by adding one new child node.

        Selects an untried action, simulates it, creates the child node,
        adds it to children, and removes the action from untried_actions.

        Returns:
            MCTSNode: The newly created child node.
        """
        if self.is_fully_expanded():
            raise RuntimeError("Cannot expand a fully expanded node.")

        action = self._untried_actions.pop(random.randrange(len(self._untried_actions)))

        next_board_array = _apply_move(self.board_array, action, self.player_turn)
        next_player_turn = 1 if self.player_turn == 2 else 2

        child_node = MCTSNode(next_board_array, next_player_turn, parent=self, action_taken=action)
        self.children[action] = child_node
        return child_node

    def update(self, simulation_result: float):
        """Updates the node's visits and wins based on the simulation result.

        Args:
            simulation_result (float): The result of the simulation (-1 loss, 0 draw, +1 win
                                        from the perspective of the player whose turn it was
                                        at the START of the simulation/rollout).
        """
        self._visits += 1
        if self.player_turn == 1:
             self._wins += (1.0 - simulation_result) / 2.0
        else: 
             self._wins += (1.0 + simulation_result) / 2.0

# --- MCTS Agent --- #

class MCTSAgent:
    """Monte Carlo Tree Search agent for Tic Tac Toe.

    Uses UCT (Upper Confidence bounds applied to Trees) for balancing exploration
    and exploitation.
    """
    def __init__(self, num_simulations: int = 100, exploration_constant: float = 1.414):
        """Initializes the MCTS agent.

        Args:
            num_simulations (int): The number of simulations (rollouts) to run per move.
                                   More simulations lead to better decisions but take longer.
            exploration_constant (float): Controls the exploration/exploitation balance (C in UCT).
                                          Higher values encourage exploring less-visited nodes.
        """
        if num_simulations < 1:
            raise ValueError("Number of simulations must be at least 1.")
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        # self._board_size = 3 # Not strictly needed here if board passed in

    def choose_action(self, current_board_flat: np.ndarray, current_player_turn: int, legal_actions: List[int]) -> int:
        """
        Chooses the best action from the current state using MCTS.

        Args:
            current_board_flat (np.ndarray): The flattened 1D numpy array of the current board state.
            current_player_turn (int): The player whose turn it is (1 or 2).
            legal_actions (List[int]): A list of legal actions for the current state.

        Returns:
            int: The chosen action (0-8).
        """
        # start_time = time.time() # Optional: for timing

        if not legal_actions:
            raise ValueError("MCTS Cannot choose action: No legal actions provided.")
        if len(legal_actions) == 1:
            return legal_actions[0] # Only one possible move, no search needed

        current_board_2d = current_board_flat.reshape(3, 3)
        root_node = MCTSNode(current_board_2d, current_player_turn)

        # Run simulations
        for _ in range(self.num_simulations):
            node = self._select(root_node)
            result = 0.0 # Default to draw if terminal node selected
            if not node.is_terminal:
                node = self._expand(node)
                result = self._simulate(node) # Simulate from the newly expanded child
            elif node.winner is not None:
                 # If selection lands on a terminal node, use its outcome (P1 perspective)
                 if node.winner == 1: result = 1.0
                 elif node.winner == 2: result = -1.0
                 # else: result remains 0.0 for draw winner (winner=0)

            self._backpropagate(node, result)

        # Choose the best move based on visits (most robust)
        best_action = -1
        max_visits = -1
        
        if not root_node.children: # Should only happen if num_simulations is very low or only one move
            # This case might occur if the root is terminal or no expansion happened
            # print("Warning: MCTS root has no children after simulations. Choosing from legal actions.")
            return random.choice(legal_actions)

        for action_key, child_node in root_node.children.items():
            if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_action = action_key

        if best_action == -1:
            # Fallback if all children have 0 visits (highly unlikely with enough sims)
            # print("Warning: MCTS fallback to random action as no child was clearly best by visits.")
            return random.choice(legal_actions)
        
        # elapsed_time = time.time() - start_time
        # print(f"MCTS chose action {best_action} for P{current_player_turn} after {self.num_simulations} sims ({elapsed_time*1000:.1f} ms)")
        return best_action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Phase 1: Selection - Traverse the tree using UCT to find a leaf node."""
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node # Found a node to expand
            else:
                if not node.children: # If no children, cannot select further, return this node
                    return node
                node = node.select_child(self.exploration_constant)
        return node # Reached a terminal node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Phase 2: Expansion - Add a new child node for an untried action."""
        if node.is_fully_expanded():
            raise RuntimeError("MCTS Expansion error: Node is already fully expanded or terminal.")
        return node.expand()

    def _simulate(self, node: MCTSNode) -> float:
        """Phase 3: Simulation (Rollout) - Play randomly from the node until the game ends.

        Returns:
            float: The result from Player 1's perspective (+1 P1 win, -1 P2 win, 0 draw).
        """
        current_board = node.board_array.copy()
        current_player = node.player_turn
        is_terminal, winner = _check_terminal_and_winner(current_board) # Initial check for the node state itself

        # If the node itself is terminal, its pre-calculated winner is used directly.
        if is_terminal:
            if winner == 1: return 1.0
            if winner == 2: return -1.0
            return 0.0 # Draw
        
        # Rollout loop
        while not is_terminal:
            legal_actions = _get_legal_actions_from_state(current_board)
            if not legal_actions:
                 is_terminal = True
                 winner = 0 # Draw if no legal actions but not caught by _check_terminal_and_winner
                 break

            action = random.choice(legal_actions)
            current_board = _apply_move(current_board, action, current_player)
            is_terminal, winner = _check_terminal_and_winner(current_board)
            current_player = 1 if current_player == 2 else 2 # Switch player

        if winner == 1: return 1.0
        if winner == 2: return -1.0
        return 0.0

    def _backpropagate(self, node: MCTSNode, result: float):
        """Phase 4: Backpropagation - Update statistics up the tree.

        Args:
            node (MCTSNode): The node from which the simulation started.
            result (float): The simulation result (+1 P1 win, -1 P2 win, 0 draw).
        """
        current_node = node
        while current_node is not None:
            current_node.update(result)
            current_node = current_node.parent

    def learn(self, *args, **kwargs):
        pass # MCTS learns via tree construction during choose_action

# --- Example Usage (Illustrative) --- #
if __name__ == '__main__':
    # Requires a gym-like environment to test `choose_action`
    print("MCTS Agent class defined. Requires an environment for full testing.")

    # Test Node and Helpers
    board = np.zeros((3, 3), dtype=int)
    board[0,0] = 1
    board[1,1] = 2
    print("Test Board:")
    print(board)
    legal = _get_legal_actions_from_state(board)
    print(f"Legal actions: {legal}")
    term, win = _check_terminal_and_winner(board)
    print(f"Terminal: {term}, Winner: {win}")

    root = MCTSNode(board, player_turn=1)
    print(f"Root node created. Untried actions: {root.untried_actions}")
    child = root.expand()
    print(f"Expanded child node for action {child.action_taken}. Board:")
    print(child.board_array)
    print(f"Root children: {root.children.keys()}")
    print(f"Root untried actions: {root.untried_actions}")

    # Simulate a result and backpropagate
    sim_result = 1.0 # Assume P1 won the rollout from child
    print(f"Sim result (P1 perspective): {sim_result}")
    child.update(sim_result)
    root.update(sim_result)
    print(f"Child stats: Visits={child.visits}, Wins(for P{3-child.player_turn})={child.wins:.1f}")
    print(f"Root stats: Visits={root.visits}, Wins(for P{3-root.player_turn})={root.wins:.1f}")

    # Select best child (will be the only one here)
    best_child = root.select_child(exploration_constant=1.41)
    print(f"Selected best child action: {best_child.action_taken}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # Will be used later for training the network

# --- AlphaZero Neural Network --- #

class AlphaZeroNet(nn.Module):
    """
    Neural Network for AlphaZero.
    Takes a board state and outputs a policy (move probabilities) and a value (expected outcome).
    Input board state is expected to be 2 planes: one for current player's pieces, one for opponent's.
    For a 3x3 board, input_channels=2, board_height=3, board_width=3, so input_size = 2*3*3 = 18.
    """
    def __init__(self, 
                 input_channels: int = 2, 
                 board_height: int = 3, 
                 board_width: int = 3, 
                 num_actions: int = 9, 
                 shared_hidden_size1: int = 128,
                 shared_hidden_size2: int = 128, 
                 policy_head_hidden_size: int = 64,
                 value_head_hidden_size: int = 64):
        super(AlphaZeroNet, self).__init__()
        self.input_size = input_channels * board_height * board_width
        self.num_actions = num_actions

        # Shared Body
        self.fc_shared1 = nn.Linear(self.input_size, shared_hidden_size1)
        self.bn_shared1 = nn.BatchNorm1d(shared_hidden_size1) # Batch norm can help stabilize training
        self.fc_shared2 = nn.Linear(shared_hidden_size1, shared_hidden_size2)
        self.bn_shared2 = nn.BatchNorm1d(shared_hidden_size2)

        # Policy Head
        self.fc_policy_hidden = nn.Linear(shared_hidden_size2, policy_head_hidden_size)
        self.bn_policy = nn.BatchNorm1d(policy_head_hidden_size)
        self.fc_policy_output = nn.Linear(policy_head_hidden_size, num_actions)
        # Using LogSoftmax for numerical stability with NLLLoss later

        # Value Head
        self.fc_value_hidden = nn.Linear(shared_hidden_size2, value_head_hidden_size)
        self.bn_value = nn.BatchNorm1d(value_head_hidden_size)
        self.fc_value_output = nn.Linear(value_head_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor representing the board state (batch_size, input_size).
                               Input should be flattened: e.g., (batch_size, 18).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - policy_logits (torch.Tensor): Raw logits for the policy (batch_size, num_actions).
                                                Apply LogSoftmax/Softmax outside if needed for loss/probabilities.
                - value (torch.Tensor): Estimated value of the state (batch_size, 1).
        """
        if x.shape[0] > 1: # Apply batch norm only if batch size > 1
            # Shared body
            s = F.relu(self.bn_shared1(self.fc_shared1(x)))
            s = F.relu(self.bn_shared2(self.fc_shared2(s)))
            
            # Policy head
            p_hidden = F.relu(self.bn_policy(self.fc_policy_hidden(s)))
            policy_logits = self.fc_policy_output(p_hidden)
            
            # Value head
            v_hidden = F.relu(self.bn_value(self.fc_value_hidden(s)))
            value = torch.tanh(self.fc_value_output(v_hidden)) # Tanh to keep value in [-1, 1]
        else: # Batch size is 1, skip batch norm or use instance norm if preferred
            s = F.relu(self.fc_shared1(x))
            s = F.relu(self.fc_shared2(s))

            p_hidden = F.relu(self.fc_policy_hidden(s))
            policy_logits = self.fc_policy_output(p_hidden)

            v_hidden = F.relu(self.fc_value_hidden(s))
            value = torch.tanh(self.fc_value_output(v_hidden))

        return policy_logits, value

if __name__ == '__main__':
    # Example usage of the network
    print("--- AlphaZeroNet Test ---")
    # For Tic Tac Toe: 2 channels (player, opponent), 3x3 board, 9 actions
    net = AlphaZeroNet(input_channels=2, board_height=3, board_width=3, num_actions=9)
    print(net)

    # Create a dummy batch of 2 board states
    # Each state is 2x3x3 = 18 features when flattened
    # Player 1's pieces on first plane, Player 2's on second plane
    dummy_state1_p1 = np.array([[1,0,0],[0,1,0],[0,0,0]]).flatten()
    dummy_state1_p2 = np.array([[0,0,0],[0,0,0],[0,0,1]]).flatten()
    dummy_state1_flat = np.concatenate((dummy_state1_p1, dummy_state1_p2))

    dummy_state2_p1 = np.array([[0,0,1],[0,0,0],[1,0,0]]).flatten()
    dummy_state2_p2 = np.array([[1,0,0],[0,1,0],[0,0,0]]).flatten()
    dummy_state2_flat = np.concatenate((dummy_state2_p1, dummy_state2_p2))

    batch_input_np = np.array([dummy_state1_flat, dummy_state2_flat])
    batch_input_tensor = torch.tensor(batch_input_np, dtype=torch.float32)
    print(f"\nInput batch shape: {batch_input_tensor.shape}")

    # Forward pass
    net.eval() # Set to evaluation mode for inference if not training
    with torch.no_grad():
        policy_logits, value_output = net(batch_input_tensor)
    
    print(f"Policy logits shape: {policy_logits.shape}") # Should be (batch_size, num_actions)
    print(f"Policy logits (raw):\n{policy_logits}")
    policy_probs = F.softmax(policy_logits, dim=1)
    print(f"Policy probabilities:\n{policy_probs}")

    print(f"Value output shape: {value_output.shape}") # Should be (batch_size, 1)
    print(f"Value output:\n{value_output}")

    print("\nNote: Batch norm layers might behave differently with batch_size=1 during actual training vs. this test.")
    # Test with batch size 1
    single_input_tensor = torch.tensor(dummy_state1_flat.reshape(1, -1), dtype=torch.float32)
    policy_logits_single, value_output_single = net(single_input_tensor)
    print(f"\nSingle input policy logits: {policy_logits_single}")
    print(f"Single input value: {value_output_single}")

# --- Board State Representation for AlphaZeroNet --- #
def get_board_representation(board_array_2d: np.ndarray, player_turn: int, board_height: int = 3, board_width: int = 3) -> np.ndarray:
    """
    Converts a 2D NumPy board array into the 2-plane representation for AlphaZeroNet.
    Plane 1: Current player's pieces (1s), others (0s).
    Plane 2: Opponent's pieces (1s), others (0s).

    Args:
        board_array_2d (np.ndarray): The 3x3 board state.
        player_turn (int): The ID of the current player (1 or 2).
        board_height (int): Height of the board.
        board_width (int): Width of the board.

    Returns:
        np.ndarray: A flattened 1D NumPy array of size (2 * board_height * board_width).
    """
    if player_turn not in [1, 2]:
        raise ValueError("player_turn must be 1 or 2")

    opponent_turn = 1 if player_turn == 2 else 2

    plane1 = np.where(board_array_2d == player_turn, 1.0, 0.0)
    plane2 = np.where(board_array_2d == opponent_turn, 1.0, 0.0)
    
    # Concatenate the planes and flatten
    # Order: (channel, height, width) then flatten, or flatten each then concat.
    # Let's flatten each then concatenate for a 1D vector of size 2*H*W
    return np.concatenate((plane1.flatten(), plane2.flatten()), axis=0).astype(np.float32)

# if __name__ == '__main__':
#    # ... (AlphaZeroNet tests from before can remain) ...
#    # Add tests for get_board_representation
#    print("\n--- Board Representation Test ---")
#    test_board = np.array([[1,0,2],[0,1,0],[2,0,1]])
#    print("Original Board:")
#    print(test_board)
#    rep_p1 = get_board_representation(test_board, player_turn=1)
#    print(f"Representation for P1 (shape {rep_p1.shape}):\n{rep_p1}")
#    # Expected for P1: [1,0,0,0,1,0,0,0,1,  0,0,1,0,0,0,1,0,0]
#    rep_p2 = get_board_representation(test_board, player_turn=2)
#    print(f"Representation for P2 (shape {rep_p2.shape}):\n{rep_p2}")
#    # Expected for P2: [0,0,1,0,0,0,1,0,0,  1,0,0,0,1,0,0,0,1] 

# --- MCTS Node for AlphaZero (MCTSNode_AZ) --- #
class MCTSNode_AZ:
    """Node in the Monte Carlo Search Tree for an AlphaZero-like agent."""
    def __init__(self, 
                 board_array_2d: np.ndarray, 
                 player_turn: int, 
                 parent: 'MCTSNode_AZ' = None, 
                 action_taken: Optional[int] = None, 
                 # prior_policy is what the network output for THIS node's state, set during expansion
                 # value_from_network is also set during expansion
                 # Let's remove them from __init__ args and set them explicitly after network call
                 num_total_actions: int = 9 # Pass total number of actions
                 ):
        self.board_array_2d = board_array_2d
        self.player_turn = player_turn
        self.parent = parent
        self.action_taken = action_taken
        self.num_total_actions = num_total_actions

        self.is_terminal, self.game_winner = _check_terminal_and_winner(self.board_array_2d)
        self.children: Dict[int, MCTSNode_AZ] = {}

        self._N_sa = defaultdict(int)
        self._W_sa = defaultdict(float)
        
        # _P_sa will store the dense policy vector from the network for this node (state s)
        # It is set when this node, as a leaf, is expanded using the network.
        self._P_sa: Optional[np.ndarray] = None 
        self.value_from_network: Optional[float] = None # V(s) from network

    @property
    def N_s(self) -> int:
        """Total visit count for this state node N(s) = sum over a of N(s,a)."""
        return sum(self._N_sa.values())

    def Q_sa(self, action: int) -> float:
        """Mean action value Q(s,a) = W(s,a) / N(s,a)."""
        if self._N_sa[action] == 0:
            return 0.0 # Default Q value for unvisited actions (or use parent Q, or 0)
        return self._W_sa[action] / self._N_sa[action]

    def is_leaf(self) -> bool:
        """A node is a leaf if it has no children (has not been expanded yet)."""
        return not self.children

    def select_child_puct(self, c_puct: float) -> Tuple[Optional[int], Optional['MCTSNode_AZ']]:
        if self.is_terminal or not self.children:
            return None, None 
        if self._P_sa is None: 
            # This means this node was not properly expanded by the network before selection among children is attempted.
            # This indicates a logic error in the MCTS search flow.
            # print(f"Error: _P_sa is None for node with board:\n{self.board_array_2d}\nand player {self.player_turn}")
            # Fallback: could select randomly or based on Q only, but it deviates from PUCT.
            # For now, let's allow it to proceed, but it will use P_sa=0 for U term if P_sa is None. Better to ensure P_sa is set.
            # A node with children should have had its _P_sa set during its expansion.
            pass # Or raise error

        best_action = -1
        max_puct = -float('inf')
        sqrt_N_s = math.sqrt(self.N_s) if self.N_s > 0 else 0 # Handle N_s = 0 case

        # Iterate over children, which correspond to legal actions from this state
        for action, child_node in self.children.items(): 
            q_val = self.Q_sa(action)
            
            # U component: Uses the _P_sa of the PARENT (self)
            u_val = 0
            if self._P_sa is not None and sqrt_N_s > 0: # Check if _P_sa is initialized and N_s > 0
                 if 0 <= action < len(self._P_sa):
                    u_val = c_puct * self._P_sa[action] * sqrt_N_s / (1 + self._N_sa[action])
                 # else: action is not in policy, P_sa[action] is effectively 0, so u_val is 0
            elif self._P_sa is not None and self._N_sa[action] == 0: # For unvisited children, U is high if P > 0
                 if 0 <= action < len(self._P_sa):
                    u_val = c_puct * self._P_sa[action] * 1.0 # sqrt_N_s / (1+0) -> effectively sqrt_N_s is 1 here to give prior some weight
            
            current_puct = q_val + u_val

            if current_puct > max_puct:
                max_puct = current_puct
                best_action = action
        
        if best_action == -1 and self.children: # If no best action but children exist (e.g. all PUCT scores were -inf or equal)
             # Fallback: if multiple have same max_puct or issues, pick one with highest P, then N, then random
             # This part can be made more robust if needed.
             # For now, if still -1, it means something is off or all children are equally bad.
             # Could select a random child if any exist.
            # print(f"Warning: No best_action found by PUCT for node {self.board_array_2d}, P_sa: {self._P_sa}, N_sa: {self._N_sa}, W_sa: {self._W_sa}")
            best_action = random.choice(list(self.children.keys())) if self.children else None
            if best_action is None: return None, None

        return best_action, self.children.get(best_action)

    def expand_leaf_with_network_output(self, policy_probs: np.ndarray, value_estimate: float):
        if not self.is_leaf(): raise RuntimeError("Cannot expand non-leaf node.")
        if self.is_terminal: return

        self._P_sa = policy_probs # policy_probs is dense array from network for this node's state
        self.value_from_network = value_estimate 

        legal_actions = _get_legal_actions_from_state(self.board_array_2d)
        next_player = 1 if self.player_turn == 2 else 2

        for action in legal_actions:
            if action not in self.children:
                next_board = _apply_move(self.board_array_2d, action, self.player_turn)
                self.children[action] = MCTSNode_AZ(next_board, next_player, parent=self, action_taken=action, num_total_actions=self.num_total_actions)

# --- AlphaZero MCTS Agent (MCTSAgent_AZ) --- #
class MCTSAgent_AZ:
    """AlphaZero-style MCTS Agent.
    Uses a neural network to guide search and evaluate positions.
    """
    def __init__(self, 
                 network: AlphaZeroNet, 
                 num_simulations: int = 400, # Number of MCTS simulations per move
                 c_puct: float = 1.0,        # Exploration constant in PUCT
                 device: torch.device = torch.device("cpu") ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.network.to(self.device) # Ensure network is on the correct device

    def _search_one_iteration(self, root_node: MCTSNode_AZ):
        current_node = root_node
        path = [current_node]

        # 1. Selection
        while not current_node.is_leaf():
            action, next_node = current_node.select_child_puct(self.c_puct)
            if next_node is None:
                break 
            current_node = next_node
            path.append(current_node)
        
        leaf_node = current_node
        value_from_leaf_perspective = 0.0

        # 2. Expansion & Evaluation
        if not leaf_node.is_terminal:
            board_repr = get_board_representation(leaf_node.board_array_2d, leaf_node.player_turn)
            board_tensor = torch.tensor(board_repr, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                policy_logits, network_value_estimate = self.network(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value_from_leaf_perspective = network_value_estimate.item() 
            leaf_node.expand_leaf_with_network_output(policy_probs, value_from_leaf_perspective)
        else:
            if leaf_node.game_winner == leaf_node.player_turn: value_from_leaf_perspective = 1.0
            elif leaf_node.game_winner == 0: value_from_leaf_perspective = 0.0
            else: value_from_leaf_perspective = -1.0

        # 3. Backpropagation
        # Iterate backwards from the leaf up to (but not including) the root's parent.
        # The value `v` obtained from the leaf node (or its rollout/evaluation)
        # is from the perspective of the player whose turn it is at that leaf node.
        for node_in_path in reversed(path):
            # The W_sa and N_sa we update belong to node_in_path.parent for the action that led to node_in_path.
            # The value needs to be from the perspective of node_in_path.parent.player_turn.
            if node_in_path.parent is not None:
                action_taken_by_parent = node_in_path.action_taken
                current_value_for_parent = value_from_leaf_perspective
                # If the parent's player is different from the leaf's player, negate the value.
                if node_in_path.parent.player_turn != leaf_node.player_turn:
                    current_value_for_parent = -value_from_leaf_perspective
                
                node_in_path.parent._N_sa[action_taken_by_parent] += 1
                node_in_path.parent._W_sa[action_taken_by_parent] += current_value_for_parent
    
    def get_action_probs(self, board_array_2d: np.ndarray, player_turn: int, temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Runs MCTS simulations from the given state and returns action probabilities.
        Also returns the MCTS-derived value of the root state.

        Args:
            board_array_2d: Current 2D board state.
            player_turn: Current player to move.
            temperature: Controls exploration in action selection from policy.
                         High temp -> more random, Low temp -> more greedy.

        Returns:
            Tuple[np.ndarray, float]: 
                - policy (np.ndarray): Action probabilities (size num_actions).
                - mcts_value (float): MCTS estimated value of the root state from perspective of player_turn.
        """
        root_node = MCTSNode_AZ(board_array_2d, player_turn, num_total_actions=self.network.num_actions)
        
        # If root is terminal, no search needed, return true value and uniform policy over legal (or zeros for illegal)
        if root_node.is_terminal:
            policy = np.zeros(self.network.num_actions, dtype=np.float32)
            if root_node.game_winner == player_turn: mcts_val = 1.0
            elif root_node.game_winner == 0: mcts_val = 0.0
            else: mcts_val = -1.0
            # For policy, maybe uniform over legal if that helps training for terminal, or zeros.
            # For now, let's return zeros for policy target from terminal, value is most important.
            return policy, mcts_val
        
        # Expand the root node immediately if it's a leaf (it always will be on first call)
        # This initializes its P_sa values from the network.
        board_repr_root = get_board_representation(root_node.board_array_2d, root_node.player_turn)
        board_tensor_root = torch.tensor(board_repr_root, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            policy_logits_root, value_root_net = self.network(board_tensor_root)
        policy_probs_root = F.softmax(policy_logits_root, dim=1).squeeze(0).cpu().numpy()
        root_node.expand_leaf_with_network_output(policy_probs_root, value_root_net.item())
        # The value_root_net is from network, not directly used for MCTS value of root yet.

        for _ in range(self.num_simulations):
            self._search_one_iteration(root_node)

        # Collect visit counts for policy target
        # N(s,a) are for actions from root_node
        visit_counts = np.zeros(self.network.num_actions, dtype=np.float32)
        for action, count in root_node._N_sa.items():
            if 0 <= action < self.network.num_actions:
                 visit_counts[action] = count
        
        policy = np.zeros(self.network.num_actions, dtype=np.float32)
        sum_visits = np.sum(visit_counts)

        if sum_visits == 0:
            # Fallback: if no visits, use uniform over legal actions (or network prior if more robust)
            legal_actions_at_root = _get_legal_actions_from_state(root_node.board_array_2d)
            if legal_actions_at_root:
                for la in legal_actions_at_root:
                    if 0 <= la < len(policy): policy[la] = 1.0 / len(legal_actions_at_root)
        else:
            # Adjusted temperature threshold for greedy selection to handle cases like temp=0.01
            if temperature < 0.05: # Increased threshold for greedy path
                best_action = np.argmax(visit_counts)
                policy[best_action] = 1.0
            else:
                # Apply temperature and normalize
                exponent = 1.0 / temperature
                # Prevent overflow by handling large exponents carefully, or by scaling counts if necessary
                # For now, let's try to catch overflow with powered_visits
                try:
                    powered_visits = np.power(visit_counts, exponent)
                    if np.any(np.isinf(powered_visits)) or np.any(np.isnan(powered_visits)):
                        # print(f"Warning: Overflow/NaN in powered_visits with temp {temperature}. Defaulting to greedy.")
                        best_action = np.argmax(visit_counts)
                        policy[best_action] = 1.0
                    else:
                        sum_powered_visits = np.sum(powered_visits)
                        if sum_powered_visits > 1e-8: # Increased tolerance slightly
                            policy = powered_visits / sum_powered_visits
                        else: 
                            # print(f"Warning: Sum of powered visits near zero with temp {temperature}. Defaulting to greedy based on original visits.")
                            best_action = np.argmax(visit_counts) # Fallback to greedy on original counts
                            policy[best_action] = 1.0
                except OverflowError:
                    # print(f"OverflowError with temp {temperature} and visit_counts. Defaulting to greedy.")
                    best_action = np.argmax(visit_counts)
                    policy[best_action] = 1.0
        
        mcts_value = sum(root_node._W_sa.values()) / root_node.N_s if root_node.N_s > 0 else 0.0
        return policy, mcts_value

    def choose_action(self, board_array_flat: np.ndarray, current_player_turn: int, legal_actions: List[int], temperature: float = 1.0) -> int:
        """
        Chooses an action by running MCTS, then sampling from the MCTS policy.
        This is typically used during self-play to generate training data.
        For evaluation/play, temperature is often set to a very small value (e.g., 0 or 0.01) for greedy play.
        """
        if not legal_actions:
            raise ValueError("MCTS choose_action: No legal actions.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        board_2d = board_array_flat.reshape(3,3)
        # Get policy (action probabilities) from MCTS search
        mcts_policy_probs, _ = self.get_action_probs(board_2d, current_player_turn, temperature)
        
        # Filter policy for legal actions only and re-normalize if needed
        # Though MCTS policy should ideally only assign probs to legal moves if priors are handled well.
        # For robustness, mask illegal moves from mcts_policy_probs
        masked_policy = np.zeros_like(mcts_policy_probs)
        sum_legal_probs = 0
        for action_idx in legal_actions:
            if 0 <= action_idx < len(mcts_policy_probs):
                masked_policy[action_idx] = mcts_policy_probs[action_idx]
                sum_legal_probs += mcts_policy_probs[action_idx]
        
        if sum_legal_probs > 0:
            final_policy = masked_policy / sum_legal_probs
        else:
            # If all legal moves had zero probability (should be rare), pick uniformly among legal
            # print("Warning: MCTS policy resulted in zero probability for all legal actions. Choosing uniformly.")
            final_policy = np.zeros_like(mcts_policy_probs)
            for la in legal_actions: final_policy[la] = 1.0 / len(legal_actions)

        # Sample an action based on the MCTS policy
        chosen_action = np.random.choice(np.arange(self.network.num_actions), p=final_policy)
        return chosen_action

    def learn(self, *args, **kwargs):
        # MCTS itself doesn't "learn" in this Q-learning sense. The NN learns.
        # This agent is used for inference and generating search data.
        pass 

# if __name__ == '__main__': ... (tests for AlphaZeroNet, get_board_representation can remain) 