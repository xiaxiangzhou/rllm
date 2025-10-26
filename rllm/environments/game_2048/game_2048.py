"""
2048 Game Environment for RLLM

A classic 2048 puzzle game where the player combines tiles to reach a target value.
Uses the gym-2048 package for the game logic.
"""

import math
from typing import Any

import gym
import gym_2048
import numpy as np

from rllm.environments.base.base_env import BaseEnv


class Game2048Env(BaseEnv):
    """
    2048 Game Environment.

    The player moves tiles on a 4x4 grid. When two tiles with the same number
    touch, they merge into one. The goal is to reach a target value (default: 128).
    
    This environment wraps the gym-2048 package and adapts it to work with
    LLM agents that expect string-based observations and actions.
    """

    # Action mapping: string -> gym-2048 integer
    ACTION_MAP = {
        "left": 0,
        "up": 1,
        "right": 2,
        "down": 3,
    }

    def __init__(
        self,
        seed: int = 0,
        target_value: int = 128,
        max_steps: int = 40,
        **kwargs,
    ):
        """
        Initialize the 2048 environment.

        Args:
            seed: Random seed for reproducibility
            target_value: Target tile value to win (default: 128)
            max_steps: Maximum number of steps before termination
        """
        self.seed_value = seed
        self.target_value = target_value
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize gym-2048 environment
        self.gym_env = gym.make('2048-v0')
        
        # Game state
        self.game_over = False
        self.won = False
        self.current_board = None
        self.previous_board = None

    def reset(self, **kwargs) -> tuple[str, dict]:
        """
        Reset the environment to start a new game.

        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset gym environment
        self.gym_env.seed(self.seed_value)
        self.current_board = self.gym_env.reset()
        
        self.game_over = False
        self.won = False
        self.current_step = 0
        self.previous_board = None

        observation = self._format_board(self.current_board)
        info = self._get_info()

        return observation, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: One of "left", "right", "up", "down"

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1

        # Validate action
        action = action.lower().strip()
        if action not in self.ACTION_MAP:
            # Invalid action
            observation = self._format_board(self.current_board)
            reward = -1.0
            done = True
            info = self._get_info()
            info["invalid_action"] = True
            self.game_over = True
            return observation, reward, done, info

        # Save board state before move
        self.previous_board = self.current_board.copy()

        # Convert string action to gym-2048 integer action
        action_int = self.ACTION_MAP[action]

        # Execute action in gym environment
        self.current_board, gym_reward, gym_done, gym_info = self.gym_env.step(action_int)

        # Check if board changed
        board_changed = not np.array_equal(self.previous_board, self.current_board)

        # Check win condition
        max_tile = self._get_max_tile()
        if max_tile >= self.target_value:
            self.won = True
            self.game_over = True

        # Check if game is over (no more moves or max steps reached)
        if gym_done or self.current_step >= self.max_steps:
            self.game_over = True

        # Calculate custom reward (not using gym's default reward)
        reward = self._calculate_reward(board_changed)

        observation = self._format_board(self.current_board)
        done = self.game_over
        info = self._get_info()

        return observation, reward, done, info

    def _format_board(self, board: np.ndarray) -> str:
        """
        Format the numpy board array as a human-readable string.
        
        Args:
            board: numpy array representing the board (4x4)
            
        Returns:
            Formatted string representation
        """
        # gym-2048 uses 0 for empty cells, we'll display as "."
        # Find max width for alignment
        max_val = board.max()
        max_width = len(str(max_val)) if max_val > 0 else 1
        
        lines = []
        for row in board:
            cells = [
                str(int(val)).rjust(max_width) if val > 0 else ".".rjust(max_width)
                for val in row
            ]
            lines.append(" | ".join(cells))
        
        return "\n".join(lines)

    def _get_max_tile(self) -> int:
        """Get the maximum tile value on the board."""
        return int(self.current_board.max())

    def _get_sum(self) -> int:
        """Get the sum of all tile values."""
        return int(self.current_board.sum())

    def _calculate_reward(self, board_changed: bool) -> float:
        """
        Calculate reward for the current state.
        
        Reward structure:
        - Invalid move (board didn't change): -1
        - Win (reached target): 2
        - Otherwise: combination of max tile and total board value
        
        Args:
            board_changed: Whether the last move changed the board

        Returns:
            Reward value
        """
        if not board_changed:
            # Invalid move penalty
            return -1.0

        if self.won:
            # Double reward for winning
            return 2.0

        # Reward based on max tile and total board value
        max_tile = self._get_max_tile()
        total_value = self._get_sum()

        # Avoid log(0) by ensuring minimum value
        max_tile = max(max_tile, 2)
        total_value = max(total_value, 2)

        # Scale max value logarithmically between 0 for 2 and 1 for target_value
        max_value_reward = (math.log(max_tile, 2) - 1) / (
            math.log(self.target_value, 2) - 1
        )
        
        # Scale board value logarithmically between 0 for 2*16 and 1 for target_value*16
        board_value_reward = (math.log(total_value, 2) - 1) / (
            math.log(self.target_value * 16, 2) - 1
        )
        
        # Combine rewards: max value is weighted higher
        reward = max_value_reward + (board_value_reward * 0.2)

        return reward

    def _get_info(self) -> dict[str, Any]:
        """Get information about the current state."""
        max_tile = self._get_max_tile()
        
        # Check if there are valid moves remaining
        # A simple check: if board is full and no adjacent equal tiles
        has_valid_moves = self._check_valid_moves()
        
        return {
            "max_tile": max_tile,
            "total_value": self._get_sum(),
            "target_value": self.target_value,
            "won": self.won,
            "step": self.current_step,
            "has_valid_moves": has_valid_moves,
        }

    def _check_valid_moves(self) -> bool:
        """Check if any valid moves remain."""
        board = self.current_board
        
        # Check for empty cells (0 values)
        if (board == 0).any():
            return True
        
        # Check for adjacent equal tiles (horizontal)
        for i in range(4):
            for j in range(3):
                if board[i][j] == board[i][j + 1]:
                    return True
        
        # Check for adjacent equal tiles (vertical)
        for i in range(3):
            for j in range(4):
                if board[i][j] == board[i + 1][j]:
                    return True
        
        return False

    @staticmethod
    def from_dict(env_info: dict) -> "Game2048Env":
        """
        Create a Game2048Env instance from a dictionary (dataset entry + env_args).
        
        Args:
            env_info: Dictionary containing environment parameters
                     From dataset:
                     - seed: Random seed (required)
                     - target_value: Target tile value (required)
                     
                     From env_args (optional runtime config):
                     - max_steps: Maximum steps per episode (default: 40)
        
        Returns:
            Game2048Env instance
        """
        return Game2048Env(
            seed=env_info["seed"],
            target_value=env_info["target_value"],
            max_steps=env_info.get("max_steps", 40),
        )


if __name__ == "__main__":
    # Test the environment
    env = Game2048Env(seed=42, target_value=128)
    obs, info = env.reset()
    print("Initial Board:")
    print(obs)
    print(f"Info: {info}\n")

    # Play a few moves
    for move in ["left", "up", "right", "down"]:
        obs, reward, done, info = env.step(move)
        print(f"Move: {move.upper()}")
        print(obs)
        print(f"Reward: {reward}, Done: {done}")
        print(f"Info: {info}\n")

        if done:
            break
