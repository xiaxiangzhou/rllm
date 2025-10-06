"""
2048 Game Environment for RLLM

A classic 2048 puzzle game where the player combines tiles to reach a target value.
"""

import random
import math
from typing import Any

from rllm.environments.base.base_env import BaseEnv

# Import core game logic from examples
import sys
from pathlib import Path

# Add examples directory to path to import utils
examples_path = Path(__file__).parent.parent.parent.parent / "examples" / "2048"
sys.path.insert(0, str(examples_path))

from utils import GameBoard, apply_move


class Game2048Env(BaseEnv):
    """
    2048 Game Environment.

    The player moves tiles on a 4x4 grid. When two tiles with the same number
    touch, they merge into one. The goal is to reach a target value (default: 128).
    """

    INVALID_ACTION = "invalid"

    def __init__(
        self,
        seed: int = 0,
        board_size: int = 4,
        target_value: int = 128,
        max_steps: int = 40,
        **kwargs,
    ):
        """
        Initialize the 2048 environment.

        Args:
            seed: Random seed for reproducibility
            board_size: Size of the board (default: 4 for 4x4 grid)
            target_value: Target tile value to win (default: 128)
            max_steps: Maximum number of steps before termination
        """
        self.seed_value = seed
        self.board_size = board_size
        self.target_value = target_value
        self.max_steps = max_steps
        self.current_step = 0

        # Game state (initialized in reset)
        self.rng: random.Random | None = None
        self.game_board: GameBoard | None = None
        self.game_over = False
        self.won = False

    def reset(self, **kwargs) -> tuple[str, dict]:
        """
        Reset the environment to start a new game.

        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset random generator
        self.rng = random.Random(self.seed_value)

        # Create new game board
        self.game_board = GameBoard.create_empty(size=self.board_size)

        # Add two initial tiles
        self.game_board.add_random_tile(self.rng)
        self.game_board.add_random_tile(self.rng)

        self.game_over = False
        self.won = False
        self.current_step = 0

        observation = self.game_board.display()
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
        if action not in ["left", "right", "up", "down"]:
            # Invalid action
            observation = self.game_board.display()
            reward = -1.0
            done = True
            info = self._get_info()
            info["invalid_action"] = True
            self.game_over = True
            return observation, reward, done, info

        # Save board state before move
        board_before = [row[:] for row in self.game_board.grid]

        # Apply the move (using core game logic)
        self.game_board = apply_move(self.game_board, action)

        # Check if board changed
        board_changed = board_before != self.game_board.grid

        if board_changed:
            # Add a new tile only if the board changed
            self.game_board.add_random_tile(self.rng)

        # Check win condition
        max_tile = self.game_board.get_max_value()
        if max_tile >= self.target_value:
            self.won = True
            self.game_over = True

        # Check if game is over (no more moves or max steps reached)
        if not self.game_board.has_valid_moves() or self.current_step >= self.max_steps:
            self.game_over = True

        # Calculate reward
        reward = self._calculate_reward(board_changed)

        observation = self.game_board.display()
        done = self.game_over
        info = self._get_info()

        return observation, reward, done, info

    def _calculate_reward(self, board_changed: bool) -> float:
        """
        Calculate reward for the current state.
        
        Reward structure (matching ART):
        - Invalid move: -1
        - Win (reached target): 2
        - Otherwise: combination of max tile and total board value
        
        Args:
            board_changed: Whether the last move changed the board

        Returns:
            Reward value
        """
        if not board_changed:
            # Invalid move penalty
            return -1

        if self.won:
            # Double reward for winning
            return 2

        # Reward based on max tile and total board value
        max_tile = self.game_board.get_max_value()
        total_value = self.game_board.get_sum()

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
        return {
            "max_tile": self.game_board.get_max_value(),
            "total_value": self.game_board.get_sum(),
            "target_value": self.target_value,
            "won": self.won,
            "step": self.current_step,
            "has_valid_moves": self.game_board.has_valid_moves(),
        }

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
                     - max_steps: Maximum steps per episode (default: 100)
        
        Returns:
            Game2048Env instance
        
        Note:
            board_size is always 4x4 (standard 2048 game)
        """
        return Game2048Env(
            seed=env_info["seed"],
            board_size=4,  # Always 4x4 for standard 2048
            target_value=env_info["target_value"],
            max_steps=env_info.get("max_steps", 40),  # From env_args or default
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

