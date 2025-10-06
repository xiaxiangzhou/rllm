"""
Utility functions for 2048 game operations.

Provides helper classes and functions for game logic, board manipulation,
and visualization. Used for testing and demonstration purposes.
"""

import random
from typing import Optional


class GameBoard:
    """Represents a 2048 game board state."""

    def __init__(self, grid: list[list[Optional[int]]], size: int = 4):
        """
        Initialize a game board.

        Args:
            grid: 2D list representing the board state
            size: Size of the board (default: 4 for 4x4)
        """
        self.grid = grid
        self.size = size

    @classmethod
    def create_empty(cls, size: int = 4) -> "GameBoard":
        """Create an empty board."""
        return cls(grid=[[None] * size for _ in range(size)], size=size)

    def get_empty_cells(self) -> list[tuple[int, int]]:
        """Return list of (row, col) coordinates of empty cells."""
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r][c] is None
        ]

    def add_random_tile(self, rng: random.Random) -> None:
        """Add a random tile (2 or 4) to an empty cell."""
        empty = self.get_empty_cells()
        if not empty:
            return

        row, col = rng.choice(empty)
        # 90% probability for 2, 10% for 4
        self.grid[row][col] = 2 if rng.random() < 0.9 else 4

    def get_max_value(self) -> int:
        """Get the maximum tile value on the board."""
        return max(
            (cell for row in self.grid for cell in row if cell is not None), default=0
        )

    def get_sum(self) -> int:
        """Get the sum of all tile values."""
        return sum(cell for row in self.grid for cell in row if cell is not None)

    def has_valid_moves(self) -> bool:
        """Check if any valid moves remain."""
        # Check for empty cells
        if any(cell is None for row in self.grid for cell in row):
            return True

        # Check for adjacent equal tiles (horizontal)
        for row in self.grid:
            for i in range(len(row) - 1):
                if row[i] == row[i + 1]:
                    return True

        # Check for adjacent equal tiles (vertical)
        for col in range(self.size):
            for row in range(self.size - 1):
                if self.grid[row][col] == self.grid[row + 1][col]:
                    return True

        return False

    def is_game_over(self, target_value: int = 128) -> bool:
        """
        Check if the game is finished.

        Args:
            target_value: Winning tile value

        Returns:
            True if game is over (won or no moves left)
        """
        # Check if won
        if self.get_max_value() >= target_value:
            return True

        # Check if no valid moves left
        if not self.has_valid_moves():
            return True

        return False

    def display(self) -> str:
        """Return a formatted string representation of the board."""
        if not any(cell is not None for row in self.grid for cell in row):
            max_width = 1
        else:
            max_width = max(
                len(str(cell)) for row in self.grid for cell in row if cell is not None
            )

        lines = []
        for row in self.grid:
            cells = [
                str(cell).rjust(max_width) if cell is not None else ".".rjust(max_width)
                for cell in row
            ]
            lines.append(" | ".join(cells))

        return "\n".join(lines)


def merge_line(line: list[Optional[int]]) -> list[Optional[int]]:
    """
    Merge tiles in a single line (left to right).

    Args:
        line: List of tile values (None for empty)

    Returns:
        Merged line with same length
    """
    # Remove empty cells
    tiles = [t for t in line if t is not None]

    # Merge adjacent equal tiles
    merged = []
    skip_next = False

    for i in range(len(tiles)):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged.append(tiles[i] * 2)
            skip_next = True
        else:
            merged.append(tiles[i])

    # Pad with None to maintain length
    result = merged + [None] * (len(line) - len(merged))
    return result


def apply_move(board: GameBoard, direction: str) -> GameBoard:
    """
    Apply a move to the board and return new board state.

    Args:
        board: Current board state
        direction: Direction string: "left", "right", "up", or "down"

    Returns:
        New board state after the move

    Raises:
        ValueError: If direction is invalid
    """
    if direction not in ["left", "right", "up", "down"]:
        raise ValueError(f"Invalid direction: {direction}")

    new_grid = [row[:] for row in board.grid]  # Deep copy
    size = board.size

    if direction == "left":
        for i in range(size):
            new_grid[i] = merge_line(new_grid[i])

    elif direction == "right":
        for i in range(size):
            reversed_row = new_grid[i][::-1]
            new_grid[i] = merge_line(reversed_row)[::-1]

    elif direction == "up":
        for col in range(size):
            column = [new_grid[row][col] for row in range(size)]
            merged_col = merge_line(column)
            for row in range(size):
                new_grid[row][col] = merged_col[row]

    elif direction == "down":
        for col in range(size):
            column = [new_grid[row][col] for row in range(size)]
            reversed_col = column[::-1]
            merged_col = merge_line(reversed_col)[::-1]
            for row in range(size):
                new_grid[row][col] = merged_col[row]

    return GameBoard(grid=new_grid, size=size)


def play_simple_game(seed: int = 42, moves: list[str] = None, target_value: int = 128) -> None:
    """
    Play a simple game with predefined moves for demonstration.

    Args:
        seed: Random seed for reproducibility
        moves: List of direction strings (e.g., ["left", "up", "right", ...])
        target_value: Winning tile value
    """
    if moves is None:
        moves = ["left", "up", "right", "down"]

    rng = random.Random(seed)
    board = GameBoard.create_empty()

    # Add initial tiles
    board.add_random_tile(rng)
    board.add_random_tile(rng)

    print(f"Initial board (seed={seed}, target={target_value}):")
    print(board.display())
    print(f"Max: {board.get_max_value()}, Sum: {board.get_sum()}")
    print(f"Game over: {board.is_game_over(target_value)}\n")

    for direction in moves:
        print(f"Move: {direction}")
        print("-" * 40)

        board = apply_move(board, direction)
        board.add_random_tile(rng)

        print(board.display())
        print(f"Max: {board.get_max_value()}, Sum: {board.get_sum()}")
        print(f"Game over: {board.is_game_over(target_value)}")

        if board.is_game_over(target_value):
            if board.get_max_value() >= target_value:
                print("You won!")
            else:
                print("No more moves!")
            break
        print()


if __name__ == "__main__":
    print("=" * 50)
    print("2048 Game Utilities Demo")
    print("=" * 50)
    print()

    # Demo 1: Simple game with XML moves
    play_simple_game(seed=42)

    print("\n" + "=" * 50)
    print("Testing Determinism")
    print("=" * 50)
    print()

    # Demo 2: Show determinism
    rng1 = random.Random(123)
    rng2 = random.Random(123)

    board1 = GameBoard.create_empty()
    board2 = GameBoard.create_empty()

    board1.add_random_tile(rng1)
    board2.add_random_tile(rng2)

    print("Board 1:")
    print(board1.display())
    print("\nBoard 2 (same seed):")
    print(board2.display())
    print("\nBoards are identical:", board1.grid == board2.grid)

