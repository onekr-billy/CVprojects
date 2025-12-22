"""
Configuration for Gomoku (Gobang) game
"""

# Default board dimensions (can be overridden)
DEFAULT_BOARD_ROWS = 15
DEFAULT_BOARD_COLS = 15

# For backward compatibility
BOARD_SIZE = 15

# Game status
class Status:
    IDLE = 'idle'       # Game not started
    GAMING = 'gaming'   # Game in progress
    ENDED = 'ended'     # Game ended

# Player roles
BLACK = 1   # Black player (goes first)
WHITE = -1  # White player
EMPTY = 0   # Empty cell
WALL = 2    # Board boundary (used in evaluation)
