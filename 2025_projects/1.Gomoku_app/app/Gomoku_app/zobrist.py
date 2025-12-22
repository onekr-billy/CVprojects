"""
Zobrist hashing for efficient board state caching.
Zobrist hashing is a technique to compute a unique hash for each board state,
which allows for efficient caching and comparison of board states.
"""

import random


class Zobrist:
    def __init__(self, size=15, seed=42):
        """
        Initialize Zobrist hashing table.
        
        Args:
            size: Board size (default 15x15)
            seed: Random seed for reproducibility
        """
        self.size = size
        random.seed(seed)
        
        # Create random numbers for each piece type at each position
        # zobrist_table[role][x][y] = random 64-bit number
        # role: 1 for black, -1 for white
        self.zobrist_table = {
            1: [[random.getrandbits(64) for _ in range(size)] for _ in range(size)],
            -1: [[random.getrandbits(64) for _ in range(size)] for _ in range(size)]
        }
        
        self.hash = 0
    
    def toggle_piece(self, x, y, role):
        """
        Toggle a piece at position (x, y).
        XOR operation ensures that toggling twice returns to original hash.
        
        Args:
            x: Row position
            y: Column position
            role: 1 for black, -1 for white
        """
        self.hash ^= self.zobrist_table[role][x][y]
    
    def get_hash(self):
        """Return the current hash value."""
        return self.hash
    
    def reset(self):
        """Reset the hash to initial state."""
        self.hash = 0
