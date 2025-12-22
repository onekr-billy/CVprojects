"""
Board class for Gomoku game.
Handles game state, move validation, and winner detection.
"""

from zobrist import Zobrist
from cache import Cache
from evaluate import Evaluate, FIVE
from config import BOARD_SIZE


class Board:
    """
    Gomoku game board class.
    Manages the board state, moves, and game logic.
    """
    
    def __init__(self, size=15, rows=None, cols=None, first_role=1):
        """
        Initialize the game board.
        
        Args:
            size: Board size (default 15x15, used if rows/cols not specified)
            rows: Number of rows (overrides size if specified)
            cols: Number of columns (overrides size if specified)
            first_role: Who goes first (1=black, -1=white)
        """
        # Handle rectangular vs square board
        if rows is not None and cols is not None:
            self.rows = rows
            self.cols = cols
            self.size = max(rows, cols)  # For backward compatibility
        else:
            self.rows = size
            self.cols = size
            self.size = size
            
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.first_role = first_role
        self.role = first_role  # Current player
        self.history = []       # Move history
        
        # Caching systems
        self.zobrist = Zobrist(self.size)  # Use max dimension for hash table size
        self.winner_cache = Cache()
        self.gameover_cache = Cache()
        self.evaluate_cache = Cache()
        self.valuable_moves_cache = Cache()
        
        # Evaluator for scoring (use max dimension for compatibility)
        self.evaluator = Evaluate(self.size)
        
        # Set actual board dimensions in evaluator for bounds checking
        if hasattr(self.evaluator, 'set_dimensions'):
            self.evaluator.set_dimensions(self.rows, self.cols)
    
    def is_game_over(self):
        """Check if the game is over (someone won or board is full)."""
        h = self.hash()
        cached = self.gameover_cache.get(h)
        if cached is not None:
            return cached
        
        if self.get_winner() != 0:
            self.gameover_cache.put(h, True)
            return True
        
        # Check for full board
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    self.gameover_cache.put(h, False)
                    return False
        
        self.gameover_cache.put(h, True)
        return True
    
    def get_winner(self):
        """
        Check if there's a winner.
        
        Returns:
            1 for black win, -1 for white win, 0 for no winner yet
        """
        h = self.hash()
        cached = self.winner_cache.get(h)
        if cached is not None:
            return cached
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonals
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    continue
                
                for dx, dy in directions:
                    count = 0
                    while (0 <= i + dx * count < self.rows and
                           0 <= j + dy * count < self.cols and
                           self.board[i + dx * count][j + dy * count] == self.board[i][j]):
                        count += 1
                    
                    if count >= 5:
                        self.winner_cache.put(h, self.board[i][j])
                        return self.board[i][j]
        
        self.winner_cache.put(h, 0)
        return 0
    
    def get_valid_moves(self):
        """Get all empty positions."""
        moves = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def put(self, i, j, role=None):
        """
        Place a piece on the board.
        
        Args:
            i: Row position
            j: Column position
            role: Player role (optional, uses current role if not specified)
            
        Returns:
            True if move was successful, False otherwise
        """
        if role is None:
            role = self.role
        
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            print(f"Invalid move: out of bounds ({i}, {j})")
            return False
        
        if self.board[i][j] != 0:
            print(f"Invalid move: position occupied ({i}, {j})")
            return False
        
        self.board[i][j] = role
        self.history.append({'i': i, 'j': j, 'role': role})
        self.zobrist.toggle_piece(i, j, role)
        self.evaluator.move(i, j, role)
        self.role *= -1  # Switch player
        return True
    
    def undo(self):
        """
        Undo the last move.
        
        Returns:
            True if successful, False if no moves to undo
        """
        if not self.history:
            print("No moves to undo!")
            return False
        
        last_move = self.history.pop()
        i, j, role = last_move['i'], last_move['j'], last_move['role']
        self.board[i][j] = 0
        self.role = role
        self.zobrist.toggle_piece(i, j, role)
        self.evaluator.undo(i, j)
        return True
    
    def get_valuable_moves(self, role, depth=0, only_three=False, only_four=False):
        """
        Get valuable moves for AI decision making.
        
        Args:
            role: Current player role
            depth: Search depth
            only_three: Only consider three/four patterns (VCT)
            only_four: Only consider four patterns (VCF)
            
        Returns:
            List of (x, y) positions
        """
        h = self.hash()
        cached = self.valuable_moves_cache.get(h)
        if cached is not None:
            if (cached['role'] == role and cached['depth'] == depth and 
                cached['only_three'] == only_three and cached['only_four'] == only_four):
                return cached['moves']
        
        moves = self.evaluator.get_moves(role, depth, only_three, only_four)
        
        # Add center point if not occupied and not in restricted mode
        if not only_three and not only_four:
            center_row, center_col = self.rows // 2, self.cols // 2
            if (0 <= center_row < self.rows and 0 <= center_col < self.cols and 
                self.board[center_row][center_col] == 0 and (center_row, center_col) not in moves):
                moves.append((center_row, center_col))
        
        # Filter out invalid moves for rectangular boards
        valid_moves = [(x, y) for x, y in moves if 0 <= x < self.rows and 0 <= y < self.cols]
        moves = valid_moves
        
        self.valuable_moves_cache.put(h, {
            'role': role,
            'moves': moves,
            'depth': depth,
            'only_three': only_three,
            'only_four': only_four
        })
        
        return moves
    
    def evaluate(self, role):
        """
        Evaluate the board position for a given role.
        
        Args:
            role: Player role to evaluate for
            
        Returns:
            Score value (positive is good for role)
        """
        return self.evaluator.evaluate(role)
    
    def hash(self):
        """Get the Zobrist hash of the current board state."""
        return self.zobrist.get_hash()
    
    def display(self, extra_points=None):
        """
        Display the board state.
        
        Args:
            extra_points: List of positions to mark with '?'
            
        Returns:
            String representation of the board
        """
        if extra_points is None:
            extra_points = []
        
        extra_positions = set(i * self.cols + j for i, j in extra_points)
        
        result = '  '
        for j in range(self.cols):
            result += f'{j:2}'
        result += '\n'
        
        for i in range(self.rows):
            result += f'{i:2}'
            for j in range(self.cols):
                position = i * self.cols + j
                if position in extra_positions:
                    result += ' ?'
                elif self.board[i][j] == 1:
                    result += ' O'
                elif self.board[i][j] == -1:
                    result += ' X'
                else:
                    result += ' -'
            result += '\n'
        
        return result
    
    def copy(self):
        """Create a copy of the board."""
        new_board = Board(rows=self.rows, cols=self.cols, first_role=self.first_role)
        for move in self.history:
            new_board.put(move['i'], move['j'], move['role'])
        return new_board
    
    def reverse(self):
        """Create a copy with reversed roles (for opponent analysis)."""
        return self.copy()
