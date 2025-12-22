#!/usr/bin/env python3
"""
Gomoku Core API Demo
=====================
This demonstrates how to use the core Gomoku AI without GUI.
Perfect for integrating with a robot arm system.

Core API:
- Board: Game state management
- minmax(): AI move calculation
- BLACK=1, WHITE=-1: Player constants
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import Board
from minmax import minmax
from config import BLACK, WHITE, BOARD_SIZE


class GomokuCoreAPI:
    """
    Simplified API wrapper for robot integration.
    Supports configurable rectangular board sizes (e.g., 8x12, 15x15, 10x20).
    """
    
    def __init__(self, board_size=15, rows=None, cols=None, ai_depth=4):
        """
        Initialize the Gomoku game.
        
        Args:
            board_size: Board size for square boards (9, 11, 13, 15, 19, etc.). Default is 15x15.
            rows: Number of rows (overrides board_size if specified)
            cols: Number of columns (overrides board_size if specified)
            ai_depth: AI search depth (2=easy, 4=normal, 6=hard)
        """
        if rows is not None and cols is not None:
            self.board_rows = rows
            self.board_cols = cols
            self.board = Board(rows=rows, cols=cols, first_role=BLACK)
        else:
            self.board_rows = board_size
            self.board_cols = board_size
            self.board = Board(size=board_size, first_role=BLACK)
            
        self.ai_depth = ai_depth
        self.human_role = BLACK  # Human plays black (first)
        self.ai_role = WHITE     # AI plays white
        
    def reset(self):
        """Reset the game to initial state."""
        if hasattr(self, 'board_rows') and hasattr(self, 'board_cols'):
            self.board = Board(rows=self.board_rows, cols=self.board_cols, first_role=BLACK)
            print(f"Game reset! Board size: {self.board_rows}x{self.board_cols}")
        else:
            self.board = Board(size=self.board_size, first_role=BLACK)
            print(f"Game reset! Board size: {self.board_size}x{self.board_size}")
        
    def human_move(self, x: int, y: int) -> bool:
        """
        Register a human move (e.g., detected by vision system).
        
        Args:
            x: Row position (0 to board_rows-1)
            y: Column position (0 to board_cols-1)
            
        Returns:
            True if move was valid, False otherwise
        """
        if not (0 <= x < self.board_rows and 0 <= y < self.board_cols):
            print(f"Error: Position ({x}, {y}) out of bounds for {self.board_rows}x{self.board_cols} board!")
            return False
            
        if self.board.board[x][y] != 0:
            print(f"Error: Position ({x}, {y}) already occupied!")
            return False
        
        success = self.board.put(x, y, self.human_role)
        if success:
            print(f"Human placed at ({x}, {y})")
        return success
    
    def get_ai_move(self) -> tuple:
        """
        Calculate the best AI move.
        
        Returns:
            (x, y) tuple for the best move, or None if game over
        """
        if self.board.is_game_over():
            print("Game is already over!")
            return None
        
        print(f"AI thinking (depth={self.ai_depth})...")
        score, move, path = minmax(self.board, self.ai_role, self.ai_depth)
        
        if move is None:
            print("No valid moves!")
            return None
        
        print(f"AI chose ({move[0]}, {move[1]}) with score {score}")
        return move
    
    def apply_ai_move(self, x: int, y: int) -> bool:
        """
        Apply the AI's move to the board.
        Call this after robot arm has placed the piece.
        
        Args:
            x: Row position
            y: Column position
            
        Returns:
            True if move was valid
        """
        success = self.board.put(x, y, self.ai_role)
        if success:
            print(f"AI move applied at ({x}, {y})")
        return success
    
    def check_winner(self) -> int:
        """
        Check if there's a winner.
        
        Returns:
            BLACK (1) if human wins, WHITE (-1) if AI wins, 0 if no winner
        """
        return self.board.get_winner()
    
    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self.board.is_game_over()
    
    def get_board_state(self) -> list:
        """
        Get current board state.
        
        Returns:
            MxN 2D list (M=rows, N=cols): 0=empty, 1=black, -1=white
        """
        return [row[:] for row in self.board.board]  # Return copy
    
    def get_board_size(self) -> tuple:
        """Get the board dimensions as (rows, cols)."""
        return (self.board_rows, self.board_cols)
    
    def get_board_rows(self) -> int:
        """Get the number of rows."""
        return self.board_rows
    
    def get_board_cols(self) -> int:
        """Get the number of columns."""
        return self.board_cols
    
    def print_board(self):
        """Print the board to console (for debugging)."""
        symbols = {0: '·', BLACK: '●', WHITE: '○'}
        print(f"\nBoard: {self.board_rows}x{self.board_cols}")
        print("   ", end="")
        for j in range(self.board_cols):
            print(f"{j:2}", end=" ")
        print()
        
        for i in range(self.board_rows):
            print(f"{i:2} ", end="")
            for j in range(self.board_cols):
                print(f" {symbols[self.board.board[i][j]]} ", end="")
            print()
        print()


def demo_game(board_size=15, rows=None, cols=None):
    """
    Demo: Simulate a short game between human and AI.
    
    Args:
        board_size: Size of square board (default 15)
        rows: Number of rows for rectangular board
        cols: Number of columns for rectangular board
    """
    if rows is not None and cols is not None:
        print("=" * 50)
        print(f"Gomoku Core API Demo ({rows}x{cols} board)")
        print("=" * 50)
        game = GomokuCoreAPI(rows=rows, cols=cols, ai_depth=4)
        center_row, center_col = rows // 2, cols // 2
    else:
        print("=" * 50)
        print(f"Gomoku Core API Demo ({board_size}x{board_size} board)")
        print("=" * 50)
        game = GomokuCoreAPI(board_size=board_size, ai_depth=4)
        center_row, center_col = board_size // 2, board_size // 2
    human_moves = [
        (center_row, center_col),       # Center
        (center_row, center_col + 1) if center_col + 1 < game.board_cols else (center_row, center_col - 1),   # Next to center
        (center_row - 1, center_col) if center_row - 1 >= 0 else (center_row + 1, center_col),   # Building vertical
    ]
    
    for move in human_moves:
        print(f"\n--- Turn {len(game.board.history) // 2 + 1} ---")
        
        # Human move (simulating vision detection)
        if not game.human_move(move[0], move[1]):
            print("Invalid human move!")
            continue
        
        # Check if human won
        winner = game.check_winner()
        if winner == BLACK:
            print("🎉 Human wins!")
            break
        
        # AI calculates and plays
        ai_move = game.get_ai_move()
        if ai_move:
            # In real robot: send ai_move to robot arm, wait for completion
            game.apply_ai_move(ai_move[0], ai_move[1])
            
            # Check if AI won
            winner = game.check_winner()
            if winner == WHITE:
                print("🤖 AI wins!")
                break
        
        # Show board state
        game.print_board()
    
    print("\nFinal board state:")
    game.print_board()
    
    # Show board as data (for robot integration)
    print("Board data (for robot arm):")
    state = game.get_board_state()
    rows, cols = game.get_board_size()
    occupied = [(i, j, state[i][j]) for i in range(rows) 
                for j in range(cols) if state[i][j] != 0]
    for x, y, piece in occupied:
        print(f"  ({x}, {y}): {'Black' if piece == BLACK else 'White'}")


def demo_api_usage(board_size=9, rows=None, cols=None):
    """
    Demo: Show basic API usage for robot integration.
    Uses a smaller board for faster demo.
    
    Args:
        board_size: Size of square board (default 9 for faster demo)
        rows: Number of rows for rectangular board
        cols: Number of columns for rectangular board
    """
    if rows is not None and cols is not None:
        print("\n" + "=" * 50)
        print(f"Robot Integration Example ({rows}x{cols} board)")
        print("=" * 50)
        game = GomokuCoreAPI(rows=rows, cols=cols, ai_depth=4)
    else:
        print("\n" + "=" * 50)
        print(f"Robot Integration Example ({board_size}x{board_size} board)")
        print("=" * 50)
        game = GomokuCoreAPI(board_size=board_size, ai_depth=4)
    
    print("""
Typical robot workflow:
1. Vision detects human placed a piece at (x, y)
2. Call: game.human_move(x, y)
3. Call: ai_move = game.get_ai_move()
4. Send ai_move to robot arm controller
5. Robot picks piece and places at ai_move position
6. Call: game.apply_ai_move(*ai_move)
7. Check: game.check_winner() for game end
8. Repeat from step 1
""")
    
    # Quick test
    print("Quick test - AI vs AI (5 moves each):")
    for i in range(5):
        # Pretend human (BLACK) move
        ai_as_black = minmax(game.board, BLACK, depth=2)
        if ai_as_black[1]:
            game.board.put(ai_as_black[1][0], ai_as_black[1][1], BLACK)
            print(f"  Black: {ai_as_black[1]}")
        
        if game.is_game_over():
            break
            
        # AI (WHITE) responds
        ai_move = game.get_ai_move()
        if ai_move:
            game.apply_ai_move(ai_move[0], ai_move[1])
            print(f"  White: {ai_move}")
        
        if game.is_game_over():
            break
    
    game.print_board()


def test_rectangular_boards():
    """Test various rectangular board configurations."""
    print("\n" + "#" * 60)
    print("# Testing Rectangular Board Configurations")
    print("#" * 60)
    
    # Test different rectangular configurations
    test_configs = [
        ("8x12 Rectangle (Wide)", {"rows": 8, "cols": 12}),
        ("15x8 Rectangle (Tall)", {"rows": 15, "cols": 8}),
        ("10x10 Square", {"rows": 10, "cols": 10}),
        ("6x20 Very Wide", {"rows": 6, "cols": 20}),
        ("20x6 Very Tall", {"rows": 20, "cols": 6}),
    ]
    
    for name, config in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")
        
        try:
            game = GomokuCoreAPI(**config, ai_depth=2)
            
            # Place a few pieces and test AI
            center_row, center_col = config["rows"] // 2, config["cols"] // 2
            
            # Ensure moves are within bounds
            test_moves = []
            if center_row >= 0 and center_col >= 0:
                test_moves.append((center_row, center_col))
            if center_row >= 0 and center_col + 1 < config["cols"]:
                test_moves.append((center_row, center_col + 1))
            elif center_row >= 0 and center_col - 1 >= 0:
                test_moves.append((center_row, center_col - 1))
            
            # Make test moves
            for i, move in enumerate(test_moves[:2]):  # Limit to 2 moves
                if game.human_move(move[0], move[1]):
                    print(f"Human move {i+1}: ({move[0]}, {move[1]})")
                    
                    # AI responds
                    ai_move = game.get_ai_move()
                    if ai_move:
                        game.apply_ai_move(ai_move[0], ai_move[1])
                        print(f"AI response: ({ai_move[0]}, {ai_move[1]})")
                    
                    if game.check_winner() != 0:
                        break
            
            game.print_board()
            rows, cols = game.get_board_size()
            print(f"✅ {name} test completed - Board: {rows}x{cols}")
            
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Demo with different board sizes
    print("\n" + "#" * 60)
    print("# Testing Square and Rectangular Board Sizes")
    print("#" * 60)
    
    # Traditional square boards
    print("\n🔲 SQUARE BOARDS:")
    
    # Standard 15x15 game
    demo_game(board_size=15)
    
    # Smaller 9x9 for faster robot demo
    demo_api_usage(board_size=9)
    
    # Quick test with 11x11
    print("\n" + "=" * 50)
    print("Quick 11x11 board test")
    print("=" * 50)
    game = GomokuCoreAPI(board_size=11, ai_depth=2)
    game.human_move(5, 5)  # Center
    ai_move = game.get_ai_move()
    if ai_move:
        game.apply_ai_move(*ai_move)
    game.print_board()
    
    # New rectangular board tests
    print("\n🔲 RECTANGULAR BOARDS:")
    
    # Test 8x12 rectangle
    demo_game(rows=8, cols=12)
    
    # Test robot API with rectangular board
    demo_api_usage(rows=10, cols=15)
    
    # Comprehensive rectangular tests
    test_rectangular_boards()
    
    print("\n✅ All demos completed successfully!")
    print("\nSupported configurations:")
    print("  • Square boards: 5x5 to 25x25 (or larger)")
    print("  • Rectangular boards: any MxN where M,N >= 5")
    print("  • Examples: 8x12, 15x8, 6x20, 20x6, etc.")
