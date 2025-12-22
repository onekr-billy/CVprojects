"""
Minimax algorithm with Alpha-Beta pruning for Gomoku AI.
Simplified and optimized version for better performance.
"""

from cache import Cache
from evaluate import FIVE

MAX_SCORE = 1000000000


class MinmaxAI:
    """
    Minimax AI with Alpha-Beta pruning.
    Simplified version without heavy VCT for better responsiveness.
    """
    
    def __init__(self):
        self.cache = Cache()
        self.node_count = 0
        self.max_nodes = 50000  # Limit search to prevent hanging
    
    def reset(self):
        """Reset the cache and statistics."""
        self.cache.clear()
        self.node_count = 0
    
    def _alphabeta(self, board, role, depth, alpha=-MAX_SCORE, beta=MAX_SCORE):
        """
        Simple Alpha-Beta pruning minimax.
        
        Args:
            board: Game board
            role: Current player
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Tuple of (score, best_move)
        """
        self.node_count += 1
        
        # Check node limit to prevent hanging
        if self.node_count > self.max_nodes:
            return (board.evaluate(role), None)
        
        # Terminal conditions
        if depth <= 0 or board.is_game_over():
            return (board.evaluate(role), None)
        
        # Check cache
        h = board.hash()
        cached = self.cache.get(h)
        if cached is not None and cached.get('depth', 0) >= depth:
            return (cached['value'], cached['move'])
        
        # Get candidate moves (limited number for performance)
        points = board.get_valuable_moves(role, depth)
        
        if not points:
            return (board.evaluate(role), None)
        
        # Limit moves to search for performance
        max_moves = 15 if depth > 2 else 20
        points = points[:max_moves]
        
        best_value = -MAX_SCORE
        best_move = points[0] if points else None
        
        for point in points:
            board.put(point[0], point[1], role)
            
            # Recursively search
            value, _ = self._alphabeta(board, -role, depth - 1, -beta, -alpha)
            value = -value
            
            board.undo()
            
            if value > best_value:
                best_value = value
                best_move = point
            
            alpha = max(alpha, value)
            
            # Winning move found
            if value >= FIVE:
                break
            
            # Alpha-Beta pruning
            if alpha >= beta:
                break
        
        # Update cache
        self.cache.put(h, {
            'depth': depth,
            'value': best_value,
            'move': best_move,
        })
        
        return (best_value, best_move)
    
    def minmax(self, board, role, depth=4, enable_vct=False):
        """
        Main minimax search.
        
        Args:
            board: Game board
            role: Current player
            depth: Search depth
            enable_vct: Ignored (kept for compatibility)
            
        Returns:
            Tuple of (score, best_move, best_path)
        """
        self.node_count = 0
        
        # Adjust max nodes based on depth
        self.max_nodes = 10000 * depth
        
        value, move = self._alphabeta(board, role, depth)
        
        if move is None:
            # No moves found, return center or any valid move
            center = board.size // 2
            if board.board[center][center] == 0:
                return (0, (center, center), [])
            valid_moves = board.get_valid_moves()
            if valid_moves:
                return (0, valid_moves[0], [])
            return (0, None, [])
        
        return (value, move, [move])


# Global AI instance
ai = MinmaxAI()


def minmax(board, role, depth=4, enable_vct=True):
    """
    Convenience function for minimax search.
    
    Args:
        board: Game board
        role: Current player
        depth: Search depth
        enable_vct: Whether to use VCT search
        
    Returns:
        Tuple of (score, best_move, best_path)
    """
    return ai.minmax(board, role, depth, enable_vct)
