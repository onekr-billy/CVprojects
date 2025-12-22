"""
Board evaluation module for Gomoku AI.
Handles scoring of board positions and finding valuable moves.
"""

from shape import Shapes, get_shape, is_five, is_four, get_all_shapes_of_point, ALL_DIRECTIONS, direction_to_index
from config import BOARD_SIZE

# Score constants
FIVE = 10000000
BLOCK_FIVE = FIVE
FOUR = 100000
FOUR_FOUR = FOUR       # Double block four
FOUR_THREE = FOUR      # Block four + open three
THREE_THREE = FOUR // 2  # Double open three
BLOCK_FOUR = 1500
THREE = 1000
BLOCK_THREE = 150
TWO_TWO = 200          # Double open two
TWO = 100
BLOCK_TWO = 15
ONE = 10
BLOCK_ONE = 1


def get_real_shape_score(shape):
    """
    Convert shape to score value.
    Note: These scores are for positions where no piece has been placed yet.
    """
    score_map = {
        Shapes.FIVE: FOUR,
        Shapes.BLOCK_FIVE: BLOCK_FOUR,
        Shapes.FOUR: THREE,
        Shapes.FOUR_FOUR: THREE,
        Shapes.FOUR_THREE: THREE,
        Shapes.BLOCK_FOUR: BLOCK_THREE,
        Shapes.THREE: TWO,
        Shapes.THREE_THREE: THREE_THREE // 10,
        Shapes.BLOCK_THREE: BLOCK_TWO,
        Shapes.TWO: ONE,
        Shapes.TWO_TWO: TWO_TWO // 10,
    }
    return score_map.get(shape, 0)


def coordinate_to_position(x, y, size):
    """Convert (x, y) coordinates to linear position."""
    return x * size + y


def position_to_coordinate(position, size):
    """Convert linear position to (x, y) coordinates."""
    return position // size, position % size


class Evaluate:
    """
    Evaluator class for Gomoku board positions.
    Calculates scores for positions and finds valuable moves.
    """
    
    def __init__(self, size=15):
        self.size = size
        # For rectangular boards, these will be set later
        self.actual_rows = size
        self.actual_cols = size
        
        # Board with walls (size+2 x size+2)
        # 0 = empty, 1 = black, -1 = white, 2 = wall
        self.board = [[2 if (i == 0 or j == 0 or i == size + 1 or j == size + 1) else 0
                       for j in range(size + 2)] for i in range(size + 2)]
        
        # Score arrays for black and white
        self.black_scores = [[0] * size for _ in range(size)]
        self.white_scores = [[0] * size for _ in range(size)]
        
        self.init_points()
        self.history = []  # Record of moves: [(position, role), ...]
    
    def set_dimensions(self, rows, cols):
        """Set actual board dimensions for rectangular boards."""
        self.actual_rows = rows
        self.actual_cols = cols
    
    def init_points(self):
        """Initialize shape cache and points cache."""
        # Cache for each point's shape in each direction
        # shape_cache[role][direction][x][y] = shape
        self.shape_cache = {}
        for role in [1, -1]:
            self.shape_cache[role] = {}
            for direction in range(4):
                self.shape_cache[role][direction] = [[Shapes.NONE] * self.size for _ in range(self.size)]
        
        # Cache for points with each shape
        # points_cache[role][shape] = set of positions
        self.points_cache = {}
        for role in [1, -1]:
            self.points_cache[role] = {}
            for shape in [Shapes.FIVE, Shapes.BLOCK_FIVE, Shapes.FOUR, Shapes.FOUR_FOUR,
                         Shapes.FOUR_THREE, Shapes.THREE_THREE, Shapes.BLOCK_FOUR,
                         Shapes.THREE, Shapes.BLOCK_THREE, Shapes.TWO_TWO, Shapes.TWO, Shapes.NONE]:
                self.points_cache[role][shape] = set()
    
    def move(self, x, y, role):
        """Place a piece and update scores."""
        # Bounds check for rectangular boards
        if not (0 <= x < min(self.size, self.actual_rows) and 0 <= y < min(self.size, self.actual_cols)):
            return False
            
        # Clear cache for this position
        for d in range(4):
            self.shape_cache[role][d][x][y] = Shapes.NONE
            self.shape_cache[-role][d][x][y] = Shapes.NONE
        self.black_scores[x][y] = 0
        self.white_scores[x][y] = 0
        
        # Update board and scores
        self.board[x + 1][y + 1] = role
        self.update_point(x, y)
        self.history.append((coordinate_to_position(x, y, self.size), role))
    
    def undo(self, x, y):
        """Remove a piece and update scores."""
        # Bounds check for rectangular boards
        if not (0 <= x < min(self.size, self.actual_rows) and 0 <= y < min(self.size, self.actual_cols)):
            return
            
        self.board[x + 1][y + 1] = 0
        self.update_point(x, y)
        if self.history:
            self.history.pop()
    
    def update_point(self, x, y):
        """Update scores around position (x, y) after a change - simplified."""
        # Only update the changed point and immediate neighbors
        for di in range(-1, 2):
            for dj in range(-1, 2):
                nx, ny = x + di, y + dj
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx + 1][ny + 1] == 0:
                        self.update_single_point(nx, ny, 1)
                        self.update_single_point(nx, ny, -1)
    
    def update_single_point(self, x, y, role, direction=None):
        """Calculate score for a single point."""
        if self.board[x + 1][y + 1] != 0:
            return
        
        # Temporarily place the piece
        self.board[x + 1][y + 1] = role
        
        directions = [direction] if direction else ALL_DIRECTIONS
        shape_cache = self.shape_cache[role]
        
        # Clear cache for these directions
        for ox, oy in directions:
            shape_cache[direction_to_index(ox, oy)][x][y] = Shapes.NONE
        
        score = 0
        block_four_count = 0
        three_count = 0
        two_count = 0
        
        # Calculate existing scores
        for int_direction in range(4):
            shape = shape_cache[int_direction][x][y]
            if shape > Shapes.NONE:
                score += get_real_shape_score(shape)
                if shape == Shapes.BLOCK_FOUR:
                    block_four_count += 1
                if shape == Shapes.THREE:
                    three_count += 1
                if shape == Shapes.TWO:
                    two_count += 1
        
        # Calculate new scores for changed directions
        for ox, oy in directions:
            int_direction = direction_to_index(ox, oy)
            shape, self_count, _, _ = get_shape(self.board, x, y, ox, oy, role)
            
            if shape:
                shape_cache[int_direction][x][y] = shape
                if shape == Shapes.BLOCK_FOUR:
                    block_four_count += 1
                if shape == Shapes.THREE:
                    three_count += 1
                if shape == Shapes.TWO:
                    two_count += 1
                
                # Check for compound shapes
                if block_four_count >= 2:
                    shape = Shapes.FOUR_FOUR
                elif block_four_count and three_count:
                    shape = Shapes.FOUR_THREE
                elif three_count >= 2:
                    shape = Shapes.THREE_THREE
                elif two_count >= 2:
                    shape = Shapes.TWO_TWO
                
                score += get_real_shape_score(shape)
        
        # Remove temporary piece
        self.board[x + 1][y + 1] = 0
        
        if role == 1:
            self.black_scores[x][y] = score
        else:
            self.white_scores[x][y] = score
        
        return score
    
    def evaluate(self, role):
        """Calculate total board score for a role."""
        black_score = sum(sum(row) for row in self.black_scores)
        white_score = sum(sum(row) for row in self.white_scores)
        
        if role == 1:
            return black_score - white_score
        else:
            return white_score - black_score
    
    def get_points(self, role, depth=0, vct=False, vcf=False):
        """Get points grouped by shape."""
        first = role if depth % 2 == 0 else -role
        
        points = {shape: set() for shape in [
            Shapes.FIVE, Shapes.BLOCK_FIVE, Shapes.FOUR, Shapes.FOUR_FOUR,
            Shapes.FOUR_THREE, Shapes.THREE_THREE, Shapes.BLOCK_FOUR,
            Shapes.THREE, Shapes.BLOCK_THREE, Shapes.TWO_TWO, Shapes.TWO, Shapes.NONE
        ]}
        
        for r in [role, -role]:
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i + 1][j + 1] != 0:
                        continue
                    
                    four_count = 0
                    block_four_count = 0
                    three_count = 0
                    
                    for direction in range(4):
                        shape = self.shape_cache[r][direction][i][j]
                        if not shape:
                            continue
                        
                        # VCF (Victory by Continuous Four) filtering
                        if vcf:
                            if r == first and not is_four(shape) and not is_five(shape):
                                continue
                            if r == -first and is_five(shape):
                                continue
                        
                        point = i * self.size + j
                        
                        # VCT (Victory by Continuous Three) filtering
                        if vct:
                            if depth % 2 == 0:  # Attacking
                                if depth == 0 and r != first:
                                    continue
                                if shape != Shapes.THREE and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == Shapes.THREE and r != first:
                                    continue
                            else:  # Defending
                                if shape != Shapes.THREE and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == Shapes.THREE and r == -first:
                                    continue
                        
                        if vcf:
                            if not is_four(shape) and not is_five(shape):
                                continue
                        
                        points[shape].add(point)
                        
                        if shape == Shapes.FOUR:
                            four_count += 1
                        elif shape == Shapes.BLOCK_FOUR:
                            block_four_count += 1
                        elif shape == Shapes.THREE:
                            three_count += 1
                        
                        # Check for compound shapes
                        union_shape = None
                        if four_count >= 2:
                            union_shape = Shapes.FOUR_FOUR
                        elif block_four_count and three_count:
                            union_shape = Shapes.FOUR_THREE
                        elif three_count >= 2:
                            union_shape = Shapes.THREE_THREE
                        
                        if union_shape:
                            points[union_shape].add(point)
        
        return points
    
    def get_moves(self, role, depth=0, only_three=False, only_four=False):
        """
        Get valuable moves for the current position.
        Uses a simpler, faster approach based on neighbor positions.
        
        Args:
            role: Current player role
            depth: Current search depth
            only_three: Only return three/four moves (for VCT)
            only_four: Only return four moves (for VCF)
            
        Returns:
            List of (x, y) positions sorted by score
        """
        # Fast approach: find empty positions near existing pieces
        candidates = []
        has_pieces = False
        
        # Use actual board dimensions for rectangular boards
        max_row = min(self.size, self.actual_rows)
        max_col = min(self.size, self.actual_cols)
        
        for i in range(max_row):
            for j in range(max_col):
                if self.board[i + 1][j + 1] != 0:
                    has_pieces = True
                    continue
                
                # Check if this position is near any piece (within 2 cells)
                is_neighbor = False
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < max_row and 0 <= nj < max_col:
                            if self.board[ni + 1][nj + 1] != 0:
                                is_neighbor = True
                                break
                    if is_neighbor:
                        break
                
                if is_neighbor:
                    # Calculate combined score for both players
                    score = self.black_scores[i][j] + self.white_scores[i][j]
                    candidates.append((score, i, j))
        
        # If no pieces on board, return center
        if not has_pieces:
            center_row = min(self.size, self.actual_rows) // 2
            center_col = min(self.size, self.actual_cols) // 2
            return [(center_row, center_col)]
        
        # Sort by score (highest first) and return positions
        candidates.sort(reverse=True)
        return [(c[1], c[2]) for c in candidates[:20]]  # Limit to top 20
    
    def display(self):
        """Display the current board state."""
        result = ''
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                cell = self.board[i][j]
                if cell == 1:
                    result += 'O '
                elif cell == -1:
                    result += 'X '
                else:
                    result += '- '
            result += '\n'
        print(result)
        return result
