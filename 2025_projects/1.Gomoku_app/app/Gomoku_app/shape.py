"""
Shape detection for Gomoku patterns.
Detects patterns like five-in-a-row, four, three, two, etc.
"""

import re


# Shape constants
class Shapes:
    FIVE = 5           # Five in a row (winning)
    BLOCK_FIVE = 50    # Blocked five
    FOUR = 4           # Open four (guaranteed win next move)
    FOUR_FOUR = 44     # Double block four
    FOUR_THREE = 43    # Block four + open three
    THREE_THREE = 33   # Double open three
    BLOCK_FOUR = 40    # Blocked four
    THREE = 3          # Open three
    BLOCK_THREE = 30   # Blocked three
    TWO_TWO = 22       # Double open two
    TWO = 2            # Open two
    NONE = 0           # No significant pattern


# Regular expression patterns for shape detection
# 1 = self, 2 = opponent or wall, 0 = empty
PATTERNS = {
    'five': re.compile(r'11111'),
    'block_five': re.compile(r'211111|111112'),
    'four': re.compile(r'011110'),
    'block_four': re.compile(r'10111|11011|11101|211110|211101|211011|210111|011112|101112|110112|111012'),
    'three': re.compile(r'011100|011010|010110|001110'),
    'block_three': re.compile(r'211100|211010|210110|001112|010112|011012'),
    'two': re.compile(r'001100|011000|000110|010100|001010'),
}


def get_shape(board, x, y, offset_x, offset_y, role):
    """
    Get the shape at position (x, y) in direction (offset_x, offset_y).
    
    Args:
        board: 2D array with walls (size+2 x size+2), 0=empty, 1=black, -1=white, 2=wall
        x, y: Position to check (0-indexed, without wall offset)
        offset_x, offset_y: Direction to check
        role: Current player (1 for black, -1 for white)
        
    Returns:
        Tuple of (shape, self_count, opponent_count, empty_count)
    """
    opponent = -role
    empty_count = 0
    self_count = 1
    opponent_count = 0
    shape = Shapes.NONE
    
    # Quick check: skip if empty neighbors
    if (board[x + offset_x + 1][y + offset_y + 1] == 0 and
        board[x - offset_x + 1][y - offset_y + 1] == 0 and
        board[x + 2 * offset_x + 1][y + 2 * offset_y + 1] == 0 and
        board[x - 2 * offset_x + 1][y - 2 * offset_y + 1] == 0):
        return (Shapes.NONE, self_count, opponent_count, empty_count)
    
    # Quick check for TWO pattern (optimization)
    for i in range(-3, 4):
        if i == 0:
            continue
        nx, ny = x + i * offset_x + 1, y + i * offset_y + 1
        if nx < 0 or ny < 0 or nx >= len(board) or ny >= len(board[0]):
            continue
        current_role = board[nx][ny]
        if current_role == 2:
            opponent_count += 1
        elif current_role == role:
            self_count += 1
        elif current_role == 0:
            empty_count += 1
    
    if self_count == 2:
        if not opponent_count:
            return (Shapes.TWO, self_count, opponent_count, empty_count)
        else:
            return (Shapes.NONE, self_count, opponent_count, empty_count)
    
    # Build pattern string for matching
    empty_count = 0
    self_count = 1
    opponent_count = 0
    result_string = '1'
    
    # Check positive direction
    for i in range(1, 6):
        nx, ny = x + i * offset_x + 1, y + i * offset_y + 1
        if nx < 0 or ny < 0 or nx >= len(board) or ny >= len(board[0]):
            result_string += '2'
            opponent_count += 1
            break
        current_role = board[nx][ny]
        if current_role == 2:
            result_string += '2'
        elif current_role == 0:
            result_string += '0'
        else:
            result_string += '1' if current_role == role else '2'
        
        if current_role == 2 or current_role == opponent:
            opponent_count += 1
            break
        if current_role == 0:
            empty_count += 1
        if current_role == role:
            self_count += 1
    
    # Check negative direction
    for i in range(1, 6):
        nx, ny = x - i * offset_x + 1, y - i * offset_y + 1
        if nx < 0 or ny < 0 or nx >= len(board) or ny >= len(board[0]):
            result_string = '2' + result_string
            opponent_count += 1
            break
        current_role = board[nx][ny]
        if current_role == 2:
            result_string = '2' + result_string
        elif current_role == 0:
            result_string = '0' + result_string
        else:
            result_string = ('1' if current_role == role else '2') + result_string
        
        if current_role == 2 or current_role == opponent:
            opponent_count += 1
            break
        if current_role == 0:
            empty_count += 1
        if current_role == role:
            self_count += 1
    
    # Match patterns
    if PATTERNS['five'].search(result_string):
        shape = Shapes.FIVE
    elif PATTERNS['four'].search(result_string):
        shape = Shapes.FOUR
    elif PATTERNS['block_four'].search(result_string):
        shape = Shapes.BLOCK_FOUR
    elif PATTERNS['three'].search(result_string):
        shape = Shapes.THREE
    elif PATTERNS['block_three'].search(result_string):
        shape = Shapes.BLOCK_THREE
    elif PATTERNS['two'].search(result_string):
        shape = Shapes.TWO
    
    return (shape, self_count, opponent_count, empty_count)


def is_five(shape):
    """Check if shape is a winning five."""
    return shape == Shapes.FIVE or shape == Shapes.BLOCK_FIVE


def is_four(shape):
    """Check if shape is a four (open or blocked)."""
    return shape == Shapes.FOUR or shape == Shapes.BLOCK_FOUR or shape == Shapes.FOUR_FOUR or shape == Shapes.FOUR_THREE


def get_all_shapes_of_point(shape_cache, x, y, role):
    """
    Get all shapes at a point from cache.
    
    Args:
        shape_cache: Cache dictionary [role][direction][x][y] = shape
        x, y: Position
        role: Player role
        
    Returns:
        List of shapes in all 4 directions
    """
    shapes_list = []
    for direction in range(4):
        shape = shape_cache[role][direction][x][y]
        if shape:
            shapes_list.append(shape)
    return shapes_list


# Direction mappings
ALL_DIRECTIONS = [
    (0, 1),   # Horizontal
    (1, 0),   # Vertical
    (1, 1),   # Diagonal \
    (1, -1),  # Diagonal /
]


def direction_to_index(ox, oy):
    """Convert direction offset to index."""
    if ox == 0:
        return 0  # Horizontal |
    if oy == 0:
        return 1  # Vertical -
    if ox == oy:
        return 2  # Diagonal \
    return 3  # Diagonal /
