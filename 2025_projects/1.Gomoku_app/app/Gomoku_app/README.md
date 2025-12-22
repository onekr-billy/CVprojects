# Gomoku (五子棋) Python GUI

A Python GUI application for playing Gomoku (Five in a Row) against an AI opponent.

This is a Python port of the original JavaScript implementation by [lihongxun945](https://github.com/lihongxun945/gobang).

## Features

- 15x15 game board with traditional Gomoku rules
- AI opponent using Minimax algorithm with Alpha-Beta pruning
- Iterative deepening for better move selection
- VCT (Victory by Continuous Three) for threat detection
- Adjustable difficulty levels (search depth 2-8)
- Option for AI to play first or second
- Move number display
- Undo functionality

## Requirements

- Python 3.7+
- tkinter (usually included with Python)

## Installation

No additional packages required! Just run with Python:

```bash
cd python_gui
python main.py
```

## How to Play

1. Click "开始游戏" (Start Game) to begin
2. Click on the board to place your piece
3. The AI will automatically respond
4. First to get 5 pieces in a row wins!

## Settings

- **电脑先手** (AI First): Let AI make the first move
- **显示序号** (Show Numbers): Display move order numbers on pieces
- **难度** (Difficulty): Adjust AI strength
  - 2 - 弱智 (Very Easy): Fast but weak
  - 4 - 简单 (Easy): Good for beginners
  - 6 - 普通 (Normal): Balanced
  - 8 - 困难 (Hard): Strong but slow

## Project Structure

```
python_gui/
├── main.py       # Main GUI application
├── board.py      # Board class - game state management
├── minmax.py     # Minimax AI algorithm
├── evaluate.py   # Board evaluation and scoring
├── shape.py      # Pattern detection (five, four, three, etc.)
├── zobrist.py    # Zobrist hashing for caching
├── cache.py      # Simple LRU cache
├── config.py     # Configuration constants
└── README.md     # This file
```

## AI Algorithm

The AI uses the following techniques:

1. **Minimax with Alpha-Beta Pruning**: Standard game tree search algorithm
2. **Iterative Deepening**: Gradually increases search depth for better time management
3. **Zobrist Hashing**: Efficient board state caching
4. **Pattern Matching**: Detects important shapes (five, four, three, two)
5. **VCT Search**: Looks for winning threat sequences

## Credits

- Original JavaScript implementation: [lihongxun945/gobang](https://github.com/lihongxun945/gobang)
- Python port: Based on the original codebase

## License

This project follows the same license as the original repository.
