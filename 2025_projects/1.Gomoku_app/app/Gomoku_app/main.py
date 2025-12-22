"""
Gomoku (五子棋) - Python GUI Application
A Gomoku game with AI opponent using Minimax algorithm with Alpha-Beta pruning.

Based on the original JavaScript implementation by lihongxun945.
https://github.com/lihongxun945/gobang
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import Board
from minmax import minmax, ai
from config import DEFAULT_BOARD_ROWS, DEFAULT_BOARD_COLS, Status, BLACK, WHITE


class GomokuGUI:
    """
    Main GUI class for Gomoku game.
    """
    
    # Constants for drawing
    CELL_SIZE = 35
    PADDING = 30
    PIECE_RADIUS = 14
    
    # Colors
    BOARD_COLOR = '#DEB887'  # Burlywood
    LINE_COLOR = '#8B4513'   # SaddleBrown
    BLACK_COLOR = '#000000'
    WHITE_COLOR = '#FFFFFF'
    HIGHLIGHT_COLOR = '#FF0000'
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("五子棋 AI - Gomoku")
        self.root.resizable(False, False)
        
        # Board dimensions
        self.board_rows = DEFAULT_BOARD_ROWS
        self.board_cols = DEFAULT_BOARD_COLS
        
        # Game state
        self.board = None
        self.status = Status.IDLE
        self.ai_first = True
        self.depth = 4
        self.loading = False
        self.show_numbers = False
        self.winner = None
        self.last_move = None
        
        # Create GUI components
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Canvas for game board
        canvas_width = 2 * self.PADDING + (self.board_cols - 1) * self.CELL_SIZE
        canvas_height = 2 * self.PADDING + (self.board_rows - 1) * self.CELL_SIZE
        self.canvas = tk.Canvas(
            main_frame, 
            width=canvas_width, 
            height=canvas_height,
            bg=self.BOARD_COLOR,
            highlightthickness=2,
            highlightbackground='#5D4037'
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # Control panel
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=0, column=1, sticky="n", padx=10)
        
        # Title
        title_label = ttk.Label(control_frame, text="五子棋 AI", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons
        self.start_btn = ttk.Button(control_frame, text="开始游戏", command=self.start_game, width=15)
        self.start_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.undo_btn = ttk.Button(control_frame, text="悔棋", command=self.undo_move, width=15, state='disabled')
        self.undo_btn.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.resign_btn = ttk.Button(control_frame, text="认输", command=self.resign_game, width=15, state='disabled')
        self.resign_btn.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=15)
        
        # Settings
        settings_label = ttk.Label(control_frame, text="设置", font=('Arial', 12, 'bold'))
        settings_label.grid(row=5, column=0, columnspan=2, pady=(0, 10))
        
        # Board size controls
        board_size_label = ttk.Label(control_frame, text="棋盘尺寸:")
        board_size_label.grid(row=6, column=0, columnspan=2, sticky='w', pady=5)
        
        # Rows input
        rows_frame = ttk.Frame(control_frame)
        rows_frame.grid(row=7, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Label(rows_frame, text="行:").pack(side='left')
        self.rows_var = tk.IntVar(value=DEFAULT_BOARD_ROWS)
        rows_spin = ttk.Spinbox(rows_frame, from_=5, to=25, width=8, textvariable=self.rows_var, 
                               command=self.on_board_size_change)
        rows_spin.pack(side='right')
        
        # Cols input
        cols_frame = ttk.Frame(control_frame)
        cols_frame.grid(row=8, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Label(cols_frame, text="列:").pack(side='left')
        self.cols_var = tk.IntVar(value=DEFAULT_BOARD_COLS)
        cols_spin = ttk.Spinbox(cols_frame, from_=5, to=25, width=8, textvariable=self.cols_var,
                               command=self.on_board_size_change)
        cols_spin.pack(side='right')
        
        # AI First checkbox
        self.ai_first_var = tk.BooleanVar(value=True)
        ai_first_cb = ttk.Checkbutton(control_frame, text="电脑先手", variable=self.ai_first_var)
        ai_first_cb.grid(row=9, column=0, columnspan=2, sticky='w', pady=5)
        
        # Show numbers checkbox
        self.show_numbers_var = tk.BooleanVar(value=False)
        show_numbers_cb = ttk.Checkbutton(control_frame, text="显示序号", variable=self.show_numbers_var, 
                                          command=self.toggle_numbers)
        show_numbers_cb.grid(row=10, column=0, columnspan=2, sticky='w', pady=5)
        
        # Difficulty selection
        diff_label = ttk.Label(control_frame, text="难度:")
        diff_label.grid(row=11, column=0, sticky='w', pady=5)
        
        self.diff_var = tk.StringVar(value='4')
        diff_combo = ttk.Combobox(control_frame, textvariable=self.diff_var, state='readonly', width=18)
        diff_combo['values'] = ('2 - 弱智', '4 - 简单', '6 - 普通', '8 - 困难')
        diff_combo.current(1)
        diff_combo.grid(row=12, column=0, columnspan=2, pady=5)
        diff_combo.bind('<<ComboboxSelected>>', self.on_difficulty_change)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=13, column=0, columnspan=2, sticky='ew', pady=15)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="点击\"开始游戏\"开始", font=('Arial', 10))
        self.status_label.grid(row=14, column=0, columnspan=2, pady=10)
        
        # Loading indicator
        self.loading_label = ttk.Label(control_frame, text="", font=('Arial', 10), foreground='blue')
        self.loading_label.grid(row=15, column=0, columnspan=2, pady=5)
        
        # Info label
        info_text = "基于极小化极大算法\n与Alpha-Beta剪枝的五子棋AI"
        info_label = ttk.Label(control_frame, text=info_text, font=('Arial', 8), foreground='gray')
        info_label.grid(row=16, column=0, columnspan=2, pady=(20, 0))
        
        # Draw initial empty board
        self.draw_board()
    
    def draw_board(self):
        """Draw the game board."""
        self.canvas.delete('all')
        
        # Draw vertical lines
        for i in range(self.board_cols):
            x = self.PADDING + i * self.CELL_SIZE
            self.canvas.create_line(
                x, self.PADDING,
                x, self.PADDING + (self.board_rows - 1) * self.CELL_SIZE,
                fill=self.LINE_COLOR, width=1
            )
        
        # Draw horizontal lines
        for i in range(self.board_rows):
            y = self.PADDING + i * self.CELL_SIZE
            self.canvas.create_line(
                self.PADDING, y,
                self.PADDING + (self.board_cols - 1) * self.CELL_SIZE, y,
                fill=self.LINE_COLOR, width=1
            )
        
        # Draw star points (center and corners if board is large enough)
        if self.board_rows >= 9 and self.board_cols >= 9:
            center_row, center_col = self.board_rows // 2, self.board_cols // 2
            star_points = []
            
            # Center point
            star_points.append((center_row, center_col))
            
            # Corner points if board is large enough
            if self.board_rows >= 15 and self.board_cols >= 15:
                margin = 3
                star_points.extend([
                    (margin, margin), (margin, self.board_cols - 1 - margin),
                    (self.board_rows - 1 - margin, margin), (self.board_rows - 1 - margin, self.board_cols - 1 - margin)
                ])
            
            for i, j in star_points:
                if 0 <= i < self.board_rows and 0 <= j < self.board_cols:
                    x = self.PADDING + j * self.CELL_SIZE
                    y = self.PADDING + i * self.CELL_SIZE
                    self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=self.LINE_COLOR, outline=self.LINE_COLOR)
        
        # Draw pieces if game is active
        if self.board is not None:
            for move_num, move in enumerate(self.board.history):
                i, j, role = move['i'], move['j'], move['role']
                self.draw_piece(i, j, role, move_num + 1)
            
            # Highlight last move
            if self.board.history:
                last = self.board.history[-1]
                self.highlight_last_move(last['i'], last['j'])
    
    def draw_piece(self, row, col, role, number=None):
        """Draw a piece on the board."""
        x = self.PADDING + col * self.CELL_SIZE
        y = self.PADDING + row * self.CELL_SIZE
        
        color = self.BLACK_COLOR if role == BLACK else self.WHITE_COLOR
        outline = self.WHITE_COLOR if role == BLACK else self.BLACK_COLOR
        
        self.canvas.create_oval(
            x - self.PIECE_RADIUS, y - self.PIECE_RADIUS,
            x + self.PIECE_RADIUS, y + self.PIECE_RADIUS,
            fill=color, outline=outline, width=1
        )
        
        # Draw number if enabled
        if self.show_numbers_var.get() and number is not None:
            text_color = self.WHITE_COLOR if role == BLACK else self.BLACK_COLOR
            self.canvas.create_text(x, y, text=str(number), fill=text_color, font=('Arial', 9, 'bold'))
    
    def highlight_last_move(self, row, col):
        """Highlight the last move with a small red dot."""
        x = self.PADDING + col * self.CELL_SIZE
        y = self.PADDING + row * self.CELL_SIZE
        
        self.canvas.create_oval(
            x - 4, y - 4, x + 4, y + 4,
            fill=self.HIGHLIGHT_COLOR, outline=self.HIGHLIGHT_COLOR
        )
    
    def on_canvas_click(self, event):
        """Handle canvas click events."""
        if self.loading or self.status != Status.GAMING:
            return
        
        # Convert click position to board coordinates
        col = round((event.x - self.PADDING) / self.CELL_SIZE)
        row = round((event.y - self.PADDING) / self.CELL_SIZE)
        
        # Check bounds
        if not (0 <= row < self.board_rows and 0 <= col < self.board_cols):
            return
        
        # Check if position is empty
        if self.board.board[row][col] != 0:
            return
        
        # Make player move
        self.make_move(row, col)
    
    def make_move(self, row, col):
        """Make a move and trigger AI response."""
        if not self.board.put(row, col):
            return
        
        self.draw_board()
        
        # Check for winner
        winner = self.board.get_winner()
        if winner != 0:
            self.end_game(winner)
            return
        
        # Check for draw
        if self.board.is_game_over():
            self.end_game(0)
            return
        
        # AI's turn
        self.ai_move()
    
    def ai_move(self):
        """Make AI move in a separate thread."""
        self.loading = True
        self.loading_label.config(text="AI思考中...")
        self.update_buttons()
        
        def ai_thread():
            try:
                score, move, path = minmax(self.board, self.board.role, self.depth)
                
                # Schedule GUI update on main thread
                self.root.after(0, lambda: self.on_ai_move_complete(move, score))
            except Exception as e:
                print(f"AI error: {e}")
                self.root.after(0, self.on_ai_move_error)
        
        thread = threading.Thread(target=ai_thread, daemon=True)
        thread.start()
    
    def on_ai_move_complete(self, move, score):
        """Handle AI move completion."""
        self.loading = False
        self.loading_label.config(text="")
        
        if move is not None:
            self.board.put(move[0], move[1])
            self.draw_board()
            
            # Check for winner
            winner = self.board.get_winner()
            if winner != 0:
                self.end_game(winner)
                return
            
            # Check for draw
            if self.board.is_game_over():
                self.end_game(0)
                return
        
        self.update_buttons()
    
    def on_ai_move_error(self):
        """Handle AI move error."""
        self.loading = False
        self.loading_label.config(text="")
        self.update_buttons()
        messagebox.showerror("错误", "AI计算出错，请重试")
    
    def start_game(self):
        """Start a new game."""
        self.ai_first = self.ai_first_var.get()
        self.depth = int(self.diff_var.get().split()[0])
        
        # Reset AI cache
        ai.reset()
        
        # Create new board with current dimensions
        self.board = Board(rows=self.board_rows, cols=self.board_cols, first_role=BLACK)
        self.status = Status.GAMING
        self.winner = None
        
        self.status_label.config(text=f"游戏进行中 ({self.board_rows}x{self.board_cols})...")
        self.update_buttons()
        self.draw_board()
        
        # AI moves first if selected
        if self.ai_first:
            self.ai_move()
    
    def undo_move(self):
        """Undo the last two moves (player's and AI's)."""
        if self.board and len(self.board.history) >= 2:
            self.board.undo()  # Undo AI move
            self.board.undo()  # Undo player move
            self.draw_board()
    
    def resign_game(self):
        """Resign the current game."""
        if self.status == Status.GAMING:
            # Player resigns, AI wins
            ai_role = BLACK if self.ai_first else WHITE
            self.end_game(ai_role)
    
    def end_game(self, winner):
        """End the game and show result."""
        self.status = Status.ENDED
        self.winner = winner
        
        if winner == BLACK:
            msg = "黑棋获胜！" if not self.ai_first else "AI获胜！"
        elif winner == WHITE:
            msg = "白棋获胜！" if self.ai_first else "AI获胜！"
        else:
            msg = "平局！"
        
        self.status_label.config(text=msg)
        self.update_buttons()
        
        messagebox.showinfo("游戏结束", msg)
    
    def toggle_numbers(self):
        """Toggle showing move numbers."""
        self.show_numbers = self.show_numbers_var.get()
        self.draw_board()
    
    def on_difficulty_change(self, event):
        """Handle difficulty change."""
        self.depth = int(self.diff_var.get().split()[0])
    
    def on_board_size_change(self):
        """Handle board size change."""
        new_rows = self.rows_var.get()
        new_cols = self.cols_var.get()
        
        # Validate dimensions
        if new_rows < 5 or new_cols < 5:
            return
        if new_rows > 25 or new_cols > 25:
            return
            
        self.board_rows = new_rows
        self.board_cols = new_cols
        
        # Update canvas size
        canvas_width = 2 * self.PADDING + (self.board_cols - 1) * self.CELL_SIZE
        canvas_height = 2 * self.PADDING + (self.board_rows - 1) * self.CELL_SIZE
        self.canvas.config(width=canvas_width, height=canvas_height)
        
        # Redraw board if game is not active
        if self.status == Status.IDLE:
            self.draw_board()
    
    def update_buttons(self):
        """Update button states based on game status."""
        gaming = self.status == Status.GAMING
        has_moves = self.board and len(self.board.history) >= 2
        
        self.start_btn.config(state='disabled' if (gaming and self.loading) else 'normal')
        self.undo_btn.config(state='normal' if (gaming and has_moves and not self.loading) else 'disabled')
        self.resign_btn.config(state='normal' if (gaming and not self.loading) else 'disabled')


def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme
    
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
