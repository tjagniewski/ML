import tkinter as tk
import colors as c
import random
import numpy as np
from Expectimax2048 import ai_move  

SEARCHES_PER_MOVE = 40
SEARCH_LENGTH = 20

class Game(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("2048 AI Bot MCTS")

        self.main_grid = tk.Frame(
            self, bg=c.GRID_COLOR, bd=3, width=600, height=600
        )
        self.main_grid.grid(pady=(100, 0))
        self.make_GUI()
        self.start_game()
        
        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<Up>", self.up)
        self.master.bind("<Down>", self.down)

        self.mainloop()

    def make_GUI(self):
        # grid
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.main_grid,
                    bg=c.EMPTY_CELL_COLOR,
                    width=150,
                    height=150
                )
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg=c.EMPTY_CELL_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)

        # score header
        score_frame = tk.Frame(self)
        score_frame.place(relx=0.5, y=45, anchor='center')
        tk.Label(
            score_frame,
            text = "Score",
            font = c.SCORE_LABEL_FONT
        ).grid(row=0)
        self.score_label = tk.Label(score_frame, text="0", font=c.SCORE_FONT)
        self.score_label.grid(row=1)
        
        self.bot_button = tk.Button(self, text="Start Game", command=self.run_bot_loop, font=("Verdana", 14, "bold"), bg="#8f7a66", fg="white")
        self.bot_button.place(relx=0.5, y=720, anchor='center')
        self.ai_running = False

    def start_game(self):
        self.matrix = [[0]*4 for _ in range(4)]
        self.add_new_tile_at_random()
        self.add_new_tile_at_random()
        self.score = 0

    def add_new_tile_at_random(self):
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        while(self.matrix[row][col] != 0):
            row = random.randint(0, 3)
            col = random.randint(0, 3)
        self.matrix[row][col] = 2
        self.update_single_cell(row, col)

    def update_single_cell(self, row, col):
        val = self.matrix[row][col]
        if val == 0:
             self.cells[row][col]["frame"].configure(bg=c.EMPTY_CELL_COLOR)
             self.cells[row][col]["number"].configure(bg=c.EMPTY_CELL_COLOR, text="")
        else:
            self.cells[row][col]["frame"].configure(bg=c.CELL_COLORS[val])
            self.cells[row][col]["number"].configure(bg=c.CELL_COLORS[val], 
                                                   fg=c.CELL_NUMBER_COLORS[val],
                                                   font=c.CELL_NUMBER_FONTS[val],
                                                   text=str(val))

    def run_bot_loop(self):
        if not self.ai_running:
            self.ai_running = True
            self.bot_button.place_forget()
            self.perform_ai_move()

    def perform_ai_move(self):
        if any(2048 in row for row in self.matrix):
            self.game_over()
            return

        current_board_np = np.array(self.matrix, dtype='int')

        try:
            new_board_np, valid_move, move_score = ai_move(current_board_np, SEARCHES_PER_MOVE, SEARCH_LENGTH)
        except ValueError:
            print("BŁĄD: Prawdopodobnie nie zaktualizowałeś pliku MCTS2048.py zgodnie z instrukcją!")
            return

        if valid_move:
            self.matrix = new_board_np.tolist()
            
            self.score += int(move_score)
            
            self.add_new_tile()
            self.update_GUI()
            
            self.after(50, self.perform_ai_move) 
        else:
            self.game_over()

    def stack(self):
        new_matrix = [[0]*4 for _ in range(4)]
        for i in range(4):
            fill_position = 0
            for j in range(4):
                if self.matrix[i][j] != 0:
                    new_matrix[i][fill_position] = self.matrix[i][j]
                    fill_position += 1
        self.matrix = new_matrix

    def combine(self):
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] != 0 and self.matrix[i][j] == self.matrix[i][j+1]:
                    self.matrix[i][j] *= 2
                    self.matrix[i][j+1] = 0
                    self.score += self.matrix[i][j]

    def reverse(self):
        new_matrix = []
        for i in range(4):
            new_matrix.append([])
            for j in range(4):
                new_matrix[i].append(self.matrix[i][3-j])
        self.matrix = new_matrix

    def transpose(self):
        new_matrix = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_matrix[i][j] = self.matrix[j][i]
        self.matrix = new_matrix

    def add_new_tile(self):
        if any(0 in row for row in self.matrix):
            row = random.randint(0, 3)
            col = random.randint(0, 3)
            while(self.matrix[row][col] != 0):
                row = random.randint(0, 3)
                col = random.randint(0, 3)
            self.matrix[row][col] = random.choice([2,4])

    def update_GUI(self):
        for i in range(4):
            for j in range(4):
                cell_value = int(self.matrix[i][j])
                if cell_value == 0:
                    self.cells[i][j]["frame"].configure(bg=c.EMPTY_CELL_COLOR)
                    self.cells[i][j]["number"].configure(bg=c.EMPTY_CELL_COLOR, text="")
                else:
                    if cell_value > 2048: bg_color = c.CELL_COLORS[2048]
                    else: bg_color = c.CELL_COLORS.get(cell_value, "#3c3a32")
                    
                    self.cells[i][j]["frame"].configure(bg=bg_color)
                    self.cells[i][j]["number"].configure(bg=bg_color,
                                                           fg=c.CELL_NUMBER_COLORS.get(cell_value, "#f9f6f2"),
                                                           font=c.CELL_NUMBER_FONTS.get(cell_value, ("Verdana", 30, "bold")),
                                                           text=str(cell_value))
        self.score_label.configure(text=self.score)
        self.update_idletasks()

    def left(self, event):
        self.stack()
        self.combine()
        self.stack()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()

    def right(self, event):
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()

    def up(self, event):
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()

    def down(self, event):
        self.transpose()
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.transpose()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()
    
    def horizontal_move_exist(self):
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] == self.matrix[i][j+1]:
                    return True
        return False
    
    def vertical_move_exist(self):
        for i in range(3):
            for j in range(4):
                if self.matrix[i][j] == self.matrix[i+1][j]:
                    return True
        return False

    def game_over(self):
        is_over = False
        msg = ""
        bg_color = ""
        
        if any(2048 in row for row in self.matrix):
            msg = "You win!"
            bg_color = c.WINNER_BG
            is_over = True
        elif not any(0 in row for row in self.matrix) and not self.horizontal_move_exist() and not self.vertical_move_exist():
            msg = "Game over!"
            bg_color = c.LOSER_BG
            is_over = True

        if is_over:
            if hasattr(self, 'bot_button'):
                self.bot_button.place_forget()

            game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
            tk.Label(
                game_over_frame,
                text=msg,
                bg=bg_color,
                fg=c.GAME_OVER_FONT_COLOR,
                font=c.GAME_OVER_FONT
            ).pack()

def main():
    Game()

if __name__ == "__main__":
    main()