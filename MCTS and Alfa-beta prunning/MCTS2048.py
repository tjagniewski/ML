import numpy as np

from game_functions import initialize_game, move_down, move_left, move_right, move_up, add_new_tile, check_for_win, random_move

def ai_move(board, searches_per_move, search_length):
    first_moves = [move_down, move_left, move_right, move_up]
    scores = np.zeros(4)

    for first_index in range(4):
        first_move = first_moves[first_index]
        first_board, first_valid, first_score = first_move(board)

        if first_valid:
            first_board = add_new_tile(first_board)
            scores[first_index] += first_score
        else:
            continue
        for later_moves in range(searches_per_move):
            move_number = 1
            search_board = np.copy(first_board)
            is_valid = True
            
            while is_valid and move_number < search_length:
                search_board, is_valid, score = random_move(search_board)
                if is_valid:
                    search_board = add_new_tile(search_board)
                    scores[first_index] += score
                    move_number += 1
                
    best_move_index = np.argmax(scores)
    best_move = first_moves[best_move_index]
    final_board, position_valid, move_score = best_move(board)
    
    return final_board, position_valid, move_score