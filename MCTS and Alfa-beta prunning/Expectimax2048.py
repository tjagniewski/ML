import numpy as np
from game_functions import move_down, move_left, move_right, move_up

MAX_DEPTH = 3          
SURVIVAL_THRESHOLD = 2  

# wages matrix
WEIGHT_MATRIX = np.array([
    [4**15, 4**14, 4**13, 4**12],
    [4**8,  4**9,  4**10, 4**11],
    [4**7,  4**6,  4**5,  4**4],
    [4**0,  4**1,  4**2,  4**3]
])

def get_empty_cells_count(board):
    return len(np.where(board == 0)[0])

def calculate_score(board):
    """
    Funkcja oceniająca stan planszy.
    """
    empty_count = get_empty_cells_count(board)
    
    score = np.sum(board * WEIGHT_MATRIX)

    left_diff = board[:, :-1] - board[:, 1:]
    up_diff   = board[:-1, :] - board[1:, :]
    
    monotonicity_score = 0
    monotonicity_score -= np.sum(np.abs(left_diff))
    monotonicity_score -= np.sum(np.abs(up_diff))
    
    score += monotonicity_score * 10

    if empty_count == 0:
        return -1e9 
    else:
        score += empty_count * 500000  

    return score

def expectimax(board, depth, is_player_turn):
    if depth == 0:
        return calculate_score(board)

    if is_player_turn:
        moves = [move_up, move_down, move_left, move_right]
        best_score = -float('inf')
        
        for func in moves:
            new_board, valid, moves_score = func(board)
            
            if valid:
                future_score = expectimax(new_board, depth - 1, False)
                
                total_score = future_score + (moves_score * 10)
                
                if total_score > best_score:
                    best_score = total_score
        
        return best_score

    else:
        empty_indices = list(zip(*np.where(board == 0)))
        
        if not empty_indices:
            return calculate_score(board)

        if len(empty_indices) > 3:
            idxs = np.random.choice(len(empty_indices), 3, replace=False)
            search_cells = [empty_indices[i] for i in idxs]
        else:
            search_cells = empty_indices

        avg_score = 0
        weight_sum = 0
        
        for r, c in search_cells:
            board[r][c] = 2
            avg_score += 0.9 * expectimax(board, depth - 1, True)
            
            board[r][c] = 4
            avg_score += 0.1 * expectimax(board, depth - 1, True)
            
            board[r][c] = 0
            weight_sum += 1

        return avg_score / weight_sum

def ai_move(board, searches_per_move, search_length):
    """
    Główna funkcja sterująca.
    """
    possible_moves = [move_up, move_down, move_left, move_right]
    best_move_result = None
    best_score = -float('inf')
    
    empty_cells = get_empty_cells_count(board)
    current_depth = MAX_DEPTH
    if empty_cells < 2:
        current_depth = 4 

    valid_moves_found = 0

    for func in possible_moves:
        sim_board = np.copy(board)
        new_board, valid, move_points = func(sim_board)

        if valid:
            valid_moves_found += 1
            score = expectimax(new_board, current_depth, False)
            
            if get_empty_cells_count(new_board) > empty_cells:
                score += 1000000 

            if score > best_score:
                best_score = score
                best_move_result = (new_board, True, move_points)

    if best_move_result is None and valid_moves_found > 0:
        for func in possible_moves:
            sim_board = np.copy(board)
            new_board, valid, move_points = func(sim_board)
            if valid:
                return new_board, True, move_points

    if best_move_result is None:
        return board, False, 0

    return best_move_result