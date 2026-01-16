import numpy as np
import matplotlib.pyplot as plt
import time
from game_functions import initialize_game, add_new_tile

import MCTS2048
import Expectimax2048 

GAMES_PER_CONFIG = 5  

CONFIGURATIONS = [
    ("MCTS (fast)", MCTS2048, 10, 5),      
    ("MCTS (strong)", MCTS2048, 50, 20),   
    ("Expectimax", Expectimax2048, 0, 0)     
]

TILE_COLORS = {
    0:    '#CDC1B4',
    2:    '#eee4da',
    4:    '#ede0c8',
    8:    '#f2b179',
    16:   '#f59563',
    32:   '#f67c5f',
    64:   '#f65e3b',
    128:  '#edcf72', 
    256:  '#33b5e5', 
    512:  '#0099cc', 
    1024: '#99cc00', 
    2048: '#ffbb33', 
    4096: '#ff4444', 
    8192: '#aa66cc', 
    16384:'#2d2d2d'  
}

def get_color(value):
    return TILE_COLORS.get(value, '#3c3a32')

def play_single_game(ai_module, spm, sl):
    board = initialize_game()
    board = add_new_tile(board)
    board = add_new_tile(board)
    
    score = 0
    game_over = False
    
    while not game_over:
        try:
            new_board, valid, move_score = ai_module.ai_move(board, spm, sl)
        except Exception as e:
            print(f"Bot mistake: {e}")
            break

        if valid:
            board = new_board
            score += move_score
            board = add_new_tile(board)
        else:
            game_over = True
            
    return score, np.max(board)

def run_statistics():
    results = {} 

    
    # collecting data
    for name, module, spm, sl in CONFIGURATIONS:
        print(f"\nTesting: {name}...")
        results[name] = {'scores': [], 'max_tiles': []}
        
        start_time = time.time()
        
        for i in range(GAMES_PER_CONFIG):
            final_score, max_tile = play_single_game(module, spm, sl)
            results[name]['scores'].append(final_score)
            results[name]['max_tiles'].append(max_tile)
            print(f"  Simulation {i+1}: Max Square = {max_tile}, Score = {int(final_score)}")
            
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f}s")

    bot_names = list(results.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    avg_scores = [np.mean(results[n]['scores']) for n in bot_names]
    max_scores_stat = [np.max(results[n]['scores']) for n in bot_names]
    
    x = np.arange(len(bot_names))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, avg_scores, width, label='Mean', color='#5da5da')
    rects2 = ax1.bar(x + width/2, max_scores_stat, width, label='Max', color='#faa43a')
    
    ax1.set_ylabel('Points (Score)')
    ax1.set_title('Points Score for each algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bot_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.bar_label(rects1, padding=3, fmt='%d')
    ax1.bar_label(rects2, padding=3, fmt='%d')

    n_games = GAMES_PER_CONFIG
    bar_width = 0.8 / n_games  
    
    for i in range(n_games):
        tiles_in_game_i = [results[name]['max_tiles'][i] for name in bot_names]
        
        positions = x + (i - n_games/2 + 0.5) * bar_width
        
        colors = [get_color(t) for t in tiles_in_game_i]
        
        heights = [np.log2(t) if t > 0 else 0 for t in tiles_in_game_i]
        
        bars = ax2.bar(positions, heights, bar_width, color=colors, edgecolor='black', alpha=0.9)
        
        for rect, val in zip(bars, tiles_in_game_i):
            ax2.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.1,
                     str(val), ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45)

    ax2.set_ylabel('Highest rated square')
    ax2.set_title(f'Highest rated square of each simulation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bot_names)
    
    all_max_tiles = [t for n in bot_names for t in results[n]['max_tiles']]
    max_val_log = np.log2(max(all_max_tiles)) if all_max_tiles else 10
    ax2.set_ylim(0, max_val_log + 2)
    
    ax2.set_yticks([]) 
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_statistics()