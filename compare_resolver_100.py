"""
Compare Resolver-only agent vs DouZero on 100 random seeds.
"""
import sys
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.deep_agent import DeepAgent
from perfectdou.evaluation.resolver_agent import ResolverAgent
from perfectdou.evaluation.random_agent import RandomAgent


def generate_game(seed):
    rng = np.random.RandomState(seed)
    deck = []
    for i in range(3, 15):
        deck.extend([i] * 4)
    deck.extend([17] * 4)
    deck.extend([20, 30])
    deck = np.array(deck)
    rng.shuffle(deck)
    return {
        'landlord': sorted(deck[:20].tolist()),
        'landlord_up': sorted(deck[20:37].tolist()),
        'landlord_down': sorted(deck[37:54].tolist()),
        'three_landlord_cards': sorted(deck[54:57].tolist()),
    }


def play_game(agent, game_data, seed):
    players = {
        'landlord': agent,
        'landlord_up': RandomAgent(seed + 1),
        'landlord_down': RandomAgent(seed + 2)
    }
    env = GameEnv(players)
    env.card_play_init(game_data)
    
    turn = 0
    while not env.game_over and turn < 60:
        env.step()
        turn += 1
    
    winner = env.get_winner() if env.game_over else "TIMEOUT"
    return winner, turn


def main():
    num_games = 100
    seeds = list(range(1000, 2000 + num_games))
    
    model_path = str(Path(__file__).parent / "perfectdou" / "model" / "douzero" / "douzero_ADP" / "landlord.ckpt")
    dz_agent = DeepAgent('landlord', model_path)
    rs_wh_on = ResolverAgent('landlord', use_weak_hand_mode=True)
    rs_wh_off = ResolverAgent('landlord', use_weak_hand_mode=False)
    
    dz_wins = 0
    rs_wh_on_wins = 0
    rs_wh_off_wins = 0
    dz_total_turns = 0
    rs_wh_on_total_turns = 0
    rs_wh_off_total_turns = 0
    
    print(f"Running {num_games} games (Weak-Hand A/B comparison)...")
    print(f"{'Seed':>6} | {'DouZero':>8} | {'RS+Weak':>8} | {'RS-Weak':>8} | {'DZ':>4} | {'+W':>4} | {'-W':>4}")
    print("-" * 65)
    
    for seed in seeds:
        game = generate_game(seed)
        
        # DouZero
        dz_winner, dz_turns = play_game(dz_agent, copy.deepcopy(game), seed)
        dz_win = 1 if dz_winner == 'landlord' else 0
        dz_wins += dz_win
        dz_total_turns += dz_turns
        
        # Resolver WITH Weak-Hand mode
        rs_wh_on_winner, rs_wh_on_turns = play_game(rs_wh_on, copy.deepcopy(game), seed)
        rs_wh_on_win = 1 if rs_wh_on_winner == 'landlord' else 0
        rs_wh_on_wins += rs_wh_on_win
        rs_wh_on_total_turns += rs_wh_on_turns
        
        # Resolver WITHOUT Weak-Hand mode
        rs_wh_off_winner, rs_wh_off_turns = play_game(rs_wh_off, copy.deepcopy(game), seed)
        rs_wh_off_win = 1 if rs_wh_off_winner == 'landlord' else 0
        rs_wh_off_wins += rs_wh_off_win
        rs_wh_off_total_turns += rs_wh_off_turns
        
        marker = ""
        if rs_wh_on_win != rs_wh_off_win:
            marker = " <<<"
        
        print(f"{seed:>6} | {'WIN' if dz_win else 'LOSS':>8} | {'WIN' if rs_wh_on_win else 'LOSS':>8} | {'WIN' if rs_wh_off_win else 'LOSS':>8} | {dz_turns:>4} | {rs_wh_on_turns:>4} | {rs_wh_off_turns:>4}{marker}")
    
    print("-" * 65)
    print(f"\nRESULTS over {num_games} games:")
    print(f"  DouZero:         {dz_wins}/{num_games} wins ({dz_wins/num_games*100:.1f}%), avg {dz_total_turns/num_games:.1f} turns")
    print(f"  Resolver +Weak:  {rs_wh_on_wins}/{num_games} wins ({rs_wh_on_wins/num_games*100:.1f}%), avg {rs_wh_on_total_turns/num_games:.1f} turns")
    print(f"  Resolver -Weak:  {rs_wh_off_wins}/{num_games} wins ({rs_wh_off_wins/num_games*100:.1f}%), avg {rs_wh_off_total_turns/num_games:.1f} turns")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resolver_weakhand_ab_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(f"Resolver A/B - Weak-Hand Mode ON vs OFF - {num_games} games\n")
        f.write(f"DouZero:         {dz_wins}/{num_games} ({dz_wins/num_games*100:.1f}%)\n")
        f.write(f"Resolver +Weak:  {rs_wh_on_wins}/{num_games} ({rs_wh_on_wins/num_games*100:.1f}%)\n")
        f.write(f"Resolver -Weak:  {rs_wh_off_wins}/{num_games} ({rs_wh_off_wins/num_games*100:.1f}%)\n")
    
    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    main()
