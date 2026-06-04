"""
Compare Resolver-only agent vs DouZero on 1000 random seeds.
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


def play_game(agent, game_data):
    players = {
        'landlord': agent,
        'landlord_up': RandomAgent(),
        'landlord_down': RandomAgent()
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
    num_games = 1000
    seeds = list(range(2000, 3000))
    
    dz_agent = DeepAgent('landlord', "perfectdou/model/douzero/douzero_ADP/landlord.ckpt")
    rs_agent = ResolverAgent('landlord')
    
    dz_wins = 0
    rs_wins = 0
    dz_total_turns = 0
    rs_total_turns = 0
    
    # Track disagreement cases
    disagreements = []
    
    print(f"Running {num_games} games...")
    print(f"{'Seed':>6} | {'DouZero':>10} | {'Resolver':>10} | {'DZ Turns':>8} | {'RS Turns':>8}")
    print("-" * 60)
    
    for seed in seeds:
        game = generate_game(seed)
        
        # DouZero
        dz_winner, dz_turns = play_game(dz_agent, copy.deepcopy(game))
        dz_win = 1 if dz_winner == 'landlord' else 0
        dz_wins += dz_win
        dz_total_turns += dz_turns
        
        # Resolver
        rs_winner, rs_turns = play_game(rs_agent, copy.deepcopy(game))
        rs_win = 1 if rs_winner == 'landlord' else 0
        rs_wins += rs_win
        rs_total_turns += rs_turns
        
        marker = ""
        if dz_win != rs_win:
            marker = " ***"
            disagreements.append({
                'seed': seed,
                'dz_win': dz_win,
                'rs_win': rs_win,
                'dz_turns': dz_turns,
                'rs_turns': rs_turns
            })
        
        if seed % 100 == 0 or dz_win != rs_win:
            print(f"{seed:>6} | {'WIN' if dz_win else 'LOSS':>10} | {'WIN' if rs_win else 'LOSS':>10} | {dz_turns:>8} | {rs_turns:>8}{marker}")
    
    print("-" * 60)
    print(f"\nRESULTS over {num_games} games:")
    print(f"  DouZero:   {dz_wins}/{num_games} wins ({dz_wins/num_games*100:.1f}%), avg {dz_total_turns/num_games:.1f} turns")
    print(f"  Resolver:  {rs_wins}/{num_games} wins ({rs_wins/num_games*100:.1f}%), avg {rs_total_turns/num_games:.1f} turns")
    print(f"\n  Disagreements: {len(disagreements)} games")
    
    # Disagreement breakdown
    dz_win_rs_loss = sum(1 for d in disagreements if d['dz_win'] == 1 and d['rs_win'] == 0)
    rs_win_dz_loss = sum(1 for d in disagreements if d['dz_win'] == 0 and d['rs_win'] == 1)
    print(f"    DouZero win / Resolver loss: {dz_win_rs_loss}")
    print(f"    Resolver win / DouZero loss: {rs_win_dz_loss}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resolver_comparison_1000_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(f"Resolver vs DouZero - {num_games} games\n")
        f.write(f"DouZero: {dz_wins}/{num_games} ({dz_wins/num_games*100:.1f}%)\n")
        f.write(f"Resolver: {rs_wins}/{num_games} ({rs_wins/num_games*100:.1f}%)\n")
        f.write(f"Disagreements: {len(disagreements)}\n")
        f.write(f"  DZ win / RS loss: {dz_win_rs_loss}\n")
        f.write(f"  RS win / DZ loss: {rs_win_dz_loss}\n\n")
        
        if disagreements:
            f.write("Disagreement details:\n")
            for d in disagreements:
                f.write(f"  Seed {d['seed']}: DZ={'WIN' if d['dz_win'] else 'LOSS'}({d['dz_turns']}t) | RS={'WIN' if d['rs_win'] else 'LOSS'}({d['rs_turns']}t)\n")
    
    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    main()
