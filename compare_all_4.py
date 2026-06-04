"""
Compare all 4 approaches on unseen seeds:
1. DouZero (DZ)
2. Resolver+Weak (RS)
3. CosineMemory+RS
4. NeuroSymbolic (Neural + Resolver fallback)
"""
import sys
import copy
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.resolver_agent import ResolverAgent
from perfectdou.evaluation.deep_agent import DeepAgent
from perfectdou.evaluation.random_agent import RandomAgent
from perfectdou.evaluation.cosine_memory_agent import CosineMemoryModel, CosineResolverAgent
from perfectdou.evaluation.neuro_symbolic_agent import NeuroSymbolicAgent


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


def run_agent(agent, game_data, seed):
    players = {
        'landlord': agent,
        'landlord_up': RandomAgent(seed + 1),
        'landlord_down': RandomAgent(seed + 2)
    }
    env = GameEnv(players)
    env.card_play_init(copy.deepcopy(game_data))
    turn = 0
    while not env.game_over and turn < 60:
        env.step()
        turn += 1
    winner = env.get_winner() if env.game_over else "TIMEOUT"
    return winner == 'landlord', turn


def main():
    num_games = 500
    seeds = list(range(60000, 60000 + num_games))
    model_path = str(Path(__file__).parent / 'perfectdou' / 'model' / 'douzero' / 'douzero_ADP' / 'landlord.ckpt')

    # Load cosine model
    table_file = "cosine_table_5000_20260603_113152.pkl"
    with open(table_file, 'rb') as f:
        data = pickle.load(f)
    cosine_model = CosineMemoryModel(threshold=0.99)
    for entry in data['table_entries']:
        cosine_model.add_pattern(entry['hand_state'], entry['last_move_state'],
                                 entry['has_control'], entry['min_plays'],
                                 entry['turn_num'], entry['chosen_action_state'])

    dz_wins = 0
    rs_wins = 0
    cos_wins = 0
    neuro_wins = 0

    dz_turns = 0
    rs_turns = 0
    cos_turns = 0
    neuro_turns = 0

    neuro_hits = 0
    neuro_total = 0

    print(f"Comparing 4 approaches on {num_games} unseen seeds...")

    for i, seed in enumerate(seeds):
        game = generate_game(seed)

        # DouZero
        dz_agent = DeepAgent('landlord', model_path=model_path)
        dz_win, dz_t = run_agent(dz_agent, game, seed)
        if dz_win:
            dz_wins += 1
            dz_turns += dz_t

        # Resolver+Weak
        rs_agent = ResolverAgent('landlord', use_orphan_strategy=True, use_weak_hand_mode=True)
        rs_win, rs_t = run_agent(rs_agent, game, seed)
        if rs_win:
            rs_wins += 1
            rs_turns += rs_t

        # Cosine+RS
        cos_agent = CosineResolverAgent('landlord', memory_model=cosine_model,
                                        use_orphan_strategy=True, use_weak_hand_mode=True)
        cos_win, cos_t = run_agent(cos_agent, game, seed)
        if cos_win:
            cos_wins += 1
            cos_turns += cos_t

        # NeuroSymbolic
        neuro_agent = NeuroSymbolicAgent('landlord', model_path='neuro_symbolic_classifier.pt',
                                           use_orphan_strategy=True, use_weak_hand_mode=True)
        neuro_win, neuro_t = run_agent(neuro_agent, game, seed)
        if neuro_win:
            neuro_wins += 1
            neuro_turns += neuro_t
        neuro_hits += neuro_agent.neural_count
        neuro_total += neuro_agent.neural_count + neuro_agent.fallback_count

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_games}: DZ={dz_wins}, RS={rs_wins}, COS={cos_wins}, NEU={neuro_wins}")

    print(f"\n{'='*60}")
    print(f"RESULTS over {num_games} unseen seeds")
    print(f"{'='*60}")
    print(f"  DouZero:         {dz_wins}/{num_games} ({100*dz_wins/num_games:.1f}%), avg {dz_turns/dz_wins:.1f} turns")
    print(f"  Resolver+Weak:   {rs_wins}/{num_games} ({100*rs_wins/num_games:.1f}%), avg {rs_turns/rs_wins:.1f} turns")
    print(f"  Cosine+RS:       {cos_wins}/{num_games} ({100*cos_wins/num_games:.1f}%), avg {cos_turns/cos_wins:.1f} turns")
    print(f"  NeuroSymbolic:   {neuro_wins}/{num_games} ({100*neuro_wins/num_games:.1f}%), avg {neuro_turns/neuro_wins:.1f} turns")
    if neuro_total > 0:
        print(f"  Neural hit rate: {neuro_hits}/{neuro_total} ({100*neuro_hits/neuro_total:.1f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"compare_all4_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(f"4-Approach Comparison\n")
        f.write(f"Games: {num_games}\n\n")
        f.write(f"DouZero:         {dz_wins}/{num_games}\n")
        f.write(f"Resolver+Weak:   {rs_wins}/{num_games}\n")
        f.write(f"Cosine+RS:       {cos_wins}/{num_games}\n")
        f.write(f"NeuroSymbolic:   {neuro_wins}/{num_games}\n")
        f.write(f"Neural hit rate: {neuro_hits}/{neuro_total}\n")
    print(f"\nSaved to {filename}")


if __name__ == "__main__":
    main()
