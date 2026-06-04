"""
Tune NeuroSymbolic confidence threshold.
Test 0.3, 0.5, 0.7, 0.9 on 200 seeds to find optimal.
"""
import sys
import copy
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.resolver_agent import ResolverAgent
from perfectdou.evaluation.neuro_symbolic_agent import NeuroSymbolicAgent
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
    num_games = 200
    seeds = list(range(70000, 70000 + num_games))
    thresholds = [0.3, 0.5, 0.7, 0.9]

    # Baseline: pure resolver
    rs_wins = 0
    for i, seed in enumerate(seeds):
        game = generate_game(seed)
        agent = ResolverAgent('landlord', use_orphan_strategy=True, use_weak_hand_mode=True)
        win, _ = run_agent(agent, game, seed)
        if win:
            rs_wins += 1

    print(f"Baseline Resolver+Weak: {rs_wins}/{num_games} ({100*rs_wins/num_games:.1f}%)")
    print()

    # NeuroSymbolic with different thresholds
    for threshold in thresholds:
        neuro_wins = 0
        total_neural = 0
        total_fallback = 0
        for i, seed in enumerate(seeds):
            game = generate_game(seed)
            agent = NeuroSymbolicAgent('landlord', model_path='neuro_symbolic_classifier.pt',
                                       use_orphan_strategy=True, use_weak_hand_mode=True,
                                       confidence_threshold=threshold)
            win, _ = run_agent(agent, game, seed)
            if win:
                neuro_wins += 1
            total_neural += agent.neural_count
            total_fallback += agent.fallback_count

        total = total_neural + total_fallback
        hit_rate = 100 * total_neural / total if total > 0 else 0
        print(f"Threshold={threshold:.1f}: {neuro_wins}/{num_games} ({100*neuro_wins/num_games:.1f}%), "
              f"neural hit={hit_rate:.1f}%")


if __name__ == "__main__":
    main()
