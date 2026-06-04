"""
Tune confidence threshold for genome-enhanced NeuroSymbolic.
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
    seeds = list(range(130000, 130000 + num_games))
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Baseline
    rs_wins = 0
    for seed in seeds:
        game = generate_game(seed)
        agent = ResolverAgent('landlord', use_orphan_strategy=True, use_weak_hand_mode=True)
        win, _ = run_agent(agent, game, seed)
        if win:
            rs_wins += 1
    print(f"Baseline Resolver: {rs_wins}/{num_games} ({100*rs_wins/num_games:.1f}%)")

    # Baseline NeuroSymbolic (for reference)
    neuro_wins = 0
    for seed in seeds:
        game = generate_game(seed)
        agent = NeuroSymbolicAgent('landlord', model_path='neuro_symbolic_classifier.pt')
        win, _ = run_agent(agent, game, seed)
        if win:
            neuro_wins += 1
    print(f"Baseline NeuroSym: {neuro_wins}/{num_games} ({100*neuro_wins/num_games:.1f}%)")

    # Genome-enhanced with different thresholds
    for thr in thresholds:
        wins = 0
        hits = 0
        total = 0
        for seed in seeds:
            game = generate_game(seed)
            agent = NeuroSymbolicAgent('landlord', model_path='neuro_symbolic_genome_classifier.pt',
                                        confidence_threshold=thr)
            win, _ = run_agent(agent, game, seed)
            if win:
                wins += 1
            hits += agent.neural_count
            total += agent.neural_count + agent.fallback_count
        hit_rate = 100 * hits / total if total > 0 else 0
        print(f"Genome thr={thr:.1f}: {wins}/{num_games} ({100*wins/num_games:.1f}%)  hit={hit_rate:.1f}%")


if __name__ == "__main__":
    main()
