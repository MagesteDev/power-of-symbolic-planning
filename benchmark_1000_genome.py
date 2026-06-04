"""
10,000-game benchmark: Genome-Enhanced NeuroSymbolic vs all baselines.
"""
import sys
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.resolver_agent import ResolverAgent
from perfectdou.evaluation.deep_agent import DeepAgent
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
    num_games = 10000
    seeds = list(range(200000, 200000 + num_games))
    model_path = str(Path(__file__).parent / 'perfectdou' / 'model' / 'douzero' / 'douzero_ADP' / 'landlord.ckpt')

    dz_wins = rs_wins = neuro_wins = genome_wins = 0
    dz_turns = rs_turns = neuro_turns = genome_turns = 0
    neuro_hits = neuro_total = 0
    genome_hits = genome_total = 0

    print(f"Benchmarking {num_games} games...")
    for i, seed in enumerate(seeds):
        game = generate_game(seed)

        dz_agent = DeepAgent('landlord', model_path=model_path)
        dz_win, dz_t = run_agent(dz_agent, game, seed)
        if dz_win:
            dz_wins += 1
            dz_turns += dz_t

        rs_agent = ResolverAgent('landlord', use_orphan_strategy=True, use_weak_hand_mode=True)
        rs_win, rs_t = run_agent(rs_agent, game, seed)
        if rs_win:
            rs_wins += 1
            rs_turns += rs_t

        neuro_agent = NeuroSymbolicAgent('landlord', model_path='neuro_symbolic_classifier.pt',
                                          use_orphan_strategy=True, use_weak_hand_mode=True)
        neuro_win, neuro_t = run_agent(neuro_agent, game, seed)
        if neuro_win:
            neuro_wins += 1
            neuro_turns += neuro_t
        neuro_hits += neuro_agent.neural_count
        neuro_total += neuro_agent.neural_count + neuro_agent.fallback_count

        genome_agent = NeuroSymbolicAgent('landlord', model_path='neuro_symbolic_genome_classifier.pt',
                                           use_orphan_strategy=True, use_weak_hand_mode=True)
        genome_win, genome_t = run_agent(genome_agent, game, seed)
        if genome_win:
            genome_wins += 1
            genome_turns += genome_t
        genome_hits += genome_agent.neural_count
        genome_total += genome_agent.neural_count + genome_agent.fallback_count

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{num_games}: DZ={dz_wins}, RS={rs_wins}, NEU={neuro_wins}, GEN={genome_wins}")

    print(f"\n{'='*60}")
    print(f"10000-GAME BENCHMARK")
    print(f"{'='*60}")
    print(f"  DouZero:         {dz_wins}/{num_games} ({100*dz_wins/num_games:.1f}%), avg {dz_turns/dz_wins:.1f}t")
    print(f"  Resolver+Weak:   {rs_wins}/{num_games} ({100*rs_wins/num_games:.1f}%), avg {rs_turns/rs_wins:.1f}t")
    print(f"  NeuroSymbolic:   {neuro_wins}/{num_games} ({100*neuro_wins/num_games:.1f}%), avg {neuro_turns/neuro_wins:.1f}t  hit={100*neuro_hits/neuro_total:.1f}%")
    print(f"  Genome-Enhanced: {genome_wins}/{num_games} ({100*genome_wins/num_games:.1f}%), avg {genome_turns/genome_wins:.1f}t  hit={100*genome_hits/genome_total:.1f}%")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"benchmark10000_genome_{ts}.txt", 'w') as f:
        f.write(f"10000-Game Genome Benchmark\n")
        f.write(f"DZ:   {dz_wins}/{num_games}\n")
        f.write(f"RS:   {rs_wins}/{num_games}\n")
        f.write(f"NEU:  {neuro_wins}/{num_games}\n")
        f.write(f"GEN:  {genome_wins}/{num_games}\n")
    print(f"\nSaved to benchmark10000_genome_{ts}.txt")


if __name__ == "__main__":
    main()
