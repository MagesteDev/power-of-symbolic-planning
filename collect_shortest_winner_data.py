"""
Collect training data from the SHORTEST-PATH WINNER per game.
For each seed, run DouZero and Resolver+Weak.
Identify which winner finished in fewer turns.
Record the full state-action trace from that winner.
Saves to CSV for neural network training.
"""
import sys
import copy
import pickle
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.resolver_agent import ResolverAgent, hand_to_resolver_state, move_to_resolver_state, EnvCard2Rank
from perfectdou.evaluation.deep_agent import DeepAgent
from perfectdou.evaluation.random_agent import RandomAgent
from doudizhu_engine.hand import Hand
from doudizhu_engine.card import Card, Suit, Rank


def env_to_cards(env_cards):
    """Standalone env card int -> doudizhu_engine Card list."""
    cards = []
    for ec in env_cards:
        rank = EnvCard2Rank[ec]
        suit = Suit.JOKER if rank in (Rank.SMALL_JOKER, Rank.BIG_JOKER) else Suit.SPADES
        cards.append(Card(rank=rank, suit=suit))
    return cards


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


def run_and_record(agent, game_data, seed, record_pos='landlord'):
    """
    Run a game with the given agent and record state-action trace.
    Returns (win, turns, trace).
    """
    players = {
        'landlord': agent,
        'landlord_up': RandomAgent(seed + 1),
        'landlord_down': RandomAgent(seed + 2)
    }
    env = GameEnv(players)
    env.card_play_init(copy.deepcopy(game_data))

    trace = []
    turn = 0

    while not env.game_over and turn < 60:
        current_player = env.acting_player_position
        if current_player == record_pos:
            infoset = env.get_infoset()
            hand_cards = env_to_cards(infoset.player_hand_cards)
            hand = Hand(hand_cards)
            hand_state = hand_to_resolver_state(hand)
            last_move = infoset.last_move
            has_control = (last_move is None or len(last_move) == 0)
            last_move_state = None
            if not has_control and last_move is not None and len(last_move) > 0:
                last_move_state = move_to_resolver_state(env_to_cards(last_move))
            from perfectdou.resolver import min_plays
            _min_plays = min_plays(tuple(hand_state))

            snapshot = {
                'hand_state': list(hand_state),
                'last_move_state': list(last_move_state) if last_move_state else [0]*15,
                'has_control': 1 if has_control else 0,
                'min_plays': _min_plays,
                'turn_num': turn,
            }

        env.step()

        if current_player == record_pos:
            action = env.last_move_dict.get(current_player, [])
            action_state = [0] * 15
            if len(action) > 0:
                action_state = list(move_to_resolver_state(env_to_cards(action)))
            snapshot['chosen_action_state'] = action_state
            trace.append(snapshot)

        turn += 1

    winner = env.get_winner() if env.game_over else "TIMEOUT"
    win = (winner == 'landlord')
    return win, turn, trace


def main():
    num_games = 5000
    seeds = list(range(50000, 50000 + num_games))
    model_path = str(Path(__file__).parent / 'perfectdou' / 'model' / 'douzero' / 'douzero_ADP' / 'landlord.ckpt')

    dz_wins = 0
    rs_wins = 0
    dz_chosen = 0
    rs_chosen = 0
    total_pairs = 0

    # CSV writer for training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"shortest_winner_data_{num_games}_{timestamp}.csv"

    fieldnames = ['seed', 'winner_type', 'game_turns'] + \
                 [f'hand_{i}' for i in range(15)] + \
                 [f'last_move_{i}' for i in range(15)] + \
                 ['has_control', 'min_plays', 'turn_num'] + \
                 [f'action_{i}' for i in range(15)]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Collecting shortest-path winner data from {num_games} games...")

        for i, seed in enumerate(seeds):
            game = generate_game(seed)

            # Run DouZero
            dz_agent = DeepAgent('landlord', model_path=model_path)
            dz_win, dz_turns, dz_trace = run_and_record(dz_agent, game, seed)

            # Run Resolver+Weak
            rs_agent = ResolverAgent('landlord', use_orphan_strategy=True, use_weak_hand_mode=True)
            rs_win, rs_turns, rs_trace = run_and_record(rs_agent, game, seed)

            if dz_win:
                dz_wins += 1
            if rs_win:
                rs_wins += 1

            # Pick shortest-path winner
            chosen_trace = None
            winner_type = None
            chosen_turns = None

            if dz_win and rs_win:
                if dz_turns <= rs_turns:
                    chosen_trace = dz_trace
                    winner_type = 'DZ'
                    chosen_turns = dz_turns
                    dz_chosen += 1
                else:
                    chosen_trace = rs_trace
                    winner_type = 'RS'
                    chosen_turns = rs_turns
                    rs_chosen += 1
            elif dz_win:
                chosen_trace = dz_trace
                winner_type = 'DZ'
                chosen_turns = dz_turns
                dz_chosen += 1
            elif rs_win:
                chosen_trace = rs_trace
                winner_type = 'RS'
                chosen_turns = rs_turns
                rs_chosen += 1
            else:
                winner_type = 'NONE'
                chosen_turns = 60

            # Write trace to CSV
            if chosen_trace is not None:
                for step in chosen_trace:
                    row = {
                        'seed': seed,
                        'winner_type': winner_type,
                        'game_turns': chosen_turns,
                    }
                    for j in range(15):
                        row[f'hand_{j}'] = step['hand_state'][j]
                        row[f'last_move_{j}'] = step['last_move_state'][j]
                        row[f'action_{j}'] = step['chosen_action_state'][j]
                    row['has_control'] = step['has_control']
                    row['min_plays'] = step['min_plays']
                    row['turn_num'] = step['turn_num']
                    writer.writerow(row)
                    total_pairs += 1

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{num_games}: DZ_wins={dz_wins}, RS_wins={rs_wins}, "
                      f"DZ_chosen={dz_chosen}, RS_chosen={rs_chosen}, pairs={total_pairs}")

    print(f"\nDone!")
    print(f"  DouZero wins:      {dz_wins}/{num_games}")
    print(f"  Resolver wins:     {rs_wins}/{num_games}")
    print(f"  DZ chosen:         {dz_chosen}")
    print(f"  RS chosen:         {rs_chosen}")
    print(f"  Total state-action pairs: {total_pairs}")
    print(f"  Saved to: {csv_file}")


if __name__ == "__main__":
    main()
