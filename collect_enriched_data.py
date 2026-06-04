"""
Collect enriched training data with opponent card counts + bomb info.
State vector: 33 + opponent_up + opponent_down + bomb_count = 36 dims.
"""
import sys
import copy
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.resolver_agent import ResolverAgent, hand_to_resolver_state, move_to_resolver_state, EnvCard2Rank
from perfectdou.evaluation.deep_agent import DeepAgent
from perfectdou.evaluation.random_agent import RandomAgent
from doudizhu_engine.hand import Hand
from doudizhu_engine.card import Card, Suit, Rank


def env_to_cards(env_cards):
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


def run_and_record(agent, env, record_pos='landlord'):
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
            last_move_state = [0] * 15
            if not has_control and last_move is not None and len(last_move) > 0:
                last_move_state = list(move_to_resolver_state(env_to_cards(last_move)))
            from perfectdou.resolver import min_plays
            _min_plays = min_plays(tuple(hand_state))

            # Opponent card counts from env
            opp_up_cards = len(env.player_hands.get('landlord_up', [])) if hasattr(env, 'player_hands') else 17
            opp_down_cards = len(env.player_hands.get('landlord_down', [])) if hasattr(env, 'player_hands') else 17
            
            # Bomb count
            bomb_count = getattr(env, 'bomb_num', 0)

            snapshot = {
                'hand_state': list(hand_state),
                'last_move_state': list(last_move_state) if last_move_state else [0]*15,
                'has_control': 1 if has_control else 0,
                'min_plays': _min_plays,
                'turn_num': turn,
                'opp_up_cards': opp_up_cards,
                'opp_down_cards': opp_down_cards,
                'bomb_count': bomb_count,
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
    seeds = list(range(120000, 120000 + num_games))
    model_path = str(Path(__file__).parent / 'perfectdou' / 'model' / 'douzero' / 'douzero_ADP' / 'landlord.ckpt')

    dz_wins = 0
    rs_wins = 0
    total_pairs = 0

    fieldnames = ['seed', 'winner_type', 'game_turns'] + \
                 [f'hand_{i}' for i in range(15)] + \
                 [f'last_move_{i}' for i in range(15)] + \
                 ['has_control', 'min_plays', 'turn_num', 'opp_up_cards', 'opp_down_cards', 'bomb_count'] + \
                 [f'action_{i}' for i in range(15)]

    with open('enriched_winner_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Collecting enriched data from {num_games} games...")
        for i, seed in enumerate(seeds):
            game = generate_game(seed)

            players = {
                'landlord': DeepAgent('landlord', model_path=model_path),
                'landlord_up': RandomAgent(seed + 1),
                'landlord_down': RandomAgent(seed + 2)
            }
            env_dz = GameEnv(players)
            env_dz.card_play_init(copy.deepcopy(game))
            dz_win, dz_turns, dz_trace = run_and_record(players['landlord'], env_dz, 'landlord')

            players = {
                'landlord': ResolverAgent('landlord', use_orphan_strategy=True, use_weak_hand_mode=True),
                'landlord_up': RandomAgent(seed + 1),
                'landlord_down': RandomAgent(seed + 2)
            }
            env_rs = GameEnv(players)
            env_rs.card_play_init(copy.deepcopy(game))
            rs_win, rs_turns, rs_trace = run_and_record(players['landlord'], env_rs, 'landlord')

            if dz_win:
                dz_wins += 1
            if rs_win:
                rs_wins += 1

            chosen_trace = None
            winner_type = 'NONE'
            chosen_turns = 60

            if dz_win and rs_win:
                if dz_turns <= rs_turns:
                    chosen_trace = dz_trace
                    winner_type = 'DZ'
                    chosen_turns = dz_turns
                else:
                    chosen_trace = rs_trace
                    winner_type = 'RS'
                    chosen_turns = rs_turns
            elif dz_win:
                chosen_trace = dz_trace
                winner_type = 'DZ'
                chosen_turns = dz_turns
            elif rs_win:
                chosen_trace = rs_trace
                winner_type = 'RS'
                chosen_turns = rs_turns

            if chosen_trace is not None:
                for step in chosen_trace:
                    row = {'seed': seed, 'winner_type': winner_type, 'game_turns': chosen_turns}
                    for j in range(15):
                        row[f'hand_{j}'] = step['hand_state'][j]
                        row[f'last_move_{j}'] = step['last_move_state'][j]
                        row[f'action_{j}'] = step['chosen_action_state'][j]
                    row['has_control'] = step['has_control']
                    row['min_plays'] = step['min_plays']
                    row['turn_num'] = step['turn_num']
                    row['opp_up_cards'] = step['opp_up_cards']
                    row['opp_down_cards'] = step['opp_down_cards']
                    row['bomb_count'] = step['bomb_count']
                    writer.writerow(row)
                    total_pairs += 1

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{num_games}: DZ_wins={dz_wins}, RS_wins={rs_wins}, pairs={total_pairs}")

    print(f"\nDone! Total pairs: {total_pairs}")


if __name__ == "__main__":
    main()
