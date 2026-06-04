from functools import lru_cache
import itertools

def parse_douzero_hand(card_ints):
    """
    Converts DouZero's integer format (3-17) to our 15-element tuple format.
    Indices: 0-12 (3 to A), 13 (2), 14 (Jokers)
    """
    counts = [0] * 15
    for card in card_ints:
        if 3 <= card <= 14:
            counts[card - 3] += 1
        elif card == 15: # 2
            counts[13] += 1
        elif card in (16, 17): # Jokers
            counts[14] += 1
    return tuple(counts)

def parse_string_hand(hand_str):
    """
    Parses a string like '3346688999TJJQQQKA22' into our tuple format.
    Use 'T' for 10.
    """
    card_map = {'3':0, '4':1, '5':2, '6':3, '7':4, '8':5, '9':6, 'T':7, 
                'J':8, 'Q':9, 'K':10, 'A':11}  # '2' handled separately at index 13, Jokers at 14
    counts = [0] * 15
    
    i = 0
    while i < len(hand_str):
        if hand_str[i] == '1' and i+1 < len(hand_str) and hand_str[i+1] == '0':
            counts[7] += 1
            i += 2
        elif hand_str[i] in ('X', 'B'): # Black Joker
            counts[14] += 1
            i += 1
        elif hand_str[i] in ('D', 'R'): # Red Joker
            counts[14] += 1
            i += 1
        else:
            char = hand_str[i]
            if char == '2':
                counts[13] += 1
            elif char in card_map:
                counts[card_map[char]] += 1
            i += 1
    return tuple(counts)

def get_k_kickers(state, k, is_pair, current_delta, start_idx):
    """Helper to generate all valid combinations of kickers (singles or pairs)"""
    if k == 0:
        yield tuple(current_delta)
        return
    
    limit = 15 if not is_pair else 14 # Jokers (14) can't be pairs
    for idx in range(start_idx, limit):
        req = 2 if is_pair else 1
        max_take = state[idx] // req
        
        for take in range(1, min(max_take, k) + 1):
            current_delta[idx] += take * req
            yield from get_k_kickers(state, k - take, is_pair, current_delta, idx + 1)
            current_delta[idx] -= take * req

def get_valid_moves(state):
    """Generates all valid moves that include the lowest available card."""
    # Find the lowest available card index
    lowest_idx = -1
    for i in range(15):
        if state[i] > 0:
            lowest_idx = i
            break
            
    if lowest_idx == -1:
        return []

    moves = set()
    
    # 1. Singles, Pairs, Trios, Bombs, Rocket
    for i in range(15):
        if state[i] >= 1:
            d = [0]*15; d[i] = 1; moves.add(tuple(d))
    for i in range(14):
        if state[i] >= 2:
            d = [0]*15; d[i] = 2; moves.add(tuple(d))
    for i in range(13):
        if state[i] >= 3:
            d = [0]*15; d[i] = 3; moves.add(tuple(d))
        if state[i] >= 4:
            d = [0]*15; d[i] = 4; moves.add(tuple(d))
    if state[14] >= 2:
        d = [0]*15; d[14] = 2; moves.add(tuple(d))

    # 2. Trio + 1 Single / 1 Pair
    for t in range(13):
        if state[t] >= 3:
            for s in range(15):
                if s != t and state[s] >= 1:
                    d = [0]*15; d[t] = 3; d[s] = 1; moves.add(tuple(d))
            for p in range(14):
                if p != t and state[p] >= 2:
                    d = [0]*15; d[t] = 3; d[p] = 2; moves.add(tuple(d))

    # 3. Straights (3 to A, length 5-12)
    for start in range(9): # 3 to 7
        for length in range(5, 13 - start):
            if all(state[start+k] >= 1 for k in range(length)):
                d = [0]*15
                for k in range(length): d[start+k] = 1
                moves.add(tuple(d))

    # 4. Pair Straights (3 to A, length 3-10)
    for start in range(11): # 3 to 9
        for length in range(3, 13 - start):
            if all(state[start+k] >= 2 for k in range(length)):
                d = [0]*15
                for k in range(length): d[start+k] = 2
                moves.add(tuple(d))

    # 5. Airplanes (Pure, +Singles, +Pairs) (3 to A, length 2-6)
    for start in range(12): # 3 to 10
        for length in range(2, min(7, 13 - start)):
            if all(state[start+k] >= 3 for k in range(length)):
                # Pure
                d_pure = [0]*15
                for k in range(length): d_pure[start+k] = 3
                moves.add(tuple(d_pure))
                
                # Base state for kickers (subtract pure airplane temporarily)
                temp_state = list(state)
                for k in range(length): temp_state[start+k] -= 3
                
                # Airplane + Singles
                for delta_singles in get_k_kickers(temp_state, length, False, [0]*15, 0):
                    d = list(d_pure)
                    for i in range(15): d[i] += delta_singles[i]
                    moves.add(tuple(d))
                    
                # Airplane + Pairs
                for delta_pairs in get_k_kickers(temp_state, length, True, [0]*15, 0):
                    d = list(d_pure)
                    for i in range(15): d[i] += delta_pairs[i]
                    moves.add(tuple(d))

    # 6. Four + 2 Singles / 2 Pairs
    for bomb_idx in range(13):
        if state[bomb_idx] >= 4:
            d_bomb = [0]*15; d_bomb[bomb_idx] = 4
            temp_state = list(state)
            temp_state[bomb_idx] -= 4
            
            for delta_s in get_k_kickers(temp_state, 2, False, [0]*15, 0):
                d = list(d_bomb)
                for i in range(15): d[i] += delta_s[i]
                moves.add(tuple(d))
                
            for delta_p in get_k_kickers(temp_state, 2, True, [0]*15, 0):
                d = list(d_bomb)
                for i in range(15): d[i] += delta_p[i]
                moves.add(tuple(d))

    # FILTER: ONLY keep moves that use the lowest available card!
    filtered_moves = [m for m in moves if m[lowest_idx] > 0]
    return filtered_moves

@lru_cache(maxsize=None)
def min_plays(state):
    """Recursive DFS to find the minimum number of plays to empty the hand."""
    if all(c == 0 for c in state):
        return 0
        
    moves = get_valid_moves(state)
    if not moves:
        return float('inf') # Should not happen in a valid game
        
    min_p = float('inf')
    for move in moves:
        next_state = tuple(s - m for s, m in zip(state, move))
        min_p = min(min_p, 1 + min_plays(next_state))
        
    return min_p



@lru_cache(maxsize=None)
def _min_plays_with_stats_impl(state):
    if all(c == 0 for c in state):
        return 0, {'singles':0, 'pairs':0, 'triples':0, 'straights':0, 'bombs':0, 'other':0}
    moves = get_valid_moves(state)
    if not moves:
        return float('inf'), None
    best_plays = float('inf')
    best_stats = None
    for move in moves:
        next_state = tuple(s - m for s, m in zip(state, move))
        sub_plays, sub_stats = _min_plays_with_stats_impl(next_state)
        if sub_plays == float('inf'):
            continue
        total_plays = 1 + sub_plays
        if total_plays < best_plays:
            best_plays = total_plays
            cards = sum(move)
            mx = max(move)
            if cards == 1: mt = 'singles'
            elif cards == 2 and mx == 2: mt = 'pairs'
            elif cards == 3 and mx == 3: mt = 'triples'
            elif mx == 1 and cards >= 3: mt = 'straights'
            elif mx >= 4: mt = 'bombs'
            else: mt = 'other'
            best_stats = sub_stats.copy()
            best_stats[mt] += 1
    return best_plays, best_stats

def resolver_score(state):
    plays, stats = _min_plays_with_stats_impl(tuple(state))
    if plays == float('inf') or stats is None:
        return 0.0
    # Primary: fewer total plays = better
    score = 2.0 / (plays + 1)
    # Secondary: each single in optimal path heavily penalizes
    singles = stats.get('singles', 0)
    for _ in range(singles):
        score *= 0.6  # 40% penalty per single
    # Bonus for good structure
    if stats.get('straights', 0) >= 1:
        score *= 1.15
    elif stats.get('pairs', 0) >= 2:
        score *= 1.05
    return score
# ==========================================
# TEST WITH YOUR SPECIFIC HAND
# ==========================================
if __name__ == "__main__":
    # Your hand from the log: 3,3, 4, 6,6, 8,8, 9,9,9, 10, J,J, Q,Q,Q, K, A, 2,2
    # String format (using 'T' for 10):
    hand_string = "4455667899TJJJQKK2BR"
    
    state = parse_string_hand(hand_string)
    
    print(f"Analyzing hand: {hand_string}")
    print(f"Total cards: {sum(state)}")
    
    import time
    start_time = time.time()
    
    # Calculate minimum plays
    plays = min_plays(state)
    
    end_time = time.time()
    print(f"\nMinimum number of plays (turns) to win: {plays}")
    print(f"Calculation time: {end_time - start_time:.4f} seconds")