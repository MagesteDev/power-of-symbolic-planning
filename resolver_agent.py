"""
Resolver-Only Agent Adapter for PerfectDou Evaluation Framework.

Pure hardcoded rules + DFS resolver. No neural network.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path
# resolver_agent.py -> evaluation -> perfectdou (inner) -> perfectdou (outer) -> new_PROJECT -> project
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from doudizhu_engine.card import Card, Rank, Suit
from doudizhu_engine.hand import Hand
from doudizhu_engine.combo import Combo, ComboType
from perfectdou.resolver import get_valid_moves, _min_plays_with_stats_impl


EnvCard2Rank = {
    3: Rank.THREE, 4: Rank.FOUR, 5: Rank.FIVE, 6: Rank.SIX, 7: Rank.SEVEN,
    8: Rank.EIGHT, 9: Rank.NINE, 10: Rank.TEN, 11: Rank.JACK, 12: Rank.QUEEN,
    13: Rank.KING, 14: Rank.ACE, 17: Rank.TWO, 20: Rank.SMALL_JOKER, 30: Rank.BIG_JOKER
}

Rank2EnvCard = {
    Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6, Rank.SEVEN: 7,
    Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12,
    Rank.KING: 13, Rank.ACE: 14, Rank.TWO: 17, Rank.SMALL_JOKER: 20, Rank.BIG_JOKER: 30
}


def hand_to_resolver_state(hand):
    """Convert Hand to resolver state tuple."""
    rank_to_idx = {
        Rank.THREE: 0, Rank.FOUR: 1, Rank.FIVE: 2, Rank.SIX: 3, Rank.SEVEN: 4,
        Rank.EIGHT: 5, Rank.NINE: 6, Rank.TEN: 7, Rank.JACK: 8, Rank.QUEEN: 9,
        Rank.KING: 10, Rank.ACE: 11, Rank.TWO: 12,
        Rank.SMALL_JOKER: 13, Rank.BIG_JOKER: 14
    }
    state = [0] * 15
    for card in hand.cards:
        state[rank_to_idx[card.rank]] += 1
    return tuple(state)


def move_to_resolver_state(cards):
    """Convert list of Cards to resolver move tuple."""
    rank_to_idx = {
        Rank.THREE: 0, Rank.FOUR: 1, Rank.FIVE: 2, Rank.SIX: 3, Rank.SEVEN: 4,
        Rank.EIGHT: 5, Rank.NINE: 6, Rank.TEN: 7, Rank.JACK: 8, Rank.QUEEN: 9,
        Rank.KING: 10, Rank.ACE: 11, Rank.TWO: 12,
        Rank.SMALL_JOKER: 13, Rank.BIG_JOKER: 14
    }
    state = [0] * 15
    for card in cards:
        state[rank_to_idx[card.rank]] += 1
    return tuple(state)


def get_optimal_path(state):
    """Get optimal sequence of resolver moves."""
    if all(c == 0 for c in state):
        return []
    
    moves = get_valid_moves(state)
    best_move = None
    best_mp = float('inf')
    
    for move in moves:
        new_state = tuple(state[i] - move[i] for i in range(15))
        mp = _min_plays_with_stats_impl(new_state)[0]
        
        if mp < best_mp:
            best_mp = mp
            best_move = move
    
    if best_move is None:
        return []
    
    new_state = [state[i] - best_move[i] for i in range(15)]
    path = get_optimal_path(new_state)
    return [best_move] + path


class ResolverAgent:
    """Agent that uses resolver + hardcoded rules only."""
    
    def __init__(self, position, model_path=None, use_orphan_strategy=False, use_weak_hand_mode=True):
        self.position = position
        self.use_orphan_strategy = use_orphan_strategy
        self.use_weak_hand_mode = use_weak_hand_mode
    
    def _env_to_cards(self, env_cards):
        cards = []
        for ec in env_cards:
            rank = EnvCard2Rank[ec]
            suit = Suit.JOKER if rank in (Rank.SMALL_JOKER, Rank.BIG_JOKER) else Suit.SPADES
            cards.append(Card(rank, suit))
        return cards
    
    def _convert_combo_to_env(self, combo):
        if combo is None or len(combo.cards) == 0:
            return []
        env_cards = []
        for card in combo.cards:
            env_cards.append(Rank2EnvCard[card.rank])
        env_cards.sort()
        return env_cards
    
    def _find_orphan_cards(self, plan):
        """
        Detect low cards played as singles (not part of any pair/straight/triple/bomb).
        These are 'orphan' cards we want to dump cheaply.
        Returns list of rank indices (0-14) that are low singles in the plan.
        """
        orphans = []
        for move in plan:
            if sum(move) == 1:  # Single card move
                for i, count in enumerate(move):
                    if count > 0 and i < 11:  # Low card: below Ace (index 11)
                        orphans.append(i)
        return orphans

    def _get_min_plays(self, state):
        """Get minimum plays remaining from current state."""
        return _min_plays_with_stats_impl(state)[0]
    
    def _count_big_cards(self, state):
        """Count A, 2, and Jokers in hand (indices 11-14)."""
        return sum(state[i] for i in range(11, 15))
    
    def _is_weak_hand(self, state):
        """Weak hand = few big cards (A, 2, Jokers)."""
        return self._count_big_cards(state) <= 2
    
    def _has_pair_of_aces_or_twos(self, state):
        """Check if we hold a pair of Aces (11) or 2s (12)."""
        return state[11] >= 2 or state[12] >= 2
    
    def act(self, infoset):
        # Convert hand
        cards = self._env_to_cards(infoset.player_hand_cards)
        hand = Hand(cards)
        state = hand_to_resolver_state(hand)
        
        # Compute optimal plan
        plan = get_optimal_path(state)
        
        # Detect orphan cards every turn (low singles in plan)
        orphan_cards = self._find_orphan_cards(plan)
        
        # Check endgame status (min plays remaining)
        min_plays = self._get_min_plays(state)
        is_endgame = (min_plays <= 3)
        
        # Get legal actions
        legal_actions = infoset.legal_actions
        if len(legal_actions) == 0:
            raise AssertionError("Empty legal_actions")
        
        # Convert legal actions to combos and resolver states
        action_data = []
        for action in legal_actions:
            action_cards = self._env_to_cards(action)
            action_combo = Combo(action_cards)
            action_state = move_to_resolver_state(action_cards)
            action_data.append((action, action_combo, action_state))
        
        # Determine if we have control
        last_move = infoset.last_move
        has_control = (last_move is None or len(last_move) == 0)
        
        if has_control:
            chosen = self._choose_control_move(plan, action_data, orphan_cards, is_endgame, state)
        else:
            last_combo = Combo(self._env_to_cards(last_move))
            chosen = self._choose_response_move(plan, last_combo, action_data, orphan_cards, is_endgame, state)
        
        return chosen
    
    def _choose_control_move(self, plan, action_data, orphan_cards, is_endgame, state):
        """Choose move when we have control (no last move)."""
        best_action = None
        best_score = -float('inf')
        
        weak_hand = self.use_weak_hand_mode and self._is_weak_hand(state)
        has_ace_pair = self._has_pair_of_aces_or_twos(state)
        
        for action, combo, move_state in action_data:
            if len(combo.cards) == 0:
                continue  # Don't pass
            
            score = 0
            
            # BIG bonus if this move is in our optimal plan
            if move_state in plan:
                idx = plan.index(move_state)
                score += 100 - idx * 10  # Earlier in plan = better
            
            # Bonus for combo size (play big combos first)
            score += len(combo.cards) * 3
            
            # Penalty for singles (delay them)
            if combo.type == ComboType.SINGLE:
                score -= 40
            
            # Slight penalty for bombs/rockets (save them)
            bomb_penalty = 20
            if weak_hand:
                bomb_penalty = 5  # Bombs more valuable when weak
            if combo.type == ComboType.BOMB or combo.type == ComboType.ROCKET:
                score -= bomb_penalty
            
            # Bonus for straights (they're efficient)
            if combo.type == ComboType.STRAIGHT:
                score += len(combo.cards) * 2
            
            # WEAK HAND: protect big cards
            if weak_hand:
                # Never play a single Ace, 2, or Joker to take control
                if combo.type == ComboType.SINGLE:
                    single_rank_idx = None
                    for i, count in enumerate(move_state):
                        if count > 0:
                            single_rank_idx = i
                            break
                    if single_rank_idx is not None and single_rank_idx >= 11:
                        score -= 15  # Mild penalty for burning big single
                
                # Pair preservation: bonus for keeping Ace/2 pairs intact
                if combo.type == ComboType.PAIR and has_ace_pair:
                    # If we're about to play a pair of Aces or 2s, it's OK (pair is strong)
                    pass  # Pairs are fine, they retain value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        # Fallback
        if best_action is None:
            best_action = action_data[0][0]
        
        return best_action
    
    def _choose_response_move(self, plan, last_combo, action_data, orphan_cards, is_endgame, state):
        """Choose move to respond to opponent."""
        best_action = None
        best_score = -float('inf')
        
        # Check if opponent played a single
        opponent_played_single = (last_combo.type == ComboType.SINGLE)
        weak_hand = self.use_weak_hand_mode and self._is_weak_hand(state)
        
        for action, combo, move_state in action_data:
            if len(combo.cards) == 0:
                # Pass - slight penalty
                score = -10
                
                # Endgame: if opponent played single and we have orphan single,
                # passing is bad because we might have to break our straight later
                if self.use_orphan_strategy and is_endgame and opponent_played_single and orphan_cards:
                    score -= 30  # Strong penalty for passing in this situation
                
                # WEAK HAND: passing is OK if we'd have to burn a big card
                if weak_hand and opponent_played_single:
                    score += 5  # Mild bonus for passing
            else:
                score = 0
                
                # Bonus if in plan
                if move_state in plan:
                    idx = plan.index(move_state)
                    score += 30 - idx * 5
                
                # Prefer smaller moves when responding
                score -= len(combo.cards)
                
                # Bonus for taking control with minimal cost
                bomb_bonus = 15
                if weak_hand:
                    bomb_bonus = 25  # Bombs more valuable when weak
                if combo.type == ComboType.BOMB or combo.type == ComboType.ROCKET:
                    score += bomb_bonus  # Worth using to take control
                
                # WEAK HAND: penalize playing big singles to take control
                if weak_hand and combo.type == ComboType.SINGLE:
                    single_rank_idx = None
                    for i, count in enumerate(move_state):
                        if count > 0:
                            single_rank_idx = i
                            break
                    if single_rank_idx is not None and single_rank_idx >= 11:
                        score -= 10  # Mild penalty for wasting big single
            
            if score > best_score:
                best_score = score
                best_action = action
        
        if best_action is None:
            best_action = action_data[0][0]
        
        return best_action
