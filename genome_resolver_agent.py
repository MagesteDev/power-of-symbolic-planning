"""
Genome-Modulated Resolver Agent.
Uses evolved genome genes to weight the resolver's scoring heuristics.
Genes encode strategic preferences learned via evolution.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from perfectdou.evaluation.resolver_agent import ResolverAgent, hand_to_resolver_state, move_to_resolver_state, get_optimal_path
from doudizhu_engine.hand import Hand
from doudizhu_engine.combo import Combo, ComboType

# Load best genome
_GENOME_PATH = r"C:\Users\Vulcan X\Desktop\phd\AI\article\project\evolution\best_genome_multi_seed.json"
with open(_GENOME_PATH, 'r') as f:
    _GENOME_DATA = json.load(f)
GENES = np.array(_GENOME_DATA['genes'], dtype=np.float32)

# Gene meanings (from decision_engine.py):
# g0: early phase threshold (35 + (g0-0.5)*10)
# g1: mid phase threshold  (15 + (g1-0.5)*10)
# g2: bomb threshold       (3 + (g2-0.5)*4)
# g3: pass gap threshold (5 + (g3-0.5)*4)
# g4: combo threshold      (2 + (g4-0.5)*2)
# g5: combo preference     (0=conservative, 1=aggressive triples)
# g6: exploration rate     (random move chance)
# g7-g9: unused

# Derived strategic parameters:
BOMB_THRESHOLD = 3 + int((GENES[2] - 0.5) * 4)  # = 2 (aggressive)
PASS_GAP = 5 + int((GENES[3] - 0.5) * 4)         # = 5
COMBO_PREF = GENES[5]                             # = 0.97 (very aggressive triples)


class GenomeResolverAgent(ResolverAgent):
    """
    Resolver with genome-modulated scoring.
    Uses evolved genes to bias move selection.
    """

    def __init__(self, position, use_orphan_strategy=True, use_weak_hand_mode=True,
                 genome_genes=None):
        super().__init__(position, use_orphan_strategy=use_orphan_strategy,
                         use_weak_hand_mode=use_weak_hand_mode)
        self.genome = genome_genes if genome_genes is not None else GENES
        self.bomb_threshold = 3 + int((self.genome[2] - 0.5) * 4)
        self.combo_pref = self.genome[5]

    def _choose_control_move(self, plan, action_data, orphan_cards, is_endgame, state):
        best_score = float('-inf')
        best_action = None

        for action, combo, move_state in action_data:
            score = 0

            # Base: prefer earlier plan items
            try:
                plan_idx = plan.index(move_state)
                score += (len(plan) - plan_idx) * 20
            except ValueError:
                score -= 20

            # Combo size bonus
            combo_size = sum(move_state)
            score += combo_size * 3

            # Genome: aggressive triple preference
            if combo.type == ComboType.TRIPLE:
                score += 40 * self.combo_pref

            # Genome: pair preference (conservative side)
            if combo.type == ComboType.PAIR:
                score += 15 * (1 - self.combo_pref)

            # Penalty for singles
            if combo.type == ComboType.SINGLE:
                score -= 15

            # Bomb penalty (genome: bomb aggressively when opponent low)
            if combo.type in (ComboType.BOMB, ComboType.ROCKET):
                score -= 30

            # Endgame orphan strategy
            if combo.type == ComboType.SINGLE and is_endgame:
                single_rank_idx = next((i for i, c in enumerate(move_state) if c == 1), -1)
                if single_rank_idx in orphan_cards:
                    if single_rank_idx >= 11:
                        score += 30
                    else:
                        score -= 50

            # Weak-hand conservation
            if self.use_weak_hand_mode:
                high_cards_in_move = sum(1 for c in combo.cards
                                         if c.rank.value >= 14)
                if high_cards_in_move > 0 and not is_endgame:
                    score -= 15 * high_cards_in_move

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else action_data[0][0]

    def _choose_response_move(self, plan, last_combo, action_data, orphan_cards, is_endgame, state):
        if not action_data:
            return []

        best_score = float('-inf')
        best_action = None

        for action, combo, move_state in action_data:
            score = 0

            # Taking control is good
            if action != []:
                score += 50

                # Genome: aggressive triple preference in response too
                if combo.type == ComboType.TRIPLE:
                    score += 30 * self.combo_pref

                # Bomb bonus when opponent has few cards (genome aggressive)
                if combo.type in (ComboType.BOMB, ComboType.ROCKET):
                    score += 25

            else:
                score -= 10

            # Endgame orphan response
            if is_endgame and combo.type == ComboType.SINGLE:
                single_rank_idx = next((i for i, c in enumerate(move_state) if c == 1), -1)
                if single_rank_idx in orphan_cards:
                    if single_rank_idx >= 11:
                        score += 60
                    else:
                        score -= 40

            # Prefer smaller combos in endgame
            if is_endgame:
                score -= sum(move_state) * 3

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else action_data[0][0]
