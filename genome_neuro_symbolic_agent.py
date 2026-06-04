"""
Unified Genome-NeuroSymbolic Agent.
- Neural path: uses genome as extra features (43-dim input)
- Fallback path: uses genome-modulated resolver scoring
This ensures the genome ALWAYS influences decisions, never lost to plain resolver fallback.
"""
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from perfectdou.evaluation.resolver_agent import ResolverAgent, hand_to_resolver_state, move_to_resolver_state, get_optimal_path
from doudizhu_engine.hand import Hand
from doudizhu_engine.combo import Combo, ComboType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best genome
_GENOME_PATH = r"C:\Users\Vulcan X\Desktop\phd\AI\article\project\evolution\best_genome_multi_seed.json"
with open(_GENOME_PATH, 'r') as f:
    _GENOME_DATA = json.load(f)
DEFAULT_GENOME = np.array(_GENOME_DATA['genes'], dtype=np.float32)


class GenomeClassifier(nn.Module):
    def __init__(self, input_dim=43, hidden=512, num_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def classify_action(action_state):
    counts = list(action_state)
    total = sum(counts)
    max_count = max(counts) if counts else 0
    if total == 0:
        return 0
    if total == 1:
        return 1
    if total == 2 and max_count == 2:
        return 2
    if total == 3:
        return 3
    if total >= 4 and max_count >= 4:
        if counts[13] == 1 and counts[14] == 1:
            return 8
        return 7
    if total >= 3 and max_count == 1:
        return 4
    if total >= 4 and max_count == 2:
        return 5
    if total >= 6 and max_count == 3:
        return 6
    if total == 4 and max_count == 3:
        return 9
    if total == 5 and max_count == 3:
        return 10
    return 11


class GenomeNeuroSymbolicAgent(ResolverAgent):
    """
    Genome is active on BOTH neural and fallback paths.
    """

    def __init__(self, position, model_path='neuro_symbolic_genome_classifier.pt',
                 use_orphan_strategy=True, use_weak_hand_mode=True,
                 confidence_threshold=0.5, genome_genes=None):
        super().__init__(position, use_orphan_strategy=use_orphan_strategy,
                         use_weak_hand_mode=use_weak_hand_mode)
        self.confidence_threshold = confidence_threshold
        self.neural_count = 0
        self.fallback_count = 0
        self.genome = genome_genes if genome_genes is not None else DEFAULT_GENOME.tolist()
        self.combo_pref = self.genome[5]

        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model = GenomeClassifier(input_dim=43).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def act(self, infoset):
        cards = self._env_to_cards(infoset.player_hand_cards)
        hand = Hand(cards)
        state = hand_to_resolver_state(hand)

        last_move = infoset.last_move
        has_control = (last_move is None or len(last_move) == 0)
        last_move_state = [0] * 15
        if not has_control and last_move is not None and len(last_move) > 0:
            last_move_state = list(move_to_resolver_state(self._env_to_cards(last_move)))

        min_plays = self._get_min_plays(state)

        # Build 43-dim state vector
        state_vec = list(state) + last_move_state + [1.0 if has_control else 0.0, min_plays, 0.0]
        state_vec.extend(self.genome)
        state_tensor = torch.tensor([state_vec], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            predicted_type = int(np.argmax(probs))
            confidence = float(probs[predicted_type])

        if confidence < self.confidence_threshold:
            # Genome-modulated fallback
            self.fallback_count += 1
            return self._genome_fallback_act(infoset, state, has_control)

        # Filter legal actions to predicted move type
        matching_actions = []
        for action in infoset.legal_actions:
            action_cards = self._env_to_cards(action)
            action_state = move_to_resolver_state(action_cards)
            action_type = classify_action(action_state)
            if action_type == predicted_type:
                matching_actions.append(action)

        if not matching_actions:
            self.fallback_count += 1
            return self._genome_fallback_act(infoset, state, has_control)

        if len(matching_actions) == 1:
            self.neural_count += 1
            return matching_actions[0]

        # Multiple matches: use genome-modulated resolver scoring
        self.neural_count += 1
        plan = get_optimal_path(state)
        orphan_cards = self._find_orphan_cards(plan) if self.use_orphan_strategy else set()
        is_endgame = sum(state) <= 6
        action_data = []
        for action in matching_actions:
            action_cards = self._env_to_cards(action)
            action_combo = Combo(action_cards)
            action_state = move_to_resolver_state(action_cards)
            action_data.append((action, action_combo, action_state))

        if has_control:
            return self._genome_choose_control(plan, action_data, orphan_cards, is_endgame, state)
        else:
            last_combo = Combo(self._env_to_cards(last_move))
            return self._genome_choose_response(plan, last_combo, action_data, orphan_cards, is_endgame, state)

    def _genome_fallback_act(self, infoset, state, has_control):
        """Fallback to resolver, but with genome-modulated scoring."""
        last_move = infoset.last_move
        action_data = []
        for action in infoset.legal_actions:
            action_cards = self._env_to_cards(action)
            action_combo = Combo(action_cards)
            action_state = move_to_resolver_state(action_cards)
            action_data.append((action, action_combo, action_state))

        plan = get_optimal_path(state)
        orphan_cards = self._find_orphan_cards(plan) if self.use_orphan_strategy else set()
        is_endgame = sum(state) <= 6

        if has_control:
            return self._genome_choose_control(plan, action_data, orphan_cards, is_endgame, state)
        else:
            if last_move is not None and len(last_move) > 0:
                last_combo = Combo(self._env_to_cards(last_move))
                return self._genome_choose_response(plan, last_combo, action_data, orphan_cards, is_endgame, state)
            return action_data[0][0] if action_data else []

    def _genome_choose_control(self, plan, action_data, orphan_cards, is_endgame, state):
        best_score = float('-inf')
        best_action = None
        for action, combo, move_state in action_data:
            score = 0
            try:
                plan_idx = plan.index(move_state)
                score += (len(plan) - plan_idx) * 20
            except ValueError:
                score -= 20
            score += sum(move_state) * 3

            # Genome modulation
            if combo.type == ComboType.TRIPLE:
                score += 40 * self.combo_pref
            if combo.type == ComboType.PAIR:
                score += 15 * (1 - self.combo_pref)
            if combo.type == ComboType.SINGLE:
                score -= 15
            if combo.type in (ComboType.BOMB, ComboType.ROCKET):
                score -= 30

            if combo.type == ComboType.SINGLE and is_endgame:
                single_rank_idx = next((i for i, c in enumerate(move_state) if c == 1), -1)
                if single_rank_idx in orphan_cards:
                    score += 30 if single_rank_idx >= 11 else -50

            if self.use_weak_hand_mode:
                high_cards = sum(1 for c in combo.cards if c.rank.value >= 14)
                if high_cards > 0 and not is_endgame:
                    score -= 15 * high_cards

            if score > best_score:
                best_score = score
                best_action = action
        return best_action if best_action else action_data[0][0]

    def _genome_choose_response(self, plan, last_combo, action_data, orphan_cards, is_endgame, state):
        if not action_data:
            return []
        best_score = float('-inf')
        best_action = None
        for action, combo, move_state in action_data:
            score = 0
            if action != []:
                score += 50
                if combo.type == ComboType.TRIPLE:
                    score += 30 * self.combo_pref
                if combo.type in (ComboType.BOMB, ComboType.ROCKET):
                    score += 25
            else:
                score -= 10

            if is_endgame and combo.type == ComboType.SINGLE:
                single_rank_idx = next((i for i, c in enumerate(move_state) if c == 1), -1)
                if single_rank_idx in orphan_cards:
                    score += 60 if single_rank_idx >= 11 else -40

            if is_endgame:
                score -= sum(move_state) * 3

            if score > best_score:
                best_score = score
                best_action = action
        return best_action if best_action else action_data[0][0]
