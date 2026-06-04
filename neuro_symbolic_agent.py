"""
Neuro-Symbolic Agent v2.
Neural classifier predicts MOVE TYPE, resolver picks specific cards.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from perfectdou.evaluation.resolver_agent import ResolverAgent, hand_to_resolver_state, move_to_resolver_state, get_optimal_path
from doudizhu_engine.hand import Hand
from doudizhu_engine.combo import Combo, ComboType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoveTypeClassifier(nn.Module):
    def __init__(self, input_dim=33, hidden=512, num_classes=12):
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
    """Return move type class from action vector."""
    counts = list(action_state)
    total = sum(counts)
    max_count = max(counts) if counts else 0
    if total == 0:
        return 0  # PASS
    if total == 1:
        return 1  # SINGLE
    if total == 2 and max_count == 2:
        return 2  # PAIR
    if total == 3:
        return 3  # TRIPLE
    if total >= 4 and max_count >= 4:
        if counts[13] == 1 and counts[14] == 1:
            return 8  # ROCKET
        return 7  # BOMB
    if total >= 3 and max_count == 1:
        return 4  # STRAIGHT
    if total >= 4 and max_count == 2:
        return 5  # PAIR_STRAIGHT
    if total >= 6 and max_count == 3:
        return 6  # TRIPLE_STRAIGHT
    if total == 4 and max_count == 3:
        return 9  # 3+1
    if total == 5 and max_count == 3:
        return 10  # 3+2
    return 11  # OTHER


class NeuroSymbolicAgent(ResolverAgent):
    """
    Move-type classifier + resolver execution.
    Neural predicts action category; resolver scores and picks best within category.
    """

    def __init__(self, position, model_path='neuro_symbolic_classifier.pt',
                 use_orphan_strategy=True, use_weak_hand_mode=True,
                 confidence_threshold=0.5):
        super().__init__(position, use_orphan_strategy=use_orphan_strategy,
                         use_weak_hand_mode=use_weak_hand_mode)
        self.confidence_threshold = confidence_threshold
        self.neural_count = 0
        self.fallback_count = 0

        checkpoint = torch.load(model_path, map_location=DEVICE)
        input_dim = checkpoint.get('input_dim', 33)

        if input_dim == 43:
            self.model = GenomeClassifier(input_dim=43).to(DEVICE)
            self.genome_genes = checkpoint.get('genome_genes', [0.5] * 10)
        else:
            self.model = MoveTypeClassifier(input_dim=33).to(DEVICE)
            self.genome_genes = None

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

        state_vec = list(state) + last_move_state + [1.0 if has_control else 0.0, min_plays, 0.0]
        if self.genome_genes is not None:
            state_vec.extend(self.genome_genes)
        state_tensor = torch.tensor([state_vec], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            predicted_type = int(np.argmax(probs))
            confidence = float(probs[predicted_type])

        if confidence < self.confidence_threshold:
            self.fallback_count += 1
            return super().act(infoset)

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
            return super().act(infoset)

        # If only one match, use it
        if len(matching_actions) == 1:
            self.neural_count += 1
            return matching_actions[0]

        # Multiple matches: use resolver scoring to pick best
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
            return self._choose_control_move(plan, action_data, orphan_cards, is_endgame, state)
        else:
            last_combo = Combo(self._env_to_cards(last_move))
            return self._choose_response_move(plan, last_combo, action_data, orphan_cards, is_endgame, state)
