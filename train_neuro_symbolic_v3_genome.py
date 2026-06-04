"""
Train a move-type classifier WITH genome features as extra inputs.
State (33) + genome genes (10) = 43-dim input.
Uses GPU if available.
"""
import sys
import csv
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load best genome
GENOME_PATH = r"C:\Users\Vulcan X\Desktop\phd\AI\article\project\evolution\best_genome_multi_seed.json"
with open(GENOME_PATH, 'r') as f:
    genome_data = json.load(f)
GENOME_GENES = np.array(genome_data['genes'], dtype=np.float32)
print(f"Loaded genome: fitness={genome_data['fitness']}, genes={GENOME_GENES.tolist()}")


def action_to_move_type(action_state):
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


class GenomeDataset(Dataset):
    def __init__(self, csv_file, genome_genes):
        self.states = []
        self.labels = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = []
                for i in range(15):
                    state.append(int(row[f'hand_{i}']))
                for i in range(15):
                    state.append(int(row[f'last_move_{i}']))
                state.append(float(row['has_control']))
                state.append(float(row['min_plays']))
                state.append(float(row['turn_num']))
                # Append genome genes (10 features)
                state.extend(genome_genes.tolist())

                action = []
                for i in range(15):
                    action.append(int(row[f'action_{i}']))

                label = action_to_move_type(action)

                self.states.append(state)
                self.labels.append(label)

        self.states = np.array(self.states, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Loaded {len(self.states)} samples, input_dim={self.states.shape[1]}")
        from collections import Counter
        dist = Counter(self.labels)
        print(f"Class distribution: {dict(sorted(dist.items()))}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (torch.tensor(self.states[idx], device=DEVICE),
                torch.tensor(self.labels[idx], device=DEVICE))


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


def train_model(csv_file, genome_genes, epochs=60, batch_size=512, lr=2e-3):
    dataset = GenomeDataset(csv_file, genome_genes)
    n = len(dataset)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = GenomeClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for states, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for states, labels in val_loader:
                outputs = model(states)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    model.load_state_dict(best_model_state)
    return model


def main():
    csv_file = "shortest_winner_data_5000_20260603_120146.csv"
    print(f"Training genome-enhanced classifier on {csv_file}...")

    model = train_model(csv_file, GENOME_GENES, epochs=60, batch_size=512, lr=2e-3)

    model_path = "neuro_symbolic_genome_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': 43,
        'hidden': 512,
        'num_classes': 12,
        'genome_genes': GENOME_GENES.tolist(),
    }, model_path)
    print(f"Saved genome-enhanced model to {model_path}")


if __name__ == "__main__":
    main()
