"""
Train a small neural network on shortest-path winner data.
State -> action regression with resolver fallback at inference.
Uses GPU if available.
"""
import sys
import csv
import pickle
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


class DouDizhuDataset(Dataset):
    def __init__(self, csv_file):
        self.states = []
        self.actions = []
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

                action = []
                for i in range(15):
                    action.append(int(row[f'action_{i}']))

                self.states.append(state)
                self.actions.append(action)

        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        print(f"Loaded {len(self.states)} samples")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (torch.tensor(self.states[idx], device=DEVICE),
                torch.tensor(self.actions[idx], device=DEVICE))


class SmallMLP(nn.Module):
    def __init__(self, input_dim=33, hidden=256, output_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, output_dim),
            nn.Sigmoid()  # action counts are non-negative
        )

    def forward(self, x):
        return self.net(x)


def train_model(csv_file, epochs=30, batch_size=256, lr=1e-3):
    dataset = DouDizhuDataset(csv_file)
    # Train/val split
    n = len(dataset)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = SmallMLP().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for states, actions in train_loader:
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                preds = model(states)
                loss = criterion(preds, actions)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    return model


def main():
    csv_file = "shortest_winner_data_5000_20260603_120146.csv"
    print(f"Training on {csv_file}...")

    model = train_model(csv_file, epochs=40, batch_size=512, lr=2e-3)

    # Save model
    model_path = "neuro_symbolic_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': 33,
        'hidden': 256,
        'output_dim': 15,
    }, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
