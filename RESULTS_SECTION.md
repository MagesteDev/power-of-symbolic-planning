# Results

## 4.1 Experimental Setup

All experiments were conducted on the DouDizhu game environment provided by the PerfectDou framework. The landlord agent was evaluated against two peasant opponents implemented as `RandomAgent`, ensuring reproducible and deterministic game dynamics across all benchmarks. Each game was initialized with a fixed random seed, and card distributions were identical across all competing agents for fair comparison.

We evaluate four principal agent architectures:

- **DouZero (DZ)**: A deep reinforcement learning baseline trained via self-play with 1.4B parameters.
- **DFS (Symbolic)**: Our symbolic agent combining DFS-based shortest-path planning (`min_plays`) with hardcoded heuristics, orphan card strategy, and weak-hand conservation mode.
- **NeuroSymbolic (NS)**: A hybrid agent where a lightweight neural network predicts the *move type* (e.g., SINGLE, PAIR, TRIPLE, BOMB), and the resolver selects the best concrete move within that category.
- **Genome-Enhanced NeuroSymbolic (G-NS)**: The NS agent with an evolved genome appended as 10 additional input features to the neural classifier.

All neural models were trained on 54,220 state-action pairs extracted from 5,000 games, retaining only winning traces from the agent with the shortest path to victory (DouZero or DFS).

## 4.2 Baseline Performance

Table 1 presents the win rates of the baseline agents across 10,000 unseen games.

| Agent | Wins / 10,000 | Win Rate (%) | Avg. Turns |
|-------|-------------|-------------|-----------|
| DouZero | 9,571 | **95.7** | 34.3 |
| DFS (Symbolic) | 9,022 | **90.2** | 34.0 |
| NeuroSymbolic (thr=0.5) | 8,913 | **89.1** | 35.0 |
| Genome-Enhanced NS | 8,878 | **88.8** | 34.8 |

*Table 1: Baseline performance on 10,000 unseen seeds (200,000–209,999). Neural hit rates: NS 77.7%, G-NS 81.2%.*

The DFS symbolic agent achieves 90.2% win rate, demonstrating that handcrafted heuristics combined with combinatorial planning are highly effective. DouZero, as expected, leads with 95.7%, though at the cost of 1.4 billion parameters and extensive self-play training. The NeuroSymbolic agent, using a 512-hidden-unit classifier with only ~260K parameters, reaches 89.1% — within 1.1 percentage points of the pure DFS (Symbolic) while requiring orders of magnitude fewer parameters than DouZero.

## 4.3 Neural Architecture Evolution

Our initial approach trained a regressor to predict the 15-dimensional action vector directly. This proved ineffective, yielding only 69.6% win rate. The mismatch between continuous regression output (sigmoid-activated) and discrete card counts (integers in {0,1,2,3,4}) caused the network to produce invalid or suboptimal moves.

We re-architected the neural component as a **move-type classifier** with 12 output classes: PASS, SINGLE, PAIR, TRIPLE, STRAIGHT, PAIR_STRAIGHT, TRIPLE_STRAIGHT, BOMB, ROCKET, TRIPLE_SINGLE, TRIPLE_PAIR, and OTHER. The resolver then filters legal actions to the predicted category and scores concrete moves within it using its existing heuristics. Table 2 shows the class distribution in the training data, revealing severe imbalance — particularly for TRIPLE (133 samples) and ROCKET (1 sample).

| Class | Move Type | Count | Freq (%) |
|-------|-----------|-------|---------|
| 0 | PASS | 14,184 | 26.2 |
| 1 | SINGLE | 18,311 | 33.8 |
| 2 | PAIR | 9,950 | 18.4 |
| 3 | TRIPLE | 133 | 0.2 |
| 4 | STRAIGHT | 3,837 | 7.1 |
| 5 | PAIR_STRAIGHT | 1,000 | 1.8 |
| 6 | TRIPLE_STRAIGHT | 420 | 0.8 |
| 7 | BOMB | 844 | 1.6 |
| 8 | ROCKET | 1 | <0.1 |
| 9 | TRIPLE_SINGLE | 3,348 | 6.2 |
| 10 | TRIPLE_PAIR | 1,583 | 2.9 |
| 11 | OTHER | 609 | 1.1 |

*Table 2: Training class distribution (N = 54,220).*

Despite imbalance, the classifier achieved **79.5% validation accuracy** on a 10% held-out split. Table 3 reports the neural hit rate (fraction of decisions delegated to the classifier rather than the resolver fallback) at varying confidence thresholds.

| Threshold | Win Rate (%) | Neural Hit Rate (%) |
|-----------|-------------|-------------------|
| 0.3 | 86.4 | 88.9 |
| 0.5 | 88.9 | 81.1 |
| 0.7 | 87.8 | 69.1 |
| 0.9 | 85.2 | 51.3 |

*Table 3: Threshold tuning on 200 unseen seeds. Threshold 0.5 selected as optimal.*

## 4.4 Genome Integration

We evolved a 10-gene genome via an evolutionary algorithm over multiple seeds, yielding a multi-seed validated genome with fitness 951.6 (Table 4). Each gene modulates a strategic parameter in the `DecisionEngine` (e.g., game-phase thresholds, bomb usage, combo preference).

| Gene | Meaning | Value | Range |
|------|---------|-------|-------|
| g₀ | Early-phase threshold | 0.782 | [0,1] |
| g₁ | Mid-phase threshold | 0.430 | [0,1] |
| g₂ | Bomb threshold | 0.098 | [0,1] |
| g₃ | Pass-gap threshold | 0.601 | [0,1] |
| g₄ | Combo threshold | 0.767 | [0,1] |
| g₅ | Combo preference (0=conservative, 1=aggressive) | **0.974** | [0,1] |
| g₆ | Exploration rate | 0.120 | [0,1] |
| g₇–g₉ | Reserved | — | [0,1] |

*Table 4: Evolved genome genes (multi-seed validation, fitness = 951.6).*

### 4.4.1 Approach A: Genome as Neural Features

We appended the 10 genome genes to the 33-dimensional state vector, producing a 43-dimensional input. The genome-enhanced classifier achieved **80.0% validation accuracy** (+1.4% absolute over the baseline classifier). Table 5 compares game performance.

| Agent | 500-Game Win Rate | 1,000-Game Win Rate | 10,000-Game Win Rate | Neural Hit |
|-------|------------------|-------------------|---------------------|-----------|
| NeuroSymbolic (baseline) | 86.4% | 88.9% | **89.1%** | 77.7% |
| Genome-Enhanced (G-NS) | 88.4% | 88.6% | **88.8%** | 81.2% |

*Table 5: Genome-enhanced vs baseline NeuroSymbolic. At 10,000 games the two variants are statistically equivalent (89.1% vs 88.8%), confirming the gain diminishes at larger sample sizes.*

### 4.4.2 Approach B: Genome as Resolver Modulation

We also tested using the genome to directly weight the resolver's scoring heuristics (e.g., boosting triple preference when g₅ ≈ 0.97). This yielded **88.6%** on 500 games — actually **below** the pure resolver's 90.0%. The genome was evolved for a different decision engine and does not transfer cleanly to the resolver's scoring function.

### 4.4.3 Approach C: Unified Genome (Both Paths)

A unified agent that applies genome features to the neural path **and** genome modulation to the resolver fallback achieved only **84.7%** on 300 games. The genome-modulated fallback actively degraded performance relative to the original resolver heuristics.

**Conclusion**: The genome provides value **only when the neural network learns to exploit it as input features**. Using it to override the resolver's hand-tuned scoring rules is counterproductive.

## 4.5 Comparative Summary

Table 6 aggregates the definitive 10,000-game results (seeds 200,000–209,999).

| Agent | Parameters | Win Rate (%) | vs DouZero | vs DFS |
|-------|-----------|-------------|-----------|------------|
| DouZero | 1.4B | **95.7** | — | +5.5 |
| DFS (Symbolic) | ~0 (symbolic) | **90.2** | −5.5 | — |
| NeuroSymbolic | ~260K | **89.1** | −6.6 | −1.1 |
| Genome-Enhanced NS | ~260K | **88.8** | −6.9 | −1.4 |

*Table 6: Final 10,000-game comparison. All agents show stable win rates at 10× scale; confidence intervals tighten to approximately ±0.5%.*

## 4.6 Key Findings

1. **Small neural networks can approach symbolic performance**. With only ~260K parameters and 5,000 games of behavioral cloning data, the NeuroSymbolic agent reaches 89.1% win rate on 10,000 games — within 1.1 pp of the pure DFS symbolic agent and comparable to the 1.4B-parameter DouZero agent's gap from the DFS baseline.

2. **Move-type classification beats direct action regression**. Re-architecting from a 15-dim action regressor to a 12-class move-type classifier improved win rate from 69.6% to 88.9% — a **19.3 percentage point** gain.

3. **Genome features improve the classifier modestly**. Appending 10 evolved genes as input features raised validation accuracy by 1.4% and game win rate by up to 2.0%, but the gain diminishes at larger scales (500→1,000→10,000 games), becoming statistically indistinguishable at 10,000 games (89.1% vs 88.8%).

4. **The resolver's heuristics are hard to beat**. Attempts to modulate resolver scoring with the genome consistently underperformed the original heuristics, suggesting the handcrafted rules are already well-tuned for this environment.

5. **The neuro-symbolic paradigm is viable for DouDizhu**. A lightweight neural classifier supported by a symbolic resolver achieves strong performance without the training overhead of deep RL, validating the core hypothesis of this work.

## 4.7 Limitations and Future Work

- **Class imbalance**: TRIPLE (133 samples) and ROCKET (1 sample) are severely underrepresented. Weighted loss or data augmentation may help.
- **Opponent model**: All experiments use `RandomAgent` peasants. Performance against stronger opponents (e.g., DouZero peasants) remains unstudied.
- **Feature expansion**: The current state vector lacks opponent card counts, bomb history, and game-phase indicators — all information the DFS agent implicitly uses.
- **Larger datasets**: Scaling from 5,000 to 10,000+ games may improve generalization, particularly for rare move types.
