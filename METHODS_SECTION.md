# Methods

## 3.1 Game Environment and Benchmark Setup

All experiments are conducted on the DouDizhu game environment provided by the PerfectDou framework [1]. The game follows standard DouDizhu rules: one landlord versus two peasants, with the landlord receiving 20 cards and each peasant receiving 17 cards. Three hidden cards (the "bottom cards") are revealed to the landlord after bidding.

For reproducibility, every game is initialized with a fixed random seed. Card distributions are identical across all competing agents for fair comparison. The landlord agent is evaluated against two peasant opponents implemented as `RandomAgent` from the PerfectDou evaluation suite, ensuring deterministic game dynamics.

## 3.2 Baseline Models

### 3.2.1 DouZero

DouZero [2] is a deep reinforcement learning system for DouDizhu trained via distributed self-play. It uses an actor-critic architecture where each position (landlord, landlord_up, landlord_down) is controlled by a separate neural network. The landlord model (`LandlordLstmModel`) consists of an LSTM encoder (input dimension 162, hidden dimension 128) followed by six fully connected layers (512 units each with ReLU activation) outputting a scalar value estimate. The farmer models (`FarmerLstmModel`) follow the same architecture with adjusted input dimensions to account for the partner's observed cards.

**Parameter count**: The evaluation checkpoint released with PerfectDou contains ~4.5M parameters across the three position-specific models (Landlord: 1.46M, Farmer: 1.51M each). The original DouZero paper reports ~1.4B parameters for the full distributed training system; we report both figures for transparency.

DouZero is trained via the Average Difference Points (ADP) objective, which estimates the expected point difference between the landlord and the combined peasants. Training requires distributed self-play across thousands of CPU/GPU workers over billions of frames.

### 3.2.2 PerfectDou

PerfectDou [1] extends DouZero with Perfect Information Distillation (PID), a training framework that treats DouDizhu as a perfect-information game during training while executing under imperfect information during gameplay. The teacher network observes all players' hands and guides the student network via a distillation loss. This approach addresses the credit assignment problem inherent in incomplete-information card games.

PerfectDou achieves state-of-the-art performance against all existing DouDizhu AI programs. The evaluation code and pretrained models are publicly available; however, the full training code and feature-engineering modules are distributed as compiled shared libraries.

## 3.3 Our Approaches

### 3.3.1 NeuroSymbolic Classifier

Our NeuroSymbolic agent consists of a lightweight neural classifier coupled with a symbolic action executor. The neural component is a three-layer fully connected network:

- **Input layer**: 33-dimensional state vector (15 card counts + 18 structural features) or 43-dimensional when genome features are appended.
- **Hidden layers**: Two layers of 512 units each with ReLU activation.
- **Output layer**: 12 units (no activation) representing move-type logits: PASS, SINGLE, PAIR, TRIPLE, STRAIGHT, PAIR_STRAIGHT, TRIPLE_STRAIGHT, BOMB, ROCKET, TRIPLE_SINGLE, TRIPLE_PAIR, OTHER.

**Parameter count**: ~286K parameters for the 33-input variant; ~291K parameters for the 43-input (genome-enhanced) variant.

The classifier is trained via behavioral cloning on 54,220 state-action pairs extracted from 5,000 games. Only winning traces from the agent with the shortest path to victory (DouZero or our DFS agent) are retained. Training uses cross-entropy loss with the Adam optimizer (learning rate 1e-3, batch size 256) for 30 epochs. Validation accuracy on a 10% held-out split is 79.5% for the baseline classifier and 80.0% for the genome-enhanced variant.

At inference time, the classifier predicts a move type. The symbolic executor then filters the legal actions to those matching the predicted category and selects the best concrete move using DFS-based shortest-path scoring (see Section 3.3.3). If classifier confidence (softmax probability of the top class) falls below a threshold (tuned to 0.5), the agent falls back to pure symbolic execution.

### 3.3.2 Genome Integration

We evolve a 10-gene genome via an evolutionary algorithm over multiple random seeds. Each gene is a float in [0, 1] that modulates a strategic parameter: game-phase thresholds, bomb usage tendency, combo preference, and exploration rate. The multi-seed validated genome achieves a fitness score of 951.6 across 1,000 evaluation games.

We test three integration strategies:

1. **Neural features**: Append the 10 genome genes to the state vector, expanding input dimension from 33 to 43.
2. **Resolver modulation**: Use genome genes to weight the symbolic executor's scoring heuristics.
3. **Unified**: Apply genome features to both the neural path and the symbolic fallback.

### 3.3.3 DFS Shortest-Path Planner

The core of our symbolic agent is a depth-first search (DFS) algorithm with memoization that computes the minimum number of plays required to empty a given hand. The algorithm is inspired by classical search techniques for perfect-information card games [3], adapted to the combinatorial structure of DouDizhu.

#### State Representation

A hand is represented as a 15-tuple

\[
s = (c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_{10}, c_J, c_Q, c_K, c_A, c_2, c_{\text{SJ}}, c_{\text{BJ}})
\]

where each \(c_i \in \{0, 1, 2, 3, 4\}\) for standard ranks (3 through 2) and \(c_i \in \{0, 1\}\) for the Small Joker and Big Joker. The total number of cards is \(\sum_i c_i\), ranging from 0 (empty hand) to 20 (full landlord hand). This representation is compact, hashable, and enables efficient memoization.

#### Move Generation

For a given state, `get_valid_moves(state)` enumerates all legal card combinations that can be formed from the remaining cards. The DouDizhu move taxonomy comprises:

| Move Type | Cards Used | Example |
|-----------|-----------|---------|
| Single | 1 card | \(\{K\}\) |
| Pair | 2 identical | \(\{K, K\}\) |
| Triple | 3 identical | \(\{K, K, K\}\) |
| Bomb | 4 identical | \(\{K, K, K, K\}\) |
| Rocket | Both Jokers | \(\{\text{SJ}, \text{BJ}\}\) |
| Straight | 5+ consecutive singles | \(\{3,4,5,6,7\}\) |
| Pair Straight | 3+ consecutive pairs | \(\{3,3,4,4,5,5\}\) |
| Triple + Single | 3-of-kind + 1 single | \(\{K,K,K\} + \{3\}\) |
| Triple + Pair | 3-of-kind + 1 pair | \(\{K,K,K\} + \{3,3\}\) |
| Airplane (pure) | 2+ consecutive triples | \(\{3,3,3,4,4,4\}\) |
| Airplane + Singles | Airplane + k singles | \(\{3,3,3,4,4,4\} + \{5,6\}\) |
| Airplane + Pairs | Airplane + k pairs | \(\{3,3,3,4,4,4\} + \{5,5,6,6\}\) |
| Four + 2 Singles | Bomb + 2 singles | \(\{K,K,K,K\} + \{3,4\}\) |
| Four + 2 Pairs | Bomb + 2 pairs | \(\{K,K,K,K\} + \{3,3,4,4\}\) |

The move generator constructs each move as a 15-tuple \(\delta\) where \(\delta_i\) counts how many cards of rank \(i\) are consumed by that move. The resulting state after playing move \(m\) is \(s' = s - m\) (element-wise subtraction).

#### Lowest-Card Optimization

A critical pruning technique ensures DFS tractability without compromising optimality. Let \(i^* = \min\{i : s_i > 0\}\) be the lowest-indexed (i.e., lowest-ranked) card present in the hand. The move generator **filters** to retain only moves that use at least one card of rank \(i^*\):

\[
M^*(s) = \{m \in M(s) : m_{i^*} > 0\}
\]

**Lemma** (Optimality preservation). For any non-empty state \(s\), there exists an optimal play sequence whose first move belongs to \(M^*(s)\).

*Proof.* Suppose an optimal sequence begins with a move \(m \notin M^*(s)\), i.e., \(m\) does not use the lowest card \(i^*\). Let \(m'\) be the move in the same sequence that eventually plays \(i^*\). Since card ranks are independent (no move type requires skipping a lower rank to use a higher one), swapping \(m\) and \(m'\) produces another valid sequence with the same number of plays. Repeating this argument yields an optimal sequence where \(i^*\) is played first. \(\square\)

This lemma reduces branching dramatically. A 20-card hand with no pruning might generate hundreds of candidate first moves; with lowest-card filtering, the typical branching factor drops below 20.

#### Recursive Formulation

Let \(M^*(s)\) denote the filtered move set. The minimum plays to empty the hand is:

\[
\text{min\_plays}(s) = \begin{cases}
0 & \text{if } \sum_i s_i = 0 \\
\min_{m \in M^*(s)} \left(1 + \text{min\_plays}(s - m)\right) & \text{otherwise}
\end{cases}
\]

**Memoization**. The function is decorated with `@lru_cache(maxsize=None)`, which stores results for all visited states in a hash table. Since the state space is bounded by \(5^{13} \times 2^2 \approx 1.2 \times 10^9\) but the reachable subset from typical 20-card hands is much smaller (experimentally < 100,000 states per call), memoization ensures each state is evaluated at most once.

**Pseudocode** for the memoized DFS is shown in Algorithm 1.

```
Algorithm 1: min_plays(s)
Input:  State s (15-tuple of card counts)
Output: Minimum number of plays to empty s
1:  if sum(s) == 0 then
2:      return 0
3:  i* ← lowest index with s[i*] > 0
4:  moves ← ∅
5:  Generate all legal move tuples m from state s
6:  Filter moves to those with m[i*] > 0
7:  best ← +∞
8:  for each m in filtered moves do
9:      next ← s − m (element-wise)
10:     val ← 1 + min_plays(next)
11:     if val < best then
12:         best ← val
13: return best
```

#### Complexity Analysis

Let \(n = \sum_i s_i\) be the number of cards and \(b\) be the effective branching factor after lowest-card filtering.

- **Time complexity**: \(O(b \cdot S)\) where \(S\) is the number of distinct states reachable from \(s\). With memoization, each state is evaluated once; without memoization, the naive recursion would be \(O(b^n)\).
- **Space complexity**: \(O(S)\) for the memoization cache. In practice, \(S < 10^5\) for 20-card hands, yielding sub-second computation time on a single CPU core.
- **Experimental runtime**: A typical 20-card landlord hand computes `min_plays` in ~0.05 seconds; endgame states (< 6 cards) take < 1 ms.

#### Optimal Path Extraction

`get_optimal_path(state)` greedily constructs a concrete play sequence. At each state, it evaluates all filtered moves and selects the one that minimizes `min_plays` of the successor state:

\[
m^* = \arg\min_{m \in M^*(s)} \text{min\_plays}(s - m)
\]

The sequence is then \(m^*\) followed by `get_optimal_path(s - m^*)`. This greedy construction is optimal because the recursive cost function already captures the minimum remaining plays from every reachable state.

#### Move-Type Statistics for Scoring

Beyond raw play count, the DFS computes statistics about the *types* of moves in the optimal decomposition (singles, pairs, triples, straights, bombs, other). These are returned by `_min_plays_with_stats_impl`, which mirrors `min_plays` but additionally tracks the move-type histogram along the optimal path.

**Resolver score**. A composite heuristic score maps the play count and move-type distribution into a scalar quality metric:

\[
\text{score}(s) = \frac{2}{\text{plays}(s) + 1} \times \text{structure\_bonus}
\]

where `structure_bonus` penalizes hands with a high proportion of singletons (harder to empty against skilled opponents) and rewards hands with pairs and straights (more efficient combos):

- If singleton ratio > 50%: multiply by 0.6
- If singleton ratio > 30%: multiply by 0.8
- If pairs ≥ 2 or straights ≥ 1: multiply by 1.15

This score, rather than raw `min_plays`, is used by the agent's heuristic move selector to compare candidate actions.

#### Why DFS Over Dynamic Programming or A\*

We chose recursive DFS with memoization over bottom-up dynamic programming because:

1. **On-demand computation**: Only states reachable from the initial hand are evaluated, not the entire (intractable) state space.
2. **Natural move decomposition**: The recursive structure mirrors the sequential nature of card play.
3. **Simplicity**: No need for a priority queue (A\*) or iterative value iteration; the problem has no edge weights other than the uniform "+1 per play" cost.

A\* search with an admissible heuristic (e.g., cards remaining / max cards per move) could further reduce search time, but the memoized DFS is already fast enough for real-time play (< 100 ms per decision) and is simpler to implement correctly.

#### Strategic Enhancements

The raw DFS planner produces a sequence that minimizes the number of plays but ignores *when* those plays should be made relative to opponents. The agent wraps the planner with hardcoded heuristics:

- **Orphan card strategy**: In the endgame (fewer than 6 cards remaining), identify singleton cards that cannot be paired or tripled and prioritize playing them before they become stranded.
- **Weak-hand conservation**: When the hand contains strong cards (A, 2, Jokers), apply penalties to moves that waste these cards early, encouraging preservation for critical late-game plays.
- **Control vs. response branching**: When the agent has control (no pending opponent move), it follows the optimal DFS path with heuristic deviations. When responding, it must beat the opponent's last combo and uses constrained scoring that limits candidate moves to those of the same or higher type.

