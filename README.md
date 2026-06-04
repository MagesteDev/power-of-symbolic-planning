# The Underappreciated Power of Symbolic Planning in Imperfect-Information Games

## Abstract

DouDizhu ("Fighting the Landlord") is a challenging imperfect-information card game that has become a benchmark for modern artificial intelligence. Recent advances, particularly DouZero and its extensions, have achieved impressive performance through deep reinforcement learning with massive parameter counts—often exceeding one billion parameters and requiring days or weeks of GPU training. This paper asks a fundamental question: Is such massive parameterization truly necessary?

We answer in the negative. We present a purely symbolic DouDizhu agent that combines depth-first search (DFS) for shortest-path planning with handcrafted heuristics, including an orphan card strategy and weak-hand conservation mode. Without any learning, neural networks, or training data, our DFS symbolic agent achieves a 90.2% win rate against random opponents across 10,000 unseen games—just 5.5 percentage points behind DouZero's 95.7% win rate, despite using zero trainable parameters.

To contextualize this result, we also develop a lightweight neuro-symbolic hybrid (≈260K parameters) that uses a neural move-type classifier with a symbolic resolver. This agent reaches 89.1% win rate—marginally below our pure symbolic baseline. Genome augmentation provides no meaningful improvement at scale. Our findings demonstrate that careful symbolic design remains highly competitive in structured imperfect-information games, challenging the prevailing assumption that deep learning is always necessary for high performance.

**Keywords:** DouDizhu, symbolic AI, depth-first search, neuro-symbolic AI, imperfect-information games, reinforcement learning

A pure symbolic planning agent for DouDizhu ("Fighting the Landlord") that achieves 90%+ win rates against RandomAgent at zero parameter cost, rivaling deep reinforcement learning baselines.

## Overview

This project demonstrates that a carefully designed symbolic agent — combining depth-first search (DFS) shortest-path planning with domain-specific heuristics — can compete with billion-parameter deep RL systems on the DouDizhu card game benchmark.

## Key Results

| Agent | Win Rate (10,000 games) | Parameters |
|-------|------------------------|------------|
| DouZero | 95.7% | ~1.4B |
| **DFS (Symbolic)** | **90.2%** | **0** |
| NeuroSymbolic | 89.1% | ~286K |
| Genome-Enhanced NS | 88.8% | ~291K |

## Agent Architectures

- **DFS (Symbolic)**: Recursive memoized DFS (`min_plays`) computes the minimum number of plays to empty any hand, wrapped with orphan-card and weak-hand heuristics.
- **NeuroSymbolic**: A lightweight 3-layer FC network predicts the move type; a symbolic resolver selects the concrete move.
- **Genome-Enhanced NS**: Appends 10 evolved strategic genes as additional neural inputs.

## Repository Structure

- `resolver.py` — Core DFS shortest-path planner with memoization
- `resolver_agent.py` — DFS agent wrapper with heuristics
- `neuro_symbolic_agent.py` — Neuro-symbolic hybrid agent
- `genome_neuro_symbolic_agent.py` — Genome-enhanced variant
- `METHODS_SECTION.md` — Detailed methodology
- `RESULTS_SECTION.md` — Benchmark results
- `literature.md` — Related work survey with 37 papers

## Dependencies

See `requirements.txt`.

## Citation

If you use this code, please cite the original PerfectDou and DouZero papers for the game environment.
