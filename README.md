# Power of Symbolic Planning: DFS-based DouDizhu Agent

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
