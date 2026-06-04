# NeuroSymbolic Model Integration with PerfectDou

This guide explains how to run your neuro-symbolic model in PerfectDou's evaluation environment to compare against baselines on the same game seeds.

## Overview

Your neuro-symbolic model has been integrated into PerfectDou's RLCard-based evaluation framework through an adapter layer (`perfectdou/evaluation/neurosymbolic_agent.py`). This allows fair comparison on identical game seeds.

## Architecture Differences

### Your Model (45 features)
- **Input**: 45-dimensional feature vector
  - Game state (10): cards, control, turns, phase
  - Move features (8): combo type, size
  - Card values (5): max/min/avg, jokers, 2s
  - Strategy (3): best strategy, junk cards
  - Force-take (2): should force, critical
  - Bluff (2): bluff score, should bluff
  - Last combo (15): opponent's move analysis

- **Architecture**: 
  ```
  Input(45) → Linear(256) + BN + Dropout(0.3)
           → Linear(128) + BN + Dropout(0.3)
           → Linear(64) + BN + Dropout(0.2)
           → Linear(1) [move value]
  ```

### PerfectDou Model
- Uses perfect information distillation
- Different feature representation
- Pre-trained on massive dataset

## Quick Start

### 1. Generate Evaluation Data (if not exists)

```bash
cd perfectdou
python generate_eval_data.py --output eval_data.pkl.pkl --num_games 100
```

This creates 100 random game seeds that all models will play.

### 2. Test Your Model vs Baselines

```bash
# Test on 100 games (landlord position)
python test_neurosymbolic_vs_baselines.py --num-games 100

# Test on 1000 games
python test_neurosymbolic_vs_baselines.py --num-games 1000

# Test as peasant_up position
python test_neurosymbolic_vs_baselines.py --num-games 100 --position landlord_up

# Use specific model file
python test_neurosymbolic_vs_baselines.py --num-games 100 --model-path ../models/single_seed_0_epochs_50_shortest_20260601_171926.pth

# Skip baselines (faster, only test your model)
python test_neurosymbolic_vs_baselines.py --num-games 100 --skip-baselines
```

### 3. Compare Against PerfectDou (if available)

```bash
# Using PerfectDou's evaluate.py
python evaluate.py --landlord neurosymbolic --landlord_up random --landlord_down random --eval_data eval_data.pkl.pkl --num_workers 1
```

## Results Interpretation

The test script outputs:

1. **Win Rate (WP)**: Percentage of games won
2. **Average Difference Points (ADP)**: Average score difference
3. **Comparison**: Improvement vs baseline

Example output:
```
Model                          Win Rate        ADP        Wins            Games
--------------------------------------------------------------------------------
Random (Baseline)              33.45%          -0.34      33/99           99
RLCard (Rule-based)            45.67%          0.12       45/99           99
NeuroSymbolic (landlord)       52.53%          0.45       52/99           99

================================================================================
IMPROVEMENT vs Random (Baseline): +19.08% win rate
================================================================================
```

## Adapter Details

The adapter (`neurosymbolic_agent.py`) handles:

1. **Card Format Conversion**
   - PerfectDou: `3-14` (regular), `17` (2), `20` (small joker), `30` (big joker)
   - Your engine: `Card(Rank, Suit)` objects

2. **Game State Conversion**
   - Creates `PerfectDouGameStateAdapter` that mimics your `GameState`
   - Provides opponent card counts
   - Tracks turn numbers

3. **Move Validation**
   - Ensures chosen moves are legal in PerfectDou's environment
   - Falls back to legal move if model chooses illegal action

## Model Files

Your trained models are in `../models/`:
- `model.pth` - Latest/default model
- `single_seed_*.pth` - Single-seed trained models
- `curriculum_handoff_model.pth` - Curriculum learning model

## Troubleshooting

### Model Not Found
```
Warning: Model not found at models/model.pth
```
**Solution**: Specify model path explicitly:
```bash
python test_neurosymbolic_vs_baselines.py --model-path ../models/your_model.pth
```

### Import Errors
```
ModuleNotFoundError: No module named 'doudizhu_engine'
```
**Solution**: The adapter automatically adds paths. If issues persist, check:
- `doudizhu_engine/` exists in parent directory
- `neuro_symbolic/` exists in `../new_PROJECT/`

### Illegal Move Warnings
```
Warning: NeuroSymbolic chose illegal move [...], using [...]
```
**Cause**: Feature extraction or model prediction issue
**Impact**: Falls back to legal move (may reduce performance)
**Solution**: Check feature extraction in `feature_extractor.py`

### Game Skipped
Some games may be skipped due to:
- Corrupted evaluation data
- Edge cases in card dealing
- This is normal, report shows skipped count

## Performance Expectations

Based on your training:
- **Single-seed models**: 100% win rate on trained seed
- **Multi-seed models**: Generalization to new seeds
- **vs Random**: Should significantly outperform (>40% WR expected)
- **vs RLCard**: Competitive performance
- **vs PerfectDou**: Challenging (PerfectDou is SOTA)

## Next Steps

1. **Baseline Comparison**: Run 100-1000 games vs Random/RLCard
2. **Analyze Results**: Check which game phases your model excels/struggles
3. **Feature Analysis**: Use path evaluations to understand decisions
4. **Model Iteration**: Train on more diverse seeds if needed
5. **PerfectDou Comparison**: Ultimate benchmark (if model available)

## File Structure

```
perfectdou/
├── NEUROSYMBOLIC_INTEGRATION.md          # This file
├── test_neurosymbolic_vs_baselines.py    # Main test script
├── perfectdou/
│   └── evaluation/
│       ├── neurosymbolic_agent.py        # Your model adapter
│       ├── random_agent.py               # Random baseline
│       ├── rlcard_agent.py               # Rule-based baseline
│       └── perfectdou_agent.py           # PerfectDou model
└── eval_data.pkl.pkl                     # Game seeds

../new_PROJECT/
├── models/
│   └── model.pth                         # Your trained model
├── neuro_symbolic/
│   ├── neuro_symbolic_player.py          # Your player class
│   ├── feature_extractor.py              # Feature extraction
│   └── neural_network.py                 # Network architecture
└── doudizhu_engine/                      # Your game engine
```

## Citation

If you use this integration in your research:

```bibtex
@inproceedings{yang2022perfectdou,
  title={PerfectDou: Dominating DouDizhu with Perfect Information Distillation},
  author={Yang, Guan and Liu, Minghuan and Hong, Weijun and Zhang, Weinan and Fang, Fei and Zeng, Guangjun and Lin, Yue},
  booktitle={NeurIPS},
  year={2022}
}
```
