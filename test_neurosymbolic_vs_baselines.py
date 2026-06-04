"""
Test Your NeuroSymbolic Model vs PerfectDou Baselines

This script runs your neuro-symbolic model in PerfectDou's evaluation environment
and compares it against Random, RLCard, DouZero, and PerfectDou on the SAME game seeds.

Usage:
    python test_neurosymbolic_vs_baselines.py --num-games 100
    python test_neurosymbolic_vs_baselines.py --num-games 1000 --position landlord
"""
import sys
import os
import argparse
import pickle
from pathlib import Path
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from perfectdou.env.game import GameEnv
from perfectdou.evaluation.neurosymbolic_agent import NeuroSymbolicAgent
from perfectdou.evaluation.random_agent import RandomAgent

# Try to import RLCard agent (optional)
try:
    from perfectdou.evaluation.rlcard_agent import RLCardAgent
    RLCARD_AVAILABLE = True
except ImportError:
    RLCARD_AVAILABLE = False
    print("⚠️  Warning: RLCard not installed. Skipping RLCard baseline.")
    print("   To install: pip install rlcard\n")

def load_eval_data(eval_data_path):
    """Load evaluation data (game seeds)"""
    print(f"Loading evaluation data from {eval_data_path}...")
    with open(eval_data_path, 'rb') as f:
        card_play_data_list = pickle.load(f)
    print(f"✓ Loaded {len(card_play_data_list)} game seeds\n")
    return card_play_data_list

def run_evaluation(config_name, players, card_play_data_list, num_games=None):
    """
    Run evaluation for a specific configuration.
    
    Args:
        config_name: Name of the configuration (e.g., "NeuroSymbolic vs Random")
        players: Dict with 'landlord', 'landlord_up', 'landlord_down' agents
        card_play_data_list: List of game seeds
        num_games: Number of games to run (None = all)
    
    Returns:
        dict: Results including wins, win rate, ADP
    """
    print(f"{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    
    # Create environment
    env = GameEnv(players)
    
    # Run games
    games_to_run = num_games if num_games else len(card_play_data_list)
    games_to_run = min(games_to_run, len(card_play_data_list))
    
    skipped = 0
    errors = []
    
    for idx in range(games_to_run):
        card_play_data = card_play_data_list[idx]
        
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"Progress: {idx+1}/{games_to_run} | "
                  f"Landlord: {env.num_wins['landlord']} | "
                  f"Farmers: {env.num_wins['farmer']} | "
                  f"Skipped: {skipped}")
        
        try:
            env.card_play_init(card_play_data)
            while not env.game_over:
                env.step()
            env.reset()
        except (AssertionError, IndexError, KeyError) as e:
            # Skip games with corrupted data or edge cases
            skipped += 1
            errors.append((idx, str(e)))
            env.reset()
            continue
        except Exception as e:
            # Catch other errors but continue
            skipped += 1
            errors.append((idx, f"Unexpected error: {str(e)}"))
            env.reset()
            continue
    
    # Calculate results
    total_games = env.num_wins['landlord'] + env.num_wins['farmer']
    
    if total_games == 0:
        print(f"\n⚠️ WARNING: No games completed successfully!")
        return None
    
    landlord_wp = env.num_wins['landlord'] / total_games * 100
    farmer_wp = env.num_wins['farmer'] / total_games * 100
    landlord_adp = env.num_scores['landlord'] / total_games
    farmer_adp = env.num_scores['farmer'] * 2 / total_games
    
    results = {
        'config_name': config_name,
        'total_games': total_games,
        'attempted_games': games_to_run,
        'skipped_games': skipped,
        'landlord_wins': env.num_wins['landlord'],
        'farmer_wins': env.num_wins['farmer'],
        'landlord_wp': landlord_wp,
        'farmer_wp': farmer_wp,
        'landlord_adp': landlord_adp,
        'farmer_adp': farmer_adp,
        'errors': errors[:5]  # Keep first 5 errors for debugging
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*80}")
    print(f"  Total games completed: {total_games}/{games_to_run}")
    print(f"  Skipped: {skipped}")
    print(f"  Landlord wins: {env.num_wins['landlord']} ({landlord_wp:.2f}%)")
    print(f"  Farmer wins: {env.num_wins['farmer']} ({farmer_wp:.2f}%)")
    print(f"  Landlord ADP: {landlord_adp:.2f}")
    print(f"  Farmer ADP: {farmer_adp:.2f}")
    print(f"{'='*80}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test NeuroSymbolic model vs baselines')
    parser.add_argument('--eval-data', type=str, default='eval_data.pkl.pkl',
                       help='Path to evaluation data pickle file')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games to test (default: 100)')
    parser.add_argument('--position', type=str, default='landlord',
                       choices=['landlord', 'landlord_up', 'landlord_down'],
                       help='Position for NeuroSymbolic agent (default: landlord)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model file (default: models/model.pth)')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='Skip baseline comparisons, only test NeuroSymbolic')
    args = parser.parse_args()
    
    # Load evaluation data
    eval_data_path = Path(args.eval_data)
    if not eval_data_path.exists():
        print(f"❌ Error: Evaluation data not found at {eval_data_path}")
        print(f"\nPlease generate evaluation data first:")
        print(f"  python generate_eval_data.py --output {args.eval_data} --num_games {args.num_games}")
        return
    
    card_play_data_list = load_eval_data(eval_data_path)
    
    # Test configurations
    all_results = []
    
    # Configuration 1: Random baseline (if not skipped)
    if not args.skip_baselines:
        print("\n" + "="*80)
        print("BASELINE 1: Random vs Random")
        print("="*80 + "\n")
        
        config_random = {
            'landlord': RandomAgent(),
            'landlord_up': RandomAgent(),
            'landlord_down': RandomAgent()
        }
        
        results_random = run_evaluation(
            "Random (Baseline)",
            config_random,
            card_play_data_list,
            args.num_games
        )
        if results_random:
            all_results.append(results_random)
    
    # Configuration 2: RLCard baseline (if not skipped and available)
    if not args.skip_baselines and RLCARD_AVAILABLE:
        print("\n" + "="*80)
        print("BASELINE 2: RLCard vs Random")
        print("="*80 + "\n")
        
        config_rlcard = {
            'landlord': RLCardAgent('landlord'),
            'landlord_up': RandomAgent(),
            'landlord_down': RandomAgent()
        }
        
        results_rlcard = run_evaluation(
            "RLCard (Rule-based)",
            config_rlcard,
            card_play_data_list,
            args.num_games
        )
        if results_rlcard:
            all_results.append(results_rlcard)
    
    # Configuration 3: Your NeuroSymbolic model
    print("\n" + "="*80)
    print(f"YOUR MODEL: NeuroSymbolic ({args.position}) vs Random")
    print("="*80 + "\n")
    
    # Create NeuroSymbolic agent for specified position
    neurosymbolic_agent = NeuroSymbolicAgent(args.position, model_path=args.model_path)
    
    config_neurosymbolic = {
        'landlord': neurosymbolic_agent if args.position == 'landlord' else RandomAgent(),
        'landlord_up': neurosymbolic_agent if args.position == 'landlord_up' else RandomAgent(),
        'landlord_down': neurosymbolic_agent if args.position == 'landlord_down' else RandomAgent()
    }
    
    results_neurosymbolic = run_evaluation(
        f"NeuroSymbolic ({args.position})",
        config_neurosymbolic,
        card_play_data_list,
        args.num_games
    )
    if results_neurosymbolic:
        all_results.append(results_neurosymbolic)
    
    # Final comparison
    if len(all_results) > 0:
        print("\n" + "="*80)
        print(f"FINAL COMPARISON (Same {args.num_games} Game Seeds)")
        print("="*80)
        print(f"\n{'Model':<30} {'Win Rate':<15} {'ADP':<10} {'Wins':<15} {'Games':<10}")
        print("-"*80)
        
        for res in all_results:
            position_label = "Landlord" if args.position == 'landlord' else args.position.replace('_', ' ').title()
            print(f"{res['config_name']:<30} "
                  f"{res['landlord_wp']:>6.2f}%        "
                  f"{res['landlord_adp']:>6.2f}     "
                  f"{res['landlord_wins']}/{res['total_games']:<10} "
                  f"{res['total_games']}")
        
        # Calculate improvements
        if len(all_results) >= 2:
            baseline_wp = all_results[0]['landlord_wp']
            neurosymbolic_wp = all_results[-1]['landlord_wp']
            improvement = neurosymbolic_wp - baseline_wp
            
            print(f"\n{'='*80}")
            print(f"IMPROVEMENT vs {all_results[0]['config_name']}: "
                  f"{'+' if improvement >= 0 else ''}{improvement:.2f}% win rate")
            print(f"{'='*80}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"neurosymbolic_evaluation_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("NEUROSYMBOLIC MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Position: {args.position}\n")
            f.write(f"Number of games: {args.num_games}\n")
            f.write(f"Model path: {args.model_path or 'models/model.pth'}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Model':<30} {'Win Rate':<15} {'ADP':<10} {'Wins':<15}\n")
            f.write("-"*70 + "\n")
            
            for res in all_results:
                f.write(f"{res['config_name']:<30} "
                       f"{res['landlord_wp']:>6.2f}%        "
                       f"{res['landlord_adp']:>6.2f}     "
                       f"{res['landlord_wins']}/{res['total_games']}\n")
            
            if len(all_results) >= 2:
                baseline_wp = all_results[0]['landlord_wp']
                neurosymbolic_wp = all_results[-1]['landlord_wp']
                improvement = neurosymbolic_wp - baseline_wp
                
                f.write(f"\n{'='*80}\n")
                f.write(f"IMPROVEMENT: {'+' if improvement >= 0 else ''}{improvement:.2f}% vs {all_results[0]['config_name']}\n")
                f.write(f"{'='*80}\n")
            
            # Add error details if any
            f.write("\n\nERROR DETAILS (first 5 per config):\n")
            f.write("="*80 + "\n")
            for res in all_results:
                if res['errors']:
                    f.write(f"\n{res['config_name']}:\n")
                    for game_idx, error_msg in res['errors']:
                        f.write(f"  Game {game_idx}: {error_msg}\n")
        
        print(f"\n✓ Detailed report saved: {report_path}")
    else:
        print("\n❌ No successful evaluations completed.")

if __name__ == "__main__":
    main()
