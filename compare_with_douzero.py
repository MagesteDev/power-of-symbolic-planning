"""
Compare Your NeuroSymbolic Model vs DouZero

Both models play as LANDLORD on the SAME 100 game seeds against random peasants.
DouZero is a strong baseline (used in PerfectDou paper for comparison).
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
from perfectdou.evaluation.deep_agent import DeepAgent

def load_eval_data(eval_data_path):
    """Load evaluation data (game seeds)"""
    print(f"Loading evaluation data from {eval_data_path}...")
    with open(eval_data_path, 'rb') as f:
        card_play_data_list = pickle.load(f)
    print(f"✓ Loaded {len(card_play_data_list)} game seeds\n")
    return card_play_data_list

def run_evaluation(model_name, landlord_agent, card_play_data_list, num_games=None):
    """Run evaluation with specified landlord agent vs random peasants."""
    print(f"{'='*80}")
    print(f"Testing: {model_name} (Landlord) vs Random Peasants")
    print(f"{'='*80}")
    
    # Create fresh agents for each evaluation to avoid state issues
    players = {
        'landlord': landlord_agent,
        'landlord_up': RandomAgent(),
        'landlord_down': RandomAgent()
    }
    
    # Create fresh environment
    env = GameEnv(players)
    
    games_to_run = num_games if num_games else len(card_play_data_list)
    games_to_run = min(games_to_run, len(card_play_data_list))
    
    skipped = 0
    game_outcomes = []
    
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
            
            winner = env.get_winner()
            landlord_won = (winner == 'landlord')
            game_outcomes.append({
                'game_idx': idx,
                'winner': winner,
                'landlord_won': landlord_won
            })
            
            env.reset()
        except Exception as e:
            skipped += 1
            game_outcomes.append({
                'game_idx': idx,
                'winner': 'SKIPPED',
                'landlord_won': False,
                'error': str(e)
            })
            env.reset()
            continue
    
    total_games = env.num_wins['landlord'] + env.num_wins['farmer']
    
    if total_games == 0:
        print(f"\n⚠️ WARNING: No games completed!")
        return None
    
    landlord_wp = env.num_wins['landlord'] / total_games * 100
    landlord_adp = env.num_scores['landlord'] / total_games
    
    results = {
        'model_name': model_name,
        'total_games': total_games,
        'skipped_games': skipped,
        'landlord_wins': env.num_wins['landlord'],
        'farmer_wins': env.num_wins['farmer'],
        'landlord_wp': landlord_wp,
        'landlord_adp': landlord_adp,
        'game_outcomes': game_outcomes
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"  Completed: {total_games}/{games_to_run} (skipped: {skipped})")
    print(f"  Landlord wins: {env.num_wins['landlord']} ({landlord_wp:.2f}%)")
    print(f"  Landlord ADP: {landlord_adp:.2f}")
    print(f"{'='*80}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare NeuroSymbolic vs DouZero')
    parser.add_argument('--eval-data', type=str, default='eval_data.pkl.pkl',
                       help='Path to evaluation data')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games (default: 100)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to your model (default: models/model.pth)')
    args = parser.parse_args()
    
    eval_data_path = Path(args.eval_data)
    if not eval_data_path.exists():
        print(f"❌ Error: {eval_data_path} not found")
        return
    
    card_play_data_list = load_eval_data(eval_data_path)
    
    all_results = []
    
    # Test 1: DouZero
    print("\n" + "="*80)
    print("MODEL 1: DouZero (Strong Baseline from PerfectDou Paper)")
    print("="*80 + "\n")
    
    # Create fresh agent
    douzero_path = "perfectdou/model/douzero/douzero_ADP/landlord.ckpt"
    douzero_agent = DeepAgent('landlord', douzero_path)
    
    results_douzero = run_evaluation(
        "DouZero",
        douzero_agent,
        card_play_data_list,
        args.num_games
    )
    if results_douzero:
        all_results.append(results_douzero)
    
    # IMPORTANT: Reload eval data to ensure fresh game states
    print("\n🔄 Reloading evaluation data for second model...\n")
    card_play_data_list = load_eval_data(eval_data_path)
    
    # Test 2: Your NeuroSymbolic
    print("\n" + "="*80)
    print("MODEL 2: Your NeuroSymbolic Model")
    print("="*80 + "\n")
    
    # Create fresh agent
    neurosymbolic_agent = NeuroSymbolicAgent('landlord', model_path=args.model_path)
    
    results_neuro = run_evaluation(
        "NeuroSymbolic (Yours)",
        neurosymbolic_agent,
        card_play_data_list,
        args.num_games
    )
    if results_neuro:
        all_results.append(results_neuro)
    
    # Comparison
    if len(all_results) >= 2:
        print("\n" + "="*80)
        print(f"HEAD-TO-HEAD: Same {args.num_games} Seeds, Both as Landlord")
        print("="*80)
        print(f"\n{'Model':<25} {'Completed':<12} {'Wins':<10} {'Win Rate':<12} {'ADP':<10}")
        print("-"*80)
        
        for res in all_results:
            print(f"{res['model_name']:<25} "
                  f"{res['total_games']:<12} "
                  f"{res['landlord_wins']:<10} "
                  f"{res['landlord_wp']:>6.2f}%      "
                  f"{res['landlord_adp']:>6.2f}")
        
        douzero_wins = all_results[0]['landlord_wins']
        neuro_wins = all_results[1]['landlord_wins']
        douzero_wp = all_results[0]['landlord_wp']
        neuro_wp = all_results[1]['landlord_wp']
        
        win_diff = neuro_wins - douzero_wins
        wp_diff = neuro_wp - douzero_wp
        
        print(f"\n{'='*80}")
        print(f"DIFFERENCE (Your Model - DouZero):")
        print(f"  Wins:      {'+' if win_diff >= 0 else ''}{win_diff}")
        print(f"  Win Rate:  {'+' if wp_diff >= 0 else ''}{wp_diff:.2f}%")
        
        if wp_diff > 0:
            print(f"\n🎉 Your model BEATS DouZero by {wp_diff:.2f}%!")
        elif wp_diff == 0:
            print(f"\n🤝 Your model TIES with DouZero!")
        else:
            print(f"\n📊 DouZero wins by {abs(wp_diff):.2f}% (still very competitive!)")
        
        print(f"{'='*80}")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"neurosymbolic_vs_douzero_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("NEUROSYMBOLIC VS DOUZERO COMPARISON\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Games: {args.num_games}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Model':<25} {'Completed':<12} {'Wins':<10} {'Win Rate':<12} {'ADP':<10}\n")
            f.write("-"*80 + "\n")
            
            for res in all_results:
                f.write(f"{res['model_name']:<25} "
                       f"{res['total_games']:<12} "
                       f"{res['landlord_wins']:<10} "
                       f"{res['landlord_wp']:>6.2f}%      "
                       f"{res['landlord_adp']:>6.2f}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write(f"DIFFERENCE (Your Model - DouZero):\n")
            f.write(f"  Wins:      {'+' if win_diff >= 0 else ''}{win_diff}\n")
            f.write(f"  Win Rate:  {'+' if wp_diff >= 0 else ''}{wp_diff:.2f}%\n")
            f.write(f"{'='*80}\n")
        
        print(f"\n✓ Report saved: {report_path}")

if __name__ == "__main__":
    main()
