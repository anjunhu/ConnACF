#!/usr/bin/env python3
"""
Calculate GCS (Generalization Consistency Score) for TOMA experiments.

GCS measures how consistently the attack performs across different configurations.
Higher GCS = more robust attack.

Usage:
    python connacf/tools/calculate_gcs.py --exp_dirs \
        attack_output/toma/toma_2cand/ml-100k-20-user-dense/260224121317 \
        attack_output/toma/toma_2cand/ml-100k-100user-dense/260224143022 \
        attack_output/toma/toma_2cand/CDs-100user-dense/260224165511
"""

import argparse
import json
import glob
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_final_ccr(exp_dir: str) -> Optional[Tuple[str, float]]:
    """
    Load final CCR from experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Tuple of (config_name, final_ccr) or None if failed
    """
    exp_path = Path(exp_dir)
    
    # Get config name from path
    # e.g., attack_output/toma/toma_2cand/ml-100k-20-user-dense/260224121317
    # → ml-100k-20-user-dense
    config_name = exp_path.parent.name
    
    # Find all turn files
    turn_files = sorted(glob.glob(str(exp_path / 'turn_*.json')))
    if not turn_files:
        print(f"⚠️  Warning: No turn files found in {exp_dir}")
        return None
    
    # Load final turn
    try:
        with open(turn_files[-1]) as f:
            data = json.load(f)
            ccr = data.get('reverse_engineering', {}).get('toma', {}).get('ccr', None)
            
            if ccr is None:
                print(f"⚠️  Warning: No CCR found in {turn_files[-1]}")
                return None
            
            return (config_name, float(ccr))
    except Exception as e:
        print(f"❌ Error loading {turn_files[-1]}: {e}")
        return None


def calculate_gcs(experiment_dirs: List[str], verbose: bool = True) -> Optional[float]:
    """
    Calculate GCS from multiple TOMA experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
        verbose: Print detailed output
        
    Returns:
        GCS value (0-1) or None if insufficient data
    """
    results = []
    
    if verbose:
        print("=" * 60)
        print("GCS (Generalization Consistency Score) Calculation")
        print("=" * 60)
        print()
    
    # Load CCR from each experiment
    for exp_dir in experiment_dirs:
        result = load_final_ccr(exp_dir)
        if result:
            config_name, ccr = result
            results.append((config_name, ccr))
            if verbose:
                print(f"✅ {config_name:40s} CCR = {ccr:.4f}")
    
    if len(results) < 2:
        print()
        print("❌ Error: Need at least 2 experiments to calculate GCS")
        print(f"   Found: {len(results)} valid experiments")
        return None
    
    # Extract CCR values
    config_names = [name for name, _ in results]
    final_ccrs = np.array([ccr for _, ccr in results])
    
    # Calculate statistics
    mean_ccr = np.mean(final_ccrs)
    std_ccr = np.std(final_ccrs, ddof=1)  # Sample std
    min_ccr = np.min(final_ccrs)
    max_ccr = np.max(final_ccrs)
    
    # Calculate Coefficient of Variation
    cv = std_ccr / mean_ccr if mean_ccr > 0 else float('inf')
    
    # Calculate GCS (inverse of CV, bounded to 0-1)
    gcs = 1 - cv if cv < 1 else 0.0
    
    if verbose:
        print()
        print("=" * 60)
        print("Results:")
        print("-" * 60)
        print(f"  Experiments:        {len(results)}")
        print(f"  Mean CCR:           {mean_ccr:.4f}")
        print(f"  Std CCR:            {std_ccr:.4f}")
        print(f"  Min CCR:            {min_ccr:.4f}")
        print(f"  Max CCR:            {max_ccr:.4f}")
        print(f"  Range:              {max_ccr - min_ccr:.4f}")
        print(f"  CV (Coef. Var.):    {cv:.4f} ({cv*100:.2f}%)")
        print()
        print(f"  GCS:                {gcs:.4f}")
        print()
        
        # Interpretation
        if gcs >= 0.95:
            interpretation = "Excellent - Attack is highly robust"
        elif gcs >= 0.85:
            interpretation = "Good - Attack is generally robust"
        elif gcs >= 0.70:
            interpretation = "Moderate - Some config dependency"
        elif gcs >= 0.50:
            interpretation = "Poor - Attack is config-dependent"
        else:
            interpretation = "Very Poor - Attack is brittle"
        
        print(f"  Interpretation:     {interpretation}")
        print("=" * 60)
    
    return gcs


def main():
    parser = argparse.ArgumentParser(
        description='Calculate GCS (Generalization Consistency Score) for TOMA experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate GCS from 3 experiments
  python connacf/tools/calculate_gcs.py --exp_dirs \\
    attack_output/toma/toma_2cand/ml-100k-20-user-dense/260224121317 \\
    attack_output/toma/toma_2cand/ml-100k-100user-dense/260224143022 \\
    attack_output/toma/toma_2cand/CDs-100user-dense/260224165511

  # Quiet mode (only print GCS)
  python connacf/tools/calculate_gcs.py --exp_dirs exp1 exp2 exp3 --quiet

Interpretation:
  GCS > 0.95: Excellent consistency (robust attack)
  GCS > 0.85: Good consistency
  GCS > 0.70: Moderate consistency
  GCS < 0.70: Poor consistency (brittle attack)
        """
    )
    
    parser.add_argument(
        '--exp_dirs',
        nargs='+',
        required=True,
        help='Paths to experiment directories (need at least 2)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print GCS value (for scripting)'
    )
    
    args = parser.parse_args()
    
    # Calculate GCS
    gcs = calculate_gcs(args.exp_dirs, verbose=not args.quiet)
    
    if gcs is None:
        exit(1)
    
    if args.quiet:
        print(f"{gcs:.4f}")
    
    exit(0)


if __name__ == '__main__':
    main()
