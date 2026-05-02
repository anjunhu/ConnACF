#!/usr/bin/env python3
"""
Fix turn_*.json files with incorrect active_users_total and active_items_total values.

Updates:
- active_users_total: Correct total unique users in the subset
- active_items_total: Correct total unique items in the subset
- active_users_non_attacker: total - num_attacker_users
- active_items_non_attacker: total - num_attacker_items
- user_evolution_percentage: user_evolution_count / active_users_non_attacker * 100
- item_evolution_percentage: item_evolution_count / active_items_non_attacker * 100

Usage:
    python fix_turn_metrics.py --task_dir <path> --users 50 --items 133
"""

import os
import json
import argparse
import glob
from pathlib import Path


def fix_turn_file(filepath: str, correct_users: int, correct_items: int) -> bool:
    """
    Fix a single turn JSON file with correct user/item totals.
    
    Args:
        filepath: Path to the turn_*.json file
        correct_users: Correct total number of unique users
        correct_items: Correct total number of unique items
    
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        modified = False
        
        # Check if natural_semantic_evolution exists
        if 'natural_semantic_evolution' not in data:
            print(f"  Skipping {filepath}: no natural_semantic_evolution section")
            return False
        
        nse = data['natural_semantic_evolution']
        
        # Get current values
        old_users_total = nse.get('active_users_total', 0)
        old_items_total = nse.get('active_items_total', 0)
        num_attacker_users = nse.get('num_attacker_users', 0)
        num_attacker_items = nse.get('num_attacker_items', 0)
        user_evolution_count = nse.get('user_evolution_count', 0)
        item_evolution_count = nse.get('item_evolution_count', 0)
        
        # Calculate correct values
        correct_users_non_attacker = correct_users - num_attacker_users
        correct_items_non_attacker = correct_items - num_attacker_items
        
        # Calculate correct percentages (evolution count / non-attacker total * 100)
        correct_user_pct = (user_evolution_count / max(1, correct_users_non_attacker)) * 100
        correct_item_pct = (item_evolution_count / max(1, correct_items_non_attacker)) * 100
        
        # Check if updates needed
        if old_users_total != correct_users or old_items_total != correct_items:
            print(f"  Fixing {Path(filepath).name}:")
            print(f"    active_users_total: {old_users_total} -> {correct_users}")
            print(f"    active_items_total: {old_items_total} -> {correct_items}")
            print(f"    active_users_non_attacker: {nse.get('active_users_non_attacker', 'N/A')} -> {correct_users_non_attacker}")
            print(f"    active_items_non_attacker: {nse.get('active_items_non_attacker', 'N/A')} -> {correct_items_non_attacker}")
            print(f"    user_evolution_percentage: {nse.get('user_evolution_percentage', 'N/A'):.2f} -> {correct_user_pct:.2f}")
            print(f"    item_evolution_percentage: {nse.get('item_evolution_percentage', 'N/A'):.2f} -> {correct_item_pct:.2f}")
            
            # Update values
            nse['active_users_total'] = correct_users
            nse['active_items_total'] = correct_items
            nse['active_users_non_attacker'] = correct_users_non_attacker
            nse['active_items_non_attacker'] = correct_items_non_attacker
            nse['user_evolution_percentage'] = correct_user_pct
            nse['item_evolution_percentage'] = correct_item_pct
            
            modified = True
        
        if modified:
            # Write back
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        
        return False
        
    except Exception as e:
        print(f"  Error processing {filepath}: {e}")
        return False


def fix_task_directory(task_dir: str, correct_users: int, correct_items: int):
    """Fix all turn_*.json files in a task directory."""
    turn_files = sorted(glob.glob(os.path.join(task_dir, "turn_*.json")))
    
    if not turn_files:
        print(f"No turn_*.json files found in {task_dir}")
        return
    
    print(f"Found {len(turn_files)} turn files in {task_dir}")
    print(f"Correcting to: {correct_users} users, {correct_items} items")
    print()
    
    modified_count = 0
    for filepath in turn_files:
        if fix_turn_file(filepath, correct_users, correct_items):
            modified_count += 1
    
    print()
    print(f"Modified {modified_count}/{len(turn_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description='Fix turn_*.json files with incorrect active user/item totals'
    )
    parser.add_argument('--task_dir', '-t', type=str, required=True,
                       help='Path to task directory containing turn_*.json files')
    parser.add_argument('--users', '-u', type=int, required=True,
                       help='Correct total number of unique users')
    parser.add_argument('--items', '-i', type=int, required=True,
                       help='Correct total number of unique items')
    parser.add_argument('--all_tasks', action='store_true',
                       help='Process all task_* subdirectories')
    
    args = parser.parse_args()
    
    if args.all_tasks:
        # Find all task directories
        base_path = Path(args.task_dir)
        task_dirs = sorted(base_path.glob("task_*"))
        
        if not task_dirs:
            print(f"No task_* directories found in {args.task_dir}")
            return
        
        print(f"Processing {len(task_dirs)} task directories")
        print()
        
        for task_dir in task_dirs:
            print(f"=== {task_dir.name} ===")
            fix_task_directory(str(task_dir), args.users, args.items)
            print()
    else:
        fix_task_directory(args.task_dir, args.users, args.items)


if __name__ == '__main__':
    main()
