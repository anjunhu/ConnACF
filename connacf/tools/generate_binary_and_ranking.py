#!/usr/bin/env python3
"""
Preprocess Training Data for Multi-Candidate Ranking

Generates ranking data using:
- Positives: From user's interaction history (train.inter), ranked by recency
- Negatives: From .random file (items user never interacted with)

Positives always get the majority (ceiling) of slots:
- num_candidates = 3: 2 positives + 1 negative
- num_candidates = 4: 2 positives + 2 negatives
- num_candidates = 5: 3 positives + 2 negatives

Within positives, rank 1 = most recent interaction, rank 2 = second most recent.

For num_candidates = 1 (binary):
- Positive: Item from history (label=True)
- Negative: Item from .random (label=False)

Output format:
- ranking_train_Ncand.json: {user_id: [[pos1, pos2, ..., neg1, ...], ...]}
  where pos1 is most preferred (most recent), negatives are least preferred
- binary_train_1cand.json: {user_id: [[item_id, True/False], ...]}
"""

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import argparse


def load_interaction_data(data_path: str, dataset_name: str) -> Dict[str, List[str]]:
    """Load training interaction data.
    
    Returns:
        Dict mapping user_id to list of item_ids in interaction order (oldest first)
    """
    train_file = os.path.join(data_path, f"{dataset_name}.train.inter")
    
    user_items = defaultdict(list)
    
    with open(train_file, 'r') as f:
        header = f.readline().strip().split('\t')
        print(f"Header: {header}")
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            user_id = parts[0]
            item_id = parts[2]  # The target item
            
            # Add to user's interaction list (preserving order - oldest first)
            if item_id not in user_items[user_id]:
                user_items[user_id].append(item_id)
    
    print(f"Loaded {len(user_items)} users from train.inter")
    for uid in list(user_items.keys())[:3]:
        print(f"  User {uid}: {len(user_items[uid])} items")
    
    return dict(user_items)


def load_random_negatives(data_path: str, dataset_name: str) -> Dict[str, List[str]]:
    """Load negative samples from .random file.
    
    Returns:
        Dict mapping user_id to list of negative item_ids
    """
    random_file = os.path.join(data_path, f"{dataset_name}.random")
    
    user_negatives = {}
    
    if not os.path.exists(random_file):
        print(f"WARNING: .random file not found: {random_file}")
        return {}
    
    with open(random_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            user_id = parts[0]
            negatives = parts[1].split()
            user_negatives[user_id] = negatives
    
    print(f"Loaded negatives for {len(user_negatives)} users from .random")
    for uid in list(user_negatives.keys())[:3]:
        print(f"  User {uid}: {len(user_negatives[uid])} negatives")
    
    return user_negatives


def generate_ranking_samples_with_negatives(
    user_items: Dict[str, List[str]],
    user_negatives: Dict[str, List[str]],
    num_candidates: int = 4,
    samples_per_user: int = 5
) -> Dict[str, List[List[str]]]:
    """Generate ranking samples using history for positives and .random for negatives.
    
    Candidate composition (positives get the majority for odd counts):
    - 3 candidates: 2 positives (history, recency-ranked) + 1 negative (.random)
    - 4 candidates: 2 positives + 2 negatives
    - 5 candidates: 3 positives + 2 negatives
    
    Positives are ranked by recency: rank 1 = most recent interaction,
    rank 2 = second most recent, etc. (more recent = closer to top of list)
    
    Args:
        user_items: Dict mapping user_id to ordered list of interacted items (oldest first)
        user_negatives: Dict mapping user_id to list of negative items
        num_candidates: Number of candidates (default 4)
        samples_per_user: Number of ranking samples to generate per user
    
    Returns:
        Dict mapping user_id to list of ranking samples
        Each sample is [most_preferred, ..., least_preferred]
        For positives: most recent interaction = most preferred (top of list)
    """
    # For odd num_candidates (e.g. 3): majority goes to positives.
    # 3 candidates → 2 positives (most recent history items) + 1 negative
    # 4 candidates → 2 positives + 2 negatives
    # 5 candidates → 3 positives + 2 negatives
    # This ensures the ranking always has more signal from real preferences.
    num_positives = (num_candidates + 1) // 2  # Ceiling division: majority to positives
    num_negatives = num_candidates - num_positives
    
    print(f"\nGenerating {num_candidates}-candidate rankings:")
    print(f"  Positives (top {num_positives}): From history, ranked by recency")
    print(f"  Negatives (bottom {num_negatives}): From .random file")
    
    ranking_data = {}
    skipped_no_history = 0
    skipped_no_negatives = 0
    
    for user_id, items in user_items.items():
        # Need at least num_positives items in history
        if len(items) < num_positives:
            skipped_no_history += 1
            continue
        
        # Need negatives from .random
        negatives = user_negatives.get(user_id, [])
        if len(negatives) < num_negatives:
            skipped_no_negatives += 1
            continue
        
        user_samples = []
        
        for _ in range(samples_per_user):
            # Select positives: most recent items from history
            # items list is oldest-first, so we take from the end
            recent_items = items[-min(len(items), num_positives * 3):]  # Pool of recent items
            if len(recent_items) >= num_positives:
                selected_positives = random.sample(recent_items, num_positives)
            else:
                selected_positives = random.sample(items, num_positives)
            
            # Rank positives by recency (most recent = rank 1)
            # Higher index in items = more recent = higher preference
            pos_with_recency = [(item, items.index(item)) for item in selected_positives]
            pos_with_recency.sort(key=lambda x: x[1], reverse=True)  # Most recent first
            ranked_positives = [item for item, _ in pos_with_recency]
            
            # Select negatives: random items from .random
            selected_negatives = random.sample(negatives, num_negatives)
            
            # Combine: positives first (ranked), then negatives (unranked)
            ranking = ranked_positives + selected_negatives
            user_samples.append(ranking)
        
        ranking_data[user_id] = user_samples
    
    print(f"\nGenerated rankings for {len(ranking_data)} users")
    print(f"  Skipped (insufficient history): {skipped_no_history}")
    print(f"  Skipped (insufficient negatives): {skipped_no_negatives}")
    
    return ranking_data


def generate_binary_samples_with_negatives(
    user_items: Dict[str, List[str]],
    user_negatives: Dict[str, List[str]],
    samples_per_user: int = 5
) -> Dict[str, List[Tuple[str, bool]]]:
    """Generate binary (yes/no) samples using history and .random.
    
    For each user:
    - Positive samples: Items from history (label=True)
    - Negative samples: Items from .random (label=False)
    
    Returns:
        Dict mapping user_id to list of (item_id, ground_truth_label) tuples
    """
    print(f"\nGenerating binary (1-candidate) samples:")
    print(f"  Positives: From history")
    print(f"  Negatives: From .random file")
    
    binary_data = {}
    
    for user_id, items in user_items.items():
        negatives = user_negatives.get(user_id, [])
        
        if not items or not negatives:
            continue
        
        samples = []
        
        # Add positive samples (items user liked)
        num_pos = min(len(items), samples_per_user // 2 + 1)
        pos_samples = random.sample(items, num_pos) if len(items) > num_pos else items
        for item in pos_samples:
            samples.append((item, True))
        
        # Add negative samples (items from .random)
        num_neg = min(len(negatives), samples_per_user - len(samples))
        if num_neg > 0:
            neg_samples = random.sample(negatives, num_neg)
            for item in neg_samples:
                samples.append((item, False))
        
        # Shuffle so positives and negatives are mixed
        random.shuffle(samples)
        
        binary_data[user_id] = samples
    
    print(f"Generated binary samples for {len(binary_data)} users")
    
    return binary_data


def save_ranking_data(
    ranking_data: Dict[str, List[List[str]]],
    output_path: str,
    num_candidates: int
):
    """Save ranking data to JSON file."""
    output_file = os.path.join(output_path, f"ranking_train_{num_candidates}cand.json")
    
    num_positives = (num_candidates + 1) // 2
    num_negatives = num_candidates - num_positives
    
    with open(output_file, 'w') as f:
        json.dump({
            'num_candidates': num_candidates,
            'description': (
                f'Ground truth rankings for {num_candidates}-candidate tasks. '
                f'Top {num_positives} are positives (from history, ranked by recency: more recent = closer to top), '
                f'bottom {num_negatives} are negatives (from .random).'
            ),
            'format': 'user_id -> list of rankings, each ranking is [most_preferred, ..., least_preferred]',
            'data': ranking_data
        }, f, indent=2)
    
    print(f"\nSaved ranking data to {output_file}")
    print(f"  Users: {len(ranking_data)}")
    print(f"  Total samples: {sum(len(v) for v in ranking_data.values())}")


def save_binary_data(
    binary_data: Dict[str, List[Tuple[str, bool]]],
    output_path: str
):
    """Save binary data to JSON file."""
    output_file = os.path.join(output_path, "binary_train_1cand.json")
    
    # Convert tuples to lists for JSON
    serializable = {
        uid: [[item, label] for item, label in samples]
        for uid, samples in binary_data.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump({
            'num_candidates': 1,
            'description': 'Binary yes/no samples. Positives from history, negatives from .random.',
            'format': 'user_id -> list of [item_id, ground_truth_label]',
            'data': serializable
        }, f, indent=2)
    
    print(f"\nSaved binary data to {output_file}")
    print(f"  Users: {len(binary_data)}")
    print(f"  Total samples: {sum(len(v) for v in binary_data.values())}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess training data for ranking using history + .random negatives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for CDs-100-user-dense
  python tools/preprocess_ranking_data.py \\
      --data_path dataset/CDs-100-user-dense \\
      --dataset_name CDs-100-user-dense

  # Generate for mini dataset
  python tools/preprocess_ranking_data.py \\
      --data_path dataset/CDs-100-user-mini \\
      --dataset_name CDs-100-user-mini
        """
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (e.g., CDs-100-user-dense)')
    parser.add_argument('--num_candidates', type=int, nargs='+', default=[1, 4],
                        help='Number of candidates to generate data for (default: 1 4)')
    parser.add_argument('--samples_per_user', type=int, default=5,
                        help='Number of samples per user (default: 5)')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Preprocessing ranking data")
    print(f"  Data path: {args.data_path}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Candidates: {args.num_candidates}")
    print(f"  Samples per user: {args.samples_per_user}")
    print(f"{'='*60}")
    
    # Load interaction data (positives)
    user_items = load_interaction_data(args.data_path, args.dataset_name)
    
    # Load negative samples from .random
    user_negatives = load_random_negatives(args.data_path, args.dataset_name)
    
    if not user_negatives:
        print("\nERROR: No negatives loaded. Cannot generate ranking data.")
        return
    
    # Generate data for each num_candidates
    for nc in args.num_candidates:
        print(f"\n{'='*60}")
        print(f"Generating data for num_candidates={nc}")
        print(f"{'='*60}")
        
        if nc == 1:
            binary_data = generate_binary_samples_with_negatives(
                user_items, user_negatives, args.samples_per_user
            )
            save_binary_data(binary_data, args.data_path)
        elif nc >= 3:
            ranking_data = generate_ranking_samples_with_negatives(
                user_items, user_negatives, nc, args.samples_per_user
            )
            save_ranking_data(ranking_data, args.data_path, nc)
        else:
            print(f"  Skipping nc={nc} (use original BPR data for pairwise)")


if __name__ == '__main__':
    main()
