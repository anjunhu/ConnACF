#!/usr/bin/env python3
"""
Regenerate ml-100k datasets with controlled sparsity.

Dataset names follow the pattern: {source}-{N}user-{label}{items}[-seed{S}]
  e.g. ml-100k-100user-medium100, ml-100k-20user-sparse500-seed7

Usage:
  python generate_ml100k_data_sparse_dense.py --num-users 100 --target-items 100 --label medium
  python generate_ml100k_data_sparse_dense.py --num-users 100 --target-items 500 --label sparse
  python generate_ml100k_data_sparse_dense.py --num-users 20  --target-items 30  --label dense
  python generate_ml100k_data_sparse_dense.py --source ml-1m --num-users 100 --target-items 100 --label medium --seed 7
"""

import os
import json
import random
import shutil
import urllib.request
import zipfile
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

# Get the directory where this script lives, then derive paths relative to it
SCRIPT_DIR = Path(__file__).resolve().parent
CONNACF_DIR = SCRIPT_DIR.parent  # connacf/

CONNACF_DATASET_DIR = CONNACF_DIR / "dataset"
CONNACF_PROPS_DIR = CONNACF_DIR / "props"
RAW_DIR = CONNACF_DIR.parent / "dataset" / "ml-100k-raw"
MIN_INTERACTIONS = 3
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))

# Known full-dataset stats (users / rated-items / ratings)
# ml-10m and ml-20m: items = items that appear in ratings, not total catalog
DATASET_STATS = {
    "ml-100k": {"users": 943,    "items": 1682,  "ratings": 100000},
    "ml-1m":   {"users": 6040,   "items": 3706,  "ratings": 1000209},
    "ml-10m":  {"users": 69878,  "items": 10677, "ratings": 10000054},
    "ml-20m":  {"users": 138493, "items": 26744, "ratings": 20000263},
}

# Zip paths for all sources (ml-100k is downloaded separately to RAW_DIR)
ML_ZIP = {k: CONNACF_DATASET_DIR / f"{k}.zip" for k in ("ml-1m", "ml-10m", "ml-20m")}

TARGET_INTERACTIONS_PER_USER = 10


def download_ml100k():
    """Download MovieLens 100k dataset if not present."""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = RAW_DIR / "ml-100k.zip"
    if (RAW_DIR / "ml-100k" / "u.data").exists():
        print("Raw ML-100k data already exists, skipping download.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens 100k...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DIR)
    os.remove(zip_path)
    print("Download complete!")


def verify_dataset_stats(source, ratings, movies, users):
    rated_items = len(set(i for _, i, _, _ in ratings))
    expected = DATASET_STATS.get(source, {})
    print(f"\n  Verifying {source} stats:")
    for name, got, want in [
        ("users",       len(users),   expected.get("users")),
        ("rated items", rated_items,  expected.get("items")),
        ("ratings",     len(ratings), expected.get("ratings")),
    ]:
        status = "✓" if got == want else f"✗ (expected {want})"
        print(f"    {name}: {got} {status}")


def load_raw_data(source):
    """Load any supported MovieLens source. Returns (ratings, movies, users)."""
    if source == "ml-100k":
        return _load_ml100k()
    zpath = ML_ZIP.get(source)
    if zpath is None or not zpath.exists():
        raise FileNotFoundError(
            f"{source}.zip not found at {zpath}. "
            f"Download from https://files.grouplens.org/datasets/movielens/{source}.zip "
            f"and place it in connacf/dataset/"
        )
    if source == "ml-1m":
        return _load_dat_source(zpath, "ml-1m", sep="::", has_users=True)
    if source == "ml-10m":
        return _load_dat_source(zpath, "ml-10M100K", sep="::", has_users=False)
    if source == "ml-20m":
        return _load_csv_source(zpath, "ml-20m")
    raise ValueError(f"Unknown source: {source}")


def _load_ml100k():
    download_ml100k()
    ratings = []
    with open(RAW_DIR / "ml-100k" / "u.data", 'r') as f:
        for line in f:
            u, i, r, t = line.strip().split('\t')
            ratings.append((int(u), int(i), int(r), int(t)))

    genres_list = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = {}
    with open(RAW_DIR / "ml-100k" / "u.item", 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            mid = int(parts[0])
            title = parts[1]
            genre_str = '; '.join(genres_list[i] for i, flag in enumerate(parts[5:24]) if flag == '1') or 'Unknown'
            movies[mid] = {'title': title, 'genres': genre_str}

    users = {}
    with open(RAW_DIR / "ml-100k" / "u.user", 'r') as f:
        for line in f:
            uid, age, gender, occ, zip_ = line.strip().split('|')
            users[int(uid)] = {'age': int(age), 'gender': gender, 'occupation': occ, 'zip_code': zip_}

    verify_dataset_stats("ml-100k", ratings, movies, users)
    return ratings, movies, users


def _load_dat_source(zpath, inner_dir, sep, has_users):
    """Load ml-1m or ml-10m (double-colon .dat format)."""
    ratings = []; movies = {}; users = {}
    with zipfile.ZipFile(zpath) as z:
        with z.open(f"{inner_dir}/ratings.dat") as f:
            for line in f:
                parts = line.decode('latin-1').strip().split(sep)
                u, i, r, t = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
                ratings.append((u, i, r, t))
                if u not in users:
                    users[u] = {}
        with z.open(f"{inner_dir}/movies.dat") as f:
            for line in f:
                parts = line.decode('latin-1').strip().split(sep)
                mid = int(parts[0])
                title = parts[1]
                genres = parts[2].replace('|', '; ')
                movies[mid] = {'title': title, 'genres': genres}
        if has_users:
            with z.open(f"{inner_dir}/users.dat") as f:
                for line in f:
                    parts = line.decode().strip().split(sep)
                    uid = int(parts[0])
                    users[uid] = {'gender': parts[1], 'age': int(parts[2]),
                                  'occupation': parts[3], 'zip_code': parts[4]}
    source = "ml-1m" if has_users else "ml-10m"
    verify_dataset_stats(source, ratings, movies, users)
    return ratings, movies, users


def _load_csv_source(zpath, inner_dir):
    """Load ml-20m (CSV format)."""
    ratings = []; movies = {}; users = {}
    with zipfile.ZipFile(zpath) as z:
        with z.open(f"{inner_dir}/ratings.csv") as f:
            next(f)
            for line in f:
                u, i, r, t = line.decode().strip().split(',')
                u, i = int(u), int(i)
                ratings.append((u, i, float(r), int(t)))
                if u not in users:
                    users[u] = {}
        with z.open(f"{inner_dir}/movies.csv") as f:
            next(f)
            for line in f:
                parts = line.decode().strip().split(',')
                mid = int(parts[0])
                genres = parts[-1].replace('|', '; ')
                title = ','.join(parts[1:-1]).strip('"')
                movies[mid] = {'title': title, 'genres': genres}
    verify_dataset_stats(inner_dir, ratings, movies, users)
    return ratings, movies, users



def select_ultra_dense_users(ratings, num_users, target_items, target_total_interactions, strict_item_limit=True, shuffle_eligible=False):
    """Select users for ultra-dense dataset with controlled density.
    
    Strategy for controlling U-I matrix density while keeping same "time":
    - Fixed: num_users (100), interactions_per_user (8 for 5 train)
    - Variable: item pool size (fewer items = denser, more items = sparser)
    
    Args:
        ratings: List of (user_id, item_id, rating, timestamp) tuples
        num_users: Number of users to select
        target_items: Target number of items (controls density)
        target_total_interactions: Target total interactions (controls "time")
        strict_item_limit: If True, prioritize hitting exact interaction count
    """
    user_items = defaultdict(set)
    item_users = defaultdict(set)
    user_counts = defaultdict(int)
    user_ratings = defaultdict(list)
    
    for user_id, item_id, rating, timestamp in ratings:
        user_items[user_id].add(item_id)
        item_users[item_id].add(user_id)
        user_counts[user_id] += 1
        user_ratings[user_id].append((item_id, rating, timestamp))
    
    # Sort each user's ratings by timestamp
    for user_id in user_ratings:
        user_ratings[user_id].sort(key=lambda x: x[2])
    
    # Find most popular items
    item_popularity = [(item, len(users)) for item, users in item_users.items()]
    item_popularity.sort(key=lambda x: -x[1])
    
    print(f"  Top 10 most popular items:")
    for item, count in item_popularity[:10]:
        print(f"    Item {item}: {count} users")
    
    # Select top N popular items as our target set
    popular_items = set([item for item, _ in item_popularity[:target_items]])
    popular_items_list = [item for item, _ in item_popularity[:target_items]]
    print(f"\n  Selected {len(popular_items)} popular items as target set")
    
    # Calculate interactions per user (this is FIXED to control "time")
    interactions_per_user = target_total_interactions // num_users
    print(f"  Target: {interactions_per_user} interactions per user (FIXED for same 'time')")
    
    # Find users who have AT LEAST interactions_per_user interactions with popular items
    eligible_users = []
    for user_id, items in user_items.items():
        # Count how many interactions this user has with popular items
        popular_interactions = sum(1 for item_id, _, _ in user_ratings[user_id] if item_id in popular_items)
        
        if popular_interactions >= interactions_per_user:
            # Track which popular items this user has in their first `interactions_per_user` interactions
            user_popular_ratings = [
                item_id for item_id, _, _ in user_ratings[user_id]
                if item_id in popular_items
            ][:interactions_per_user]
            eligible_users.append((user_id, set(user_popular_ratings)))
    
    print(f"  Found {len(eligible_users)} eligible users with {interactions_per_user}+ interactions in popular items")
    
    if len(eligible_users) < num_users:
        print(f"  WARNING: Not enough users! Relaxing constraints...")
        min_interactions = interactions_per_user - 2
        for user_id, items in user_items.items():
            if any(user_id == u[0] for u in eligible_users):
                continue
            popular_interactions = sum(1 for item_id, _, _ in user_ratings[user_id] if item_id in popular_items)
            if popular_interactions >= min_interactions:
                user_popular_ratings = [
                    item_id for item_id, _, _ in user_ratings[user_id]
                    if item_id in popular_items
                ][:interactions_per_user]
                eligible_users.append((user_id, set(user_popular_ratings)))
    
    # GREEDY SELECTION: fill num_users slots.
    # Phase 1 (coverage incomplete): prefer users who bring new items.
    # Phase 2 (all items covered): prefer users with most overlap (density).
    selected_users = []
    covered_items = set()
    if shuffle_eligible:
        random.shuffle(eligible_users)
    remaining_users = list(eligible_users)
    selected_indices = set()

    print(f"\n  Greedy selection to maximize item coverage...")
    
    while len(selected_users) < num_users and len(selected_indices) < len(remaining_users):
        best_idx = -1
        best_score = -float('inf')
        
        for i, (user_id, user_item_set) in enumerate(remaining_users):
            if i in selected_indices:
                continue
            new_items = user_item_set - covered_items
            overlap = len(user_item_set & covered_items)
            # Always prefer new coverage; once full, prefer overlap (density)
            score = len(new_items) * 1000 + overlap
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx == -1:
            break

        user_id, user_item_set = remaining_users[best_idx]
        selected_users.append(user_id)
        covered_items.update(user_item_set)
        selected_indices.add(best_idx)
            
        if len(selected_users) <= 5 or len(selected_users) % 20 == 0:
            print(f"    Selected {len(selected_users)} users, covering {len(covered_items)}/{target_items} items")
    
    print(f"\n  Selected {len(selected_users)} users, final coverage: {len(covered_items)}/{target_items} items")
    
    # CRITICAL: Filter to popular items AND limit to EXACTLY interactions_per_user
    filtered_ratings = []
    for user_id in selected_users:
        user_popular_ratings = [
            (item_id, rating, timestamp) 
            for item_id, rating, timestamp in user_ratings[user_id]
            if item_id in popular_items
        ]
        limited_ratings = user_popular_ratings[:interactions_per_user]
        
        for item_id, rating, timestamp in limited_ratings:
            filtered_ratings.append((user_id, item_id, rating, timestamp))
    
    # Calculate stats on filtered data
    filtered_user_items = defaultdict(set)
    filtered_user_counts = defaultdict(int)
    for user_id, item_id, rating, timestamp in filtered_ratings:
        filtered_user_items[user_id].add(item_id)
        filtered_user_counts[user_id] += 1
    
    all_items = set()
    for user_id in selected_users:
        all_items.update(filtered_user_items[user_id])
    
    total_interactions = sum(filtered_user_counts[u] for u in selected_users)
    
    # Calculate average pairwise overlap (measure of density)
    total_overlap = 0
    pairs = 0
    for i, u1 in enumerate(selected_users):
        for u2 in selected_users[i+1:]:
            total_overlap += len(filtered_user_items[u1] & filtered_user_items[u2])
            pairs += 1
    avg_overlap = total_overlap / pairs if pairs > 0 else 0
    
    # Calculate sparsity (1 - density)
    max_possible = len(selected_users) * len(all_items)
    density = total_interactions / max_possible if max_possible > 0 else 0
    sparsity = 1 - density
    
    print(f"\n  Final dataset stats:")
    print(f"    - {len(selected_users)} users")
    print(f"    - {len(all_items)} unique items (target: {target_items})")
    print(f"    - {total_interactions} total interactions")
    print(f"    - {total_interactions / len(selected_users):.1f} interactions per user")
    print(f"    - Avg pairwise user overlap: {avg_overlap:.1f} items")
    print(f"    - Matrix density: {density:.4f}")
    print(f"    - Matrix sparsity: {sparsity:.4f}")
    
    return set(selected_users), filtered_ratings


def select_very_sparse_users(ratings, num_users, target_total_interactions):
    """Select users for very sparse dataset with fixed total interactions but many items."""
    user_counts = defaultdict(int)
    user_items = defaultdict(set)
    user_ratings = defaultdict(list)
    item_users = defaultdict(set)
    
    for user_id, item_id, rating, timestamp in ratings:
        user_counts[user_id] += 1
        user_items[user_id].add(item_id)
        user_ratings[user_id].append((item_id, rating, timestamp))
        item_users[item_id].add(user_id)
    
    # Sort each user's ratings by timestamp
    for user_id in user_ratings:
        user_ratings[user_id].sort(key=lambda x: x[2])
    
    # Calculate interactions per user to hit target
    interactions_per_user = target_total_interactions // num_users
    print(f"  Target: {interactions_per_user} interactions per user")
    
    # Find users with enough interactions
    eligible_users = [u for u, c in user_counts.items() if c >= interactions_per_user]
    print(f"  Found {len(eligible_users)} eligible users with {interactions_per_user}+ interactions")
    
    # Greedy selection to MAXIMIZE unique item usage (minimize overlap between users)
    random.shuffle(eligible_users)
    
    selected_users = []
    remaining_users = set(eligible_users)
    item_usage_count = defaultdict(int)  # Track how many users have each item
    
    # Select users who minimize item reuse
    while len(selected_users) < num_users and remaining_users:
        best_user = None
        best_score = float('inf')
        
        for user_id in remaining_users:
            # Calculate score: prefer users whose items are least used
            reuse_score = sum(item_usage_count[item] for item in user_items[user_id])
            avg_reuse = reuse_score / len(user_items[user_id]) if user_items[user_id] else 0
            
            if avg_reuse < best_score:
                best_score = avg_reuse
                best_user = user_id
        
        if best_user:
            selected_users.append(best_user)
            remaining_users.remove(best_user)
            # Update item usage
            for item in user_items[best_user]:
                item_usage_count[item] += 1
            
            if len(selected_users) <= 5 or len(selected_users) % 20 == 0:
                print(f"  Selected user {best_user}: avg item reuse {best_score:.2f}")
    
    # Limit each user's history to interactions_per_user
    filtered_ratings = []
    for user_id in selected_users:
        limited_ratings = user_ratings[user_id][:interactions_per_user]
        for item_id, rating, timestamp in limited_ratings:
            filtered_ratings.append((user_id, item_id, rating, timestamp))
    
    # Calculate stats
    filtered_user_items = defaultdict(set)
    filtered_user_counts = defaultdict(int)
    for user_id, item_id, rating, timestamp in filtered_ratings:
        filtered_user_items[user_id].add(item_id)
        filtered_user_counts[user_id] += 1
    
    all_items = set()
    for user_id in selected_users:
        all_items.update(filtered_user_items[user_id])
    
    total_interactions = sum(filtered_user_counts[u] for u in selected_users)
    
    # Calculate average pairwise overlap
    total_overlap = 0
    pairs = 0
    for i, u1 in enumerate(selected_users):
        for u2 in selected_users[i+1:]:
            total_overlap += len(filtered_user_items[u1] & filtered_user_items[u2])
            pairs += 1
    avg_overlap = total_overlap / pairs if pairs > 0 else 0
    
    # Calculate sparsity
    max_possible = len(selected_users) * len(all_items)
    sparsity = 1 - (total_interactions / max_possible)
    
    print(f"\n  After limiting to {interactions_per_user} interactions per user:")
    print(f"    - {len(all_items)} unique items")
    print(f"    - {total_interactions} total interactions")
    print(f"    - Avg interactions per user: {total_interactions / len(selected_users):.1f}")
    print(f"    - Avg pairwise overlap: {avg_overlap:.1f} items")
    print(f"    - Sparsity: {sparsity:.4f}")
    
    return set(selected_users), filtered_ratings


def build_user_sequences(ratings, selected_users):
    """Build interaction sequences per user, sorted by timestamp."""
    user_sequences = defaultdict(list)
    for user_id, item_id, rating, timestamp in ratings:
        if user_id in selected_users:
            user_sequences[user_id].append((item_id, timestamp, rating))
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])
    return user_sequences


def create_inter_files(user_sequences, output_dir, dataset_name):
    """Create train/valid/test .inter files."""
    train_lines = ["user_id:token\titem_id_list:token_seq\titem_id:token"]
    valid_lines = ["user_id:token\titem_id_list:token_seq\titem_id:token"]
    test_lines = ["user_id:token\titem_id_list:token_seq\titem_id:token"]
    
    all_items = set()
    user_item_sets = {}
    
    print(f"  DEBUG: user_sequences has {len(user_sequences)} users: {sorted(list(user_sequences.keys())[:5])}...")
    
    for user_id, sequence in user_sequences.items():
        if len(sequence) < MIN_INTERACTIONS:
            continue
        items = [item_id for item_id, _, _ in sequence]
        all_items.update(items)
        user_item_sets[user_id] = set(items)
        
        train_items = items[:-2]
        valid_item = items[-2]
        test_item = items[-1]
        
        for i in range(1, len(train_items)):
            history = ' '.join(map(str, train_items[:i]))
            train_lines.append(f"{user_id}\t{history}\t{train_items[i]}")
        
        train_history = ' '.join(map(str, train_items))
        valid_lines.append(f"{user_id}\t{train_history}\t{valid_item}")
        valid_history = ' '.join(map(str, train_items + [valid_item]))
        test_lines.append(f"{user_id}\t{valid_history}\t{test_item}")
    
    with open(output_dir / f"{dataset_name}.train.inter", 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    with open(output_dir / f"{dataset_name}.valid.inter", 'w') as f:
        f.write('\n'.join(valid_lines) + '\n')
    with open(output_dir / f"{dataset_name}.test.inter", 'w') as f:
        f.write('\n'.join(test_lines) + '\n')
    
    print(f"  Created inter files: {len(train_lines)-1} train, {len(valid_lines)-1} valid, {len(test_lines)-1} test")
    return all_items, user_item_sets


def create_item_file(movies, all_items, output_dir, dataset_name):
    """Create .item file with movie metadata."""
    lines = ["item_id:token\ttitle:token_seq\tcategory:token_seq"]
    lines.append("0\t[PAD]\t[PAD]")
    
    for item_id in sorted(all_items):
        if item_id in movies:
            movie = movies[item_id]
            title = movie['title'].replace('\t', ' ')
            genres = movie['genres']
            lines.append(f"{item_id}\t{title}\t{genres}")
        else:
            lines.append(f"{item_id}\tUnknown Movie {item_id}\tUnknown")
    
    with open(output_dir / f"{dataset_name}.item", 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    # Create symlink for ml-100k.item
    symlink_path = output_dir / "ml-100k.item"
    if symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(f"{dataset_name}.item")
    
    print(f"  Created item file with {len(lines)-1} items")


def create_user_file(selected_users, users_demographics, output_dir):
    """Create ml-100k.user file with user demographic metadata."""
    lines = ["user_id:token\tage:token\tgender:token\toccupation:token\tzip_code:token"]
    
    missing_count = 0
    for user_id in sorted(selected_users):
        demo = users_demographics.get(user_id, {})
        if demo and 'age' in demo:
            lines.append(f"{user_id}\t{demo['age']}\t{demo['gender']}\t{demo['occupation']}\t{demo['zip_code']}")
        else:
            missing_count += 1
            lines.append(f"{user_id}\tN/A\tN/A\tN/A\tN/A")
    
    if missing_count > 0:
        print(f"  WARNING: {missing_count} users missing demographics, using fallback data")
    
    with open(output_dir / "ml-100k.user", 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"  Created user file with {len(lines)-1} users (with demographics)")
    
    # Print demographic statistics
    ages = [users_demographics[u]['age'] for u in selected_users if u in users_demographics]
    genders = [users_demographics[u]['gender'] for u in selected_users if u in users_demographics]
    occupations = [users_demographics[u]['occupation'] for u in selected_users if u in users_demographics]
    
    if ages:
        print(f"    - Age range: {min(ages)}-{max(ages)}, avg: {sum(ages)/len(ages):.1f}")
        print(f"    - Gender: {genders.count('M')} male, {genders.count('F')} female")
        print(f"    - Occupations: {len(set(occupations))} unique")


def create_random_file(user_item_sets, all_items, output_dir, dataset_name, num_negatives=100):
    """Create .random file with negative samples per user."""
    lines = []
    all_items_list = list(all_items)
    
    for user_id, positive_items in user_item_sets.items():
        negative_pool = [i for i in all_items_list if i not in positive_items]
        negatives = random.sample(negative_pool, min(num_negatives, len(negative_pool)))
        neg_str = ' '.join(map(str, negatives))
        lines.append(f"{user_id}\t{neg_str}")
    
    with open(output_dir / f"{dataset_name}.random", 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"  Created random file with {len(lines)} users")


def create_binary_train_json(user_sequences, user_item_sets, all_items, output_dir):
    """Create binary_train_1cand.json with POPULARITY-BASED negative sampling."""
    data = {}
    all_items_list = list(all_items)
    
    # Compute item popularity (frequency in all users' histories)
    item_popularity = defaultdict(int)
    for user_id, items in user_item_sets.items():
        for item in items:
            item_popularity[item] += 1
    
    for user_id, sequence in user_sequences.items():
        if user_id not in user_item_sets:
            continue
        positive_items = user_item_sets[user_id]
        train_items = [item_id for item_id, _, _ in sequence[:-2]]
        
        if len(train_items) < 3:
            continue
        
        samples = []
        pos_samples = random.sample(train_items[1:], min(3, len(train_items)-1))
        for item in pos_samples:
            samples.append([str(item), True])
        
        # POPULARITY-BASED negative sampling (matching 2_cand behavior)
        negative_pool = [i for i in all_items_list if i not in positive_items]
        if len(negative_pool) >= 2:
            # Compute popularity weights for negative pool
            neg_weights = [item_popularity.get(item, 1) for item in negative_pool]
            total_weight = sum(neg_weights)
            if total_weight > 0:
                neg_probs = [w / total_weight for w in neg_weights]
                neg_samples = np.random.choice(
                    negative_pool, 
                    size=min(2, len(negative_pool)), 
                    replace=False,
                    p=neg_probs
                )
            else:
                # Fallback to random if all weights are 0
                neg_samples = random.sample(negative_pool, min(2, len(negative_pool)))
            
            for item in neg_samples:
                samples.append([str(item), False])
        
        random.shuffle(samples)
        data[str(user_id)] = samples
    
    output = {"num_candidates": 1, "data": data}
    with open(output_dir / "binary_train_1cand.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Created binary_train_1cand.json with {len(data)} users (POPULARITY-BASED)")


def create_ranking_train_json(user_sequences, user_item_sets, all_items, output_dir, num_candidates_list=[3, 4]):
    """Create ranking_train_{n}cand.json files with POPULARITY-BASED negative sampling.
    
    Args:
        user_sequences: Dict of user_id -> list of (item_id, timestamp, rating)
        user_item_sets: Dict of user_id -> set of item_ids
        all_items: Set of all item_ids
        output_dir: Path to output directory
        num_candidates_list: List of candidate counts to generate files for (default: [3, 4])
    """
    all_items_list = list(all_items)
    
    # Compute item popularity (frequency in all users' histories)
    item_popularity = defaultdict(int)
    for user_id, items in user_item_sets.items():
        for item in items:
            item_popularity[item] += 1
    
    for num_candidates in num_candidates_list:
        data = {}
        num_positives = (num_candidates + 1) // 2  # Ceiling division for positives
        num_negatives = num_candidates - num_positives
        
        for user_id, sequence in user_sequences.items():
            if user_id not in user_item_sets:
                continue
            positive_items = user_item_sets[user_id]
            train_items = [item_id for item_id, _, _ in sequence[:-2]]
            
            if len(train_items) < num_positives:
                continue
            
            rankings = []
            negative_pool = [i for i in all_items_list if i not in positive_items]
            
            if len(negative_pool) < num_negatives:
                continue
            
            # Compute popularity weights for negative pool
            neg_weights = [item_popularity.get(item, 1) for item in negative_pool]
            total_weight = sum(neg_weights)
            if total_weight > 0:
                neg_probs = [w / total_weight for w in neg_weights]
            else:
                neg_probs = [1.0 / len(negative_pool)] * len(negative_pool)
            
            for _ in range(5):
                # Sample positives (ordered by recency in training history)
                if len(train_items) >= num_positives:
                    pos_indices = random.sample(range(len(train_items)), num_positives)
                    pos_indices.sort(reverse=True)  # More recent first
                    positives = [train_items[i] for i in pos_indices]
                else:
                    continue
                
                # Sample negatives with popularity-based weighting
                negatives = list(np.random.choice(
                    negative_pool, 
                    size=num_negatives, 
                    replace=False,
                    p=neg_probs
                ))
                
                # Combine: positives first, then negatives
                ranking = [str(p) for p in positives] + [str(n) for n in negatives]
                rankings.append(ranking)
            
            if rankings:
                data[str(user_id)] = rankings
        
        output = {"num_candidates": num_candidates, "data": data}
        filename = f"ranking_train_{num_candidates}cand.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  Created {filename} with {len(data)} users (POPULARITY-BASED)")


def create_record_folders(user_sequences, movies, users_demographics, output_dir):
    """Create record folders with demographic-aware user descriptions."""
    record_dir = output_dir / "record"
    user_record_dir = record_dir / "user_record_0"
    user_record_dir.mkdir(parents=True, exist_ok=True)
    
    for user_id in user_sequences.keys():
        with open(user_record_dir / f"user.{user_id}", 'w') as f:
            f.write("~~~~~~~~~~~~~~~~~~~~Meta information~~~~~~~~~~~~~~~~~~~~\n")
            
            # Generate demographic-aware description
            if user_id in users_demographics and 'gender' in users_demographics[user_id]:
                demo = users_demographics[user_id]
                gender_desc = "man" if demo['gender'] == 'M' else "woman"
                
                # Better occupation phrasing
                occupation = demo['occupation']
                if occupation == 'other':
                    occupation_desc = "a movie enthusiast"
                elif occupation in ['administrator', 'educator', 'engineer', 'entertainment', 
                                   'executive', 'healthcare', 'homemaker', 'lawyer', 'librarian',
                                   'marketing', 'programmer', 'scientist', 'technician', 'writer']:
                    occupation_desc = f"working in {occupation}"
                elif occupation == 'student':
                    occupation_desc = "a student"
                elif occupation == 'retired':
                    occupation_desc = "retired"
                else:
                    occupation_desc = f"a {occupation}"
                
                description = f"I am a {gender_desc}. I am {occupation_desc}."
            else:
                description = "I enjoy watching movies very much."
            
            f.write(f"The user wrote the following self-description as follows: {description}\n\n")
    
    item_record_dir = record_dir / "item_record_0"
    item_record_dir.mkdir(parents=True, exist_ok=True)
    
    with open(item_record_dir / "item.0", 'w') as f:
        f.write("~~~~~~~~~~~~~~~~~~~~Meta information~~~~~~~~~~~~~~~~~~~~\n")
        f.write("The item has the following characteristics: {'item_title': '[PAD]', 'item_class': '[PAD]'} \n\n")
    
    all_items = set()
    for seq in user_sequences.values():
        all_items.update([item_id for item_id, _, _ in seq])
    
    for item_id in all_items:
        if item_id in movies:
            movie = movies[item_id]
            with open(item_record_dir / f"item.{item_id}", 'w') as f:
                f.write("~~~~~~~~~~~~~~~~~~~~Meta information~~~~~~~~~~~~~~~~~~~~\n")
                f.write(f"The item has the following characteristics: {{'item_title': '{movie['title']}', 'item_class': '{movie['genres']}'}} \n\n")
    
    print(f"  Created record folders")


def create_props_file(dataset_name):
    """Create props YAML file."""
    props_content = f'''# Atomic File Format
field_separator: "\\t"
seq_separator: " "

# Basic Information
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
seq_len: ~
LABEL_FIELD: label
threshold: ~
NEG_PREFIX: neg_

# Sequential Model Needed
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id

# Knowledge-based Model Needed
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
kg_reverse_r: False
entity_kg_num_interval: ~
relation_kg_num_interval: ~

# Selectively Loading
load_col:
    inter: [user_id, item_id_list, item_id]
unload_col: ~
unused_col: ~

# Filtering
rm_dup_inter: ~
val_interval: ~
filter_inter_by_user_or_item: True
user_inter_num_interval: ~
item_inter_num_interval: ~

# Preprocessing
alias_of_user_id: ~
alias_of_item_id: [item_id_list]
alias_of_entity_id: ~
alias_of_relation_id: ~
preload_weight: ~
normalize_field: ~
normalize_all: True
data_path: dataset/
benchmark_filename: [train, valid, test]
sampled_user_suffix: random
recall_budget: 10
fix_pos: -1
has_gt: True
'''
    props_path = CONNACF_PROPS_DIR / f"{dataset_name}.yaml"
    with open(props_path, 'w') as f:
        f.write(props_content)
    print(f"  Created props file")


def generate_variant(ratings, movies, users_demographics, dataset_name, num_users, target_items, target_total_interactions, seed=RANDOM_SEED, force=False, shuffle_graph=False):
    """Generate dataset variant with given item pool size (controls density)."""
    output_dir = CONNACF_DATASET_DIR / dataset_name

    # Skip if already complete (has the core files). Use --force to regenerate.
    sentinel = output_dir / "binary_train_1cand.json"
    if not force and output_dir.exists() and sentinel.exists():
        print(f"  Skipping {dataset_name} — already exists (use --force to regenerate)")
        return

    # Backup existing if present
    if output_dir.exists():
        backup_dir = output_dir.parent / f"{dataset_name}.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(output_dir), str(backup_dir))
        print(f"  Backed up existing {dataset_name} to {dataset_name}.backup")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {dataset_name} (target_items={target_items})...")
    random.seed(seed)

    interactions_per_user = target_total_interactions // num_users
    if interactions_per_user < MIN_INTERACTIONS:
        shutil.rmtree(output_dir)
        raise SystemExit(
            f"FAILED: {interactions_per_user} interactions/user is below the minimum {MIN_INTERACTIONS} needed for "
            f"train/valid/test split. Increase --target-interactions (currently {target_total_interactions}) or "
            f"reduce --num-users (currently {num_users}). "
            f"Minimum total interactions: {MIN_INTERACTIONS * num_users}."
        )
    
    selected_users, filtered_ratings = select_ultra_dense_users(ratings, num_users, target_items, target_total_interactions, shuffle_eligible=shuffle_graph)

    # Validate actual item count is within 5% of target
    actual_items = set()
    _tmp = defaultdict(set)
    for u, i, r, t in filtered_ratings:
        _tmp[u].add(i)
    for u in selected_users:
        actual_items.update(_tmp[u])
    actual_count = len(actual_items)
    tolerance = max(1, round(target_items * 0.05))
    if abs(actual_count - target_items) > tolerance:
        shutil.rmtree(output_dir)
        raise SystemExit(
            f"FAILED: actual item count {actual_count} is outside 5% tolerance of target {target_items} "
            f"(allowed range: {target_items - tolerance}–{target_items + tolerance}). "
            f"Try adjusting --target-items or --target-interactions."
        )

    user_sequences = build_user_sequences(filtered_ratings, selected_users)
    all_items, user_item_sets = create_inter_files(user_sequences, output_dir, dataset_name)
    create_item_file(movies, all_items, output_dir, dataset_name)
    # Only write user file if demographics are available (e.g. ml-100k/ml-1m; ml-10m/ml-20m have none)
    if any('gender' in d for d in users_demographics.values()):
        create_user_file(selected_users, users_demographics, output_dir)
    create_random_file(user_item_sets, all_items, output_dir, dataset_name)
    create_binary_train_json(user_sequences, user_item_sets, all_items, output_dir)
    create_ranking_train_json(user_sequences, user_item_sets, all_items, output_dir)
    create_record_folders(user_sequences, movies, users_demographics, output_dir)
    create_props_file(dataset_name)
    
    print(f"Completed {dataset_name}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate a single MovieLens dataset variant')
    parser.add_argument('--source', type=str, default='ml-100k',
                        choices=['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m'],
                        help='Source MovieLens dataset. Full sizes: '
                             'ml-100k=943u/1682i, ml-1m=6040u/3706i, '
                             'ml-10m=69878u/10677i, ml-20m=138493u/26744i.')
    parser.add_argument('--num-users', type=int, required=True,
                        help='Number of users to select (strict)')
    parser.add_argument('--target-items', type=int, required=True,
                        help='Target item pool size (must land within 5%% or generation fails)')
    parser.add_argument('--label', type=str, default=None,
                        help='Density label for dataset name, e.g. "dense", "medium", "sparse". '
                             'Combined with item count to form names like dense50, medium100.')
    parser.add_argument('--target-interactions', type=int, default=None,
                        help=f'Target total interactions (default: {TARGET_INTERACTIONS_PER_USER} per user)')
    parser.add_argument('--train-per-user', type=int, default=None,
                        help='Target training interactions per user (e.g., 5). Overrides --target-interactions.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--shuffle-graph', action='store_true',
                        help='Shuffle eligible users before greedy selection so different seeds '
                             'produce different interaction matrices at the same density. '
                             'Default: off (seed only affects training candidate ordering).')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if the dataset already exists')
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    if args.train_per_user is not None:
        target_interactions = (args.train_per_user + 3) * args.num_users
        print(f"Calculated target_interactions from train_per_user: {args.train_per_user} train/user -> {target_interactions} total")
    else:
        target_interactions = args.target_interactions if args.target_interactions is not None else TARGET_INTERACTIONS_PER_USER * args.num_users * (args.num_users // 100)

    seed_suffix = f"-seed{seed}"
    label = args.label if args.label else "dataset"
    dataset_name = f"{args.source}-{args.num_users}user-{label}{args.target_items}{seed_suffix}"

    print("=" * 70)
    print(f"Generating {dataset_name}")
    print(f"  source:      {args.source}")
    print(f"  num_users:   {args.num_users}")
    print(f"  target_items:{args.target_items} (±5% tolerance)")
    print(f"  interactions:{target_interactions}")
    print(f"  seed:        {seed}")
    print("=" * 70)

    print("\nLoading raw data...")
    ratings, movies, users = load_raw_data(args.source)
    print(f"Loaded {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")

    generate_variant(
        ratings, movies, users,
        dataset_name,
        num_users=args.num_users,
        target_items=args.target_items,
        target_total_interactions=target_interactions,
        seed=seed,
        force=args.force,
        shuffle_graph=args.shuffle_graph,
    )

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
