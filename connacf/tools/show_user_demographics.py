#!/usr/bin/env python3
"""
Display demographic statistics for ML-100k datasets.
"""

import sys
from pathlib import Path
from collections import Counter

def show_demographics(dataset_name):
    """Display demographic statistics for a dataset."""
    dataset_path = Path(f"connacf/dataset/{dataset_name}")
    user_file = dataset_path / "ml-100k.user"
    
    if not user_file.exists():
        print(f"Error: User file not found at {user_file}")
        return
    
    users = []
    with open(user_file, 'r') as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                users.append({
                    'user_id': int(parts[0]),
                    'age': int(parts[1]),
                    'gender': parts[2],
                    'occupation': parts[3],
                    'zip_code': parts[4]
                })
    
    if not users:
        print("No users found in dataset")
        return
    
    print(f"\n{'='*70}")
    print(f"Demographics for {dataset_name}")
    print(f"{'='*70}\n")
    
    print(f"Total Users: {len(users)}\n")
    
    # Age statistics
    ages = [u['age'] for u in users]
    print(f"Age Statistics:")
    print(f"  Range: {min(ages)} - {max(ages)}")
    print(f"  Average: {sum(ages)/len(ages):.1f}")
    print(f"  Median: {sorted(ages)[len(ages)//2]}")
    
    # Age distribution
    age_bins = {'<20': 0, '20-29': 0, '30-39': 0, '40-49': 0, '50+': 0}
    for age in ages:
        if age < 20:
            age_bins['<20'] += 1
        elif age < 30:
            age_bins['20-29'] += 1
        elif age < 40:
            age_bins['30-39'] += 1
        elif age < 50:
            age_bins['40-49'] += 1
        else:
            age_bins['50+'] += 1
    
    print(f"\n  Age Distribution:")
    for bin_name, count in age_bins.items():
        pct = (count / len(users)) * 100
        bar = '█' * int(pct / 2)
        print(f"    {bin_name:8s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Gender statistics
    genders = [u['gender'] for u in users]
    gender_counts = Counter(genders)
    print(f"\nGender Distribution:")
    for gender, count in gender_counts.most_common():
        pct = (count / len(users)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {gender}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Occupation statistics
    occupations = [u['occupation'] for u in users]
    occupation_counts = Counter(occupations)
    print(f"\nOccupation Distribution (Top 10):")
    for occupation, count in occupation_counts.most_common(10):
        pct = (count / len(users)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {occupation:20s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n  Total unique occupations: {len(occupation_counts)}")
    
    # Geographic distribution (by first 2 digits of zip code)
    zip_prefixes = [u['zip_code'][:2] for u in users if len(u['zip_code']) >= 2]
    zip_counts = Counter(zip_prefixes)
    print(f"\nGeographic Distribution (Top 5 by ZIP prefix):")
    for zip_prefix, count in zip_counts.most_common(5):
        pct = (count / len(users)) * 100
        print(f"  {zip_prefix}xxx: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python show_user_demographics.py <dataset_name>")
        print("\nAvailable datasets:")
        dataset_dir = Path("connacf/dataset")
        if dataset_dir.exists():
            for d in sorted(dataset_dir.iterdir()):
                if d.is_dir() and (d / "ml-100k.user").exists():
                    print(f"  - {d.name}")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    show_demographics(dataset_name)


if __name__ == "__main__":
    main()
