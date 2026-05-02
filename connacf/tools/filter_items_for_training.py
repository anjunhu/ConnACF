#!/usr/bin/env python3
"""
Filter item files to only include items that appear in training samples.
This reduces the number of item agents needed during training.
"""

import json
from pathlib import Path

DATASET_DIR = Path("connacf/dataset")

def get_training_items(dataset_name):
    """Extract all unique items from training samples."""
    dataset_path = DATASET_DIR / dataset_name
    training_items = set()
    
    # Get items from binary_train_1cand.json
    with open(dataset_path / "binary_train_1cand.json") as f:
        binary_data = json.load(f)
        for user_samples in binary_data['data'].values():
            for item, label in user_samples:
                training_items.add(str(item))
    
    # Get items from ranking_train_4cand.json
    with open(dataset_path / "ranking_train_4cand.json") as f:
        ranking_data = json.load(f)
        for user_rankings in ranking_data['data'].values():
            for ranking in user_rankings:
                training_items.update(ranking)
    
    return training_items


def filter_item_file(dataset_name):
    """Create a filtered .item file with only training items."""
    dataset_path = DATASET_DIR / dataset_name
    
    # Get training items
    training_items = get_training_items(dataset_name)
    print(f"\n{dataset_name}:")
    print(f"  Items in training samples: {len(training_items)}")
    
    # Read original item file
    original_item_file = dataset_path / f"{dataset_name}.item"
    with open(original_item_file) as f:
        lines = f.readlines()
    
    header = lines[0]
    item_lines = lines[1:]
    
    # Filter to only training items
    filtered_lines = [header]
    kept_count = 0
    
    for line in item_lines:
        item_id = line.split('\t')[0]
        if item_id == '0' or item_id in training_items:  # Keep PAD and training items
            filtered_lines.append(line)
            kept_count += 1
    
    # Write filtered file
    filtered_file = dataset_path / f"{dataset_name}.item.training"
    with open(filtered_file, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"  Original items: {len(item_lines)}")
    print(f"  Filtered items: {kept_count}")
    print(f"  Saved to: {filtered_file.name}")
    
    # Also create a JSON mapping for easy lookup
    training_items_list = sorted([int(item) for item in training_items if item.isdigit()])
    mapping_file = dataset_path / "training_items.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            "num_items": len(training_items_list),
            "item_ids": training_items_list
        }, f, indent=2)
    print(f"  Item ID list saved to: {mapping_file.name}")


def main():
    datasets = [
        'ml-100k-20-user-dense',
        'ml-100k-20-user-sparse',
        'ml-100k-100user-dense',
        'ml-100k-100user-sparse'
    ]
    
    print("=" * 70)
    print("Filtering item files to only include training items")
    print("=" * 70)
    
    for dataset in datasets:
        try:
            filter_item_file(dataset)
        except Exception as e:
            print(f"\n{dataset}: Error - {e}")
    
    print("\n" + "=" * 70)
    print("Done! Use *.item.training files for training to reduce item agents.")
    print("=" * 70)


if __name__ == "__main__":
    main()
