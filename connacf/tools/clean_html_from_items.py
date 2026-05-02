#!/usr/bin/env python3
"""
Clean HTML tags from CDs.item files.

The Amazon scraping left some items with raw HTML tags as titles.
This script removes those HTML tags and replaces them with a placeholder.
"""

import os
import re
from pathlib import Path


def clean_html_tags(text: str) -> str:
    """Remove HTML tags and fragments from text."""
    # Remove complete HTML tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Remove incomplete HTML tags (like '<span class="...' without closing >)
    cleaned = re.sub(r'<\w+[^>]*$', '', cleaned)
    cleaned = re.sub(r'<\w+\s+[^>]*', '', cleaned)
    # Clean up any remaining HTML entities
    cleaned = cleaned.replace('&amp;', '&')
    cleaned = cleaned.replace('&lt;', '<')
    cleaned = cleaned.replace('&gt;', '>')
    cleaned = cleaned.replace('&quot;', '"')
    cleaned = cleaned.replace('&#39;', "'")
    return cleaned.strip()


def is_html_contaminated(title: str) -> bool:
    """Check if a title contains HTML tags or fragments."""
    # Check for opening HTML tags (may not be closed)
    if '<span' in title.lower() or '<div' in title.lower():
        return True
    # Check for complete HTML tags
    if re.search(r'<[^>]+>', title):
        return True
    return False


def clean_item_file(filepath: str, dry_run: bool = False) -> dict:
    """
    Clean HTML from a .item file.
    
    Args:
        filepath: Path to the .item file
        dry_run: If True, don't write changes, just report
        
    Returns:
        Dict with statistics about the cleaning
    """
    stats = {
        'total_items': 0,
        'html_contaminated': 0,
        'cleaned_items': [],
        'filepath': filepath
    }
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return stats
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        return stats
    
    # First line is header
    header = lines[0]
    cleaned_lines = [header]
    
    for line in lines[1:]:
        stats['total_items'] += 1
        parts = line.strip().split('\t')
        
        if len(parts) >= 2:
            item_id = parts[0]
            title = parts[1]
            rest = parts[2:] if len(parts) > 2 else []
            
            if is_html_contaminated(title):
                stats['html_contaminated'] += 1
                original_title = title
                
                # Clean the title
                cleaned_title = clean_html_tags(title)
                
                # If title is now empty or just whitespace, use a placeholder
                if not cleaned_title or cleaned_title.isspace():
                    cleaned_title = f"Unknown CD #{item_id}"
                
                stats['cleaned_items'].append({
                    'item_id': item_id,
                    'original': original_title,
                    'cleaned': cleaned_title
                })
                
                # Reconstruct the line
                new_parts = [item_id, cleaned_title] + rest
                cleaned_lines.append('\t'.join(new_parts) + '\n')
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    # Write cleaned file
    if not dry_run and stats['html_contaminated'] > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"✓ Cleaned {filepath}: {stats['html_contaminated']} items fixed")
    
    return stats


def main():
    """Clean HTML from all CDs.item files in the dataset directories."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean HTML tags from CDs.item files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be cleaned without making changes')
    parser.add_argument('--path', '-p', type=str, default=None,
                        help='Specific .item file to clean (default: all CDs.item files)')
    args = parser.parse_args()
    
    # Find all CDs.item files
    base_dirs = [
        'dataset',
        'connacf/dataset',
    ]
    
    item_files = []
    
    if args.path:
        item_files = [args.path]
    else:
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                for subdir in os.listdir(base_dir):
                    subdir_path = os.path.join(base_dir, subdir)
                    if os.path.isdir(subdir_path):
                        # Look for .item files
                        for filename in os.listdir(subdir_path):
                            if filename.endswith('.item'):
                                item_files.append(os.path.join(subdir_path, filename))
    
    if not item_files:
        print("No .item files found")
        return
    
    print(f"Found {len(item_files)} .item files to process")
    print("=" * 60)
    
    total_contaminated = 0
    
    for filepath in sorted(item_files):
        stats = clean_item_file(filepath, dry_run=args.dry_run)
        
        if stats['html_contaminated'] > 0:
            total_contaminated += stats['html_contaminated']
            print(f"\n{filepath}:")
            print(f"  Total items: {stats['total_items']}")
            print(f"  HTML contaminated: {stats['html_contaminated']}")
            
            if args.dry_run:
                print("  Would clean:")
                for item in stats['cleaned_items'][:5]:  # Show first 5
                    print(f"    Item {item['item_id']}: '{item['original'][:50]}...' -> '{item['cleaned']}'")
                if len(stats['cleaned_items']) > 5:
                    print(f"    ... and {len(stats['cleaned_items']) - 5} more")
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN: Would clean {total_contaminated} items total")
        print("Run without --dry-run to apply changes")
    else:
        print(f"Cleaned {total_contaminated} items total")


if __name__ == '__main__':
    main()
