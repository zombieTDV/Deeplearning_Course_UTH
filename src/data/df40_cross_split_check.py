#!/usr/bin/env python3
"""
DF40 Cross-Split Duplicate Detection Script
Checks for duplicate files crossing train/validation/test splits.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path("/Users/pickapu/Documents/PyCharmMiscProject/DeepLearning")
DF40_ROOT = PROJECT_ROOT / "src" / "data" / "DF40"
METADATA_ROOT = DF40_ROOT / "metadata"

def check_cross_split_duplicates():
    """Check for duplicate files crossing splits."""
    print("Checking cross-split duplicates...")
    
    # Load final training metadata
    df = pd.read_csv(METADATA_ROOT / 'final_training_metadata.csv')
    
    # Group by SHA256 hash
    hash_map = defaultdict(list)
    for _, row in df.iterrows():
        file_hash = row.get('sha256', '')
        if file_hash:
            hash_map[file_hash].append({
                'sample_id': row['sample_id'],
                'file_path': row['file_path'],
                'video_id': row['video_id'],
                'manipulation': row['manipulation'],
                'split': row['split']
            })
    
    # Find cross-split duplicates
    cross_split_dups = []
    for file_hash, samples in hash_map.items():
        if len(samples) > 1:
            splits = set(s['split'] for s in samples)
            if len(splits) > 1:
                for sample in samples:
                    cross_split_dups.append({
                        'sha256': file_hash,
                        'sample_id': sample['sample_id'],
                        'file_path': sample['file_path'],
                        'video_id': sample['video_id'],
                        'manipulation': sample['manipulation'],
                        'split': sample['split'],
                        'duplicate_count': len(samples),
                        'cross_split': True
                    })
    
    # Save cross-split duplicates
    if cross_split_dups:
        df = pd.DataFrame(cross_split_dups)
        df.to_csv(METADATA_ROOT / 'cross_split_duplicates.csv', index=False)
        print(f"Found {len(cross_split_dups)} cross-split duplicate samples")
    else:
        print("No cross-split duplicates found")
        # Create empty file
        pd.DataFrame(columns=['sha256', 'sample_id', 'file_path', 'video_id', 'manipulation', 'split', 'duplicate_count', 'cross_split']).to_csv(METADATA_ROOT / 'cross_split_duplicates.csv', index=False)
    
    return cross_split_dups

if __name__ == "__main__":
    check_cross_split_duplicates()