#!/usr/bin/env python3
"""
DF40 Dataset Split Creation Script
Creates train/validation/test splits at VIDEO level to avoid frame leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

# Configuration
PROJECT_ROOT = Path("/")
DF40_ROOT = PROJECT_ROOT / "src" / "data" / "DF40"
METADATA_ROOT = DF40_ROOT / "metadata"

# Label mapping
LABEL_MAPPING = {
    'REAL': '0',
    'FAKE': '1'
}

def create_video_level_splits():
    """Create train/validation/test splits at VIDEO level."""
    print("Creating video-level splits...")
    
    # Load video metadata
    video_df = pd.read_csv(METADATA_ROOT / 'df40_video_metadata.csv')
    
    # Separate train and test videos
    train_videos = video_df[video_df['split_source'] == 'train'].copy()
    test_videos = video_df[video_df['split_source'] == 'test'].copy()
    
    # Create unique video identifiers
    train_videos['unique_video_id'] = train_videos['manipulation'] + '_' + train_videos['video_id'].astype(str)
    test_videos['unique_video_id'] = test_videos['manipulation'] + '_' + test_videos['video_id'].astype(str)
    
    print(f"Train unique videos: {train_videos['unique_video_id'].nunique()}")
    print(f"Test unique videos: {test_videos['unique_video_id'].nunique()}")
    
    # For train videos, create train/validation split (80/20)
    # Stratify by label where possible
    train_videos['label_primary'] = train_videos['label'].apply(lambda x: x.split(',')[0] if ',' in str(x) else x)
    
    # Group by manipulation for stratification
    manipulation_groups = train_videos.groupby('manipulation')
    
    train_split_videos = []
    val_split_videos = []
    
    random.seed(42)
    
    for manipulation, group in manipulation_groups:
        # Within each manipulation, split by label if possible
        if len(group) > 1:
            # Simple random split (80/20)
            indices = list(group.index)
            random.shuffle(indices)
            split_point = int(len(indices) * 0.8)
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]
            
            train_split_videos.append(group.loc[train_indices])
            val_split_videos.append(group.loc[val_indices])
        else:
            # If only 1 video, put in train
            train_split_videos.append(group)
    
    train_split_df = pd.concat(train_split_videos, ignore_index=True)
    val_split_df = pd.concat(val_split_videos, ignore_index=True)
    
    print(f"Train split videos: {len(train_split_df)}")
    print(f"Validation split videos: {len(val_split_df)}")
    print(f"Test split videos: {len(test_videos)}")
    
    # Assign split to videos
    train_split_df['split'] = 'train'
    val_split_df['split'] = 'validation'
    test_videos['split'] = 'test'
    
    # Combine all video splits
    all_video_splits = pd.concat([train_split_df, val_split_df, test_videos], ignore_index=True)
    
    # Save video splits
    all_video_splits.to_csv(METADATA_ROOT / 'video_splits.csv', index=False)
    
    # Create split-specific video ID sets for leakage checking
    train_video_ids = set(train_split_df['unique_video_id'].tolist())
    val_video_ids = set(val_split_df['unique_video_id'].tolist())
    test_video_ids = set(test_videos['unique_video_id'].tolist())
    
    # Check for video leakage
    train_val_overlap = train_video_ids & val_video_ids
    train_test_overlap = train_video_ids & test_video_ids
    val_test_overlap = val_video_ids & test_video_ids
    
    print("\nVideo Leakage Check:")
    print(f"Train ∩ Validation: {len(train_val_overlap)} videos")
    print(f"Train ∩ Test: {len(train_test_overlap)} videos")
    print(f"Validation ∩ Test: {len(val_test_overlap)} videos")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("WARNING: VIDEO LEAKAGE DETECTED!")
    else:
        print("Video leakage check: PASS")
    
    # Load sample metadata to assign splits to samples
    sample_df = pd.read_csv(METADATA_ROOT / 'df40_metadata.csv')
    
    # Create unique video key for samples
    sample_df['unique_video_id'] = sample_df['manipulation'] + '_' + sample_df['video_id'].astype(str)
    
    # Create video_id to split mapping using unique identifiers
    video_to_split = {}
    for _, row in all_video_splits.iterrows():
        video_key = row['unique_video_id']
        video_to_split[video_key] = row['split']
    
    # Assign splits to samples
    sample_df['split'] = sample_df['unique_video_id'].map(video_to_split).fillna('unknown')
    
    # Save split information
    sample_df.to_csv(METADATA_ROOT / 'df40_metadata_with_splits.csv', index=False)
    
    # Create individual split files
    train_samples = sample_df[sample_df['split'] == 'train']
    val_samples = sample_df[sample_df['split'] == 'validation']
    test_samples = sample_df[sample_df['split'] == 'test']
    
    train_samples.to_csv(METADATA_ROOT / 'train_split.csv', index=False)
    val_samples.to_csv(METADATA_ROOT / 'validation_split.csv', index=False)
    test_samples.to_csv(METADATA_ROOT / 'test_split.csv', index=False)
    
    print(f"\nSample Split Distribution:")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Analyze class distribution per split
    print("\nClass Distribution by Split:")
    for split_name, split_df in [('train', train_samples), ('validation', val_samples), ('test', test_samples)]:
        class_dist = split_df['class_name'].value_counts()
        total = len(split_df)
        print(f"{split_name}:")
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count} ({count/total*100:.1f}%)")
    
    return {
        'train_videos': len(train_split_df),
        'val_videos': len(val_split_df),
        'test_videos': len(test_videos),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'video_leakage': {
            'train_val': len(train_val_overlap),
            'train_test': len(train_test_overlap),
            'val_test': len(val_test_overlap)
        }
    }

def check_duplicate_leakage():
    """Check for duplicate files across splits."""
    print("\nChecking duplicate leakage across splits...")
    
    # Load duplicates
    duplicates_df = pd.read_csv(METADATA_ROOT / 'duplicates.csv')
    
    if len(duplicates_df) == 0:
        print("No duplicates found - duplicate leakage check: PASS")
        return {'duplicate_leakage': 0}
    
    # Load samples with splits
    samples_df = pd.read_csv(METADATA_ROOT / 'df40_metadata_with_splits.csv')
    
    # Create hash to split mapping using unique video IDs
    hash_to_video_splits = defaultdict(set)
    for _, row in samples_df.iterrows():
        file_hash = row.get('sha256', '')
        unique_video_id = row.get('unique_video_id', '')
        if file_hash and unique_video_id:
            hash_to_video_splits[file_hash].add(unique_video_id)
    
    # Check for duplicates that cross splits at video level
    cross_split_duplicates = 0
    cross_split_hashes = []
    
    for file_hash, video_ids in hash_to_video_splits.items():
        # Get splits for these videos
        splits_for_videos = set()
        for video_id in video_ids:
            split = video_to_split.get(video_id, 'unknown')
            splits_for_videos.add(split)
        
        if len(splits_for_videos) > 1:
            cross_split_duplicates += 1
            cross_split_hashes.append(file_hash)
    
    print(f"Duplicate files crossing splits: {cross_split_duplicates}")
    
    if cross_split_duplicates > 0:
        print("WARNING: DUPLICATE LEAKAGE DETECTED!")
        # Save cross-split duplicates for analysis
        cross_split_dup_info = []
        for file_hash in cross_split_hashes:
            dup_samples = samples_df[samples_df['sha256'] == file_hash]
            splits = dup_samples['split'].unique().tolist()
            cross_split_dup_info.append({
                'sha256': file_hash,
                'splits': ','.join(splits),
                'sample_count': len(dup_samples)
            })
        
        pd.DataFrame(cross_split_dup_info).to_csv(METADATA_ROOT / 'cross_split_duplicates.csv', index=False)
    else:
        print("Duplicate leakage check: PASS")
    
    return {'duplicate_leakage': cross_split_duplicates}

if __name__ == "__main__":
    # Create splits
    split_results = create_video_level_splits()
    
    # Check duplicate leakage
    leakage_results = check_duplicate_leakage()
    
    print("\n" + "=" * 50)
    print("SPLIT CREATION COMPLETE")
    print("=" * 50)
    print(f"Train videos: {split_results['train_videos']}")
    print(f"Validation videos: {split_results['val_videos']}")
    print(f"Test videos: {split_results['test_videos']}")
    print(f"Train samples: {split_results['train_samples']}")
    print(f"Validation samples: {split_results['val_samples']}")
    print(f"Test samples: {split_results['test_samples']}")
    print(f"Video leakage: {split_results['video_leakage']}")
    print(f"Duplicate leakage: {leakage_results['duplicate_leakage']}")