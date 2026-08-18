#!/usr/bin/env python3
"""
DF40 Training Metadata Finalization Script
Creates final training metadata files combining all analysis results.
"""

import pandas as pd
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/")
DF40_ROOT = PROJECT_ROOT / "src" / "data" / "DF40"
METADATA_ROOT = DF40_ROOT / "metadata"

def create_final_training_metadata():
    """Create final training metadata file."""
    print("Creating final training metadata...")
    
    # Load the main metadata with splits
    df = pd.read_csv(METADATA_ROOT / 'df40_metadata_with_splits.csv')
    
    # Filter to only trainable samples
    trainable_df = df[df['is_trainable'] == True].copy()
    
    # Add additional training-specific fields
    trainable_df['eda_label'] = trainable_df['class_name']  # EDA-friendly label name
    trainable_df['numeric_label'] = trainable_df['label'].astype(int)  # Numeric label for training
    
    # Final verification of train/validation/test samples
    train_samples = trainable_df[trainable_df['split'] == 'train']
    val_samples = trainable_df[trainable_df['split'] == 'validation']
    test_samples = trainable_df[trainable_df['split'] == 'test']
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Save final training metadata
    trainable_df.to_csv(METADATA_ROOT / 'final_training_metadata.csv', index=False)
    
    # Create a summary statistics file
    summary_stats = {
        'total_samples': len(trainable_df),
        'total_videos': trainable_df['unique_video_id'].nunique(),
        'train_samples': len(train_samples),
        'validation_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_videos': train_samples['unique_video_id'].nunique(),
        'validation_videos': val_samples['unique_video_id'].nunique(),
        'test_videos': test_samples['unique_video_id'].nunique(),
        'real_samples': len(trainable_df[trainable_df['class_name'] == 'REAL']),
        'fake_samples': len(trainable_df[trainable_df['class_name'] == 'FAKE']),
        'manipulation_methods': trainable_df['manipulation'].nunique(),
        'resolution': f"{trainable_df['width'].iloc[0]}x{trainable_df['height'].iloc[0]}",
        'color_mode': trainable_df['color_mode'].iloc[0],
        'format': trainable_df['file_format'].iloc[0]
    }
    
    # Save summary as JSON
    import json
    with open(METADATA_ROOT / 'training_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Training summary saved")
    
    return summary_stats

if __name__ == "__main__":
    summary = create_final_training_metadata()
    
    print("\n" + "=" * 50)
    print("TRAINING METADATA FINALIZATION COMPLETE")
    print("=" * 50)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Train samples: {summary['train_samples']}")
    print(f"Validation samples: {summary['validation_samples']}")
    print(f"Test samples: {summary['test_samples']}")
    print(f"Real samples: {summary['real_samples']}")
    print(f"Fake samples: {summary['fake_samples']}")
    print(f"Manipulation methods: {summary['manipulation_methods']}")