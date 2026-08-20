#!/usr/bin/env python3
"""
DF40 Dataset Analysis and Metadata Generation Script
Analyzes the complete DF40 dataset structure and generates metadata files.
"""

import os
import csv
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter, defaultdict
import statistics
from typing import Dict, List, Tuple, Optional
import json
import random

# Configuration
PROJECT_ROOT = Path("/Users/pickapu/Documents/PyCharmMiscProject/DeepLearning")
DF40_ROOT = PROJECT_ROOT / "src" / "data" / "DF40"
TRAIN_ROOT = DF40_ROOT / "train"
TEST_ROOT = DF40_ROOT / "test"
METADATA_ROOT = DF40_ROOT / "metadata"
EDA_ROOT = DF40_ROOT / "eda"
FIGURES_ROOT = EDA_ROOT / "figures"
REPORTS_ROOT = EDA_ROOT / "reports"

# Label mapping based on test dataset manifest
LABEL_MAPPING = {
    '0': 'REAL',
    '1': 'FAKE'
}

# Random seed for reproducibility
RANDOM_SEED = 42

class DF40Analyzer:
    def __init__(self):
        self.df40_root = DF40_ROOT
        self.train_root = TRAIN_ROOT
        self.test_root = TEST_ROOT
        self.metadata_root = METADATA_ROOT
        
        # Data storage
        self.sample_metadata = []
        self.video_metadata = {}
        self.quality_issues = []
        self.duplicates = []
        
    def analyze_dataset_structure(self):
        """Analyze the complete dataset structure."""
        print("Analyzing dataset structure...")
        
        # Analyze training data
        train_methods = []
        for item in self.train_root.iterdir():
            if item.is_dir() and item.name not in ['.cache', 'frames']:
                train_methods.append(item.name)
        
        # Analyze test data
        test_structure = {}
        for item in (self.test_root / "test_data_v3").iterdir():
            if item.is_dir():
                test_structure[item.name] = len(list(item.rglob("*.png")))
        
        # Analyze frames directory (likely real samples)
        frames_count = 0
        if (self.train_root / "frames").exists():
            frames_count = len(list((self.train_root / "frames").rglob("*.png")))
        
        structure_info = {
            'train_methods': sorted(train_methods),
            'test_structure': test_structure,
            'frames_count': frames_count,
            'total_train_methods': len(train_methods)
        }
        
        # Save structure info
        with open(self.metadata_root / 'dataset_structure.json', 'w') as f:
            json.dump(structure_info, f, indent=2)
        
        print(f"Found {len(train_methods)} training manipulation methods")
        print(f"Found {len(test_structure)} test manipulation methods")
        print(f"Found {frames_count} frames in real samples directory")
        
        return structure_info
    
    def determine_label_schema(self):
        """Determine the actual label schema from the dataset."""
        print("Determining label schema...")
        
        # Load test manifest for official label schema
        manifest_path = self.test_root / "test_data_v3" / "manifest.csv"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                reader = csv.DictReader(f)
                manifest_rows = list(reader)
            
            print(f"Test manifest contains {len(manifest_rows)} entries")
            print(f"Label schema from manifest: 0=REAL, 1=FAKE")
            
            # Analyze manifest structure
            label_dist = Counter(row['label'] for row in manifest_rows)
            method_dist = Counter(row['method'] for row in manifest_rows)
            
            print(f"Label distribution in test: {dict(label_dist)}")
            print(f"Method distribution in test: {len(method_dist)} methods")
            
            return {
                'label_source': 'test_manifest',
                'label_mapping': LABEL_MAPPING,
                'test_label_distribution': dict(label_dist),
                'test_method_distribution': dict(method_dist)
            }
        else:
            print("No test manifest found - will infer from structure")
            return {
                'label_source': 'inferred',
                'label_mapping': LABEL_MAPPING,
                'note': 'Labels inferred from folder structure (manipulation methods = FAKE)'
            }
    
    def generate_master_metadata(self):
        """Generate master sample metadata for the entire dataset."""
        print("Generating master sample metadata...")
        
        sample_id = 0
        
        # Process training data - manipulation methods (FAKE)
        for method_dir in self.train_root.iterdir():
            if not method_dir.is_dir() or method_dir.name in ['.cache', 'frames']:
                continue
            
            method_name = method_dir.name
            print(f"Processing training method: {method_name}")
            
            for video_dir in method_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                video_id = video_dir.name
                
                for img_file in video_dir.glob("*.png"):
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                            channels = len(img.getbands())
                            color_mode = img.mode
                            file_size = img_file.stat().st_size
                            
                            # Calculate SHA256
                            sha256_hash = hashlib.sha256()
                            with open(img_file, 'rb') as f:
                                for chunk in iter(lambda: f.read(4096), b''):
                                    sha256_hash.update(chunk)
                            file_hash = sha256_hash.hexdigest()
                            
                            self.sample_metadata.append({
                                'sample_id': sample_id,
                                'dataset': 'DF40_train',
                                'split_source': 'train',
                                'label': '1',
                                'class_name': 'FAKE',
                                'manipulation': method_name,
                                'video_id': video_id,
                                'frame_id': img_file.stem,
                                'file_path': str(img_file.relative_to(self.df40_root)),
                                'relative_path': str(img_file.relative_to(self.train_root)),
                                'file_format': 'PNG',
                                'width': width,
                                'height': height,
                                'channels': channels,
                                'color_mode': color_mode,
                                'file_size_bytes': file_size,
                                'sha256': file_hash,
                                'aspect_ratio': round(width / height, 4),
                                'is_readable': True,
                                'is_valid': True,
                                'is_trainable': True
                            })
                            
                            sample_id += 1
                            
                    except Exception as e:
                        self.quality_issues.append({
                            'file_path': str(img_file.relative_to(self.df40_root)),
                            'issue_type': 'read_error',
                            'error_message': str(e),
                            'is_trainable': False
                        })
        
        # Process frames directory (likely REAL samples)
        frames_dir = self.train_root / "frames"
        if frames_dir.exists():
            print("Processing frames directory (REAL samples)...")
            
            for video_dir in frames_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                video_id = video_dir.name
                
                for img_file in video_dir.glob("*.png"):
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                            channels = len(img.getbands())
                            color_mode = img.mode
                            file_size = img_file.stat().st_size
                            
                            # Calculate SHA256
                            sha256_hash = hashlib.sha256()
                            with open(img_file, 'rb') as f:
                                for chunk in iter(lambda: f.read(4096), b''):
                                    sha256_hash.update(chunk)
                            file_hash = sha256_hash.hexdigest()
                            
                            self.sample_metadata.append({
                                'sample_id': sample_id,
                                'dataset': 'DF40_train',
                                'split_source': 'train',
                                'label': '0',
                                'class_name': 'REAL',
                                'manipulation': 'real',
                                'video_id': video_id,
                                'frame_id': img_file.stem,
                                'file_path': str(img_file.relative_to(self.df40_root)),
                                'relative_path': str(img_file.relative_to(self.train_root)),
                                'file_format': 'PNG',
                                'width': width,
                                'height': height,
                                'channels': channels,
                                'color_mode': color_mode,
                                'file_size_bytes': file_size,
                                'sha256': file_hash,
                                'aspect_ratio': round(width / height, 4),
                                'is_readable': True,
                                'is_valid': True,
                                'is_trainable': True
                            })
                            
                            sample_id += 1
                            
                    except Exception as e:
                        self.quality_issues.append({
                            'file_path': str(img_file.relative_to(self.df40_root)),
                            'issue_type': 'read_error',
                            'error_message': str(e),
                            'is_trainable': False
                        })
        
        # Process test data using manifest
        manifest_path = self.test_root / "test_data_v3" / "manifest.csv"
        if manifest_path.exists():
            print("Processing test data using manifest...")
            
            with open(manifest_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = self.test_root / "test_data_v3" / row['path']
                    
                    if img_path.exists():
                        try:
                            with Image.open(img_path) as img:
                                width, height = img.size
                                channels = len(img.getbands())
                                color_mode = img.mode
                                file_size = img_path.stat().st_size
                                
                                # Calculate SHA256
                                sha256_hash = hashlib.sha256()
                                with open(img_path, 'rb') as f:
                                    for chunk in iter(lambda: f.read(4096), b''):
                                        sha256_hash.update(chunk)
                                file_hash = sha256_hash.hexdigest()
                                
                                label_name = LABEL_MAPPING.get(row['label'], 'UNKNOWN')
                                
                                self.sample_metadata.append({
                                    'sample_id': sample_id,
                                    'dataset': 'df40-test-data-v3',
                                    'split_source': 'test',
                                    'label': row['label'],
                                    'class_name': label_name,
                                    'manipulation': row['method'],
                                    'video_id': row['video'],
                                    'frame_id': img_path.stem,
                                    'file_path': str(img_path.relative_to(self.df40_root)),
                                    'relative_path': str(img_path.relative_to(self.test_root)),
                                    'file_format': 'PNG',
                                    'width': width,
                                    'height': height,
                                    'channels': channels,
                                    'color_mode': color_mode,
                                    'file_size_bytes': file_size,
                                    'sha256': file_hash,
                                    'aspect_ratio': round(width / height, 4),
                                    'is_readable': True,
                                    'is_valid': True,
                                    'is_trainable': True
                                })
                                
                                sample_id += 1
                                
                        except Exception as e:
                            self.quality_issues.append({
                                'file_path': str(img_path.relative_to(self.df40_root)),
                                'issue_type': 'read_error',
                                'error_message': str(e),
                                'is_trainable': False
                            })
        
        print(f"Generated metadata for {len(self.sample_metadata)} samples")
        
        # Save master metadata
        df = pd.DataFrame(self.sample_metadata)
        df.to_csv(self.metadata_root / 'df40_metadata.csv', index=False)
        
        return df
    
    def generate_video_metadata(self):
        """Generate video-level metadata."""
        print("Generating video-level metadata...")
        
        video_data = defaultdict(lambda: {
            'frame_count': 0,
            'valid_frame_count': 0,
            'labels': set(),
            'manipulations': set(),
            'widths': [],
            'heights': [],
            'file_sizes': []
        })
        
        for sample in self.sample_metadata:
            video_key = (sample['split_source'], sample['manipulation'], sample['video_id'])
            
            video_data[video_key]['frame_count'] += 1
            if sample['is_trainable']:
                video_data[video_key]['valid_frame_count'] += 1
            video_data[video_key]['labels'].add(sample['label'])
            video_data[video_key]['manipulations'].add(sample['manipulation'])
            video_data[video_key]['widths'].append(sample['width'])
            video_data[video_key]['heights'].append(sample['height'])
            video_data[video_key]['file_sizes'].append(sample['file_size_bytes'])
        
        # Convert to structured format
        video_metadata = []
        for video_key, data in video_data.items():
            split_source, manipulation, video_id = video_key
            
            video_metadata.append({
                'dataset': 'DF40' if split_source == 'train' else 'df40-test-data-v3',
                'split_source': split_source,
                'video_id': video_id,
                'manipulation': manipulation,
                'label': ','.join(sorted(data['labels'])),
                'frame_count': data['frame_count'],
                'valid_frame_count': data['valid_frame_count'],
                'invalid_frame_count': data['frame_count'] - data['valid_frame_count'],
                'avg_width': statistics.mean(data['widths']) if data['widths'] else 0,
                'avg_height': statistics.mean(data['heights']) if data['heights'] else 0,
                'avg_file_size': statistics.mean(data['file_sizes']) if data['file_sizes'] else 0
            })
        
        # Save video metadata
        df = pd.DataFrame(video_metadata)
        df.to_csv(self.metadata_root / 'df40_video_metadata.csv', index=False)
        
        print(f"Generated metadata for {len(video_metadata)} videos")
        
        return df
    
    def analyze_class_distribution(self):
        """Analyze class and manipulation distribution."""
        print("Analyzing class distribution...")
        
        # Class distribution
        class_dist = Counter(sample['class_name'] for sample in self.sample_metadata)
        
        # Manipulation distribution
        manipulation_dist = Counter(sample['manipulation'] for sample in self.sample_metadata)
        
        # Split distribution
        split_dist = Counter(sample['split_source'] for sample in self.sample_metadata)
        
        # Save distributions
        class_df = pd.DataFrame([
            {'class_name': class_name, 'count': count, 
             'percentage': round(count / len(self.sample_metadata) * 100, 2)}
            for class_name, count in class_dist.items()
        ])
        
        manipulation_df = pd.DataFrame([
            {'manipulation': method, 'count': count, 
             'percentage': round(count / len(self.sample_metadata) * 100, 2)}
            for method, count in manipulation_dist.most_common()
        ])
        
        class_df.to_csv(self.metadata_root / 'class_distribution.csv', index=False)
        manipulation_df.to_csv(self.metadata_root / 'manipulation_distribution.csv', index=False)
        
        print(f"Class distribution: {dict(class_dist)}")
        print(f"Manipulation methods: {len(manipulation_dist)}")
        print(f"Split distribution: {dict(split_dist)}")
        
        return class_dist, manipulation_dist, split_dist
    
    def detect_duplicates(self):
        """Detect duplicate files using SHA256 hashes."""
        print("Detecting duplicates...")
        
        hash_map = defaultdict(list)
        
        for sample in self.sample_metadata:
            file_hash = sample.get('sha256', '')
            if file_hash:
                hash_map[file_hash].append(sample)
        
        # Find duplicates
        for file_hash, samples in hash_map.items():
            if len(samples) > 1:
                for sample in samples:
                    self.duplicates.append({
                        'sha256': file_hash,
                        'sample_id': sample['sample_id'],
                        'file_path': sample['file_path'],
                        'video_id': sample['video_id'],
                        'manipulation': sample['manipulation'],
                        'split_source': sample['split_source'],
                        'duplicate_count': len(samples)
                    })
        
        # Save duplicates
        if self.duplicates:
            df = pd.DataFrame(self.duplicates)
            df.to_csv(self.metadata_root / 'duplicates.csv', index=False)
            print(f"Found {len(self.duplicates)} duplicate samples")
        else:
            print("No duplicates found")
            # Create empty file
            pd.DataFrame(columns=['sha256', 'sample_id', 'file_path', 'video_id', 'manipulation', 'split_source', 'duplicate_count']).to_csv(self.metadata_root / 'duplicates.csv', index=False)
        
        return self.duplicates
    
    def generate_trainable_samples(self):
        """Generate list of trainable samples."""
        print("Generating trainable samples list...")
        
        trainable_samples = [
            sample for sample in self.sample_metadata if sample['is_trainable']
        ]
        
        df = pd.DataFrame(trainable_samples)
        df.to_csv(self.metadata_root / 'trainable_samples.csv', index=False)
        
        print(f"Trainable samples: {len(trainable_samples)}/{len(self.sample_metadata)}")
        
        return df
    
    def create_video_level_splits(self):
        """Create train/validation/test splits at VIDEO level."""
        print("Creating video-level splits...")
        
        # Load video metadata
        video_df = pd.read_csv(self.metadata_root / 'df40_video_metadata.csv')
        
        # Separate train and test videos
        train_videos = video_df[video_df['split_source'] == 'train'].copy()
        test_videos = video_df[video_df['split_source'] == 'test'].copy()
        
        # Create unique video identifiers
        train_videos['unique_video_id'] = train_videos['manipulation'] + '_' + train_videos['video_id'].astype(str)
        test_videos['unique_video_id'] = test_videos['manipulation'] + '_' + test_videos['video_id'].astype(str)
        
        print(f"Train unique videos: {train_videos['unique_video_id'].nunique()}")
        print(f"Test unique videos: {test_videos['unique_video_id'].nunique()}")
        
        # For train videos, create train/validation split (80/20)
        train_videos['label_primary'] = train_videos['label'].apply(lambda x: str(x).split(',')[0] if ',' in str(x) else str(x))
        
        # Group by manipulation for stratification
        manipulation_groups = train_videos.groupby('manipulation')
        
        train_split_videos = []
        val_split_videos = []
        
        random.seed(RANDOM_SEED)
        
        for manipulation, group in manipulation_groups:
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
        all_video_splits.to_csv(self.metadata_root / 'video_splits.csv', index=False)
        
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
        
        # Create video_id to split mapping (ensure consistent string types)
        video_to_split = {}
        for _, row in all_video_splits.iterrows():
            video_key = str(row['unique_video_id'])
            video_to_split[video_key] = row['split']
        
        # Load sample metadata to assign splits to samples
        sample_df = pd.read_csv(self.metadata_root / 'df40_metadata.csv')
        
        # Create unique video key for samples (ensure consistent string types)
        sample_df['unique_video_id'] = sample_df['manipulation'] + '_' + sample_df['video_id'].astype(str)
        sample_df['unique_video_id'] = sample_df['unique_video_id'].astype(str)
        
        # Assign splits to samples
        sample_df['split'] = sample_df['unique_video_id'].map(video_to_split)
        
        # Check for unmapped samples and assign based on split_source as fallback
        unmapped_mask = sample_df['split'].isna()
        if unmapped_mask.any():
            print(f"Warning: {unmapped_mask.sum()} samples could not be mapped to video splits")
            print("Assigning splits based on split_source as fallback")
            sample_df.loc[unmapped_mask & (sample_df['split_source'] == 'train'), 'split'] = 'train'
            sample_df.loc[unmapped_mask & (sample_df['split_source'] == 'test'), 'split'] = 'test'
        
        # Ensure no remaining unknown splits
        assert sample_df['split'].isna().sum() == 0, "Some samples still have no split assignment"
        
        # Save split information
        sample_df.to_csv(self.metadata_root / 'df40_metadata_with_splits.csv', index=False)
        
        # Create individual split files
        train_samples = sample_df[sample_df['split'] == 'train']
        val_samples = sample_df[sample_df['split'] == 'validation']
        test_samples = sample_df[sample_df['split'] == 'test']
        
        train_samples.to_csv(self.metadata_root / 'train_split.csv', index=False)
        val_samples.to_csv(self.metadata_root / 'validation_split.csv', index=False)
        test_samples.to_csv(self.metadata_root / 'test_split.csv', index=False)
        
        print(f"\nSample Split Distribution:")
        print(f"Train samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        print(f"Test samples: {len(test_samples)}")
        
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
            },
            'video_to_split': video_to_split
        }
    
    def finalize_metadata(self):
        """Create final training metadata."""
        print("Creating final training metadata...")
        
        # Load the main metadata with splits
        df = pd.read_csv(self.metadata_root / 'df40_metadata_with_splits.csv')
        
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
        trainable_df.to_csv(self.metadata_root / 'final_training_metadata.csv', index=False)
        
        # Create a summary statistics file
        summary_stats = {
            'total_samples': len(trainable_df),
            'total_videos': trainable_df['unique_video_id'].nunique(),
            'train_samples': len(train_samples),
            'validation_samples': len(val_samples),
            'test_samples': len(test_samples),
            'real_samples': len(trainable_df[trainable_df['class_name'] == 'REAL']),
            'fake_samples': len(trainable_df[trainable_df['class_name'] == 'FAKE']),
            'manipulation_methods': trainable_df['manipulation'].nunique(),
            'resolution': f"{trainable_df['width'].iloc[0]}x{trainable_df['height'].iloc[0]}",
            'color_mode': trainable_df['color_mode'].iloc[0],
            'format': trainable_df['file_format'].iloc[0]
        }
        
        # Save summary as JSON
        with open(self.metadata_root / 'training_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Training summary saved")
        
        return summary_stats
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 50)
        print("DF40 DATASET COMPREHENSIVE ANALYSIS")
        print("=" * 50)
        
        # Step 1: Analyze structure
        structure_info = self.analyze_dataset_structure()
        
        # Step 2: Determine label schema
        label_schema = self.determine_label_schema()
        
        # Step 3: Generate master metadata
        master_df = self.generate_master_metadata()
        
        # Step 4: Generate video metadata
        video_df = self.generate_video_metadata()
        
        # Step 5: Analyze distributions
        class_dist, manipulation_dist, split_dist = self.analyze_class_distribution()
        
        # Step 6: Detect duplicates
        duplicates = self.detect_duplicates()
        
        # Step 7: Generate trainable samples
        trainable_df = self.generate_trainable_samples()
        
        # Step 8: Create video-level splits
        split_results = self.create_video_level_splits()
        
        # Step 9: Finalize metadata
        summary = self.finalize_metadata()
        
        print("=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total samples: {len(self.sample_metadata)}")
        print(f"Total videos: {len(video_df)}")
        print(f"Class distribution: {dict(class_dist)}")
        print(f"Manipulation distribution: {dict(manipulation_dist)}")
        print(f"Split distribution: {dict(split_dist)}")
        print(f"Duplicates: {len(duplicates)}")
        print(f"Quality issues: {len(self.quality_issues)}")
        print(f"Trainable samples: {len(trainable_df)}")
        
        return {
            'structure_info': structure_info,
            'label_schema': label_schema,
            'total_samples': len(self.sample_metadata),
            'total_videos': len(video_df),
            'class_distribution': dict(class_dist),
            'manipulation_distribution': dict(manipulation_dist),
            'split_distribution': dict(split_dist),
            'duplicates': len(duplicates),
            'quality_issues': len(self.quality_issues),
            'trainable_samples': len(trainable_df)
        }

if __name__ == "__main__":
    analyzer = DF40Analyzer()
    results = analyzer.run_full_analysis()