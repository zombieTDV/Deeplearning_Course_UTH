#!/usr/bin/env python3
"""
DF40 Dataset EDA Visualization Script
Generates EDA visualizations for the DF40 dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter

# Configuration
PROJECT_ROOT = Path("/Users/pickapu/Documents/PyCharmMiscProject/DeepLearning")
DF40_ROOT = PROJECT_ROOT / "src" / "data" / "DF40"
METADATA_ROOT = DF40_ROOT / "metadata"
FIGURES_ROOT = DF40_ROOT / "eda" / "figures"
REPORTS_ROOT = DF40_ROOT / "eda" / "reports"

def create_visualizations():
    """Create EDA visualizations."""
    print("Creating EDA visualizations...")
    
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(METADATA_ROOT / 'df40_metadata_with_splits.csv')
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    class_counts = df['class_name'].value_counts()
    ax.bar(class_counts.index, class_counts.values, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'class_distribution.png', dpi=150)
    plt.close()
    
    # 2. Split Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    split_counts = df['split'].value_counts()
    ax.bar(split_counts.index, split_counts.values, color=['#3498db', '#f39c12', '#9b59b6'])
    ax.set_title('Split Distribution')
    ax.set_ylabel('Count')
    for i, v in enumerate(split_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'split_distribution.png', dpi=150)
    plt.close()
    
    # 3. Manipulation Method Distribution (Top 20)
    fig, ax = plt.subplots(figsize=(12, 6))
    manip_counts = df['manipulation'].value_counts().head(20)
    ax.barh(range(len(manip_counts)), manip_counts.values)
    ax.set_yticks(range(len(manip_counts)))
    ax.set_yticklabels(manip_counts.index)
    ax.set_title('Top 20 Manipulation Methods')
    ax.set_xlabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'manipulation_distribution.png', dpi=150)
    plt.close()
    
    # 4. Resolution Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Width distribution
    ax1.hist(df['width'], bins=30, color='#3498db', alpha=0.7)
    ax1.set_title('Width Distribution')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Count')
    
    # Height distribution
    ax2.hist(df['height'], bins=30, color='#e74c3c', alpha=0.7)
    ax2.set_title('Height Distribution')
    ax2.set_xlabel('Height (pixels)')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'resolution_distribution.png', dpi=150)
    plt.close()
    
    # 5. Aspect Ratio Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['aspect_ratio'], bins=50, color='#9b59b6', alpha=0.7)
    ax.set_title('Aspect Ratio Distribution')
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'aspect_ratio_distribution.png', dpi=150)
    plt.close()
    
    # 6. File Size Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['file_size_bytes'] / 1024, bins=50, color='#f39c12', alpha=0.7)
    ax.set_title('File Size Distribution')
    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'file_size_distribution.png', dpi=150)
    plt.close()
    
    # 7. Class Distribution by Split
    fig, ax = plt.subplots(figsize=(10, 6))
    split_class_dist = pd.crosstab(df['split'], df['class_name'])
    split_class_dist.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Class Distribution by Split')
    ax.set_ylabel('Count')
    ax.legend(title='Class')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'class_by_split_distribution.png', dpi=150)
    plt.close()
    
    # 8. Fake manipulation distribution
    fake_df = df[df['class_name'] == 'FAKE']
    fig, ax = plt.subplots(figsize=(12, 6))
    fake_manip_counts = fake_df['manipulation'].value_counts().head(15)
    ax.barh(range(len(fake_manip_counts)), fake_manip_counts.values)
    ax.set_yticks(range(len(fake_manip_counts)))
    ax.set_yticklabels(fake_manip_counts.index)
    ax.set_title('FAKE Class - Top 15 Manipulation Methods')
    ax.set_xlabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / 'fake_manipulation_distribution.png', dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {FIGURES_ROOT}")

if __name__ == "__main__":
    create_visualizations()