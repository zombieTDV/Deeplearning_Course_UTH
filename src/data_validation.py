import os
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DataValidator:
    """Validate dataset for duplicates, corrupted images, missing data, and label issues."""
    
    def __init__(self, dataset=None, image_dir=None):
        """
        Args:
            dataset: PyTorch Dataset object (optional)
            image_dir: Path to image directory (optional)
        """
        self.dataset = dataset
        self.image_dir = image_dir
        self.duplicates = []
        self.corrupted_images = []
        self.missing_images = []
        self.label_issues = []
        
    def check_corrupted_images(self) -> List[str]:
        """Check for corrupted images that cannot be opened."""
        corrupted = []
        
        if self.dataset is not None:
            # Check dataset images
            for idx in range(len(self.dataset)):
                try:
                    img, label = self.dataset[idx]
                    if isinstance(img, torch.Tensor):
                        # Convert tensor to PIL for validation
                        img = transforms.ToPILImage()(img)
                    else:
                        img = Image.fromarray(img)
                    img.verify()
                except Exception as e:
                    corrupted.append((idx, str(e)))
                    
        elif self.image_dir is not None:
            # Check directory images
            for root, dirs, files in os.walk(self.image_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_path = os.path.join(root, file)
                        try:
                            with Image.open(file_path) as img:
                                img.verify()
                        except Exception as e:
                            corrupted.append((file_path, str(e)))
        
        self.corrupted_images = corrupted
        return corrupted
    
    def detect_duplicates(self, method='phash', threshold=5) -> List[Tuple]:
        """
        Detect duplicate images using perceptual hashing.
        
        Args:
            method: 'phash', 'dhash', 'ahash', or 'whash'
            threshold: Hash distance threshold for considering duplicates
            
        Returns:
            List of tuples (idx1, idx2, distance) for duplicate pairs
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for duplicate detection")
        
        hashes = {}
        duplicates = []
        
        for idx in range(len(self.dataset)):
            try:
                img, label = self.dataset[idx]
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                
                # Compute hash based on method
                if method == 'phash':
                    h = imagehash.phash(img)
                elif method == 'dhash':
                    h = imagehash.dhash(img)
                elif method == 'ahash':
                    h = imagehash.average_hash(img)
                elif method == 'whash':
                    h = imagehash.whash(img)
                else:
                    raise ValueError(f"Unknown hash method: {method}")
                
                # Check for similar hashes
                for existing_idx, existing_h in hashes.items():
                    distance = h - existing_h
                    if distance <= threshold:
                        duplicates.append((existing_idx, idx, distance))
                
                hashes[idx] = h
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
        
        self.duplicates = duplicates
        return duplicates
    
    def check_missing_images(self, expected_count: int = None) -> Dict:
        """
        Check for missing images in dataset.
        
        Args:
            expected_count: Expected number of images (optional)
            
        Returns:
            Dictionary with missing information
        """
        if self.dataset is None and self.image_dir is None:
            raise ValueError("Either dataset or image_dir must be provided")
        
        missing_info = {}
        
        if self.dataset is not None:
            actual_count = len(self.dataset)
            missing_info['expected'] = expected_count
            missing_info['actual'] = actual_count
            missing_info['missing'] = expected_count - actual_count if expected_count else 0
            missing_info['status'] = 'OK' if expected_count is None or actual_count == expected_count else 'MISSING'
            
        elif self.image_dir is not None:
            image_count = 0
            for root, dirs, files in os.walk(self.image_dir):
                image_count += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))
            
            missing_info['expected'] = expected_count
            missing_info['actual'] = image_count
            missing_info['missing'] = expected_count - image_count if expected_count else 0
            missing_info['status'] = 'OK' if expected_count is None or image_count == expected_count else 'MISSING'
        
        self.missing_images = missing_info
        return missing_info
    
    def validate_labels(self, num_classes: int = None) -> Dict:
        """
        Validate labels in dataset.
        
        Args:
            num_classes: Expected number of classes
            
        Returns:
            Dictionary with label validation results
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for label validation")
        
        label_counts = defaultdict(int)
        invalid_labels = []
        
        for idx in range(len(self.dataset)):
            try:
                _, label = self.dataset[idx]
                label_counts[label] += 1
                
                if num_classes is not None and label >= num_classes:
                    invalid_labels.append((idx, label))
                    
            except Exception as e:
                invalid_labels.append((idx, f"Error: {str(e)}"))
        
        validation_result = {
            'num_classes': len(label_counts),
            'class_distribution': dict(label_counts),
            'invalid_labels': invalid_labels,
            'expected_classes': num_classes,
            'status': 'OK' if num_classes is None or len(label_counts) == num_classes else 'MISMATCH'
        }
        
        self.label_issues = validation_result
        return validation_result
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Corrupted images
        report.append(f"\nCorrupted Images: {len(self.corrupted_images)}")
        if self.corrupted_images:
            report.append("Found corrupted images:")
            for item in self.corrupted_images[:10]:  # Show first 10
                report.append(f"  - {item}")
            if len(self.corrupted_images) > 10:
                report.append(f"  ... and {len(self.corrupted_images) - 10} more")
        
        # Duplicates
        report.append(f"\nDuplicate Images: {len(self.duplicates)}")
        if self.duplicates:
            report.append("Found duplicate pairs:")
            for item in self.duplicates[:10]:
                report.append(f"  - Images {item[0]} and {item[1]} (distance: {item[2]})")
            if len(self.duplicates) > 10:
                report.append(f"  ... and {len(self.duplicates) - 10} more")
        
        # Missing images
        if self.missing_images:
            report.append(f"\nMissing Images Check:")
            report.append(f"  Expected: {self.missing_images.get('expected', 'N/A')}")
            report.append(f"  Actual: {self.missing_images.get('actual', 'N/A')}")
            report.append(f"  Missing: {self.missing_images.get('missing', 'N/A')}")
            report.append(f"  Status: {self.missing_images.get('status', 'N/A')}")
        
        # Label validation
        if self.label_issues:
            report.append(f"\nLabel Validation:")
            report.append(f"  Number of classes: {self.label_issues['num_classes']}")
            report.append(f"  Expected classes: {self.label_issues.get('expected_classes', 'N/A')}")
            report.append(f"  Status: {self.label_issues['status']}")
            report.append(f"  Invalid labels: {len(self.label_issues['invalid_labels'])}")
            if self.label_issues['invalid_labels']:
                for item in self.label_issues['invalid_labels'][:5]:
                    report.append(f"    - {item}")
        
        report.append("\n" + "=" * 60)
        
        report_str = "\n".join(report)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_str)
            print(f"Report saved to {save_path}")
        
        return report_str
    
    def remove_duplicates(self, keep: str = 'first') -> List[int]:
        """
        Get indices of images to remove to eliminate duplicates.
        
        Args:
            keep: 'first' to keep first occurrence, 'last' to keep last
            
        Returns:
            List of indices to remove
        """
        if not self.duplicates:
            return []
        
        to_remove = set()
        processed = set()
        
        # Sort duplicates by distance (closest duplicates first)
        sorted_dups = sorted(self.duplicates, key=lambda x: x[2])
        
        for idx1, idx2, _ in sorted_dups:
            if keep == 'first':
                if idx1 not in to_remove and idx2 not in to_remove:
                    to_remove.add(idx2)
            elif keep == 'last':
                if idx1 not in to_remove and idx2 not in to_remove:
                    to_remove.add(idx1)
        
        return sorted(list(to_remove))


def check_dataset_integrity(dataset, num_classes: int = None, expected_count: int = None) -> Dict:
    """
    Quick integrity check for a dataset.
    
    Args:
        dataset: PyTorch Dataset
        num_classes: Expected number of classes
        expected_count: Expected number of samples
        
    Returns:
        Dictionary with integrity check results
    """
    validator = DataValidator(dataset=dataset)
    
    results = {
        'corrupted': validator.check_corrupted_images(),
        'missing': validator.check_missing_images(expected_count),
        'labels': validator.validate_labels(num_classes)
    }
    
    return results
