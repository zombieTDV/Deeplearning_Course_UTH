import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
from typing import Optional, Tuple, Dict, Union
import torch
import numpy as np
import cv2
from PIL import Image


def compute_dataset_statistics(dataset) -> Dict[str, np.ndarray]:
    """
    Compute mean and std of dataset for normalization.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in DataLoader(dataset, batch_size=1000):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {'mean': mean.numpy(), 'std': std.numpy()}


def normalize_dataset(dataset, mean: float = None, std: float = None):
    """
    Normalize dataset using specified mean and std.
    
    Args:
        dataset: PyTorch dataset
        mean: Mean for normalization (computed from dataset if None)
        std: Std for normalization (computed from dataset if None)
    
    Returns:
        Dictionary with normalization parameters
    """
    if mean is None or std is None:
        stats = compute_dataset_statistics(dataset)
        mean = stats['mean'][0]
        std = stats['std'][0]
    
    norm_params = {
        'mean': mean,
        'std': std
    }
    
    print("=" * 60)
    print("NORMALIZATION PARAMETERS")
    print("=" * 60)
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print("=" * 60)
    
    return norm_params


def detect_blurry_images(image: Union[np.ndarray, Image.Image], threshold: float = 100) -> bool:
    """
    Detect if an image is blurry using Variance of Laplacian.
    
    Args:
        image: Input image as numpy array or PIL Image
        threshold: Variance threshold below which image is considered blurry
    
    Returns:
        True if image is blurry, False otherwise
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return laplacian_var < threshold


def sharpen_image(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Sharpen an image using a convolution kernel.
    
    Args:
        image: Input image as numpy array or PIL Image
    
    Returns:
        Sharpened image as numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Define sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    # Apply kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def unsharp_mask(image: Union[np.ndarray, Image.Image], 
                 sigma: float = 1.0, 
                 strength: float = 1.5) -> np.ndarray:
    """
    Apply unsharp masking to enhance image details.
    
    Workflow:
        Image -> Gaussian Blur -> Original - Blur -> Enhance Details -> Sharpened Image
    
    Args:
        image: Input image as numpy array or PIL Image
        sigma: Standard deviation for Gaussian blur
        strength: Strength of the sharpening effect
    
    Returns:
        Enhanced image as numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Calculate the difference (high-frequency details)
    details = image - blurred
    
    # Enhance details and add back to original
    sharpened = image + strength * details
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def deblur_image(image: Union[np.ndarray, Image.Image], 
                 method: str = 'unsharp',
                 threshold: float = 100) -> np.ndarray:
    """
    Deblur an image if it is detected as blurry.
    
    Args:
        image: Input image as numpy array or PIL Image
        method: Deblurring method ('unsharp' or 'sharpen')
        threshold: Blur detection threshold
    
    Returns:
        Deblurred image if blurry, otherwise original image
    """
    # Check if image is blurry
    if detect_blurry_images(image, threshold):
        if method == 'unsharp':
            return unsharp_mask(image)
        elif method == 'sharpen':
            return sharpen_image(image)
        else:
            raise ValueError(f"Unknown deblurring method: {method}")
    
    # Return original if not blurry
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


class DeblurTransform:
    """
    Custom transform to apply deblurring to PIL images.
    """
    def __init__(self, method: str = 'unsharp', threshold: float = 100):
        """
        Args:
            method: Deblurring method ('unsharp' or 'sharpen')
            threshold: Blur detection threshold
        """
        self.method = method
        self.threshold = threshold
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply deblurring to image.
        
        Args:
            image: PIL Image
        
        Returns:
            Deblurred PIL Image
        """
        deblurred = deblur_image(image, method=self.method, threshold=self.threshold)
        return Image.fromarray(deblurred)


def build_advanced_transforms(resize: Optional[Tuple[int, int]] = None,
                             normalize: bool = True,
                             augmentation: bool = False,
                             sharpen: bool = False,
                             deblur_method: str = 'unsharp',
                             blur_threshold: float = 100,
                             mean: float = 0.2860,
                             std: float = 0.3205):
    """
    Build advanced transforms with preprocessing options.
    
    Args:
        resize: Target size for resizing (height, width)
        normalize: Apply normalization
        augmentation: Apply data augmentation
        sharpen: Apply deblurring/sharpening to images
        deblur_method: Deblurring method ('unsharp' or 'sharpen')
        blur_threshold: Blur detection threshold
        mean: Mean for normalization
        std: Std for normalization
    
    Returns:
        Composed transform
    
    Pipeline:
        Resize -> Deblur (Optional) -> ToTensor -> Normalize
    """
    transform_list = []
    
    if resize:
        transform_list.append(transforms.Resize(resize))
    
    if sharpen:
        transform_list.append(DeblurTransform(method=deblur_method, threshold=blur_threshold))
    
    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize((mean,), (std,)))
    
    return transforms.Compose(transform_list)


def get_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights for weighted loss function.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Tensor of class weights
    """
    if hasattr(dataset, 'dataset'):
        # Handle random split dataset
        labels = [dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    class_weights = torch.tensor([
        total_samples / (num_classes * count) for count in class_counts.values()
    ], dtype=torch.float32)
    
    return class_weights
