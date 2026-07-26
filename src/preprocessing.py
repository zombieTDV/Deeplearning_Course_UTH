import cv2
import numpy as np
from PIL import Image
import torch
from typing import Union, Tuple, Optional
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Advanced image preprocessing for computer vision tasks."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def gaussian_blur(image: Union[np.ndarray, Image.Image], kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            image: Input image (numpy array or PIL Image)
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of the Gaussian kernel
            
        Returns:
            Blurred image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        else:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def median_filter(image: Union[np.ndarray, Image.Image], kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter to remove salt-and-pepper noise.
        
        Args:
            image: Input image (numpy array or PIL Image)
            kernel_size: Size of the median filter kernel (must be odd)
            
        Returns:
            Filtered image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_filter(image: Union[np.ndarray, Image.Image], d: int = 9, sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving denoising.
        
        Args:
            image: Input image (numpy array or PIL Image)
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            return filtered
        else:
            return cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def histogram_equalization(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Equalized image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    
    @staticmethod
    def clahe(image: Union[np.ndarray, Image.Image], clip_limit: float = 2.0, 
              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image (numpy array or PIL Image)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE enhanced image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    @staticmethod
    def contrast_stretching(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply contrast stretching to improve image contrast.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Contrast stretched image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Calculate min and max pixel values
        min_val = np.min(image)
        max_val = np.max(image)
        
        # Stretch contrast
        stretched = ((image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        
        return stretched
    
    @staticmethod
    def denoise(image: Union[np.ndarray, Image.Image], method: str = 'gaussian', **kwargs) -> np.ndarray:
        """
        Apply denoising using specified method.
        
        Args:
            image: Input image (numpy array or PIL Image)
            method: Denoising method ('gaussian', 'median', 'bilateral', 'nlm')
            **kwargs: Additional parameters for the chosen method
            
        Returns:
            Denoised image as numpy array
        """
        if method == 'gaussian':
            return ImagePreprocessor.gaussian_blur(image, **kwargs)
        elif method == 'median':
            return ImagePreprocessor.median_filter(image, **kwargs)
        elif method == 'bilateral':
            return ImagePreprocessor.bilateral_filter(image, **kwargs)
        elif method == 'nlm':
            if isinstance(image, Image.Image):
                image = np.array(image)
            if len(image.shape) == 2:
                return cv2.fastNlMeansDenoising(image, **kwargs)
            else:
                return cv2.fastNlMeansDenoisingColored(image, **kwargs)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    @staticmethod
    def enhance_contrast(image: Union[np.ndarray, Image.Image], method: str = 'clahe', **kwargs) -> np.ndarray:
        """
        Enhance image contrast using specified method.
        
        Args:
            image: Input image (numpy array or PIL Image)
            method: Enhancement method ('hist_eq', 'clahe', 'stretch')
            **kwargs: Additional parameters for the chosen method
            
        Returns:
            Enhanced image as numpy array
        """
        if method == 'hist_eq':
            return ImagePreprocessor.histogram_equalization(image)
        elif method == 'clahe':
            return ImagePreprocessor.clahe(image, **kwargs)
        elif method == 'stretch':
            return ImagePreprocessor.contrast_stretching(image)
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
    
    @staticmethod
    def sharpen(image: Union[np.ndarray, Image.Image], kernel: np.ndarray = None) -> np.ndarray:
        """
        Apply sharpening filter to image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            kernel: Custom sharpening kernel (optional)
            
        Returns:
            Sharpened image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if kernel is None:
            # Default sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def resize_with_aspect_ratio(image: Union[np.ndarray, Image.Image], target_size: Tuple[int, int],
                                 keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image while optionally maintaining aspect ratio.
        
        Args:
            image: Input image (numpy array or PIL Image)
            target_size: Target size (width, height)
            keep_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            padded = cv2.copyMakeBorder(resized, pad_h // 2, pad_h - pad_h // 2,
                                       pad_w // 2, pad_w - pad_w // 2,
                                       cv2.BORDER_CONSTANT, value=0)
            return padded
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


class PreprocessingTransform:
    """PyTorch transform wrapper for preprocessing operations."""
    
    def __init__(self, preprocess_func, **kwargs):
        """
        Args:
            preprocess_func: Preprocessing function from ImagePreprocessor
            **kwargs: Arguments to pass to the preprocessing function
        """
        self.preprocess_func = preprocess_func
        self.kwargs = kwargs
    
    def __call__(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Apply preprocessing to image.
        
        Args:
            image: Input image (torch tensor or PIL Image)
            
        Returns:
            Preprocessed image as torch tensor
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            image = transforms.ToPILImage()(image)
        
        # Apply preprocessing
        processed = self.preprocess_func(image, **self.kwargs)
        
        # Convert back to tensor
        if isinstance(processed, np.ndarray):
            if len(processed.shape) == 2:
                processed = np.expand_dims(processed, axis=-1)
            processed = Image.fromarray(processed)
        
        return transforms.ToTensor()(processed)


def get_preprocessing_transform(pipeline: list) -> transforms.Compose:
    """
    Create a preprocessing pipeline transform.
    
    Args:
        pipeline: List of preprocessing steps, each as a tuple (method, kwargs)
                  Example: [('denoise', {'method': 'gaussian', 'kernel_size': 3}),
                            ('enhance_contrast', {'method': 'clahe', 'clip_limit': 2.0})]
    
    Returns:
        Composed transform
    """
    transforms_list = []
    
    for step in pipeline:
        method_name, kwargs = step
        
        if method_name == 'denoise':
            transforms_list.append(PreprocessingTransform(ImagePreprocessor.denoise, **kwargs))
        elif method_name == 'enhance_contrast':
            transforms_list.append(PreprocessingTransform(ImagePreprocessor.enhance_contrast, **kwargs))
        elif method_name == 'sharpen':
            transforms_list.append(PreprocessingTransform(ImagePreprocessor.sharpen, **kwargs))
        elif method_name == 'resize':
            transforms_list.append(transforms.Resize(kwargs.get('size', (28, 28))))
        else:
            raise ValueError(f"Unknown preprocessing method: {method_name}")
    
    return transforms.Compose(transforms_list)


def apply_preprocessing_to_tensor(tensor: torch.Tensor, pipeline: list) -> torch.Tensor:
    """
    Apply preprocessing pipeline to a tensor.
    
    Args:
        tensor: Input tensor (C, H, W)
        pipeline: List of preprocessing steps
        
    Returns:
        Preprocessed tensor
    """
    transform = get_preprocessing_transform(pipeline)
    return transform(tensor)
