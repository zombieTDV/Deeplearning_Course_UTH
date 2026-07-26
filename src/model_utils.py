import torch
import torch.nn as nn
import os
from typing import Dict, Optional, Any
import copy


def save_model(model, path='../outputs/model/fashion_mnist_model.pth'):
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model_class, path='../outputs/model/fashion_mnist_model.pth', device='cpu'):
    """Load model from state dict."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f'Model loaded from {path}')
    return model


def save_checkpoint(model, optimizer, epoch, loss, accuracy, 
                   scheduler=None, path='../outputs/checkpoints/checkpoint.pth'):
    """
    Save training checkpoint for resuming training.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        scheduler: Learning rate scheduler (optional)
        path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path} (epoch {epoch})')


def load_checkpoint(model, optimizer, path='../outputs/checkpoints/checkpoint.pth',
                   device='cpu', scheduler=None):
    """
    Load training checkpoint to resume training.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path: Path to checkpoint file
        device: Device to load model on
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Dictionary with checkpoint information (epoch, loss, accuracy)
    """
    if not os.path.exists(path):
        print(f'Checkpoint not found at {path}')
        return None
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f'Checkpoint loaded from {path} (epoch {checkpoint["epoch"]})')
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'accuracy': checkpoint['accuracy']
    }


def save_best_model(model, metric_value, metric_name='accuracy', 
                   mode='max', path='../outputs/model/best_model.pth'):
    """
    Save model if it achieves the best metric value.
    
    Args:
        model: PyTorch model
        metric_value: Current metric value
        metric_name: Name of the metric
        mode: 'max' for higher is better, 'min' for lower is better
        path: Path to save best model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Load previous best if exists
    best_value = None
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            best_value = checkpoint.get(metric_name, None)
        except:
            pass
    
    # Check if current is better
    should_save = False
    if best_value is None:
        should_save = True
    elif mode == 'max' and metric_value > best_value:
        should_save = True
    elif mode == 'min' and metric_value < best_value:
        should_save = True
    
    if should_save:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            metric_name: metric_value
        }
        torch.save(checkpoint, path)
        print(f'Best model saved to {path} ({metric_name}: {metric_value:.4f})')
        return True
    
    return False


def load_best_model(model, path='../outputs/model/best_model.pth', device='cpu'):
    """
    Load best model from checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to best model checkpoint
        device: Device to load model on
    
    Returns:
        Metric value if available, None otherwise
    """
    if not os.path.exists(path):
        print(f'Best model not found at {path}')
        return None
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metric_value = None
    for key in checkpoint.keys():
        if key != 'model_state_dict':
            metric_value = checkpoint[key]
            break
    
    print(f'Best model loaded from {path}')
    return metric_value


class ModelCheckpoint:
    """
    Callback for saving model checkpoints during training.
    """
    
    def __init__(self, save_dir='../outputs/checkpoints', 
                 save_freq: int = 5, save_best: bool = True,
                 metric_name: str = 'accuracy', mode: str = 'max'):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_freq: Save checkpoint every N epochs
            save_best: Whether to save best model separately
            metric_name: Metric name for best model tracking
            mode: 'max' or 'min' for metric comparison
        """
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = None
        self.epoch = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def __call__(self, model, optimizer, epoch, loss, accuracy, 
                 scheduler=None):
        """
        Called at the end of each epoch.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            loss: Current loss
            accuracy: Current accuracy
            scheduler: Learning rate scheduler (optional)
        """
        self.epoch = epoch
        metric_value = accuracy if self.metric_name == 'accuracy' else loss
        
        # Save periodic checkpoint
        if epoch % self.save_freq == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, loss, accuracy, scheduler, path)
        
        # Save best model
        if self.save_best:
            should_update = False
            if self.best_metric is None:
                should_update = True
            elif self.mode == 'max' and metric_value > self.best_metric:
                should_update = True
            elif self.mode == 'min' and metric_value < self.best_metric:
                should_update = True
            
            if should_update:
                self.best_metric = metric_value
                path = os.path.join(self.save_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, loss, accuracy, scheduler, path)
    
    def get_best_metric(self):
        """Get the best metric value achieved."""
        return self.best_metric
