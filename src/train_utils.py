import torch
from typing import Optional, Callable, Dict, Tuple
import copy


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp: bool = False):
    """Train for one epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Current model
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.restore_checkpoint(model)
                return True
        
        return False
    
    def save_checkpoint(self, model):
        """Save current model weights."""
        self.best_weights = copy.deepcopy(model.state_dict())
    
    def restore_checkpoint(self, model):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, 
                verbose=True, val_loader=None, early_stopping=None, 
                scheduler=None, use_amp: bool = False):
    """
    Train model with validation, early stopping, learning rate scheduling, and mixed precision.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        verbose: Whether to print progress
        val_loader: Validation data loader (optional)
        early_stopping: EarlyStopping instance (optional)
        scheduler: Learning rate scheduler (optional)
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with training history
    """
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        history['train_losses'].append(train_loss)
        
        # Validate
        if val_loader is not None:
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_accuracy)
            
            if verbose:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_accuracy:.2f}%')
            
            # Early stopping check
            if early_stopping is not None and early_stopping(val_loss, model):
                if verbose:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                break
        else:
            if verbose:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader is not None else train_loss)
            else:
                scheduler.step()
    
    return history


def get_lr_scheduler(optimizer, scheduler_type: str = 'step', **kwargs):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau', 'exponential')
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 10), 
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
