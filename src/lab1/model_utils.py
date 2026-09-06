from pathlib import Path
import torch
import torch.nn as nn

_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / 'experiments' / 'lab1' / 'practice_1' / 'model' / 'fashion_mnist_model.pth')


def save_model(model, path=_DEFAULT_MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model_class, path=_DEFAULT_MODEL_PATH, device='cpu'):
    if not Path(path).exists() and Path(_DEFAULT_MODEL_PATH).exists():
        path = _DEFAULT_MODEL_PATH
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print(f'Model loaded from {path}')
    return model
