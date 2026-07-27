import torch
import torch.nn as nn


def save_model(model, path='../outputs/practice_1/model/fashion_mnist_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model_class, path='../outputs/practice_1/model/fashion_mnist_model.pth', device='cpu'):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f'Model loaded from {path}')
    return model
