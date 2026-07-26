# ── Optuna HP search — drop into any notebook ──────────────────────
# pip install optuna  (if missing)
# Study auto-saved to outputs/error_analysis/optuna_study/ — can resume
# ────────────────────────────────────────────────────────────────────
import sys, os, warnings, json
sys.path.append('..')
warnings.filterwarnings('ignore')

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from src.train_utils import train_one_epoch
from src.eval_utils import evaluate

# ── config ──────────────────────────────────────────────────────────
N_TRIALS = 30
N_EPOCHS = 35
VAL_SPLIT = 0.1
SEED = 42
STUDY_DIR = '../outputs/error_analysis/optuna_study'
os.makedirs(STUDY_DIR, exist_ok=True)

device = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  Trials: {N_TRIALS}  |  Epochs/trial: {N_EPOCHS}')

# ── data (shared across all trials) ─────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
full_train = __import__('torchvision').datasets.FashionMNIST(
    root='../data', train=True, download=True, transform=transform)
test_ds = __import__('torchvision').datasets.FashionMNIST(
    root='../data', train=False, download=True, transform=transform)

val_len = int(len(full_train) * VAL_SPLIT)
train_ds, val_ds = random_split(
    full_train, [len(full_train) - val_len, val_len],
    generator=torch.Generator().manual_seed(SEED))
print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

# ── model ───────────────────────────────────────────────────────────
class DiagnosticCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

# ── objective ───────────────────────────────────────────────────────
def objective(trial):
    # sample
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    opt_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    sched_name = trial.suggest_categorical('lr_schedule',
                                           ['None', 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])

    # data
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # model + optimizer
    model = DiagnosticCNN(dropout=dropout).to(device)
    opt_cls = {'Adam': optim.Adam, 'AdamW': optim.AdamW, 'SGD': lambda p, **kw: optim.SGD(p, momentum=0.9, nesterov=True, **kw)}[opt_name]
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # scheduler
    if sched_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=max(1, N_EPOCHS // 3), gamma=0.1)
    elif sched_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    elif sched_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        scheduler = None

    # training loop with tracking for ReduceLROnPlateau
    best_val_acc = 0.0
    for epoch in range(N_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        model.train()
        val_acc = correct / total

        # scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

        # prune
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc

# ── run ─────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    study_name='fashion_mnist_diagnostic_cnn',
    storage=f'sqlite:///{STUDY_DIR}/optuna_study.db',
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS)

# ── results ─────────────────────────────────────────────────────────
print(f'\n{"="*65}\nBest trial #{study.best_trial.number}  |  Val-acc: {study.best_value:.4f}')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

# retrain best on full training set → test
bp = study.best_params
model = DiagnosticCNN(dropout=bp['dropout']).to(device)
opt_cls = {'Adam': optim.Adam, 'AdamW': optim.AdamW,
           'SGD': lambda p, **kw: optim.SGD(p, momentum=0.9, nesterov=True, **kw)}[bp['optimizer']]
optimizer = opt_cls(model.parameters(), lr=bp['lr'], weight_decay=bp['weight_decay'])

full_loader = DataLoader(full_train, batch_size=bp['batch_size'], shuffle=True)
for epoch in range(N_EPOCHS):
    train_one_epoch(model, full_loader, nn.CrossEntropyLoss(), optimizer, device)

test_acc = evaluate(model, DataLoader(test_ds, batch_size=256, shuffle=False), device)
print(f'Test accuracy: {test_acc:.2f}%')

# importance
try:
    imp = optuna.importance.get_param_importances(study)
    print('\nHyperparameter importance:')
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f'  {k:<20}  {v:.4f}')
except Exception:
    pass

# save
with open(os.path.join(STUDY_DIR, 'best_params.json'), 'w') as f:
    json.dump({**bp, 'test_accuracy': test_acc, 'best_val_accuracy': study.best_value}, f, indent=2)
print(f'\nResults saved to {STUDY_DIR}/')
print(f'Resume: study.optimize(objective, n_trials=N_TRIALS + 30)')
